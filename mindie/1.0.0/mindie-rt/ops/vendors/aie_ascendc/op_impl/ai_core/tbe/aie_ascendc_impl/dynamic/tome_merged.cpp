/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"
using namespace AscendC;
namespace {
#if defined(__DAV_C220_VEC__)
// 12288 * 12 B available memory in 910B
constexpr uint64_t MAX_UB_MEMORY = 12288 * 12;
#else
// 12288 * 16 B available memory in 310P
constexpr uint64_t MAX_UB_MEMORY = 12288 * 16;
#endif
// 16 half nums equal to 32B
constexpr uint32_t BLOCK_SIZE = 16;
// Repeat size is 128
constexpr uint32_t REPEAT_SIZE = 128;
// Default repeat stride is 8
constexpr uint32_t DEFAULT_REPEAT_STRIDE = 8;
// Vector Mask of fp16 is 128
constexpr uint32_t FP16_MASK = 128;
// Vector Mask of fp32 is 64
constexpr uint32_t FP32_MASK = 64;
// Reduce Num is 8
constexpr uint64_t HEADS = 8;
// Repeat time is 255 at most
constexpr uint32_t DUP_REPEAT_MAX = 255;
constexpr int64_t MAX_BATCH = 64;
constexpr int64_t MAX_SEQLENA = 4096;
constexpr int64_t MAX_SEQLENB = 4096;
}

namespace {
class TokenMerged {
public:
    __aicore__ inline TokenMerged(uint64_t batch, uint64_t hiddenSize, uint64_t topR,
        uint64_t seqlenA, uint64_t seqlenB)
    {
        this->batch = batch;
        this->hiddenSize = hiddenSize;
        this->topR = topR;
        this->seqlenA = seqlenA;
        this->seqlenB = seqlenB;
        this->afterMergedLenA = seqlenA - topR;
        this->seqlenAD128 = (seqlenA + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
        this->seqlenBD128 = (seqlenB + REPEAT_SIZE - 1) / REPEAT_SIZE * REPEAT_SIZE;
#if defined(__DAV_M200__)
        // 2 tensors of length seqlenAD128, 1 tensor of length seqlenBD128
        this->usedMemory = seqlenBD128 * sizeof(float) + seqlenAD128 * sizeof(int64_t) * 2;
#else
        this->usedMemory = seqlenBD128 * sizeof(float);
#endif
        if (usedMemory > MAX_UB_MEMORY) {
            valid = false;
            return;
        }
        this->maxCacheMemory = MAX_UB_MEMORY - usedMemory;
        this->colsBase = (maxCacheMemory / sizeof(half) / hiddenSize / BLOCK_SIZE) * BLOCK_SIZE;
        if (colsBase == 0) {
            valid = false;
            return;
        }
        // 2 for half of colsBase
        this->colsBaseH = colsBase / 2;
        this->blockX = block_idx / HEADS;
        this->blockY = block_idx % HEADS;
    }

    __aicore__ inline void Init(__gm__ uint8_t *tokenA, __gm__ uint8_t *tokenB, __gm__ uint8_t *topkIndice,
        __gm__ uint8_t *argMax, __gm__ uint8_t *mergedToken, __gm__ uint8_t *unreducedToken,
        __gm__ uint8_t *unreducedCount)
    {
        if (tokenA == nullptr || tokenB == nullptr || topkIndice == nullptr || argMax == nullptr ||
            mergedToken == nullptr || unreducedToken == nullptr || unreducedCount == nullptr) {
            return;
        }
        tokenAGm = (__gm__ half *)tokenA + blockX * seqlenA * hiddenSize;
        tokenBGm = (__gm__ half *)tokenB + blockX * seqlenB * hiddenSize;
        argMaxGm = (__gm__ int64_t *)argMax + blockX * seqlenA;
        topkIndiceGm = (__gm__ int64_t *)topkIndice + blockX * seqlenA;

        mergedTokenGm = (__gm__ half *)mergedToken + blockX * afterMergedLenA * hiddenSize;
        unreducedTokenGm = (__gm__ half *)unreducedToken + blockX * HEADS * seqlenB * hiddenSize;
        unreducedCountGm = (__gm__ float *)unreducedCount + blockX * HEADS * seqlenB;

        pipe.InitBuffer(cacheQueue, 1, maxCacheMemory);
        LocalTensor<half> cacheMemory = cacheQueue.AllocTensor<half>();
        commonUbuf = (__ubuf__ half *)cacheMemory.GetPhyAddr();

        pipe.InitBuffer(countQueue, 1, (seqlenBD128 * sizeof(float)));
        LocalTensor<float> countMemory = countQueue.AllocTensor<float>();
        countUbuf = (__ubuf__ float *)countMemory.GetPhyAddr();
#if defined(__DAV_M200__)
        pipe.InitBuffer(indiceQueue, 1, (seqlenAD128 * sizeof(int64_t)));
        LocalTensor<int64_t> indiceMemory = indiceQueue.AllocTensor<int64_t>();
        indiceUbuf = (__ubuf__ int64_t *)indiceMemory.GetPhyAddr();

        pipe.InitBuffer(argmaxQueue, 1, (seqlenAD128 * sizeof(int64_t)));
        LocalTensor<int64_t> argmaxMemory = argmaxQueue.AllocTensor<int64_t>();
        argmaxUbuf = (__ubuf__ int64_t *)argmaxMemory.GetPhyAddr();
#endif
    }

    __aicore__ inline void Process()
    {
        /* stage 0: init unreducedToken and unreducedCount with zero */
        InitGm();
        /* stage 1: copy unmerge part of tokenA and the whole tokenB to dst */
        CopyUnmerged();
        /* stage 2: move and add data from tokenAGm to unreducedToken and update count */
        CopyReduced();
    }

private:
    __aicore__ inline void InitGm()
    {
        uint64_t dupTimes = colsBase * hiddenSize / FP16_MASK;
        uint64_t dupRepeat = (dupTimes + DUP_REPEAT_MAX - 1) / DUP_REPEAT_MAX;
        uint64_t dupRemain = dupTimes % DUP_REPEAT_MAX;

        for (uint64_t i = 0; i < dupRepeat; ++i) {
            uint64_t curDup = ((dupRemain != 0) && (i == dupRepeat - 1)) ? dupRemain : DUP_REPEAT_MAX;
            vector_dup(commonUbuf + i * DUP_REPEAT_MAX * FP16_MASK, (half)0.0, curDup,
                       1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        }

        vector_dup(countUbuf, (float)0.0, seqlenBD128 / FP32_MASK, 1, 1,
                   DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        pipe_barrier(PIPE_ALL);

        uint64_t fillRepeat = (seqlenB + colsBase - 1) / colsBase;
        uint64_t fillRemain = seqlenB % colsBase;
        for (uint64_t i = 0; i < fillRepeat; ++i) {
            uint64_t curCol = ((fillRemain != 0) && (i == fillRepeat - 1)) ? fillRemain : colsBase;
            copy_ubuf_to_gm(unreducedTokenGm + blockY * seqlenB * hiddenSize + i * colsBase * hiddenSize,
                            commonUbuf,
                            0,
                            1,
                            curCol * hiddenSize / BLOCK_SIZE,
                            0,
                            0);
            pipe_barrier(PIPE_ALL);
        }
#if defined(__DAV_M200__)
        copy_gm_to_ubuf(indiceUbuf, topkIndiceGm, 0, 1, seqlenA * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf(argmaxUbuf, argMaxGm, 0, 1, seqlenA * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
#endif
    }

    __aicore__ inline void CopyUnmerged()
    {
        /* update count */
        if (blockY == 0) {
            vector_dup(countUbuf, (float)1.0, seqlenBD128 / FP32_MASK, 1, 1,
                       DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
        }

        uint64_t taskRepeat = seqlenB / HEADS;
        uint64_t taskCurNum = (blockY == (HEADS - 1)) ? (seqlenB - (HEADS - 1) * taskRepeat) : taskRepeat;
        uint64_t colsRepeat = (taskCurNum + colsBaseH - 1) / colsBaseH;
        uint64_t colsRemain = taskCurNum % colsBaseH;

        for (uint64_t i = 0; i < colsRepeat; ++i) {
            uint64_t curCols = ((colsRemain != 0) && (i == colsRepeat - 1)) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;
            copy_gm_to_ubuf(commonUbuf + idx * colsBaseH * hiddenSize,
                            tokenBGm + i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize,
                            0, 1,
                            curCols * hiddenSize / BLOCK_SIZE,
                            0, 0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm(unreducedTokenGm + i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize,
                            commonUbuf + idx * colsBaseH * hiddenSize,
                            0, 1,
                            curCols * hiddenSize / BLOCK_SIZE,
                            0, 0);
        }
        pipe_barrier(PIPE_ALL);

        taskRepeat = afterMergedLenA / HEADS;
        taskCurNum = (blockY == (HEADS - 1)) ? (afterMergedLenA - (HEADS - 1) * taskRepeat) : taskRepeat;
        colsRepeat = (taskCurNum + colsBaseH - 1) / colsBaseH;
        colsRemain = taskCurNum % colsBaseH;

        for (uint64_t i = 0; i < colsRepeat; ++i) {
            uint64_t curCols = ((colsRemain != 0) && (i == colsRepeat - 1)) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;
            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                int64_t idxA = *(indiceUbuf + topR + blockY * taskRepeat + i * colsBaseH + j);
#else
                int64_t idxA = *(topkIndiceGm + topR + blockY * taskRepeat + i * colsBaseH + j);
#endif
                copy_gm_to_ubuf(commonUbuf + idx * colsBaseH * hiddenSize + j * hiddenSize,
                                tokenAGm + idxA * hiddenSize,
                                0, 1,
                                hiddenSize / BLOCK_SIZE,
                                0, 0);
            }
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm(mergedTokenGm + i * colsBaseH * hiddenSize + blockY * taskRepeat * hiddenSize,
                            commonUbuf + idx * colsBaseH * hiddenSize,
                            0, 1,
                            curCols * hiddenSize / BLOCK_SIZE,
                            0, 0);
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void CopyReduced()
    {
        uint64_t repeat = topR / HEADS;
        uint64_t curNum = (blockY == (HEADS - 1)) ? (topR - (HEADS - 1) * repeat) : repeat;
        uint64_t start = blockY * repeat;
        uint64_t end = start + curNum;

        set_atomic_f16();

        for (uint64_t i = start; i < end; ++i) {
#if defined(__DAV_M200__)
            uint64_t idxA = *(indiceUbuf + i);
            uint64_t idxB = *(argmaxUbuf + idxA);
#else
            uint64_t idxA = *(topkIndiceGm + i);
            uint64_t idxB = *(argMaxGm + idxA);
#endif
            uint64_t idx = i % colsBase;

            copy_gm_to_ubuf(commonUbuf + idx * hiddenSize,
                            tokenAGm + idxA * hiddenSize,
                            0,
                            1,
                            hiddenSize / BLOCK_SIZE,
                            0,
                            0);
            *(countUbuf + idxB) = *(countUbuf + idxB) + 1;
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm(unreducedTokenGm + blockY * seqlenB * hiddenSize + idxB * hiddenSize,
                            commonUbuf + idx * hiddenSize,
                            0,
                            1,
                            hiddenSize / BLOCK_SIZE,
                            0,
                            0);
        }

        set_atomic_none();
        pipe_barrier(PIPE_ALL);
        copy_ubuf_to_gm(unreducedCountGm + blockY * seqlenB,
                        countUbuf,
                        0,
                        1,
                        // 32 for 32 Bytes
                        seqlenB * sizeof(float) / 32,
                        0,
                        0);
        pipe_barrier(PIPE_ALL);
    }

private:
    /* global memory address */
    __gm__ half *tokenAGm;
    __gm__ half *tokenBGm;
    __gm__ half *mergedTokenGm;
    __gm__ half *unreducedTokenGm;

    __gm__ int64_t *topkIndiceGm;
    __gm__ int64_t *argMaxGm;

    __gm__ float *unreducedCountGm;

    /* variable */
    uint64_t batch = 0;
    uint64_t hiddenSize = 0;
    uint64_t seqlenA = 0;
    uint64_t seqlenB = 0;
    uint64_t topR = 0;
    uint64_t afterMergedLenA = 0;
    uint64_t usedMemory = 0;
    uint64_t maxCacheMemory = 0;

    uint64_t colsBase = 0;
    uint64_t colsBaseH = 0;
    uint64_t seqlenAD128 = 0;
    uint64_t seqlenBD128 = 0;

    uint32_t blockX = 0;
    uint32_t blockY = 0;

    /* ascendc variable */
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> cacheQueue;
    TQue<QuePosition::VECIN, 1> countQueue;
    TQue<QuePosition::VECIN, 1> indiceQueue;
    TQue<QuePosition::VECIN, 1> argmaxQueue;
    __ubuf__ half *commonUbuf;
    __ubuf__ float *countUbuf;
#if defined(__DAV_M200__)
    __ubuf__ int64_t *indiceUbuf;
    __ubuf__ int64_t *argmaxUbuf;
#endif
    bool valid = true;
};
}

extern "C" __global__ __aicore__ void tome_merged(GM_ADDR TokenA, GM_ADDR TokenB, GM_ADDR TOPK_Indice,
    GM_ADDR Arg_Max, GM_ADDR unmergeTokenA, GM_ADDR unReduceTokenB, GM_ADDR unReduceCount,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (TokenA == nullptr || TokenB == nullptr || TOPK_Indice == nullptr || Arg_Max == nullptr ||
        unmergeTokenA == nullptr || unReduceTokenB == nullptr || unReduceCount == nullptr) {
        return;
    }
    GET_TILING_DATA(tiling_data, tiling);

    TokenMerged op(tiling_data.batch, tiling_data.hiddenSize, tiling_data.topR, tiling_data.seqlenA,
                   tiling_data.seqlenB);

#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
    op.Init(TokenA, TokenB, TOPK_Indice, Arg_Max, unmergeTokenA, unReduceTokenB, unReduceCount);
    op.Process();
#endif
}