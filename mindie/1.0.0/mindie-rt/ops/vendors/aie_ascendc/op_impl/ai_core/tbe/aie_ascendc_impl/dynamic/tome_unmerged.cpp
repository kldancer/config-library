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
constexpr uint64_t MAX_UB_MEMORY = 12288 * 16;
// 16 half nums equal to 32B
constexpr uint32_t BLOCK_SIZE = 16;
// Repeat size is 128
constexpr uint32_t REPEATSIZE = 128;
// Default repeat stride is 8
constexpr uint32_t DEFAULREPEATSTRIDE = 8;
constexpr uint64_t HEADS = 4;
constexpr int64_t MAX_BATCH = 64;
constexpr int64_t MAX_SEQLENA = 4096;
constexpr int64_t MAX_SEQLENB = 4096;
}

namespace {
class TomeUnmerged {
public:
    __aicore__ inline TomeUnmerged(uint64_t batch, uint64_t hiddenSize, uint64_t topR,
        uint64_t seqlenA, uint64_t seqlenB)
    {
        this->batch = batch;
        this->hiddenSize = hiddenSize;
        this->topR = topR;
        this->seqlenA = seqlenA;
        this->seqlenB = seqlenB;
        this->seqlenAD128 = (seqlenA + REPEATSIZE - 1) / REPEATSIZE * REPEATSIZE;
        this->seqlenBD128 = (seqlenB + REPEATSIZE - 1) / REPEATSIZE * REPEATSIZE;
#if defined(__DAV_M200__)
        // 3 tensors of length seqlenAD128, 1 tensor of length seqlenBD128
        this->usedMemory = (seqlenAD128 * 3 + seqlenBD128) * sizeof(int64_t);
#else
        this->usedMemory = 0;
#endif
        if (usedMemory > MAX_UB_MEMORY) {
            valid = false;
            return;
        }
        this->maxCacheMemory = MAX_UB_MEMORY - usedMemory;
        this->afterMergedLenA = seqlenA - topR;
        this->afterMergedLen = this->afterMergedLenA + this->seqlenB;
        if (sizeof(half) == 0 || hiddenSize == 0 || BLOCK_SIZE == 0) {
            return;
        }
        this->colsBase = (maxCacheMemory / sizeof(half) / hiddenSize / BLOCK_SIZE) * BLOCK_SIZE;
        if (colsBase == 0) {
            valid = false;
            return;
        }
        srcBatchOffset = this->afterMergedLen * hiddenSize;
        // 32 for 32 bytes uint
        curBurstLen = hiddenSize * sizeof(half) / 32;
    }

    __aicore__ inline void Init(__gm__ uint8_t *attenOut, __gm__ uint8_t *oriIndiceA, __gm__ uint8_t *oriIndiceB,
        __gm__ uint8_t *topkIndice, __gm__ uint8_t *argMax, __gm__ uint8_t *unZipToken)
    {
        if (attenOut == nullptr || oriIndiceA == nullptr || oriIndiceB == nullptr ||
            topkIndice == nullptr || argMax == nullptr || unZipToken == nullptr) {
            return;
        }

        curBatch = block_idx / (taskPerBatch * HEADS);
        curTask = (block_idx % (taskPerBatch * HEADS)) / HEADS;

        attenOutGm = (__gm__ half *)attenOut + curBatch * srcBatchOffset;
        oriIndiceAGm = (__gm__ int64_t *)oriIndiceA + curBatch * seqlenA;
        oriIndiceBGm = (__gm__ int64_t *)oriIndiceB + curBatch * seqlenB;
        topkIndiceGm = (__gm__ int64_t *)topkIndice + curBatch * seqlenA;
        argMaxGm = (__gm__ int64_t *)argMax + curBatch * seqlenA;
        unZipTokenGm = (__gm__ half *)unZipToken + curBatch * (seqlenA + seqlenB) * hiddenSize;

        pipe.InitBuffer(outQueueCO2, 1, maxCacheMemory);
        LocalTensor<half> cache_perloop_ub = outQueueCO2.AllocTensor<half>();
        commonUbuf = (__ubuf__ half *)cache_perloop_ub.GetPhyAddr();
#if defined(__DAV_M200__)
        pipe.InitBuffer(oriIndiceAQueue, 1, (seqlenAD128 * sizeof(int64_t)));
        LocalTensor<int64_t> oriIndiceAMemory = oriIndiceAQueue.AllocTensor<int64_t>();
        oriIndiceAUbuf = (__ubuf__ int64_t *)oriIndiceAMemory.GetPhyAddr();

        pipe.InitBuffer(oriIndiceBQueue, 1, (seqlenBD128 * sizeof(int64_t)));
        LocalTensor<int64_t> oriIndiceBMemory = oriIndiceBQueue.AllocTensor<int64_t>();
        oriIndiceBUbuf = (__ubuf__ int64_t *)oriIndiceBMemory.GetPhyAddr();

        pipe.InitBuffer(topkIndiceQueue, 1, (seqlenAD128 * sizeof(int64_t)));
        LocalTensor<int64_t> topkIndiceMemory = topkIndiceQueue.AllocTensor<int64_t>();
        topkIndiceUbuf = (__ubuf__ int64_t *)topkIndiceMemory.GetPhyAddr();

        pipe.InitBuffer(argmaxQueue, 1, (seqlenAD128 * sizeof(int64_t)));
        LocalTensor<int64_t> argmaxMemory = argmaxQueue.AllocTensor<int64_t>();
        argmaxUbuf = (__ubuf__ int64_t *)argmaxMemory.GetPhyAddr();
#endif
    }
    __aicore__ inline void ProcessAll()
    {
#if defined(__DAV_M200__)
        copy_gm_to_ubuf(oriIndiceAUbuf, oriIndiceAGm, 0, 1, seqlenA * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf(oriIndiceBUbuf, oriIndiceBGm, 0, 1, seqlenB * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf(topkIndiceUbuf, topkIndiceGm, 0, 1, seqlenA * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
        copy_gm_to_ubuf(argmaxUbuf, argMaxGm, 0, 1, seqlenA * sizeof(int64_t) / BLOCK_SIZE, 0, 0);
        pipe_barrier(PIPE_ALL);
#endif

        if (curTask == 0) {
            ProcessUnmergeA();
        } else if (curTask == 1) {
            ProcessMergedB();
        // 2 for task mode 2
        } else if (curTask == 2) {
            ProcessMergedA();
        } else {
            ProcessMergedHalfA();
        }
    }

private:
    __aicore__ inline void ProcessUnmergeA()
    {
        uint64_t colsBaseH = colsBase / 2;
        uint64_t colsRepeat = (afterMergedLenA + colsBaseH - 1) / colsBaseH;
        uint64_t colsRemain = afterMergedLenA % colsBaseH;

        uint64_t currHead = (block_idx % (taskPerBatch * HEADS)) % HEADS;
        uint64_t repeatPerHead = (colsRepeat + HEADS -1) / HEADS;
        uint64_t src = currHead * repeatPerHead;
        uint64_t dst = currHead == (HEADS - 1) ? colsRepeat : src + repeatPerHead;
        for (uint64_t i = src; i < dst; ++i) {
            uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;
            copy_gm_to_ubuf(commonUbuf + idx * colsBaseH * hiddenSize,
                attenOutGm + i * colsBaseH * hiddenSize,
                0,
                1,
                curCols * curBurstLen,
                0,
                0);
            pipe_barrier(PIPE_ALL);
            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                beforeMergedIndicesA = *(topkIndiceUbuf + topR + i * colsBaseH + j);
                oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
                beforeMergedIndicesA = *(topkIndiceGm + topR + i * colsBaseH + j);
                oriIndicesA = *(oriIndiceAGm + beforeMergedIndicesA);
#endif
                copy_ubuf_to_gm(unZipTokenGm + oriIndicesA * hiddenSize,
                    commonUbuf + idx * colsBaseH * hiddenSize + j * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessMergedB()
    {
        uint64_t colsBaseH = colsBase / 2;
        uint64_t colsRepeat = (seqlenB + colsBaseH -1) / colsBaseH;
        uint64_t colsRemain = seqlenB % colsBaseH;

        uint64_t currHead = (block_idx % (taskPerBatch * HEADS)) % HEADS;
        uint64_t repeatPerHead = (colsRepeat + HEADS - 1) / HEADS;
        uint64_t src = currHead * repeatPerHead;
        uint64_t dst = currHead == HEADS - 1 ? colsRepeat : src + repeatPerHead;
        for (uint64_t i = src; i < dst; ++i) {
            uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;
            copy_gm_to_ubuf(commonUbuf + idx * colsBaseH * hiddenSize,
                attenOutGm + afterMergedLenA * hiddenSize + i * colsBaseH * hiddenSize,
                0,
                1,
                curCols * curBurstLen,
                0,
                0);
            pipe_barrier(PIPE_ALL);
            for (uint64_t j = 0; j < curCols; ++j) {
                beforeMergedIndicesB = i * colsBaseH + j;
#if defined(__DAV_M200__)
                oriIndicesB = *(oriIndiceBUbuf + beforeMergedIndicesB);
#else
                oriIndicesB = *(oriIndiceBGm + beforeMergedIndicesB);
#endif
                copy_ubuf_to_gm(unZipTokenGm + oriIndicesB * hiddenSize,
                    commonUbuf + idx * colsBaseH * hiddenSize + j * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }
    __aicore__ inline void ProcessMergedA()
    {
        uint64_t colsBaseH = colsBase / 2;
        uint64_t colsRepeat = (topR / 2 + colsBaseH - 1) / colsBaseH;
        uint64_t colsRemain = (topR / 2) % colsBaseH;
        uint64_t currHead = (block_idx % (taskPerBatch * HEADS)) % HEADS;
        uint64_t repeatPerHead = (colsRepeat + HEADS - 1) / HEADS;
        uint64_t src = currHead * repeatPerHead;
        uint64_t dst = currHead == HEADS - 1 ? colsRepeat : src + repeatPerHead;
        for (uint64_t i = src; i < dst; ++i) {
            uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;

            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j);
                afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
                beforeMergedIndicesA = *(topkIndiceGm + i * colsBaseH + j);
                afterMergedIndicesA = *(argMaxGm + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAGm + beforeMergedIndicesA);
#endif
                copy_gm_to_ubuf(commonUbuf + (idx * colsBaseH + j) * hiddenSize,
                    attenOutGm + (afterMergedLenA + afterMergedIndicesA) * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
            pipe_barrier(PIPE_ALL);
            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j);
                afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
                beforeMergedIndicesA = *(topkIndiceGm + i * colsBaseH + j);
                afterMergedIndicesA = *(argMaxGm + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAGm + beforeMergedIndicesA);
#endif
                copy_ubuf_to_gm(unZipTokenGm + oriIndicesA * hiddenSize,
                    commonUbuf + (idx * colsBaseH + j) * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessMergedHalfA()
    {
        uint64_t colsBaseH = colsBase / 2;
        uint64_t colsRepeat = (topR / 2 + colsBaseH - 1) / colsBaseH;
        uint64_t colsRemain = (topR / 2) % colsBaseH;
        uint64_t currHead = (block_idx % (taskPerBatch * HEADS)) % HEADS;
        uint64_t repeatPerHead = (colsRepeat + HEADS - 1) / HEADS;
        uint64_t src = currHead * repeatPerHead;
        uint64_t dst = currHead == HEADS - 1 ? colsRepeat : src + repeatPerHead;
        for (uint64_t i = src; i < dst; ++i) {
            uint64_t curCols = (i == colsRepeat - 1) ? colsRemain : colsBaseH;
            uint64_t idx = i % 2;
            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                // 2 for half
                beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j + (topR / 2));
                afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
                // 2 for half
                beforeMergedIndicesA = *(topkIndiceGm + i * colsBaseH + j + (topR / 2));
                afterMergedIndicesA = *(argMaxGm + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAGm + beforeMergedIndicesA);
#endif
                copy_gm_to_ubuf(commonUbuf + (idx * colsBaseH + j) * hiddenSize,
                    attenOutGm + (afterMergedLenA + afterMergedIndicesA) * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
            pipe_barrier(PIPE_ALL);
            for (uint64_t j = 0; j < curCols; ++j) {
#if defined(__DAV_M200__)
                // 2 for half
                beforeMergedIndicesA = *(topkIndiceUbuf + i * colsBaseH + j + (topR / 2));
                afterMergedIndicesA = *(argmaxUbuf + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAUbuf + beforeMergedIndicesA);
#else
                // 2 for half
                beforeMergedIndicesA = *(topkIndiceGm + i * colsBaseH + j + (topR / 2));
                afterMergedIndicesA = *(argMaxGm + beforeMergedIndicesA);
                oriIndicesA = *(oriIndiceAGm + beforeMergedIndicesA);
#endif
                copy_ubuf_to_gm(unZipTokenGm + oriIndicesA * hiddenSize,
                    commonUbuf + (idx * colsBaseH + j) * hiddenSize,
                    0,
                    1,
                    curBurstLen,
                    0,
                    0);
            }
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    __gm__ half *attenOutGm;
    __gm__ int64_t *oriIndiceAGm;
    __gm__ int64_t *oriIndiceBGm;
    __gm__ int64_t *topkIndiceGm;
    __gm__ int64_t *argMaxGm;
    __gm__ half *unZipTokenGm;

    uint64_t batch = 1;
    uint64_t hiddenSize = 1;
    uint64_t seqlenA = 0;
    uint64_t seqlenB = 0;
    uint64_t seqlenAD128 = 0;
    uint64_t seqlenBD128 = 0;
    uint64_t usedMemory = 0;
    uint64_t maxCacheMemory = 0;
    uint64_t topR = 0;

    uint64_t afterMergedLenA = 0;
    uint64_t afterMergedLen = 0;
    uint64_t batchOffsetCurrent = 0;
    uint64_t colsBase = 0;
    // one batch has 4 tasks
    uint64_t taskPerBatch = 4;
    uint64_t curBatch = 0;
    uint64_t srcBatchOffset = 0;
    uint64_t curTask = 0;
    uint64_t curBurstLen = 0;

    uint64_t beforeMergedIndicesA = 0;
    uint64_t oriIndicesA = 0;
    uint64_t oriIndicesB = 0;
    uint64_t beforeMergedIndicesB = 0;
    uint64_t afterMergedIndicesA = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> outQueueCO2;
    TQue<QuePosition::VECIN, 1> oriIndiceAQueue;
    TQue<QuePosition::VECIN, 1> oriIndiceBQueue;
    TQue<QuePosition::VECIN, 1> topkIndiceQueue;
    TQue<QuePosition::VECIN, 1> argmaxQueue;
    __ubuf__ half *commonUbuf;
#if defined(__DAV_M200__)
    __ubuf__ int64_t *oriIndiceAUbuf;
    __ubuf__ int64_t *oriIndiceBUbuf;
    __ubuf__ int64_t *topkIndiceUbuf;
    __ubuf__ int64_t *argmaxUbuf;
#endif
    bool valid = true;
};
}
namespace {
extern "C" __global__ __aicore__ void tome_unmerged(GM_ADDR attenOut, GM_ADDR Ori_IndiceA, GM_ADDR Ori_IndiceB,
    GM_ADDR TOPK_Indice, GM_ADDR Arg_Max, GM_ADDR unZipToken, GM_ADDR workspace, GM_ADDR tiling)
{
    if (attenOut == nullptr || Ori_IndiceA == nullptr || Ori_IndiceB == nullptr ||
        TOPK_Indice == nullptr || Arg_Max == nullptr || unZipToken == nullptr) {
        return;
    }
    GET_TILING_DATA(tiling_data, tiling);
    TomeUnmerged op(tiling_data.batch,
        tiling_data.hiddenSize,
        tiling_data.topR,
        tiling_data.seqlenA,
        tiling_data.seqlenB);
#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
    op.Init(attenOut, Ori_IndiceA, Ori_IndiceB, TOPK_Indice, Arg_Max, unZipToken);
    op.ProcessAll();
#endif
}
}