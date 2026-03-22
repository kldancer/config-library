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
// 16 half nums equal to 32B
constexpr uint32_t BLOCK_SIZE = 16;
// Default repeat stride is 8
constexpr uint8_t REPEAT_STRIDE = 8;
// Default repeat num is 128
constexpr uint8_t REPEAT_NUM = 128;
// Max repeat times is 255
constexpr uint64_t REPEAT_TIMES = 255;

 // Set 310P ubSize -> 131072
constexpr uint32_t MAX_UBSIZE_310P = 130000;
// Set 910B ubSize -> 71920
constexpr uint32_t MAX_UBSIZE_910B = 65280;
// Set 310P coreNum for singe batch -> 8
constexpr uint32_t MAX_CORENUM_310P = 8;
// Set 910B coreNum for singe batch-> 16
constexpr uint32_t MAX_CORENUM_910B = 16;

class SliceTransGeluMul {
public:
    __aicore__ inline SliceTransGeluMul(uint64_t batch, uint64_t seqLen, uint64_t hiddenSize,
        uint64_t coreNums, uint64_t tokenPerCore, uint64_t tokenPerLoop, uint64_t loopNum,
        uint64_t lastLoopTokenNums)
    {
        this->batch = batch;
        // you must guarantee seqLen is align to 16
        this->seqLen = seqLen;
        // you must guarantee hiddenSize is align to 16
        this->hiddenSize = hiddenSize;
        // Split hiddenStates into 2 patch
        this->halfHiddenSize = hiddenSize / 2;
        this->coreNums = coreNums;
        this->tokenPerCore = tokenPerCore;
        this->tokenPerLoop = tokenPerLoop;
        this->loopNum = loopNum;
        this->lastLoopTokenNums = lastLoopTokenNums;
        // Find current batch
        if (coreNums == 0) {
            return;
        }
        this->curBatch = block_idx / coreNums;
        // Find the current token
        this->curTokenPos = tokenPerCore * (block_idx % coreNums);

        uint32_t maxProcess_num = 0;
        if (this->coreNums == MAX_CORENUM_310P) {
            maxProcess_num = MAX_UBSIZE_310P;
        } else {
            maxProcess_num = MAX_UBSIZE_910B;
        }

        pipe.InitBuffer(outQueueCO2, 1, (maxProcess_num * sizeof(half)));
        LocalTensor<half> globalUbuf = outQueueCO2.AllocTensor<half>();

        commonUbuf = (__ubuf__ half *)globalUbuf.GetPhyAddr();
        inputUb0 = commonUbuf;
        inputUb1 = inputUb0 + tokenPerLoop * halfHiddenSize;
        outputUb0 = inputUb1 + tokenPerLoop * halfHiddenSize;
    }
    __aicore__ inline void InitNZ(__gm__ uint8_t *inputX, __gm__ uint8_t *outputX)
    {
        // Jump to current token
        inputXGm0 = (__gm__ half *)inputX + curTokenPos * halfHiddenSize + curBatch * seqLen * hiddenSize;
        inputXGm1 = (__gm__ half *)inputX + curTokenPos * halfHiddenSize + seqLen * halfHiddenSize
            + curBatch * seqLen * hiddenSize;
        outputXGm = (__gm__ half *)outputX + curTokenPos * halfHiddenSize + curBatch * seqLen * halfHiddenSize;
    }
    __aicore__ inline void InitND(__gm__ uint8_t *inputX, __gm__ uint8_t *outputX)
    {
        // Jump to current token
        inputXGm0 = (__gm__ half *)inputX + curTokenPos * hiddenSize + curBatch * seqLen * hiddenSize;
        inputXGm1 = (__gm__ half *)inputX + halfHiddenSize + curTokenPos * hiddenSize
            + curBatch * seqLen * hiddenSize;
        outputXGm = (__gm__ half *)outputX + curTokenPos * halfHiddenSize + curBatch * seqLen * halfHiddenSize;
    }
    __aicore__ inline void VecMuls(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t repeat,
        uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        for (uint64_t i = 0; i < repeat; i += REPEAT_TIMES) {
            uint8_t curRepeat = i + REPEAT_TIMES >= repeat ? (uint8_t)(repeat - i) : REPEAT_TIMES;
            vmuls(dst + i * REPEAT_NUM,
                src0 + i * REPEAT_NUM,
                src1,
                curRepeat,
                1,
                1,
                REPEAT_STRIDE,
                REPEAT_STRIDE
            );
        }
    }

    __aicore__ inline void VecAdds(__ubuf__ half *dst, __ubuf__ half *src0, half src1, uint64_t repeat,
        uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        for (uint64_t i = 0; i < repeat; i += REPEAT_TIMES) {
            uint8_t curRepeat = i + REPEAT_TIMES >= repeat ? (uint8_t)(repeat - i) : REPEAT_TIMES;
            vadds(dst + i * REPEAT_NUM,
                src0 + i * REPEAT_NUM,
                src1,
                curRepeat,
                1,
                1,
                REPEAT_STRIDE,
                REPEAT_STRIDE
            );
        }
    }

    __aicore__ inline void VecExp(__ubuf__ half *dst, __ubuf__ half *src, uint64_t repeat,
        uint16_t dstBlockStride, uint16_t srcBlockStride, uint8_t dstRepeatStride, uint8_t srcRepeatStride)
    {
        for (uint64_t i = 0; i < repeat; i += REPEAT_TIMES) {
            uint8_t curRepeat = i + REPEAT_TIMES >= repeat ? (uint8_t)(repeat - i) : REPEAT_TIMES;
            vexp(dst + i * REPEAT_NUM,
                src + i * REPEAT_NUM,
                curRepeat,
                1,
                1,
                REPEAT_STRIDE,
                REPEAT_STRIDE);
        }
    }
    __aicore__ inline void VecDiv(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t repeat,
        uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
        uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        for (uint64_t i = 0; i < repeat; i += REPEAT_TIMES) {
            uint8_t curRepeat = i + REPEAT_TIMES >= repeat ? (uint8_t)(repeat - i) : REPEAT_TIMES;
            vdiv(dst + i * REPEAT_NUM,
                src0 + i * REPEAT_NUM,
                src1 + i * REPEAT_NUM,
                curRepeat,
                1,
                1,
                1,
                REPEAT_STRIDE,
                REPEAT_STRIDE,
                REPEAT_STRIDE);
        }
    }

    __aicore__ inline void VecMul(__ubuf__ half *dst, __ubuf__ half *src0, __ubuf__ half *src1, uint64_t repeat,
        uint8_t dstBlockStride, uint8_t src0BlockStride, uint8_t src1BlockStride, uint8_t dstRepeatStride,
        uint8_t src0RepeatStride, uint8_t src1RepeatStride)
    {
        for (uint64_t i = 0; i < repeat; i += REPEAT_TIMES) {
            uint8_t curRepeat = i + REPEAT_TIMES >= repeat ? (uint8_t)(repeat - i) : REPEAT_TIMES;
            vmul(dst + i * REPEAT_NUM,
                src0 + i * REPEAT_NUM,
                src1 + i * REPEAT_NUM,
                curRepeat,
                1,
                1,
                1,
                REPEAT_STRIDE,
                REPEAT_STRIDE,
                REPEAT_STRIDE);
        }
    }
    __aicore__ inline void ProcessNZFormat()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        for (uint32_t loop = 0; loop < loopNum; ++loop) {
            uint64_t curProcessNum = loop == loopNum - 1 ? lastLoopTokenNums : tokenPerLoop;
            uint64_t curRepeatTimes = curProcessNum * halfHiddenSize / REPEAT_NUM;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(inputUb1,
                inputXGm1 + loop * tokenPerLoop * halfHiddenSize,
                0,
                1,
                curProcessNum * halfHiddenSize / BLOCK_SIZE,
                0,
                0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            copy_gm_to_ubuf(inputUb0,
                inputXGm0 + loop * tokenPerLoop * halfHiddenSize,
                0,
                1,
                curProcessNum * halfHiddenSize / BLOCK_SIZE,
                0,
                0);
            CommonProcess(curRepeatTimes);
            copy_ubuf_to_gm(outputXGm + loop * tokenPerLoop * halfHiddenSize,
                outputUb0,
                0,
                1,
                curProcessNum * halfHiddenSize / BLOCK_SIZE,
                0,
                0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    __aicore__ inline void ProcessNDFormat()
    {
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        for (uint32_t loop = 0; loop < loopNum; ++loop) {
            curProcessNum = loop == loopNum - 1 ? lastLoopTokenNums : tokenPerLoop;
            curRepeatTimes = curProcessNum * halfHiddenSize / REPEAT_NUM;
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_ubuf(inputUb1,
                inputXGm1 + loop * tokenPerLoop * hiddenSize,
                0,
                curProcessNum,
                halfHiddenSize / BLOCK_SIZE,
                halfHiddenSize / BLOCK_SIZE,
                0);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
            copy_gm_to_ubuf(inputUb0,
                inputXGm0 + loop * tokenPerLoop * hiddenSize,
                0,
                curProcessNum,
                halfHiddenSize / BLOCK_SIZE,
                halfHiddenSize / BLOCK_SIZE,
                0);
            CommonProcess(curRepeatTimes);
            copy_ubuf_to_gm(outputXGm + loop * tokenPerLoop * halfHiddenSize,
                outputUb0,
                0,
                1,
                curProcessNum * halfHiddenSize / BLOCK_SIZE,
                0,
                0);
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    }
    __aicore__ inline void CommonProcess(uint64_t curRepeatTimes)
    {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        VecMuls(outputUb0,
            inputUb1,
            -1.702, // coefficient is -1.7017 or -1.702
            curRepeatTimes,
            1,
            1,
            REPEAT_STRIDE,
            REPEAT_STRIDE);
        VecExp(outputUb0,
            outputUb0,
            curRepeatTimes,
            1,
            1,
            REPEAT_STRIDE,
            REPEAT_STRIDE);
        VecAdds(outputUb0,
            outputUb0,
            1,
            curRepeatTimes,
            1,
            1,
            REPEAT_STRIDE,
            REPEAT_STRIDE);
        VecDiv(outputUb0,
            inputUb1,
            outputUb0,
            curRepeatTimes,
            1,
            1,
            1,
            REPEAT_STRIDE,
            REPEAT_STRIDE,
            REPEAT_STRIDE);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        VecMul(outputUb0,
            inputUb0,
            outputUb0,
            curRepeatTimes,
            1,
            1,
            1,
            REPEAT_STRIDE,
            REPEAT_STRIDE,
            REPEAT_STRIDE);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    }
private:
    uint64_t batch = 0;
    uint64_t seqLen = 0;
    uint64_t hiddenSize = 0;
    uint64_t halfHiddenSize = 0;
    uint64_t coreNums = 0;
    uint64_t tokenPerCore = 0;
    uint64_t tokenPerLoop = 0;
    
    uint64_t curBatch = 0;
    uint64_t curTokenPos = 0;
    uint64_t loopNum = 0;
    uint64_t lastLoopTokenNums = 0;

    uint64_t curProcessNum = 0;
    uint64_t curRepeatTimes = 0;

    __gm__ half *inputXGm0;
    __gm__ half *inputXGm1;
    __gm__ half *outputXGm;

    // malloc ubuf
    __ubuf__ half *commonUbuf;

    __ubuf__ uint8_t *globalUbuf;
    __ubuf__ half *inputUb0;
    __ubuf__ half *inputUb1;
    __ubuf__ half *outputUb0;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> outQueueCO2;
};
}

extern "C" __global__ __aicore__ void slice_trans_gelu_mul(GM_ADDR inputX, GM_ADDR outputX,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    SliceTransGeluMul op(tiling_data.batch, tiling_data.seqLen,
        tiling_data.hiddenSize, tiling_data.coreNums, tiling_data.tokenPerCore,
        tiling_data.tokenPerLoop, tiling_data.loopNum, tiling_data.lastLoopTokenNums);
#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
    int64_t isNZ = tiling_data.isNZ;
    if (isNZ == 1) {
        op.InitNZ(inputX, outputX);
        op.ProcessNZFormat();
    }
    if (isNZ == 0) {
        op.InitND(inputX, outputX);
        op.ProcessNDFormat();
    }
#endif
}