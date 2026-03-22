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

namespace {
constexpr size_t MAX_LOOP_NUM = 122880;     // 122880 * sizeof(half), limited in UB Memory
constexpr size_t INT32_BLOCK_SIZE = 8;      // 32 bytes / sizeof(int32_t) = 8
constexpr size_t MAX_MAX_SEQ_LEN = 1024;    // 1024 is the max value of seqLen
constexpr size_t MAX_HIDDEN_SIZE = 2048;    // 2048 is the max value of hiddenSize
constexpr size_t ALIGN = 64;

class UnpadInput {
public:
    __aicore__ inline UnpadInput(uint32_t tillingBatch, uint32_t tillingMaxSeqLen, uint32_t tillingHiddenSize,
        uint32_t tiling_use_core_num)
        : tillingBatch(tillingBatch),
          tillingMaxSeqLen(tillingMaxSeqLen),
          tillingHiddenSize(tillingHiddenSize),
          tiling_use_core_num(tiling_use_core_num)
    {
        Check(tillingBatch, tillingMaxSeqLen, tillingHiddenSize, tiling_use_core_num);
        if (!valid) {
            return;
        }
    }

    __aicore__ inline void Process(__gm__ uint8_t *gmInputs, __gm__ uint8_t *gmSeqLen, __gm__ uint8_t *ntokens,
        __gm__ uint8_t *res)
    {
        size_t batch = tillingBatch;
        size_t maxSeqLen = tillingMaxSeqLen;
        size_t hiddenSize = tillingHiddenSize;
        size_t coreNum = tiling_use_core_num;

        size_t batchPerCore = batch / coreNum;
        size_t tailNum = batch % coreNum;

        size_t gmBatchIdx = 0;
        size_t coreIdx = block_idx;

        if (coreIdx < tailNum) {
            gmBatchIdx = coreIdx + coreIdx * batchPerCore;
        } else {
            gmBatchIdx = tailNum + coreIdx * batchPerCore;
        }
        if (coreIdx < tailNum) {
            batchPerCore += 1;
        }
        seqlen_gm.SetGlobalBuffer((__gm__ int32_t *)gmSeqLen);
        pipe.InitBuffer(inQueueSeqLen, 1, Int32RoundUp(batch) * sizeof(int32_t));
        AscendC::LocalTensor<int32_t> seqlenLocal = inQueueSeqLen.AllocTensor<int32_t>();
        AscendC::DataCopy(seqlenLocal, seqlen_gm, Int32RoundUp(batch));

        // calculate process capacity for current core
        size_t inputCoreOffset = gmBatchIdx * maxSeqLen * hiddenSize;
        int offset = 0;
        pipe_barrier(PIPE_ALL); // 不能删
        for (int i = 0; i < gmBatchIdx; i++) {
            offset += seqlenLocal.GetValue(i);
        }
        size_t outputCoreOffset = offset * hiddenSize;
        size_t rowsPerLoop = MAX_LOOP_NUM / hiddenSize;

        inputs_gm.SetGlobalBuffer((__gm__ half *)gmInputs);
        output_gm.SetGlobalBuffer((__gm__ half *)res);

        pipe.InitBuffer(inQueueInputs, 1, MAX_LOOP_NUM * sizeof(half));
        AscendC::LocalTensor<half> inputsLocal = inQueueInputs.AllocTensor<half>();

        // process
        offset = 0;
        pipe_barrier(PIPE_ALL); // 不能删
        for (int i = 0; i < batchPerCore; i++) {
            int copyRows = seqlenLocal.GetValue(gmBatchIdx + i);
            size_t rowsRepeat = copyRows / rowsPerLoop;
            size_t rowsRemain = copyRows % rowsPerLoop;
            for (int j = 0; j < rowsRepeat; j++) {
                pipe_barrier(PIPE_ALL); // 不能删
                AscendC::DataCopy(inputsLocal,
                    inputs_gm[inputCoreOffset + i * maxSeqLen * hiddenSize + j * rowsPerLoop * hiddenSize],
                    rowsPerLoop * hiddenSize);
                pipe_barrier(PIPE_ALL); // 不能删
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + offset * hiddenSize + j * rowsPerLoop * hiddenSize],
                    inputsLocal,
                    rowsPerLoop * hiddenSize);
            }
            if (rowsRemain > 0) {
                pipe_barrier(PIPE_ALL); // 不能删
                AscendC::DataCopy(inputsLocal,
                    inputs_gm[inputCoreOffset + i * maxSeqLen * hiddenSize + rowsRepeat * rowsPerLoop * hiddenSize],
                    rowsRemain * hiddenSize);
                pipe_barrier(PIPE_ALL); // 不能删
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + offset * hiddenSize + rowsRepeat * rowsPerLoop * hiddenSize],
                    inputsLocal, rowsRemain * hiddenSize);
            }
            offset += copyRows;
        }
    }

private:
    __aicore__ inline uint32_t Int32RoundUp(uint32_t a)
    {
        return (a + INT32_BLOCK_SIZE - 1) / INT32_BLOCK_SIZE * INT32_BLOCK_SIZE;
    }

    __aicore__ inline void Check(uint32_t tillingBatch, uint32_t tillingMaxSeqLen, uint32_t tillingHiddenSize,
        uint32_t tiling_use_core_num)
    {
        if (tillingMaxSeqLen == 0 ||
            tiling_use_core_num == 0 ||
            tillingHiddenSize == 0) {
            valid = false;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueInputs, inQueueSeqLen;

    AscendC::GlobalTensor<half> inputs_gm, output_gm;
    AscendC::GlobalTensor<int32_t> seqlen_gm;

    uint32_t tillingBatch;
    uint32_t tillingMaxSeqLen;
    uint32_t tillingHiddenSize;
    uint32_t tiling_use_core_num;
    bool valid = true;
};
}

namespace {
extern "C" __global__ __aicore__ void unpad_input(GM_ADDR hiddenStates, GM_ADDR seqLen, GM_ADDR ntokens,
    GM_ADDR outStates, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    ::UnpadInput op(tiling_data.batch, tiling_data.maxSeqLen, tiling_data.hiddenSize, tiling_data.coreNums);
    op.Process(hiddenStates, seqLen, ntokens, outStates);
}
}
