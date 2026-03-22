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
constexpr size_t MAX_LOOP_NUM = 122880; // 122880 * sizeof(half), limited in UB Memory
constexpr size_t ZEROS_SIZE = 4096;     // 4096
constexpr uint32_t INT32_BLOCK_SIZE = 8; // 32 bytes / sizeof(int32_t) = 8
constexpr uint32_t MAX_MAX_SEQ_LEN = 1024;
constexpr uint32_t MAX_HIDDEN_SIZE = 2048;
constexpr uint32_t ALIGN = 64;

class PadInput {
public:
    __aicore__ inline PadInput(uint32_t tillingBatch, uint32_t tillingMaxSeqLen, uint32_t tillingHiddenSize,
        uint32_t use_core_num)
        : tillingBatch(tillingBatch),
          tillingMaxSeqLen(tillingMaxSeqLen),
          tillingHiddenSize(tillingHiddenSize),
          use_core_num(use_core_num)
    {
        Check(tillingBatch, tillingMaxSeqLen, tillingHiddenSize, use_core_num);
        if (!valid) {
            return;
        }
    }

    __aicore__ inline void Process(__gm__ uint8_t *gmInputs, __gm__ uint8_t *gmSeqLen, __gm__ uint8_t *gmOutputs)
    {
        size_t batchPerCore = tillingBatch / use_core_num;
        size_t tailNum = tillingBatch % use_core_num;

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

        // calculate process capacity for current core
        size_t outputCoreOffset = gmBatchIdx * tillingMaxSeqLen * tillingHiddenSize;
        seqlen_gm.SetGlobalBuffer((__gm__ int32_t *)gmSeqLen);
        pipe.InitBuffer(inQueueSeqLen, 1, Int32RoundUp(tillingBatch) * sizeof(int32_t));
        AscendC::LocalTensor<int32_t> seqlenLocal = inQueueSeqLen.AllocTensor<int32_t>();

        AscendC::DataCopy(seqlenLocal, seqlen_gm, Int32RoundUp(tillingBatch));

        // 累计的seqlen偏移
        pipe_barrier(PIPE_ALL);
        size_t offset = 0;
        for (size_t i = 0; i < gmBatchIdx; i++) {
            offset += seqlenLocal.GetValue(i);
        }
        // 输入的偏移
        size_t inputCoreOffset = offset * tillingHiddenSize;

        // ub能放的总行数
        size_t rowsPerLoop = MAX_LOOP_NUM / tillingHiddenSize;
        size_t zerosRowsPerLoop = ZEROS_SIZE / tillingHiddenSize;

        inputs_gm.SetGlobalBuffer((__gm__ half *)gmInputs);
        output_gm.SetGlobalBuffer((__gm__ half *)gmOutputs);

        pipe.InitBuffer(inQueueInputs, 1, MAX_LOOP_NUM * sizeof(half));
        pipe.InitBuffer(inQueuePadInputs, 1, ZEROS_SIZE * sizeof(half));
        AscendC::LocalTensor<half> inputsLocal = inQueueInputs.AllocTensor<half>();
        AscendC::LocalTensor<half> padInputsLocal = inQueuePadInputs.AllocTensor<half>();

        // process
        offset = 0;
        pipe_barrier(PIPE_ALL);
        for (size_t i = 0; i < batchPerCore; i++) {
            // 获取当前core当前batch的实际seqlen值
            size_t copyRows = seqlenLocal.GetValue(gmBatchIdx + i);
            // 当前core当前batch，每次最多能放rowsPerLoop行，需要循环的次数为rowsRepeat
            size_t rowsRepeat = copyRows / rowsPerLoop;
            // 最后一次循环，需要处理的行数
            size_t rowsRemain = copyRows % rowsPerLoop;

            for (size_t j = 0; j < rowsRepeat; j++) {
                pipe_barrier(PIPE_ALL);
                AscendC::DataCopy(
                    inputsLocal,
                    inputs_gm[inputCoreOffset + offset * tillingHiddenSize + j * rowsPerLoop * tillingHiddenSize],
                    rowsPerLoop * tillingHiddenSize);
                pipe_barrier(PIPE_ALL);
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + i * tillingMaxSeqLen * tillingHiddenSize
                              + j * rowsPerLoop * tillingHiddenSize],
                    inputsLocal, rowsPerLoop * tillingHiddenSize);
            }
            if (rowsRemain > 0) {
                pipe_barrier(PIPE_ALL);
                AscendC::DataCopy(inputsLocal,
                    inputs_gm[inputCoreOffset + offset * tillingHiddenSize
                              + rowsRepeat * rowsPerLoop * tillingHiddenSize],
                    rowsRemain * tillingHiddenSize);

                pipe_barrier(PIPE_ALL);
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + i * tillingMaxSeqLen * tillingHiddenSize
                              + rowsRepeat * rowsPerLoop * tillingHiddenSize],
                    inputsLocal, rowsRemain * tillingHiddenSize);
            }
            // pad zero to maxLen
            int padRows = tillingMaxSeqLen - copyRows;
            rowsRepeat = padRows / zerosRowsPerLoop;
            rowsRemain = padRows % zerosRowsPerLoop;
            Duplicate<half>(padInputsLocal, half(0.0), ZEROS_SIZE);

            for (int j = 0; j < rowsRepeat; j++) {
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + i * tillingMaxSeqLen * tillingHiddenSize
                              + copyRows * tillingHiddenSize + j * zerosRowsPerLoop * tillingHiddenSize],
                    padInputsLocal, zerosRowsPerLoop * tillingHiddenSize);
            }
            if (rowsRemain > 0) {
                AscendC::DataCopy(
                    output_gm[outputCoreOffset + i * tillingMaxSeqLen * tillingHiddenSize
                              + copyRows * tillingHiddenSize + rowsRepeat * zerosRowsPerLoop * tillingHiddenSize],
                    padInputsLocal, rowsRemain * tillingHiddenSize);
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
        uint32_t use_core_num)
    {
        if (tillingMaxSeqLen == 0 ||
            use_core_num == 0 ||
            tillingHiddenSize == 0) {
            valid = false;
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueInputs, inQueueSeqLen, inQueuePadInputs;
    AscendC::GlobalTensor<half> inputs_gm, output_gm;
    AscendC::GlobalTensor<int32_t> seqlen_gm;
    size_t tillingBatch, tillingMaxSeqLen, tillingHiddenSize, use_core_num;
    bool valid = true;
};
}

extern "C" __global__ __aicore__ void pad_input(GM_ADDR hiddenStates, GM_ADDR seqLen, GM_ADDR batch, GM_ADDR maxSeqLen,
    GM_ADDR hiddenSize, GM_ADDR outStates, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    ::PadInput op(static_cast<size_t>(tilingData.batch),
                  static_cast<size_t>(tilingData.maxSeqLen),
                  static_cast<size_t>(tilingData.hiddenSize),
                  static_cast<size_t>(tilingData.coreNums));
    op.Process(hiddenStates, seqLen, outStates);
}