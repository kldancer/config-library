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
constexpr int32_t MAX_VALUE = 16;
constexpr int32_t MIN_VALUE = -16;

class Arange {
public:
    __aicore__ inline Arange(int32_t tillingStartNum, int32_t tillingEndNum, int32_t tillingStepForward,
        uint32_t tillingDtypeOutput)
        : tillingStartNum(tillingStartNum),
          tillingEndNum(tillingEndNum),
          tillingStepForward(tillingStepForward),
          tillingDtypeOutput(tillingDtypeOutput)
    {}

    __aicore__ inline void Process(__gm__ uint8_t *gmOutputs)
    {
        if (gmOutputs == nullptr) {
            return;
        }

        int64_t loopNum = 0;

        if (tillingStartNum < tillingEndNum && tillingStepForward > 0) {
            loopNum = (static_cast<int64_t>(tillingEndNum) -
                static_cast<int64_t>(tillingStartNum) - 1) /
                static_cast<int64_t>(tillingStepForward) + 1;
        }

        if (tillingStartNum > tillingEndNum && tillingStepForward < 0) {
            loopNum = (static_cast<int64_t>(tillingEndNum) -
                static_cast<int64_t>(tillingStartNum) + 1) /
                static_cast<int64_t>(tillingStepForward) + 1;
        }

        int64_t loopNumAlign = (loopNum + 3) / 4 * 4;

        pipe.InitBuffer(outQueue, 1, loopNumAlign * sizeof(int64_t));
        output_gm.SetGlobalBuffer((__gm__ int64_t *)gmOutputs, loopNumAlign);
        AscendC::LocalTensor<int64_t> rangeOut = outQueue.AllocTensor<int64_t>();
        for (int64_t i = 0; i < loopNum; i++) {
            rangeOut.SetValue(i, tillingStartNum + i * tillingStepForward);
        }
        AscendC::DataCopy(output_gm, rangeOut, loopNumAlign);
        outQueue.FreeTensor(rangeOut);
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<int64_t> output_gm;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueue; // 1 outputs
    int32_t tillingStartNum, tillingEndNum, tillingStepForward;
    uint32_t tillingDtypeOutput;
    bool valid = false;
};
}

extern "C" __global__ __aicore__ void arange(GM_ADDR startNum, GM_ADDR endNum, GM_ADDR stepForward,
    GM_ADDR dtypeOutput, GM_ADDR outRange, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    ::Arange op(tiling_data.startNum, tiling_data.endNum, tiling_data.stepForward, tiling_data.dtypeOutput);
    op.Process(outRange);
}