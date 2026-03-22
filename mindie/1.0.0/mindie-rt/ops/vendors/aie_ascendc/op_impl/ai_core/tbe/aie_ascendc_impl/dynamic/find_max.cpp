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
#if defined(__DAV_C220_VEC__)
constexpr uint32_t VECTOR_CORE_NUM = 40;
#else
constexpr uint32_t VECTOR_CORE_NUM = 8;
#endif
constexpr uint32_t BLOCK_HALF_NUM = 16;
constexpr uint32_t INT64_HALF_SCALE = 4;
constexpr uint32_t PROCESS_TOKEN = 32;
constexpr uint32_t PROCESS_VALUE = PROCESS_TOKEN;
constexpr uint32_t PROCESS_INDICES = PROCESS_VALUE * INT64_HALF_SCALE;
constexpr uint32_t VEC_MASK = 128;
constexpr uint32_t REPEAT_BLOCK = 8;
constexpr uint32_t MAX_X_BUF = 176 * 1024;
constexpr uint32_t MAX_VALUE_T16_BUF = 2 * PROCESS_VALUE;
constexpr uint32_t MAX_INDICES_T16_BUF = 2 * PROCESS_INDICES;
constexpr uint32_t MAX_RAW_T16_BUF = 32;
constexpr uint32_t WORK_LOCAL_BUF = 1024;
constexpr uint32_t MAX_HIDDEN = 2048;
constexpr uint32_t MIN_HIDDEN = 1;
constexpr uint32_t ALIGN = 128;
constexpr uint32_t MAX_CLUSTER = 32768;
constexpr uint32_t MIN_CLUSTER = 8;
constexpr uint32_t HALF_BLOCK_NUM = 16;
}
/*
 1.追求应用场景极致性能，ReduceMax使用0级接口，且一次拷贝多条数据：hidden % 128 == 0
 2.PROCESS_TOKEN * hidden * 2 <= 176 * 1024: hidden <= 2048
 */
namespace AscendC {
class FindMax {
public:

    __aicore__ inline FindMax(uint32_t tilingCoreTokens, uint32_t tilingTokens, uint32_t tilingHidden)
        : coreTokens(tilingCoreTokens), tokens(tilingTokens), hidden(tilingHidden)
    {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR values, GM_ADDR indices)
    {
        if (x == nullptr || values == nullptr || indices == nullptr) {
            return;
        }
        gmX.SetGlobalBuffer((__gm__ half *)x + block_idx * coreTokens * hidden);
        gmValue.SetGlobalBuffer((__gm__ half *)values + block_idx * coreTokens);
        gmIndices.SetGlobalBuffer((__gm__ half *)indices + block_idx * coreTokens * INT64_HALF_SCALE);
        pipe.InitBuffer(xBuf, MAX_X_BUF);
        pipe.InitBuffer(valueBuf, MAX_VALUE_T16_BUF);
        pipe.InitBuffer(indicesBuf, MAX_INDICES_T16_BUF);
        pipe.InitBuffer(rawOutBuf, MAX_RAW_T16_BUF);
        pipe.InitBuffer(workLocalBuf, WORK_LOCAL_BUF);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<half> ubX = xBuf.Get<half>();
        LocalTensor<half> ubValue = valueBuf.Get<half>();
        LocalTensor<half> ubIndices = indicesBuf.Get<half>();
        LocalTensor<half> ubRawOut = rawOutBuf.Get<half>();
        LocalTensor<half> ubWorkLocal = workLocalBuf.Get<half>();
        Duplicate(ubIndices, half(0), PROCESS_INDICES);

        uint32_t iters = tokens / PROCESS_TOKEN;
        uint32_t tails = tokens % PROCESS_TOKEN;
        uint32_t xStride = PROCESS_TOKEN * hidden;
        uint32_t xOffset = 0;
        uint32_t valueOffset = 0;
        uint32_t indicesOffset = 0;
        uint32_t repeats = hidden / VEC_MASK;
        for (uint32_t i = 0; i < iters; ++i) {
            DataCopy(ubX, gmX[xOffset], xStride);
            uint32_t xUbOffset = 0;
            uint32_t indicesUbOffset = 0;
            pipe_barrier(PIPE_ALL);
            for (uint32_t j = 0; j < PROCESS_VALUE; ++j) {
                ReduceMax(ubRawOut, ubX[xUbOffset], ubWorkLocal, VEC_MASK, repeats, REPEAT_BLOCK, true);
                ubValue.SetValue(j, ubRawOut.GetValue(0));
                xUbOffset += hidden;
                ubIndices.SetValue(indicesUbOffset, ubRawOut.GetValue(1));
                indicesUbOffset += INT64_HALF_SCALE;
            }
            DataCopy(gmValue[valueOffset], ubValue, PROCESS_VALUE);
            DataCopy(gmIndices[indicesOffset], ubIndices, PROCESS_INDICES);
            xOffset += xStride;
            valueOffset += PROCESS_VALUE;
            indicesOffset += PROCESS_INDICES;
        }
        if (tails > 0) {
            DataCopy(ubX, gmX[xOffset], tails * hidden);
            uint32_t xUbOffset = 0;
            uint32_t indicesUbOffset = 0;
            pipe_barrier(PIPE_ALL);
            for (uint32_t j = 0; j < tails; ++j) {
                ReduceMax(ubRawOut, ubX[xUbOffset], ubWorkLocal, VEC_MASK, repeats, REPEAT_BLOCK, true);
                ubValue.SetValue(j, ubRawOut.GetValue(0));
                xUbOffset += hidden;
                ubIndices.SetValue(indicesUbOffset, ubRawOut.GetValue(1));
                indicesUbOffset += INT64_HALF_SCALE;
            }
            DataCopy(gmValue[valueOffset], ubValue, AlignUp(tails, BLOCK_HALF_NUM));
            DataCopy(gmIndices[indicesOffset], ubIndices, AlignUp(tails * INT64_HALF_SCALE, BLOCK_HALF_NUM));
        }
    }

private:
    __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t unit)
    {
        return (x + unit -1) / unit * unit;
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECCALC> xBuf, valueBuf, indicesBuf, rawOutBuf, workLocalBuf;
    GlobalTensor<half> gmX, gmValue, gmIndices;
    uint32_t coreTokens, tokens, hidden;
    bool valid = true;
};
}

extern "C" __global__ __aicore__ void find_max(GM_ADDR x, GM_ADDR values, GM_ADDR indices,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    uint32_t tokens = block_idx + 1 == t.cores ? t.lastCoreTokens : t.coreTokens;
    AscendC::FindMax op(t.coreTokens, tokens, t.hidden);
#if not defined(__DAV_C220_CUBE__)
    op.Init(x, values, indices);
    op.Process();
#endif
}