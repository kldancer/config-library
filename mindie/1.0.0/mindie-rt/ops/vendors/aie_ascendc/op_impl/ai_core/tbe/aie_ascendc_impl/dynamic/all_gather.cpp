/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#if defined(__DAV_C220_VEC__)

#include "kernel_operator.h"

#include "allgather_bigdata.h"
#include "allgather.h"

using namespace AscendC;

namespace {
constexpr int64_t BIGDATA_THRESHOLD = 1 * 1024 * 1024;

template <typename Dtype>
__aicore__ __inline__ void RunAllGatherOpKernel(GM_ADDR input, GM_ADDR output, GM_ADDR commArgs, int64_t len,
    int64_t magic, int64_t rank, int64_t rankSize)
{
    uint32_t extraFlag = 0;
    int op = 0;
    int root = 0;
    if (len > BIGDATA_THRESHOLD) {
        AllGatherBigData<Dtype> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL());
        opKernel.Process();
    } else {
        AllGather<Dtype> opKernel(rank, rankSize, extraFlag);
        opKernel.Init(KERNELS_ARGS_CALL());
        opKernel.Process();
    }
}

extern "C" __global__ __aicore__ void all_gather(GM_ADDR sendData, GM_ADDR commArgs, GM_ADDR recvData,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    int64_t rank = tiling_data.rank;
    int64_t rankSize = tiling_data.rankSize;
    int64_t len = tiling_data.len;
    int64_t magic = tiling_data.magic;
    int64_t dtype = tiling_data.dtype;
    GM_ADDR commArgsGm = reinterpret_cast<GM_ADDR>(*((__gm__ int64_t *)commArgs));
    if (dtype == 0) { // 0: DT_FLOAT
        RunAllGatherOpKernel<float>(sendData, recvData, commArgsGm, len, magic, rank, rankSize);
    } else if (dtype == 2) { // 2: DT_INT8
        RunAllGatherOpKernel<int8_t>(sendData, recvData, commArgsGm, len, magic, rank, rankSize);
    } else if (dtype == 9) { // 9: DT_INT64
        RunAllGatherOpKernel<int64_t>(sendData, recvData, commArgsGm, len, magic, rank, rankSize);
    } else { // defalut: DT_FLOAT16
        RunAllGatherOpKernel<half>(sendData, recvData, commArgsGm, len, magic, rank, rankSize);
    }
}
} // namespace

#endif