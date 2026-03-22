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

#include "kernel_operator.h"

using namespace AscendC;
namespace {
static constexpr int32_t BLOCK_SIZE = 32;

template <typename Dtype>
__aicore__ __inline__ void reshape_and_cache(GM_ADDR keyInput, GM_ADDR valueInput, GM_ADDR keyCache,
    GM_ADDR valueCache, GM_ADDR slotMapping, GM_ADDR tilingParam)
{
    // get tiling data
    GET_TILING_DATA(tilingData, tilingParam);
    int32_t numTokens = tilingData.numTokens;
    int32_t tokenSize = tilingData.numHeads * tilingData.headSize;
    int32_t burstLen = tokenSize * sizeof(Dtype) / BLOCK_SIZE;
    int32_t numCores = get_block_num();
    int32_t numTasksPerCore = numTokens * 2 / numCores;
    int32_t numTasksRemaining = numTokens * 2 % numCores;
    __ubuf__ uint8_t *tempUbuf = (__ubuf__ uint8_t *)get_imm(0); // store token temporary

    // keyCache and valueCache is address
    int64_t keyCacheAddr = (int64_t)keyCache;
    int64_t valueCacheAddr = (int64_t)valueCache;
    __gm__ Dtype *kCacheGm = (__gm__ Dtype *)keyCacheAddr;
    __gm__ Dtype *vCacheGm = (__gm__ Dtype *)valueCacheAddr;

    int32_t blockId = get_block_idx();
    int32_t startTaskId = blockId * numTasksPerCore;

    if (blockId < numTasksRemaining) {
        numTasksPerCore++;
        startTaskId += blockId;
    } else {
        startTaskId += numTasksRemaining;
    }
    for (int32_t i = 0; i < numTasksPerCore; i++) {
        if (i + startTaskId < numTokens) {
            int32_t start = (i + startTaskId) * tokenSize;
            int32_t slotValue = (int32_t)(*((__gm__ int32_t *)slotMapping + i + startTaskId));
            if (slotValue < 0) continue;
            int32_t cacheStart = slotValue * tokenSize;
            copy_gm_to_ubuf((__ubuf__ Dtype *)tempUbuf, (__gm__ Dtype *)keyInput + start, 0, 1, burstLen, 0, 0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm((__gm__ Dtype *)kCacheGm + cacheStart, (__ubuf__ Dtype *)tempUbuf, 0, 1, burstLen, 0, 0);
            pipe_barrier(PIPE_ALL);
        } else {
            int32_t start = (i + startTaskId - numTokens) * tokenSize;
            int32_t slotValue = (int32_t)(*((__gm__ int32_t *)slotMapping + i + startTaskId - numTokens));
            if (slotValue < 0) continue;
            int32_t cacheStart = slotValue * tokenSize;
            copy_gm_to_ubuf((__ubuf__ Dtype *)tempUbuf, (__gm__ Dtype *)valueInput + start, 0, 1, burstLen, 0, 0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm((__gm__ Dtype *)vCacheGm + cacheStart, (__ubuf__ Dtype *)tempUbuf, 0, 1, burstLen, 0, 0);
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __global__ __aicore__ void reshape_and_cache(GM_ADDR keyInput, GM_ADDR valueInput, GM_ADDR keyCache,
    GM_ADDR valueCache, GM_ADDR slotMapping, GM_ADDR keyOutput,
    GM_ADDR valueOutput, GM_ADDR workspace, GM_ADDR tilingParam)
{
#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
    GET_TILING_DATA(tilingData, tilingParam);
    int32_t typeByteSize = tilingData.typeByte;
    if (typeByteSize == 1) { // 1字节
        reshape_and_cache<int8_t>(keyInput, valueInput, keyCache, valueCache, slotMapping, tilingParam);
    } else {
        reshape_and_cache<half>(keyInput, valueInput, keyCache, valueCache, slotMapping, tilingParam);
    }
#endif
}
}