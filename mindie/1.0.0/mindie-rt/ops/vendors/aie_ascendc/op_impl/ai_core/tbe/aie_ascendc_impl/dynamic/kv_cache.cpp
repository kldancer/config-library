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
constexpr uint32_t MAX_PROCESS_NUM = 12288 * 8;
constexpr uint32_t BLOCK_SIZE = 16;

class KvCache {
public:
    __aicore__ inline KvCache(uint32_t batch, uint32_t hiddenSize, uint32_t maxSeqLen)
    {
        this->hiddenSize = hiddenSize;
        this->maxSeqLen = maxSeqLen;
        this->batch = batch;
    }
    __aicore__ inline void Init(__gm__ uint8_t *newKV, __gm__ uint8_t *layerId, __gm__ uint8_t *CacheIn,
        __gm__ uint8_t *tokenOffset, __gm__ uint8_t *seqLen, __gm__ uint8_t *cacheOut, __gm__ uint8_t *tiling)
    {
        if (g_coreType == AIC) {
            return;
        }
        newKVGm = (__gm__ half *)newKV;
        layerIdGm = (__gm__ uint32_t *)layerId;
        // cacheIn addr trans to ptr
        uint64_t cacheInAddr = *((__gm__ uint64_t *)CacheIn);
        cacheInGm = (__gm__ half *)cacheInAddr;

        tokenOffsetGm = (__gm__ uint32_t *)tokenOffset;
        seqLenGm = (__gm__ uint32_t *)seqLen;

        pipe.InitBuffer(outQueue, 1, (MAX_PROCESS_NUM * sizeof(half)));
        LocalTensor<half> cache_perloop_ub = outQueue.AllocTensor<half>();
        this->tokenOffset = *(this->tokenOffsetGm + block_idx);
        this->seqLen = *(this->seqLenGm + block_idx);
        this->tokenOffsetOld = this->tokenOffset;
        this->tokenOffset -= this->seqLen;
        this->layerId = *(layerIdGm);
        for (uint32_t i = 0; i < block_idx; ++i) {
            prefixOfNtokens += *(this->seqLenGm + i);
        }

        cacheBuff = cacheInGm + this->layerId * this->batch * this->maxSeqLen * this->hiddenSize;
        writeKvCache(newKVGm, cacheBuff, 0, (__ubuf__ half *)cache_perloop_ub.GetPhyAddr());
        *((__gm__ uint64_t *)cacheOut) = cacheInAddr;
        outQueue.FreeTensor(cache_perloop_ub);
    }
    __aicore__ inline void writeKvCache(__gm__ half *oldCache, __gm__ half *newCache, uint64_t kvOffset,
        __ubuf__ half *cache_perloop_ub)
    {
        if (g_coreType == AIC) {
            return;
        }
        uint64_t tokensPerLoop = MAX_PROCESS_NUM / this->hiddenSize;
        uint64_t loopTimes = this->seqLen / tokensPerLoop;
        uint64_t tailTokens = this->seqLen % tokensPerLoop;
        uint64_t tailLoopTimes = tailTokens == 0 ? 0 : 1;

        uint64_t qkvSplitBatchOffset = prefixOfNtokens * this->hiddenSize;
        uint64_t kvCacheBatchOffset = block_idx * this->maxSeqLen * this->hiddenSize
                                    + this->tokenOffset * this->hiddenSize;

        for (uint64_t loop = 0; loop < loopTimes; ++loop) {
            copy_gm_to_ubuf((__ubuf__ half *)cache_perloop_ub,
                (__gm__ half *)oldCache + (qkvSplitBatchOffset + kvOffset + loop * tokensPerLoop * this->hiddenSize),
                0,
                tokensPerLoop,
                this->hiddenSize / BLOCK_SIZE,
                0,
                0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm((__gm__ half *)newCache + kvCacheBatchOffset + loop * tokensPerLoop * this->hiddenSize,
                (__ubuf__ half *)cache_perloop_ub,
                0,
                tokensPerLoop,
                this->hiddenSize / BLOCK_SIZE,
                0,
                0);
            pipe_barrier(PIPE_ALL);
        }
        // process tail
        for (uint64_t loop = 0; loop < tailLoopTimes; ++loop) {
            copy_gm_to_ubuf((__ubuf__ half *)cache_perloop_ub,
                (__gm__ half *)oldCache + qkvSplitBatchOffset
                    + kvOffset + loopTimes * tokensPerLoop * this->hiddenSize,
                0,
                tailTokens,
                this->hiddenSize / BLOCK_SIZE,
                0,
                0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm((__gm__ half *)newCache
                + (kvCacheBatchOffset + loopTimes * tokensPerLoop * this->hiddenSize),
                (__ubuf__ half *)cache_perloop_ub,
                0,
                tailTokens,
                this->hiddenSize / BLOCK_SIZE,
                0,
                0);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    /* data */
    __gm__ half *newKVGm;
    __gm__ uint32_t *layerIdGm;
    __gm__ half *cacheInGm;
    __gm__ uint32_t *tokenOffsetGm;
    __gm__ uint32_t *seqLenGm;
    __gm__ half *cacheBuff;
    uint64_t batch;
    uint64_t hiddenSize;
    uint64_t maxSeqLen;
    uint64_t layerId;
    uint64_t prefixOfNtokens = 0;
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> outQueue;
    uint64_t tokenOffset = 0;
    uint64_t tokenOffsetOld = 0;
    uint64_t seqLen = 0;
};
}

namespace {
extern "C" __global__ __aicore__ void kv_cache(GM_ADDR newKV, GM_ADDR layerId, GM_ADDR CacheIn, GM_ADDR tokenOffset,
    GM_ADDR seqLen, GM_ADDR cacheOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
#ifdef __DAV_C220_VEC__
    KvCache op(tilingData.batch, tilingData.hiddenSize, tilingData.maxSeqLen);
    op.Init(newKV, layerId, CacheIn, tokenOffset, seqLen, cacheOut, tiling);
#endif
}
}
