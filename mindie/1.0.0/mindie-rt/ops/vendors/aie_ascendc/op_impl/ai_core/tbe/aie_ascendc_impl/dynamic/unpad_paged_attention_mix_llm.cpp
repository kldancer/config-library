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
#include "lib/matmul_intf.h"

using namespace AscendC;

namespace {
// FFTS Flag
constexpr int32_t QK_READY = 0;
constexpr int32_t SOFTMAX_READY = 1;
constexpr int32_t UPDATE_READY_LLM = 2;
constexpr int32_t TILING_HEAD_SIZE = 16;
constexpr int32_t BIT_SHIFT_LLM = 8;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384; // 128 * 128
constexpr int32_t L1_HALF_BUF_SIZE_LLM = 65536; // 256 * 256
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t CUBE_MATRIX_SIZE = 256; // 16 * 16
constexpr int32_t L0AB_UINT8_BLOCK_SIZE_LLM = 32768; // 128 * 128 * 2B
constexpr int32_t TMP_SIZE = 32768; // 128 * 256
constexpr int32_t FLOAT_BLOCK_SIZE = 8;
constexpr int32_t HALF_VECTOR_SIZE_LLM = 128;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t MASK_LOW_LLM = 64;
constexpr int32_t MASK_HIGH_LLM = 128;
constexpr int32_t UB_UINT8_BLOCK_SIZE_DECODER = 24576; // 96 * 128 * 2B
constexpr int32_t UB_UINT8_LINE_SIZE = 512; // 128 * 4B
constexpr int32_t TILING_PARA_SIZE = 26;
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072; // 128K, 910B L1 512K
constexpr int32_t UB_HALF_BUF_SIZE = 8192; // 64 * 128
constexpr int64_t UB_UINT8_BLOCK_SIZE_PREFILL = 16384; // 64 * 128 * 2B
constexpr int64_t UB_FLOAT_LINE_SIZE = 128;
constexpr int64_t UB_HALF_LINE_SIZE = 256;
constexpr int32_t VECTOR_SIZE = 128;
} // namespace

namespace AscendC {
class UnpadPagedAttentionMixLlm {
public:
    __aicore__ inline UnpadPagedAttentionMixLlm() {}

    __aicore__ inline void unpad_flash_attention_decoder_mix_aic(__gm__ uint8_t * __restrict__ qGmLlm,
        __gm__ uint8_t * __restrict__ kGmLlm, __gm__ uint8_t * __restrict__ vGmLlm,
        __gm__ uint8_t * __restrict__ blockTablesGmLlm, __gm__ uint8_t * __restrict__ layerIdGmLlm,
        __gm__ uint8_t * __restrict__ sGmLlm, __gm__ uint8_t * __restrict__ pGmLlm,
        __gm__ uint8_t * __restrict__ oTmpGmLlm, __gm__ uint8_t *tiling, uint32_t *tilingParaGm)
    {
        set_padding(0);
        set_atomic_none();
        uint64_t config = 0x1;
        set_nd_para(config);
        set_mask_norm();

        uint32_t batchSize;
        uint32_t maxSeqlen;
        uint32_t qHeads;
        uint32_t embd;
        uint32_t kvHeads;
        uint32_t batchMask;
        uint32_t formerBatch;
        uint32_t formerHeadSplit;
        uint32_t tailBatch;
        uint32_t tailHeadSplit;
        uint32_t headSplitNum;
        uint32_t maxNumBlocksPerQuery, blockSize;

        if (tiling == nullptr) {
            batchSize = (uint32_t)(*((int32_t *)tilingParaGm));
            maxSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1));
            qHeads = (uint32_t)(*((int32_t *)tilingParaGm + 2));
            embd = (uint32_t)(*((int32_t *)tilingParaGm + 3));
            kvHeads = (uint32_t)(*((int32_t *)tilingParaGm + 4));
            formerBatch = (uint32_t)(*((int32_t *)tilingParaGm + 7));
            formerHeadSplit = (uint32_t)(*((int32_t *)tilingParaGm + 8));
            tailBatch = (uint32_t)(*((int32_t *)tilingParaGm + 9));
            tailHeadSplit = (uint32_t)(*((int32_t *)tilingParaGm + 10));
            headSplitNum = (uint32_t)(*((int32_t *)tilingParaGm + 12));
            maxNumBlocksPerQuery = (uint32_t)(*((int32_t *)tilingParaGm + 14));
            blockSize = (uint32_t)(*((int32_t *)tilingParaGm + 15));
        } else {
            batchSize = (uint32_t)(*((__gm__ int32_t *)tiling));
            maxSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            qHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embd = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            kvHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 4));
            formerBatch = (uint32_t)(*((__gm__ int32_t *)tiling + 7));
            formerHeadSplit = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            tailBatch = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            tailHeadSplit = (uint32_t)(*((__gm__ int32_t *)tiling + 10));
            headSplitNum = (uint32_t)(*((__gm__ int32_t *)tiling + 12));
            maxNumBlocksPerQuery = (uint32_t)(*((__gm__ int32_t *)tiling + 14));
            blockSize = (uint32_t)(*((__gm__ int32_t *)tiling + 15));
        }

        uint32_t groupNum = qHeads / kvHeads;
        uint64_t strideKv = kvHeads * embd;
        uint64_t batchStrideKv = batchSize * maxSeqlen * strideKv;
        // 指针地址
        uint64_t kCacheAddr = *((__gm__ uint64_t *)kGmLlm);
        uint64_t vCacheAddr = *((__gm__ uint64_t *)vGmLlm);
        __gm__ half *kCacheGm = (__gm__ half *)kCacheAddr;
        __gm__ half *vCacheGm = (__gm__ half *)vCacheAddr;
        // 指针地址

        uint32_t k = embd;
        uint32_t roundK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        uint32_t corePerBatch = (qHeads + formerHeadSplit - 1) / formerHeadSplit;
        uint32_t processNum = formerBatch * corePerBatch;

        uint32_t kvSeqlen;
        uint32_t ppNScalar;
        uint32_t addrQHigh32;
        uint32_t addrQLoww32;
        uint32_t addrKHigh32;
        uint32_t addrKLoww32;
        uint32_t addrVHigh32;
        uint32_t addrVLoww32DeAic;
        uint32_t batchIdDeAic;
        uint32_t layerId = *((__gm__ uint32_t *)layerIdGmLlm);
        uint32_t numBlocksDeAic;

        for (uint32_t process = block_idx; process < processNum; process += uint32_t(block_num)) {
            uint32_t curBatch = process / corePerBatch;
            uint32_t curCore = process % corePerBatch;
            uint32_t curHeadNum = formerHeadSplit;
            if (curCore == (corePerBatch - 1)) {
                curHeadNum = qHeads - curCore * formerHeadSplit;
            }
            uint32_t headSplitLoop = (curHeadNum + headSplitNum -1) / headSplitNum;
            uint32_t startHead = (process % corePerBatch) * formerHeadSplit;

            uint32_t offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatch;

            if (tiling == nullptr) {
                kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTiling));
                ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTiling));
                addrQHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 4 + offsetTiling));
                addrQLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 5 + offsetTiling));
                addrKHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 6 + offsetTiling));
                addrKLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 7 + offsetTiling));
                addrVHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 8 + offsetTiling));
                addrVLoww32DeAic = (uint32_t)(*((int32_t *)tilingParaGm + 9 + offsetTiling));
                batchIdDeAic = (uint32_t)(*((int32_t *)tilingParaGm + 15 + offsetTiling));
                numBlocksDeAic = (uint32_t)(*((int32_t *)tilingParaGm + 17 + offsetTiling));

            } else {
                kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTiling));
                ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTiling));
                addrQHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offsetTiling));
                addrQLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offsetTiling));
                addrKHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offsetTiling));
                addrKLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offsetTiling));
                addrVHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offsetTiling));
                addrVLoww32DeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offsetTiling));
                batchIdDeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 15 + offsetTiling));
                numBlocksDeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 17 + offsetTiling));
            }

            RunAic(curBatch, startHead, curHeadNum, headSplitLoop, offsetTiling, embd, headSplitNum, groupNum,
                   strideKv, k, roundK, kvSeqlen, ppNScalar, addrQHigh32, addrQLoww32, addrKHigh32,
                   addrKLoww32, addrVHigh32, addrVLoww32DeAic, maxNumBlocksPerQuery, blockSize, batchIdDeAic, layerId,
                   numBlocksDeAic, qGmLlm, kCacheGm, vCacheGm, blockTablesGmLlm, sGmLlm, pGmLlm, oTmpGmLlm);
        }

        if (tailBatch > 0) {
            corePerBatch = (qHeads + tailHeadSplit - 1) / tailHeadSplit;
            processNum = tailBatch * corePerBatch;
            for (uint32_t processDeAic = block_idx; processDeAic < processNum; processDeAic += uint32_t(block_num)) {
                uint32_t curBatch = processDeAic / corePerBatch + formerBatch;
                uint32_t curCore = processDeAic % corePerBatch;
                uint32_t curHeadNum = tailHeadSplit;
                if (curCore == (corePerBatch - 1)) {
                    curHeadNum = qHeads - curCore * tailHeadSplit;
                }
                uint32_t headSplitLoop = (curHeadNum + headSplitNum -1) / headSplitNum;
                uint32_t startHead = (processDeAic % corePerBatch) * tailHeadSplit;

                uint32_t offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatch;

                if (tiling == nullptr) {
                    kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTiling));
                    ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTiling));
                    addrQHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 4 + offsetTiling));
                    addrQLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 5 + offsetTiling));
                    addrKHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 6 + offsetTiling));
                    addrKLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 7 + offsetTiling));
                    addrVHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 8 + offsetTiling));
                    addrVLoww32DeAic = (uint32_t)(*((int32_t *)tilingParaGm + 9 + offsetTiling));
                    batchIdDeAic = (uint32_t)(*((int32_t *)tilingParaGm + 15 + offsetTiling));
                    numBlocksDeAic = (uint32_t)(*((int32_t *)tilingParaGm + 17 + offsetTiling));
                } else {
                    kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTiling));
                    ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTiling));
                    addrQHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offsetTiling));
                    addrQLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offsetTiling));
                    addrKHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offsetTiling));
                    addrKLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offsetTiling));
                    addrVHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offsetTiling));
                    addrVLoww32DeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offsetTiling));
                    batchIdDeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 15 + offsetTiling));
                    numBlocksDeAic = (uint32_t)(*((__gm__ int32_t *)tiling + 17 + offsetTiling));
                }

                RunAic(curBatch, startHead, curHeadNum, headSplitLoop, offsetTiling, embd, headSplitNum,
                       groupNum, strideKv, k, roundK, kvSeqlen, ppNScalar, addrQHigh32, addrQLoww32,
                       addrKHigh32, addrKLoww32, addrVHigh32, addrVLoww32DeAic, maxNumBlocksPerQuery,
                       blockSize, batchIdDeAic, layerId, numBlocksDeAic, qGmLlm, kCacheGm, vCacheGm, blockTablesGmLlm,
                       sGmLlm, pGmLlm, oTmpGmLlm);
            }
        }

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flash_attention_decoder_mix_aiv(__gm__ uint8_t * __restrict__ maskGm,
        __gm__ uint8_t * __restrict__ oGm, __gm__ uint8_t * __restrict__ sGm, __gm__ uint8_t * __restrict__ pGm,
        __gm__ uint8_t * __restrict__ oTmpGm, __gm__ uint8_t *tiling, uint32_t *tilingParaGm)
    {
        int32_t subBlockIdx = get_subblockid();
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint32_t batchSize;
        uint32_t maxSeqlen;
        uint32_t qHeads;
        uint32_t embd;
        float tor;
        uint32_t batchMask;
        uint32_t formerBatch;
        uint32_t formerHeadSplit;
        uint32_t tailBatch;
        uint32_t tailHeadSplit;
        uint32_t maskStride;

        __ubuf__ int32_t *tilingParaUb;

        if (tiling == nullptr) {
            batchSize = (uint32_t)(*((int32_t *)tilingParaGm));
            maxSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1));
            qHeads = (uint32_t)(*((int32_t *)tilingParaGm + 2));
            embd = (uint32_t)(*((int32_t *)tilingParaGm + 3));
            tor = (half)(*((float *)tilingParaGm + 5));
            batchMask = (uint32_t)(*((int32_t *)tilingParaGm + 6));
            formerBatch = (uint32_t)(*((int32_t *)tilingParaGm + 7));
            formerHeadSplit = (uint32_t)(*((int32_t *)tilingParaGm + 8));
            tailBatch = (uint32_t)(*((int32_t *)tilingParaGm + 9));
            tailHeadSplit = (uint32_t)(*((int32_t *)tilingParaGm + 10));
            maskStride = (uint32_t)(*((int32_t *)tilingParaGm + 11));
        } else {
            batchSize = (uint32_t)(*((__gm__ int32_t *)tiling));
            maxSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            qHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embd = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            tor = (half)(*((__gm__ float *)tiling + 5));
            batchMask = (uint32_t)(*((__gm__ int32_t *)tiling + 6));
            formerBatch = (uint32_t)(*((__gm__ int32_t *)tiling + 7));
            formerHeadSplit = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            tailBatch = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            tailHeadSplit = (uint32_t)(*((__gm__ int32_t *)tiling + 10));
            maskStride = (uint32_t)(*((__gm__ int32_t *)tiling + 11));
        }

        uint32_t k = embd;
        uint32_t roundK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        uint32_t corePerBatch = (qHeads + formerHeadSplit - 1) / formerHeadSplit;
        uint32_t processNum = formerBatch * corePerBatch;

        uint32_t kvSeqlen;
        uint32_t ppNScalar;
        uint32_t addrOHigh32;
        uint32_t addrOLoww32;

        for (uint32_t process = block_idx; process < processNum; process += uint32_t(block_num)) {
            uint32_t curBatch = process / corePerBatch;
            uint32_t curCore = process % corePerBatch;
            uint32_t curHeadNum = formerHeadSplit;
            if (curCore == (corePerBatch - 1)) {
            curHeadNum = qHeads - curCore * formerHeadSplit;
            }
            uint32_t startHead = (process % corePerBatch) * formerHeadSplit;
            uint32_t headIdx = startHead + subBlockIdx * curHeadNum / 2;

            // get tiling args
            uint32_t offsetTilingTmp = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatch;
            if (tiling == nullptr) {
                kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTilingTmp));
                ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTilingTmp));
                addrOHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 10 + offsetTilingTmp));
                addrOLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 11 + offsetTilingTmp));
            } else {
                kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTilingTmp));
                ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTilingTmp));
                addrOHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offsetTilingTmp));
                addrOLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offsetTilingTmp));
            }

            RunAiv(curBatch, headIdx, curHeadNum, offsetTilingTmp, subBlockIdx, maxSeqlen, embd, tor, batchMask,
                   maskStride, k, roundK, kvSeqlen, ppNScalar, addrOHigh32, addrOLoww32, maskGm, oGm, sGm, pGm, oTmpGm);
        }
        if (tailBatch > 0) {
            corePerBatch = (qHeads + tailHeadSplit - 1) / tailHeadSplit;
            processNum = tailBatch * corePerBatch;
            for (uint32_t process = block_idx; process < processNum; process += uint32_t(block_num)) {
                uint32_t curBatch = process / corePerBatch + formerBatch;
                uint32_t curCore = process % corePerBatch;
                uint32_t curHeadNum = tailHeadSplit;
                if (curCore == (corePerBatch - 1)) {
                curHeadNum = qHeads - curCore * tailHeadSplit;
                }
                uint32_t startHead = (process % corePerBatch) * tailHeadSplit;
                uint32_t headIdx = startHead + subBlockIdx * curHeadNum / 2;

                // get tiling args
                uint32_t offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatch;
                if (tiling == nullptr) {
                    kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTiling));
                    ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTiling));
                    addrOHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 10 + offsetTiling));
                    addrOLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 11 + offsetTiling));
                } else {

                    kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTiling));
                    ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTiling));
                    addrOHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offsetTiling));
                    addrOLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offsetTiling));
                }

                RunAiv(curBatch, headIdx, curHeadNum, offsetTiling, subBlockIdx, maxSeqlen, embd, tor, batchMask,
                    maskStride, k, roundK, kvSeqlen, ppNScalar, addrOHigh32, addrOLoww32, maskGm, oGm, sGm, pGm,
                    oTmpGm);
            }
        }

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flashattention_encoder_mix_aic(__gm__ uint8_t *__restrict__ qGm,
        __gm__ uint8_t *__restrict__ kGm, __gm__ uint8_t *__restrict__ vGm, __gm__ uint8_t *__restrict__ maskGm,
        __gm__ uint8_t *__restrict__ blockTablesGm, __gm__ uint8_t *__restrict__ layerIdGm,
        __gm__ uint8_t *__restrict__ oGm, __gm__ uint8_t *__restrict__ sGm, __gm__ uint8_t *__restrict__ pGm,
        __gm__ uint8_t *__restrict__ oTmpGm, __gm__ uint8_t *tiling, uint32_t *tilingParaGm)
    {
        set_padding(0);
        set_atomic_none();
        uint64_t config = 0x1;
        set_nd_para(config);
        set_mask_norm();

        __cbuf__ uint8_t *l1qBufAddr = (__cbuf__ uint8_t *)get_imm(0);
        __cbuf__ uint8_t *l1kBufAddr = (__cbuf__ uint8_t *)get_imm(2 * L0AB_UINT8_BLOCK_SIZE_LLM);
        __cbuf__ uint8_t *l1pBufAddr = (__cbuf__ uint8_t *)get_imm(4 * L0AB_UINT8_BLOCK_SIZE_LLM);
        __cbuf__ uint8_t *l1vBufAddr = (__cbuf__ uint8_t *)get_imm(6 * L0AB_UINT8_BLOCK_SIZE_LLM);
        __ca__ uint8_t *l0aBuf = (__ca__ uint8_t *)get_imm(0);
        __cb__ uint8_t *l0bBuf = (__cb__ uint8_t *)get_imm(0);
        __cc__ uint8_t *l0cBuf = (__cc__ uint8_t *)get_imm(0);

        uint32_t batchSize;
        uint32_t maxSeqlen;
        uint32_t qHeads;
        uint32_t embd;
        uint32_t kvHeads;
        uint32_t isTriuMask;
        uint32_t totalQBlkNum;

        uint32_t maxNumBlocksPerQuery, blockSize;

        if (tiling == nullptr) {
            batchSize = (uint32_t)(*((int32_t *)tilingParaGm));
            maxSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1));
            qHeads = (uint32_t)(*((int32_t *)tilingParaGm + 2));
            embd = (uint32_t)(*((int32_t *)tilingParaGm + 3));
            kvHeads = (uint32_t)(*((int32_t *)tilingParaGm + 4));
            isTriuMask = (uint32_t)(*((int32_t *)tilingParaGm + 8));
            totalQBlkNum = (uint32_t)(*((int32_t *)tilingParaGm + 9));
            maxNumBlocksPerQuery = (uint32_t)(*((int32_t *)tilingParaGm + 14));
            blockSize = (uint32_t)(*((int32_t *)tilingParaGm + 15));
        } else {
            batchSize = (uint32_t)(*((__gm__ int32_t *)tiling));
            maxSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            qHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embd = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            kvHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 4));
            isTriuMask = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            totalQBlkNum = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            maxNumBlocksPerQuery = (uint32_t)(*((__gm__ int32_t *)tiling + 14));
            blockSize = (uint32_t)(*((__gm__ int32_t *)tiling + 15));
        }

        uint32_t groupNum = qHeads / kvHeads;
        uint64_t strideQo = qHeads * embd;
        uint64_t strideKv = kvHeads * embd;

        uint32_t k = embd;
        uint32_t roundK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        uint32_t curBatch = 0;
        uint32_t preTotalQBlkNum = 0;
        uint32_t offsetTiling = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatch;
        uint32_t curTotalQBlkNum;
        if (tiling == nullptr) {
            curTotalQBlkNum = (uint32_t)(*((int32_t *)tilingParaGm + 13 + offsetTiling));
        } else {
            curTotalQBlkNum = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offsetTiling));
        }
        uint32_t processNum = totalQBlkNum * qHeads;

        uint64_t batchStrideKv = batchSize * maxSeqlen * strideKv;
        // 指针地址
        uint64_t kCacheAddr = *((__gm__ uint64_t *)kGm);
        uint64_t vCacheAddr = *((__gm__ uint64_t *)vGm);
        uint32_t layerId = *((__gm__ uint32_t *)layerIdGm);
        __gm__ half *kCacheGm = (__gm__ half *)kCacheAddr;
        __gm__ half *vCacheGm = (__gm__ half *)vCacheAddr;

        for (uint32_t process = 0; process < processNum; process++) {
            if (process >= curTotalQBlkNum * qHeads) {
                while (curBatch < batchSize) {
                    curBatch++;
                    preTotalQBlkNum = curTotalQBlkNum;
                    offsetTiling += TILING_PARA_SIZE;
                    uint32_t qSeqlen;
                    if (tiling == nullptr) {
                        curTotalQBlkNum = (uint32_t)(*((int32_t *)tilingParaGm + 13 +offsetTiling));
                        qSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + offsetTiling));
                    } else {
                        curTotalQBlkNum = (uint32_t)(*((__gm__ int32_t *)tiling + 13 +offsetTiling));
                        qSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + offsetTiling));
                    }

                    if (qSeqlen != 0) {
                        break;
                    }
                }
            }
            uint32_t curCoreIdx = process % block_num;
            if (isTriuMask) {
                if ((process / block_num) % 2 == 1) {
                    curCoreIdx = block_num - process % block_num -1;
                }
            }
            if (block_idx != curCoreIdx) {
                continue;
            }

            uint32_t qSeqlen;
            uint32_t kvSeqlen;
            uint32_t ppMScalar;
            uint32_t ppNScalar;
            uint32_t addrQHigh32;
            uint32_t addrQLoww32;
            uint32_t addrKHigh32;
            uint32_t addrKLoww32;
            uint32_t addrVHigh32;
            uint32_t addrVLoww32;
            uint32_t batchId;
            uint32_t numBlocks;
            if (tiling == nullptr) {
                qSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + offsetTiling));
                kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTiling));
                ppMScalar = (uint32_t)(*((int32_t *)tilingParaGm + 2 + offsetTiling));
                ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTiling));
                addrQHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 4 + offsetTiling));
                addrQLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 5 + offsetTiling));
                addrKHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 6 + offsetTiling));
                addrKLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 7 + offsetTiling));
                addrVHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 8 + offsetTiling));
                addrVLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 9 + offsetTiling));
                batchId = (uint32_t)(*((int32_t *)tilingParaGm + 15 + offsetTiling));
                numBlocks = (uint32_t)(*((int32_t *)tilingParaGm + 17 + offsetTiling));
            } else {
                qSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + offsetTiling));
                kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTiling));
                ppMScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 2 + offsetTiling));
                ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTiling));
                addrQHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offsetTiling));
                addrQLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offsetTiling));
                addrKHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offsetTiling));
                addrKLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offsetTiling));
                addrVHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offsetTiling));
                addrVLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offsetTiling));
                batchId = (uint32_t)(*((__gm__ int32_t *)tiling + 15 + offsetTiling));
                numBlocks = (uint32_t)(*((__gm__ int32_t *)tiling + 17 + offsetTiling));
            }
            uint64_t addrQScalar = (uint64_t)(((uint64_t)addrQHigh32) << 32 | addrQLoww32);
            uint64_t addrKScalar = (uint64_t)(((uint64_t)addrKHigh32) << 32 | addrKLoww32);
            uint64_t addrVScalar = (uint64_t)(((uint64_t)addrVHigh32) << 32 | addrVLoww32);

            uint32_t processIdx = process - preTotalQBlkNum * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint32_t headIdx = processIdx % qHeads;

            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;

            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t qkRoundM = (qkM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint64_t qkIndex = 0;
            /********************pre_load********************/
            uint32_t qkN = (qkIndex == (nLoop -1)) ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint32_t pingPongFlag = 0;
            uint32_t offset = pingPongFlag * L0AB_HALF_BUF_SIZE;

            uint32_t sPingPongFlag = 0;
            uint32_t pPingPongFlag = 0;
            uint32_t oPingPongFlag = 0;

            uint64_t qOffset = addrQScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            uint64_t strideTable = batchId * maxNumBlocksPerQuery;
            int32_t blockId0 = (int32_t)(*((__gm__ int32_t *)blockTablesGm + strideTable + qkIndex));
            int64_t layerOffset = (int64_t)layerId * numBlocks * blockSize * kvHeads * embd;
            int64_t kOffset = layerOffset + (int64_t)blockId0 * blockSize * kvHeads * embd +
                (int64_t)(headIdx / groupNum) * embd;

            // Only need load Q once
            if (qkM == 1) {
                // 0 is sid, 1 is nBurst
                copy_gm_to_cbuf((__cbuf__ half *)l1qBufAddr, (__gm__ half *)qGm + qOffset, 0, 1, roundK / BLOCK_SIZE,
                    0, 0, PAD_NONE);    // 0 is srcGap、dstGap
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1qBufAddr, (__gm__ half *)qGm + qOffset,
                    0, 1, qkM, k, 0, strideQo, qkRoundM, 1, 0); // 0:sid, 1:ndNum
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
            wait_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
            if (qkM == 1) {
                load_cbuf_to_ca(
                    (__ca__ half *)l0aBuf + offset, (__cbuf__ half *)l1qBufAddr, 0,
                    (roundK + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
            } else {
                for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0aBuf + offset + l0aLoadIdx * roundK * BLOCK_SIZE,
                        (__cbuf__ half *)l1qBufAddr + l0aLoadIdx * CUBE_MATRIX_SIZE,
                        0, roundK / BLOCK_SIZE, qkRoundM / BLOCK_SIZE, 0, 0, false, inc);
                }
            }
            // Prepare K to L1
            wait_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag);
            copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1kBufAddr + offset,
                (__gm__ half *)kCacheGm + kOffset, 0, 1, qkN, k, 0, strideKv, qkRoundN, 1, 0);
            set_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
            load_cbuf_to_cb((__cb__ half *)l0bBuf + offset, (__cbuf__ half *)l1kBufAddr + offset,
                0, roundK * qkRoundN / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
            set_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag);
            set_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
            wait_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
            wait_flag(PIPE_FIX, PIPE_M, pingPongFlag);
            mad((__cc__ float *)l0cBuf + offset, (__ca__ half *)l0aBuf + offset, (__cb__ half *)l0bBuf + offset,
                qkM, k, qkN, 0, 0, 0, 1);
            set_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
            set_flag(PIPE_M, PIPE_FIX, pingPongFlag);
            wait_flag(PIPE_M, PIPE_FIX, pingPongFlag);
            // copy S to gm
            copy_matrix_cc_to_gm((__gm__ half *)sGm + (uint64_t)block_idx * TMP_SIZE + sPingPongFlag * TMP_SIZE / 2,
                (__cc__ float *)l0cBuf + offset, 0, qkRoundN, qkM, qkRoundN, qkRoundM, 0, F322F16, 0, false, true);
            set_flag(PIPE_FIX, PIPE_M, pingPongFlag);
            pingPongFlag = 1 - pingPongFlag;
            offset = pingPongFlag * L0AB_HALF_BUF_SIZE;
            ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY << BIT_SHIFT_LLM));
            sPingPongFlag = 1 - sPingPongFlag;
            qkIndex++;

            uint32_t svN = ppNScalar;
            uint32_t svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            uint64_t vOffset = kOffset;
            uint32_t nEnd = nLoop;
            if (isTriuMask) {
                nEnd = mIdx + 1;
            }
            for (uint32_t nIdx = 0; nIdx < nEnd; nIdx++) {
                if (qkIndex < nEnd) {
                    if (qkIndex == (nLoop - 1)) {
                        qkN = (kvSeqlen - qkIndex * ppNScalar);
                        qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    }
                    blockId0 = (int32_t)(*((__gm__ int32_t *)blockTablesGm + strideTable + qkIndex));
                    kOffset = layerOffset + (int64_t)blockId0 * blockSize * kvHeads * embd +
                        (int64_t)(headIdx / groupNum) * embd;

                    wait_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
                    if (qkM == 1) {
                        load_cbuf_to_ca((__ca__ half *)l0aBuf + offset, (__cbuf__ half *)l1qBufAddr, 0,
                            (roundK + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
                    } else {
                        for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                            load_cbuf_to_ca((__ca__ half *)l0aBuf + offset + l0aLoadIdx * roundK * BLOCK_SIZE,
                                (__cbuf__ half *)l1qBufAddr + l0aLoadIdx * CUBE_MATRIX_SIZE, 0, roundK / BLOCK_SIZE,
                                qkRoundM / BLOCK_SIZE, 0, 0, false, inc);
                        }
                    }
                    // Prepare K to L1
                    wait_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag);
                    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1kBufAddr + offset,
                        (__gm__ half *)kCacheGm + kOffset, 0, 1, qkN, k, 0, strideKv, qkRoundN, 1, 0);
                    set_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                    load_cbuf_to_cb((__cb__ half *)l0bBuf + offset, (__cbuf__ half *)l1kBufAddr + offset,
                        0, roundK * qkRoundN / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
                    set_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag);
                    set_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
                    wait_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
                    wait_flag(PIPE_FIX, PIPE_M, pingPongFlag);
                    mad((__cc__ float *)l0cBuf + offset, (__ca__ half *)l0aBuf + offset,
                        (__cb__ half *)l0bBuf + offset, qkM, k, qkN, 0, 0, 0, 1);
                    set_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
                    set_flag(PIPE_M, PIPE_FIX, pingPongFlag);
                    wait_flag(PIPE_M, PIPE_FIX, pingPongFlag);
                    // copy S to gm
                    copy_matrix_cc_to_gm(
                        (__gm__ half *)sGm + (uint64_t)block_idx * TMP_SIZE + sPingPongFlag * TMP_SIZE / 2,
                        (__cc__ float *)l0cBuf + offset, 0, qkRoundN, qkM, qkRoundN, qkRoundM, 0, F322F16, 0,
                        false, true);
                    set_flag(PIPE_FIX, PIPE_M, pingPongFlag);
                    pingPongFlag = 1 - pingPongFlag;
                    offset = pingPongFlag * L0AB_HALF_BUF_SIZE;
                    ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY << BIT_SHIFT_LLM));
                    sPingPongFlag = 1 - sPingPongFlag;
                }

                if (nIdx == (nLoop -1)) {
                    svN = (kvSeqlen - nIdx * ppNScalar);
                    svRoundN = (svN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                }
                // Prepare V to L1
                wait_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag + 2);
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1vBufAddr + offset,
                    (__gm__ half *)vCacheGm + vOffset, 0, 1, svN, k, 0, strideKv, svRoundN, 1, 0);
                set_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                wait_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
                for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < svRoundN / BLOCK_SIZE; ++l0bLoadIdx) {
                    load_cbuf_to_cb(
                        (__cb__ half *)l0bBuf + offset + l0bLoadIdx * roundK * BLOCK_SIZE,
                        (__cbuf__ half *)l1vBufAddr + offset + l0bLoadIdx * CUBE_MATRIX_SIZE,
                        0, roundK / BLOCK_SIZE, svRoundN / BLOCK_SIZE, 0, 0, true, inc);
                }
                vOffset = kOffset;

                wait_flag_dev(SOFTMAX_READY);
                // Prepare P to L1
                if (qkM == 1) {
                    copy_gm_to_cbuf(
                        (__cbuf__ half *)l1pBufAddr + offset,
                        (__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE + pPingPongFlag * TMP_SIZE / 2,
                        0, 1, svRoundN / BLOCK_SIZE, 0, 0, PAD_NONE
                    );
                } else {
                    copy_gm_to_cbuf_multi_nd2nz_b16(
                        (__cbuf__ half *)l1pBufAddr + offset,
                        (__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE + pPingPongFlag * TMP_SIZE / 2,
                        0, 1, qkM, svN, 0, svRoundN, qkRoundM, 1, 0);
                }
                pPingPongFlag = 1 - pPingPongFlag;
                set_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, pingPongFlag);
                if (qkM == 1) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0aBuf + offset, (__cbuf__ half *)l1pBufAddr + offset,
                        0, (svRoundN + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
                } else {
                    for (uint32_t l0aLoadIdx = 0; l0aLoadIdx < qkRoundM / BLOCK_SIZE; ++l0aLoadIdx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0aBuf + offset + l0aLoadIdx * svRoundN * BLOCK_SIZE,
                            (__cbuf__ half *)l1pBufAddr + offset + l0aLoadIdx * CUBE_MATRIX_SIZE,
                            0, svRoundN / BLOCK_SIZE, qkRoundM / BLOCK_SIZE, 0, 0, false, inc);
                    }
                }
                set_flag(PIPE_MTE1, PIPE_MTE2, pingPongFlag + 2);
                set_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
                wait_flag(PIPE_MTE1, PIPE_M, pingPongFlag);
                wait_flag(PIPE_FIX, PIPE_M, pingPongFlag);
                mad((__cc__ float *)l0cBuf + offset, (__ca__ half *)l0aBuf + offset, (__cb__ half *)l0bBuf + offset,
                    qkM, svN, k, 0, 0, 0, 1);
                set_flag(PIPE_M, PIPE_MTE1, pingPongFlag);
                set_flag(PIPE_M, PIPE_FIX, pingPongFlag);
                wait_flag(PIPE_M, PIPE_FIX, pingPongFlag);
                // copy O to gm
                copy_matrix_cc_to_gm(
                    (__gm__ float *)oTmpGm + (uint64_t)block_idx * TMP_SIZE + oPingPongFlag * TMP_SIZE / 2,
                    (__cc__ float *)l0cBuf + offset, 0, roundK, qkM, roundK, qkRoundM, 0, NoQuant, 0, false, true);
                set_flag(PIPE_FIX, PIPE_M, pingPongFlag);
                pingPongFlag = 1 - pingPongFlag;
                offset = pingPongFlag * L0AB_HALF_BUF_SIZE;
                ffts_cross_core_sync(PIPE_FIX, 0x21 + (UPDATE_READY_LLM << BIT_SHIFT_LLM));
                oPingPongFlag = 1 - oPingPongFlag;
                qkIndex++;
            }
        }
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flashattention_encoder_mix_aiv(__gm__ uint8_t *__restrict__ qGm,
        __gm__ uint8_t *__restrict__ kGm, __gm__ uint8_t *__restrict__ vGm, __gm__ uint8_t *__restrict__ maskGm,
        __gm__ uint8_t *__restrict__ oGm, __gm__ uint8_t *__restrict__ sGm, __gm__ uint8_t *__restrict__ pGm,
        __gm__ uint8_t *__restrict__ oTmpGm, __gm__ uint8_t *tiling, uint32_t *tilingParaGm)
    {
        int32_t subBlockIdx = get_subblockid();
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        __ubuf__ uint8_t *lsUbuf = (__ubuf__ uint8_t *)get_imm(0);
        __ubuf__ uint8_t *lpUbuf = (__ubuf__ uint8_t *)get_imm(0);
        __ubuf__ uint8_t *ls32Ubuf =
            (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE_PREFILL);
        __ubuf__ uint8_t *maskUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE_PREFILL);
        __ubuf__ uint8_t *loUbuf = (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_PREFILL);
        __ubuf__ uint8_t *lmUbuf = (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL);
        __ubuf__ uint8_t *hmUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 1 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *gmUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 2 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *dmUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 4 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *llUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 6 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *glUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 10 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *tvUbuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL + 11 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *goUbuf = (__ubuf__ uint8_t *)get_imm(8 * UB_UINT8_BLOCK_SIZE_PREFILL);
        uint32_t goFlagScalar = 1;

        uint32_t batchSize;
        uint32_t maxSeqlen;
        uint32_t qHeads;
        uint32_t embd;
        half tor;
        uint32_t maskStride;
        uint32_t isTriuMask;
        uint32_t totalQBlkNum;
        uint32_t isClamp;
        half clampMin;
        half clampMax;
        uint32_t headStride;

        if (tiling == nullptr) {
            batchSize = (uint32_t)(*((int32_t *)tilingParaGm));
            maxSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1));
            qHeads = (uint32_t)(*((int32_t *)tilingParaGm + 2));
            embd = (uint32_t)(*((int32_t *)tilingParaGm + 3));
            tor = (half)(*((float *)tilingParaGm + 5));
            maskStride = (uint32_t)(*((uint32_t *)tilingParaGm + 7));
            isTriuMask = (uint32_t)(*((int32_t *)tilingParaGm + 8));
            totalQBlkNum = (uint32_t)(*((int32_t *)tilingParaGm + 9));
            isClamp = (uint32_t)(*((uint32_t *)tilingParaGm + 10));
            clampMin = (half)(*((float *)tilingParaGm + 11));
            clampMax = (half)(*((float *)tilingParaGm + 12));
            headStride = (uint32_t)(*((uint32_t *)tilingParaGm + 13));
        } else {
            batchSize = (uint32_t)(*((__gm__ int32_t *)tiling));
            maxSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            qHeads = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embd = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            tor = (half)(*((__gm__ float *)tiling + 5));
            maskStride = (uint32_t)(*((__gm__ uint32_t *)tiling + 7));
            isTriuMask = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            totalQBlkNum = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            isClamp = (uint32_t)(*((__gm__ uint32_t *)tiling + 10));
            clampMin = (half)(*((__gm__ float *)tiling + 11));
            clampMax = (half)(*((__gm__ float *)tiling + 12));
            headStride = (uint32_t)(*((__gm__ uint32_t *)tiling + 13));
        }

        uint64_t strideQo = qHeads * embd;

        uint32_t k = embd;
        uint32_t roundK = (k + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        uint32_t curBatchEncoderAiv = 0;
        uint32_t preTotalQBlkNumEncoderAiv = 0;
        uint32_t offsetTilingEncoderAiv = TILING_HEAD_SIZE + TILING_PARA_SIZE * curBatchEncoderAiv;
        uint32_t curTotalQBlkNumEncoderAiv;
        if (tiling == nullptr) {
            curTotalQBlkNumEncoderAiv = (uint32_t)(*((int32_t *)tilingParaGm + 13 + offsetTilingEncoderAiv));
        } else {
            curTotalQBlkNumEncoderAiv = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offsetTilingEncoderAiv));
        }
        uint32_t processNum = totalQBlkNum * qHeads;
        for (uint32_t processTmp = 0; processTmp < processNum; processTmp++) {
            if (processTmp >= curTotalQBlkNumEncoderAiv * qHeads) {
                while (curBatchEncoderAiv < batchSize) {
                    curBatchEncoderAiv++;
                    preTotalQBlkNumEncoderAiv = curTotalQBlkNumEncoderAiv;
                    offsetTilingEncoderAiv += TILING_PARA_SIZE;
                    uint32_t qSeqlenTmp;
                    if (tiling == nullptr) {
                        curTotalQBlkNumEncoderAiv = (uint32_t)(*((int32_t *)tilingParaGm + 13 +
                        offsetTilingEncoderAiv));
                        qSeqlenTmp = (uint32_t)(*((int32_t *)tilingParaGm + offsetTilingEncoderAiv));
                    } else {
                        curTotalQBlkNumEncoderAiv = (uint32_t)(*((__gm__ int32_t *)tiling + 13 +
                        offsetTilingEncoderAiv));
                        qSeqlenTmp = (uint32_t)(*((__gm__ int32_t *)tiling + offsetTilingEncoderAiv));
                    }
                    if (qSeqlenTmp != 0) {
                        break;
                    }
                }
            }
            uint32_t curCoreIdx = processTmp % block_num;
            if (isTriuMask) {
                if ((processTmp / block_num) % 2 == 1) {
                    curCoreIdx = block_num - processTmp % block_num - 1;
                }
            }
            if (block_idx != curCoreIdx) {
                continue;
            }

            uint32_t qSeqlen;
            uint32_t kvSeqlen;
            uint32_t ppMScalar;
            uint32_t ppNScalar;
            uint32_t addrOHigh32;
            uint32_t addrOLoww32;
            if (tiling == nullptr) {
                qSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + offsetTilingEncoderAiv));
                kvSeqlen = (uint32_t)(*((int32_t *)tilingParaGm + 1 + offsetTilingEncoderAiv));
                ppMScalar = (uint32_t)(*((int32_t *)tilingParaGm + 2 + offsetTilingEncoderAiv));
                ppNScalar = (uint32_t)(*((int32_t *)tilingParaGm + 3 + offsetTilingEncoderAiv));
                addrOHigh32 = (uint32_t)(*((int32_t *)tilingParaGm + 10 + offsetTilingEncoderAiv));
                addrOLoww32 = (uint32_t)(*((int32_t *)tilingParaGm + 11 + offsetTilingEncoderAiv));
            } else {
                qSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + offsetTilingEncoderAiv));
                kvSeqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offsetTilingEncoderAiv));
                ppMScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 2 + offsetTilingEncoderAiv));
                ppNScalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offsetTilingEncoderAiv));
                addrOHigh32 = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offsetTilingEncoderAiv));
                addrOLoww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offsetTilingEncoderAiv));
            }
            uint64_t addrOScalar = (uint64_t)(((uint64_t)addrOHigh32) << 32 | addrOLoww32);

            uint32_t processIdx = processTmp - preTotalQBlkNumEncoderAiv * qHeads;
            uint32_t mIdx = processIdx / qHeads;
            uint32_t headIdx = processIdx % qHeads;

            uint32_t mLoop = (qSeqlen + ppMScalar - 1) / ppMScalar;
            uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;

            uint32_t qkM = (mIdx == (mLoop - 1)) ? (qSeqlen - mIdx * ppMScalar) : ppMScalar;
            uint32_t subM = (subBlockIdx == 1) ? (qkM - qkM / 2) : qkM / 2;
            uint32_t subMD128 = (subM + VECTOR_SIZE - 1) / VECTOR_SIZE;
            uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
            uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint64_t qkIndex = 0;
            /********pre_load*******/
            uint32_t qkN = (qkIndex == (nLoop - 1)) ? kvSeqlen : ppNScalar;
            uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

            uint32_t pingPongFlag = 0;
            uint32_t offset = pingPongFlag * UB_HALF_BUF_SIZE;
            uint64_t maskBatchOffset = 0;
            uint64_t maskHeadOffset = 0;
            uint64_t maskOffset = maskBatchOffset + maskHeadOffset + mIdx * ppMScalar * maxSeqlen;

            uint32_t sPingPongFlag = 0;
            uint32_t pPingPongFlag = 0;
            uint32_t oPingPongFlag = 0;

            if (subM > 0 && maskGm != nullptr) {
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                uint64_t maskOffsetTmp = maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen;
                uint32_t tmp = (maxSeqlen - qkN) * 2;
                copy_gm_to_ubuf_align_b16((__ubuf__ half *)maskUbuf, (__gm__ half *)maskGm + maskOffsetTmp,
                    0, subM, qkN * 2, 0, 0, tmp, 0);
            }
            wait_flag_dev(QK_READY);
            if (subM > 0) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, pingPongFlag);
                // input QK
                copy_gm_to_ubuf(
                    (__ubuf__ half *)lsUbuf + offset,
                    (__gm__ half *)sGm + (uint64_t)block_idx * TMP_SIZE + sPingPongFlag * TMP_SIZE / 2 +
                        (uint64_t)subBlockIdx * qkM / 2 * qkRoundN,
                    0, 1, subM * qkRoundN / BLOCK_SIZE, 0, 0);
                sPingPongFlag = 1 - sPingPongFlag;
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                // ls = tor * ls
                vmuls((__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)lsUbuf + offset, tor,
                    (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE, 1, 1, 8, 8);

                pipe_barrier(PIPE_V);
                // ls = ls + mask
                if (maskGm != nullptr) {
                    VecAddLlm((__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)lsUbuf + offset,
                        (__ubuf__ half *)maskUbuf, (subM * qkRoundN + VECTOR_SIZE -1) / VECTOR_SIZE,
                        1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                }
                // lm = rowmax(ls)
                if (qkN <= VECTOR_SIZE) {
                    SetMask(qkN);
                    vcmax((__ubuf__ half *)lmUbuf, (__ubuf__ half *)lsUbuf + offset, subM, 1, 1,
                        qkRoundN / BLOCK_SIZE, ONLY_VALUE);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    CopyUbufToUbufLlm((__ubuf__ half *)ls32Ubuf, (__ubuf__ half *)lsUbuf + offset, 0, subM,
                        VECTOR_SIZE / BLOCK_SIZE, (qkRoundN - VECTOR_SIZE) / BLOCK_SIZE, 0);
                    pipe_barrier(PIPE_V);
                    SetMask(qkN - VECTOR_SIZE);
                    VecMaxLlm((__ubuf__ half *)ls32Ubuf, (__ubuf__ half *)ls32Ubuf,
                        (__ubuf__ half *)lsUbuf + offset + VECTOR_SIZE,
                        subM, 1, 1, 1, 8, 8, qkRoundN / BLOCK_SIZE
                    );
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    vcmax((__ubuf__ half *)lmUbuf, (__ubuf__ half *)ls32Ubuf, subM, 1, 1, 8, ONLY_VALUE);
                }
                pipe_barrier(PIPE_V);
                // hm = lm
                CopyUbufToUbufLlm((__ubuf__ half *)hmUbuf,
                    (__ubuf__ half *)lmUbuf, 0, 1, roundSubM / BLOCK_SIZE, 0, 0);
                pipe_barrier(PIPE_V);
                // gm == hm
                CopyUbufToUbufLlm(
                    (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, 0, 1, roundSubM / BLOCK_SIZE, 0, 0);
                pipe_barrier(PIPE_V);
                // hm_block = expand_to_block(hm)
                vbrcb(
                    (__ubuf__ uint16_t *)tvUbuf, (__ubuf__ uint16_t *)hmUbuf, 1, 8, roundSubM / FLOAT_BLOCK_SIZE);
                pipe_barrier(PIPE_V);
                // ls = ls - hm_block
                for (uint32_t vsubIdx = 0; vsubIdx < qkN / VECTOR_SIZE; ++vsubIdx) {
                    VecSubLlm((__ubuf__ half *)lsUbuf + offset + vsubIdx * VECTOR_SIZE,
                        (__ubuf__ half *)lsUbuf + offset + vsubIdx * VECTOR_SIZE,
                        (__ubuf__ half *)tvUbuf, subM, 1, 1, 0, qkRoundN / BLOCK_SIZE,
                        qkRoundN / BLOCK_SIZE, 1);
                }
                if (qkN % VECTOR_SIZE > 0) {
                    SetMask(qkN % VECTOR_SIZE);
                    VecSubLlm((__ubuf__ half *)lsUbuf + offset + qkN / VECTOR_SIZE * VECTOR_SIZE,
                        (__ubuf__ half *)lsUbuf + offset + qkN / VECTOR_SIZE * VECTOR_SIZE,
                        (__ubuf__ half *)tvUbuf, subM, 1, 1, 0, qkRoundN / BLOCK_SIZE,
                        qkRoundN / BLOCK_SIZE, 1);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                pipe_barrier(PIPE_V);
                // ls = castfp16to32(ls)
                vconv_f162f32((__ubuf__ float *)ls32Ubuf,
                    (__ubuf__ half *)lsUbuf + offset,
                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                    1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                // ls = exp(ls)
                VecExpLlm((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                     (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                     1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                // lp = castfp32to16(ls)
                vconv_f322f16((__ubuf__ half *)lpUbuf + offset, (__ubuf__ float *)ls32Ubuf,
                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 1, 1, 4, 8);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                // ll = rowsum(ls32)
                if (qkN <= FLOAT_VECTOR_SIZE) {
                    SetMask(qkN);
                    vcadd((__ubuf__ float *)llUbuf, (__ubuf__ float *)ls32Ubuf, subM,
                        1, 1, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    for (uint32_t rowsumIdx = 1; rowsumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowsumIdx) {
                        VecAddLlm((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                            (__ubuf__ float *)ls32Ubuf + rowsumIdx * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE,
                            qkRoundN / FLOAT_BLOCK_SIZE);
                        pipe_barrier(PIPE_V);
                    }
                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                        SetMask(qkN % FLOAT_VECTOR_SIZE);
                        VecAddLlm((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                            (__ubuf__ float *)ls32Ubuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE,
                            qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcadd((__ubuf__ float *)llUbuf, (__ubuf__ float *)ls32Ubuf,
                        subM, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                }
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm(
                    (__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE + pPingPongFlag * TMP_SIZE / 2 +
                        (uint64_t)subBlockIdx * qkM / 2 * qkRoundN,
                    (__ubuf__ half *)lpUbuf + offset, 0, 1, subM * qkRoundN / BLOCK_SIZE, 0, 0);
                set_flag(PIPE_MTE3, PIPE_MTE2, pingPongFlag);
                pingPongFlag = 1 - pingPongFlag;
                offset = pingPongFlag * UB_HALF_BUF_SIZE;
                pPingPongFlag = 1 - pPingPongFlag;
            }
            qkIndex++;
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY << BIT_SHIFT_LLM));

            uint64_t oOffset = addrOScalar + headIdx * embd + mIdx * ppMScalar * strideQo;
            uint32_t nEndNew = nLoop;
            if (isTriuMask) {
                nEndNew = mIdx + 1;
            }
            for (uint32_t nIdx = 0; nIdx < nEndNew; nIdx++) {
                if (qkIndex < nEndNew) {
                    if (qkIndex == (nLoop - 1)) {
                        qkN = (kvSeqlen - qkIndex * ppNScalar);
                        qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
                    }
                    if (subM > 0 && maskGm != nullptr) {
                        maskOffset += ppNScalar;
                        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                        copy_gm_to_ubuf_align_b16(
                            (__ubuf__ half *)maskUbuf,
                            (__gm__ half *)maskGm + maskOffset + (uint64_t)subBlockIdx * qkM / 2 * maxSeqlen,
                            0, subM,                      // nBurst
                            qkN * 2,                      // lenBurst
                            0, 0, (maxSeqlen - qkN) * 2, 0);
                    }
                    wait_flag_dev(QK_READY);
                    if (subM > 0) {
                        wait_flag(PIPE_MTE3, PIPE_MTE2, pingPongFlag);
                        // input QK
                        copy_gm_to_ubuf(
                            (__ubuf__ half *)lsUbuf + offset,
                            (__gm__ half *)sGm + (uint64_t)block_idx * TMP_SIZE + sPingPongFlag * TMP_SIZE / 2 +
                                (uint64_t)subBlockIdx * qkM / 2 * qkRoundN,
                            0, 1, subM * qkRoundN / BLOCK_SIZE, 0, 0);
                        sPingPongFlag = 1 - sPingPongFlag;
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        // *** ls = tor * ls
                        vmuls((__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)lsUbuf + offset, tor,
                            (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,   // repeat
                            1, 1,                                                      // srcBlockStride
                            8, 8                                                       // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);

                        if (isClamp == 1) {
                            // get min(clampMin, lsUbuf)
                            vmaxs((__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)lsUbuf + offset,
                                clampMin, (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,   // repeat
                                1, 1,                                                      // srcBlockStride
                                8, 8                                                       // srcRepeatStride
                            );
                            pipe_barrier(PIPE_V);

                            // get max(clampMin, lsUbuf)
                            vmins((__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)lsUbuf + offset,
                                clampMax, (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,   // repeat
                                1, 1,                                                      // srcBlockStride
                                8, 8                                                       // srcRepeatStride
                            );
                            pipe_barrier(PIPE_V);
                        }

                        // *** ls = ls + mask
                        if (maskGm != nullptr) {
                            VecAddLlm((__ubuf__ half *)lsUbuf + offset,
                                (__ubuf__ half *)lsUbuf + offset, (__ubuf__ half *)maskUbuf,
                                (subM * qkRoundN + VECTOR_SIZE - 1) / VECTOR_SIZE,   // repeat
                                1, 1, 1,                                                      // src1BlockStride
                                8, 8, 8                                                       // src1RepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                        }
                        // *** lm = rowmax(ls)
                        if (qkN <= VECTOR_SIZE) {
                            SetMask(qkN);
                            vcmax((__ubuf__ half *)lmUbuf, (__ubuf__ half *)lsUbuf + offset,
                                subM, 1, 1,                          // srcBlockStride
                                qkRoundN / BLOCK_SIZE, ONLY_VALUE                  // order
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        } else {
                            CopyUbufToUbufLlm((__ubuf__ half *)ls32Ubuf,
                                (__ubuf__ half *)lsUbuf + offset, 0,                                          // sid
                                subM, VECTOR_SIZE / BLOCK_SIZE, (qkRoundN - VECTOR_SIZE) / BLOCK_SIZE, 0);
                            pipe_barrier(PIPE_V);
                            SetMask(qkN - VECTOR_SIZE);
                            VecMaxLlm((__ubuf__ half *)ls32Ubuf, (__ubuf__ half *)ls32Ubuf,
                                (__ubuf__ half *)lsUbuf + offset + VECTOR_SIZE, subM,                      // repeat
                                1, 1, 1,                          // src1BlockStride
                                8, 8,                          // src0RepeatStride
                                qkRoundN / BLOCK_SIZE     // order
                            );
                            pipe_barrier(PIPE_V);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            vcmax((__ubuf__ half *)lmUbuf, (__ubuf__ half *)ls32Ubuf,
                                subM, 1, 1,                          // srcBlockStride
                                8,                          // srcRepeatStride
                                ONLY_VALUE                  // order
                            );
                        }
                        pipe_barrier(PIPE_V);
                        // *** hm = VecMaxLlm(lm, gm)
                        VecMaxLlm((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf,
                            (__ubuf__ half *)gmUbuf, subMD128, 1, 1, 1,                          // src1BlockStride
                            8, 8, 8                           // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** dm = gm - hm
                        VecSubLlm((__ubuf__ half *)dmUbuf + qkIndex % 2 * UB_HALF_LINE_SIZE,
                            (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, subMD128,                 // repeat
                            1, 1, 1,                          // src1BlockStride
                            8, 8, 8                           // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** gm = hm
                        CopyUbufToUbufLlm((__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf,
                            0, 1, roundSubM / BLOCK_SIZE, 0, 0);
                        pipe_barrier(PIPE_V);
                        // *** hm_block = expand_to_block(hm), 存放于 tv
                        vbrcb((__ubuf__ uint16_t *)tvUbuf, (__ubuf__ uint16_t *)hmUbuf, 1, 8,
                            roundSubM / FLOAT_BLOCK_SIZE);
                        pipe_barrier(PIPE_V);
                        // *** ls = ls - hm_block
                        for (uint32_t vsubIdx = 0; vsubIdx < qkN / VECTOR_SIZE; ++vsubIdx) {
                            VecSubLlm((__ubuf__ half *)lsUbuf + offset + vsubIdx * VECTOR_SIZE,
                                (__ubuf__ half *)lsUbuf + offset + vsubIdx * VECTOR_SIZE,
                                (__ubuf__ half *)tvUbuf, subM, 1, 1, 0,
                                qkRoundN / BLOCK_SIZE,    // dstRepeatStride
                                qkRoundN / BLOCK_SIZE,    // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                        }
                        if (qkN % VECTOR_SIZE > 0) {
                            SetMask(qkN % VECTOR_SIZE);
                            VecSubLlm((__ubuf__ half *)lsUbuf + offset + qkN / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)lsUbuf + offset + qkN / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)tvUbuf, subM, 1, 1, 0,
                                qkRoundN / BLOCK_SIZE,    // dstRepeatStride
                                qkRoundN / BLOCK_SIZE,    // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        pipe_barrier(PIPE_V);
                        // *** ls = castfp16to32(ls)
                        vconv_f162f32((__ubuf__ float *)ls32Ubuf, (__ubuf__ half *)lsUbuf + offset,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,   // repeat
                            1, 1,                                                                  // srcBlockStride
                            8,                                                                  // dstRepeatStride
                            4                                                                   // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** ls = exp(ls)
                        VecExpLlm((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,   // repeat
                            1, 1,                                                                  // srcBlockStride
                            8, 8                                                                   // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** lp = castfp32to16(ls)
                        vconv_f322f16((__ubuf__ half *)lpUbuf + offset, (__ubuf__ float *)ls32Ubuf,
                            (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,   // repeat
                            1, 1,                                                                  // srcBlockStride
                            4,                                                                  // dstRepeatStride
                            8                                                                   // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        // *** ll = rowsum(ls32)
                        if (qkN <= FLOAT_VECTOR_SIZE) {
                            SetMask(qkN);
                            vcadd((__ubuf__ float *)llUbuf + qkIndex % 2 * UB_FLOAT_LINE_SIZE,
                                (__ubuf__ float *)ls32Ubuf, subM,                          // repeat
                                1, 1,                              // srcBlockStride
                                qkRoundN / FLOAT_BLOCK_SIZE, 0);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        } else {
                            for (uint32_t rowsumIdx = 1; rowsumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowsumIdx) {
                                VecAddLlm((__ubuf__ float *)ls32Ubuf,
                                    (__ubuf__ float *)ls32Ubuf,
                                    (__ubuf__ float *)ls32Ubuf + rowsumIdx * FLOAT_VECTOR_SIZE,
                                    subM, 1, 1, 1,                              // src1BlockStride
                                    qkRoundN / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                                    qkRoundN / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                                    qkRoundN / FLOAT_BLOCK_SIZE   // src1RepeatStride
                                );
                                pipe_barrier(PIPE_V);
                            }
                            if (qkN % FLOAT_VECTOR_SIZE > 0) {
                                SetMask(qkN % FLOAT_VECTOR_SIZE);
                                VecAddLlm((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                                    (__ubuf__ float *)ls32Ubuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                                    subM, 1, 1, 1,                              // src1BlockStride
                                    qkRoundN / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                                    qkRoundN / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                                    qkRoundN / FLOAT_BLOCK_SIZE   // src1RepeatStride
                                );
                                set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            }
                            pipe_barrier(PIPE_V);
                            vcadd((__ubuf__ float *)llUbuf + qkIndex % 2 * UB_FLOAT_LINE_SIZE,
                                (__ubuf__ float *)ls32Ubuf,
                                subM, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                        }
                        pipe_barrier(PIPE_V);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        copy_ubuf_to_gm(
                            (__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE + pPingPongFlag * TMP_SIZE / 2 +
                                (uint64_t)subBlockIdx * qkM / 2 * qkRoundN,
                            (__ubuf__ half *)lpUbuf + offset, 0, 1,
                            subM * qkRoundN / BLOCK_SIZE, 0, 0);
                        set_flag(PIPE_MTE3, PIPE_MTE2, pingPongFlag);
                        pingPongFlag = 1 - pingPongFlag;
                        offset = pingPongFlag * UB_HALF_BUF_SIZE;
                        pPingPongFlag = 1 - pPingPongFlag;
                    }
                    qkIndex++;
                    ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY << BIT_SHIFT_LLM));   // 2
                }

                wait_flag_dev(UPDATE_READY_LLM);    // 4
                if (subM > 0) {
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                    copy_gm_to_ubuf((__ubuf__ float *)loUbuf,
                        (__gm__ float *)oTmpGm + (uint64_t)block_idx * TMP_SIZE + oPingPongFlag * TMP_SIZE / 2 +
                            (uint64_t)subBlockIdx * qkM / 2 * roundK,
                        0, 1, subM * roundK / FLOAT_BLOCK_SIZE, 0, 0);
                    oPingPongFlag = 1 - oPingPongFlag;
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    // *** 更新 L 和 O
                    if (nIdx != 0) {
                        // *** dm32 = castfp16to32(dm)， 存放于 tv
                        vconv_f162f32((__ubuf__ float *)tvUbuf,
                            (__ubuf__ half *)dmUbuf + nIdx % 2 * UB_HALF_LINE_SIZE,
                            subMD64,      // repeat
                            1,              // dstBlockStride
                            1,              // srcBlockStride
                            8,              // dstRepeatStride
                            4               // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** dm_block = expand_to_block(dm)， 存放于 tv
                        vbrcb((__ubuf__ uint32_t *)tvUbuf + VECTOR_SIZE, (__ubuf__ uint32_t *)tvUbuf,
                            1,                              // dstBlockStride
                            8,                              // dstRepeatStride
                            roundSubM / FLOAT_BLOCK_SIZE  // repeat
                        );
                        pipe_barrier(PIPE_V);
                        // *** dm = exp(dm)
                        VecExpLlm((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf, subMD64,  // repeat
                            1, 1,              // srcBlockStride
                            8, 8               // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** gl = dm * gl
                        VecMulLlm((__ubuf__ float *)glUbuf,
                            (__ubuf__ float *)tvUbuf,
                            (__ubuf__ float *)glUbuf,
                            subMD64,  // repeat
                            1, 1, 1,          // src1BlockStride
                            8, 8, 8           // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** gl = ll + gl
                        VecAddLlm((__ubuf__ float *)glUbuf, (__ubuf__ float *)glUbuf,
                            (__ubuf__ float *)llUbuf + nIdx % 2 * UB_FLOAT_LINE_SIZE, subMD64,  // repeat
                            1, 1, 1,          // src1BlockStride
                            8, 8, 8           // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** dm_block = exp(dm_block)
                        VecExpLlm((__ubuf__ float *)tvUbuf + VECTOR_SIZE, (__ubuf__ float *)tvUbuf + VECTOR_SIZE,
                            (subM * FLOAT_BLOCK_SIZE + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
                            1, 1,                                                                      // srcBlockStride
                            8, 8                          // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        if (goFlagScalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            goFlagScalar = 0;
                        }
                        // *** go = go * dm_block
                        for (uint32_t vmulIdx = 0; vmulIdx < k / FLOAT_VECTOR_SIZE; ++vmulIdx) {
                            VecMulLlm((__ubuf__ float *)goUbuf + vmulIdx * FLOAT_VECTOR_SIZE,
                                (__ubuf__ float *)goUbuf + vmulIdx * FLOAT_VECTOR_SIZE,
                                (__ubuf__ float *)tvUbuf + VECTOR_SIZE, subM,         // repeat
                                1, 1,                          // src0BlockStride
                                0, roundK / FLOAT_BLOCK_SIZE, // dstRepeatStride
                                roundK / FLOAT_BLOCK_SIZE, 1                           // src1RepeatStride
                            );
                        }
                        if (k % FLOAT_VECTOR_SIZE > 0) {
                            SetMask(k % FLOAT_VECTOR_SIZE);
                            VecMulLlm((__ubuf__ float *)goUbuf + k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                                (__ubuf__ float *)goUbuf + k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                                (__ubuf__ float *)tvUbuf + VECTOR_SIZE,
                                subM, 1, 1,                          // src0BlockStride
                                0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, // src0RepeatStride
                                1                           // src1RepeatStride
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        pipe_barrier(PIPE_V);
                        // *** go = lo + go
                        VecAddLlm((__ubuf__ float *)goUbuf,
                            (__ubuf__ float *)goUbuf,
                            (__ubuf__ float *)loUbuf,
                            (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,  // repeat
                            1, 1, 1,                                                              // src1BlockStride
                            8, 8, 8                                                               // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                    } else {
                        // gl = ll
                        CopyUbufToUbufLlm(
                            (__ubuf__ float *)glUbuf,
                            (__ubuf__ float *)llUbuf + nIdx % 2 * UB_FLOAT_LINE_SIZE,
                            0, 1,
                            roundSubM / FLOAT_BLOCK_SIZE, 0, 0);
                        pipe_barrier(PIPE_V);
                        if (goFlagScalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            goFlagScalar = 0;
                        }
                        // go = lo
                        CopyUbufToUbufLlm((__ubuf__ float *)goUbuf, (__ubuf__ float *)loUbuf,
                            0, 1, subM * roundK / FLOAT_BLOCK_SIZE, 0, 0);
                        pipe_barrier(PIPE_V);
                    }
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                    if (nIdx == nEndNew - 1) {
                        // gl = castfp32to16(gl)
                        vconv_f322f16((__ubuf__ half *)glUbuf, (__ubuf__ float *)glUbuf, subMD64,
                            1, 1, 4, 8);
                        pipe_barrier(PIPE_V);
                        // go = castfp32to16(go)
                        vconv_f322f16((__ubuf__ half *)goUbuf, (__ubuf__ float *)goUbuf,
                            (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,
                            1, 1, 4, 8
                        );
                        pipe_barrier(PIPE_V);
                        // gl_block = expand_to_block(gl)
                        vbrcb((__ubuf__ uint16_t *)tvUbuf, (__ubuf__ uint16_t *)glUbuf,
                            1, 8, roundSubM / FLOAT_BLOCK_SIZE);
                        pipe_barrier(PIPE_V);
                        // go = go / gl_block
                        for (uint32_t vdivIdx = 0; vdivIdx < k / VECTOR_SIZE; ++vdivIdx) {
                            VecDivLlm((__ubuf__ half *)goUbuf + vdivIdx * VECTOR_SIZE,
                                (__ubuf__ half *)goUbuf + vdivIdx * VECTOR_SIZE,
                                (__ubuf__ half *)tvUbuf, subM, 1, 1, 0, roundK / BLOCK_SIZE,
                                roundK / BLOCK_SIZE, 1);
                        }
                        if (k % VECTOR_SIZE > 0) {
                            SetMask(k % VECTOR_SIZE);
                            VecDivLlm((__ubuf__ half *)goUbuf + k / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)goUbuf + k / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)tvUbuf, subM, 1, 1, 0, roundK / BLOCK_SIZE, roundK / BLOCK_SIZE,
                                1);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        // *******************move O to GM*********************
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        copy_ubuf_to_gm_align_b16(
                            (__gm__ half *)oGm + oOffset + (uint64_t)subBlockIdx * qkM / 2 * strideQo,
                            (__ubuf__ half *)goUbuf,
                            0, subM, k * 2, 0, 0, 0, (strideQo - k) * 2
                        );
                        if (goFlagScalar == 0) {
                            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            goFlagScalar = 1;
                        }
                    }
                }
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }
private:
    __aicore__ inline void RunAic(uint32_t curBatch, uint32_t startHead, uint32_t curHeadNum, uint32_t headSplitLoop,
        uint32_t offsetTiling, uint32_t embd, uint32_t headSplitNum, uint32_t groupNum, uint64_t strideKv,
        uint32_t k, uint32_t roundK, uint32_t kvSeqlen, uint32_t ppNScalar, uint32_t addrQHigh32,
        uint32_t addrQLoww32, uint32_t addrKHigh32, uint32_t addrKLoww32, uint32_t addrVHigh32, uint32_t addrVLoww32,
        uint32_t maxNumBlocksPerQuery, uint32_t blockSize, uint32_t batchId, uint32_t layerId, uint32_t numBlocks,
        __gm__ uint8_t* qGm, __gm__ half* kGm, __gm__ half* vGm, __gm__ uint8_t* blockTablesGm, __gm__ uint8_t* sGm,
        __gm__ uint8_t* pGm, __gm__ uint8_t* oTmpGm)
    {
        uint64_t addrQScalar = (uint64_t)(((uint64_t)addrQHigh32) << 32 | addrQLoww32);
        uint64_t addrKScalar = (uint64_t)(((uint64_t)addrKHigh32) << 32 | addrKLoww32);
        uint64_t addrVScalar = (uint64_t)(((uint64_t)addrVHigh32) << 32 | addrVLoww32);

        __cbuf__ uint8_t *l1qBufAddr = (__cbuf__ uint8_t *)get_imm(0);
        __cbuf__ uint8_t *l1pBufAddr = (__cbuf__ uint8_t *)get_imm(2 * L0AB_UINT8_BLOCK_SIZE_LLM);
        __cbuf__ uint8_t *l1kvBufAddr = (__cbuf__ uint8_t *)get_imm(4 * L0AB_UINT8_BLOCK_SIZE_LLM);
        __ca__ uint8_t *l0aBuf = (__ca__ uint8_t *)get_imm(0);
        __cb__ uint8_t *l0bBuf = (__cb__ uint8_t *)get_imm(0);
        __cc__ uint8_t *l0cBuf = (__cc__ uint8_t *)get_imm(0);

        uint32_t nLoop = (kvSeqlen + blockSize - 1) / blockSize;
        uint64_t qOffset = addrQScalar + startHead * embd;

        uint32_t qkN = blockSize;
        uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
        uint32_t l1PingPongFlag = 0;
        uint32_t l0PingPongFlag = 0;
        uint32_t l1Offset = l1PingPongFlag * L1_HALF_BUF_SIZE_LLM;
        uint32_t l0Offset = l0PingPongFlag * L0AB_HALF_BUF_SIZE;

        for (uint32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            if (nIdx == (nLoop - 1)) {
                qkN = (kvSeqlen - nIdx * qkN);
                qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            }

            uint32_t blockTableId = (uint32_t)(*((__gm__ int32_t *)blockTablesGm +
                batchId * maxNumBlocksPerQuery + nIdx));
            int64_t layerOffset = (int64_t)layerId * numBlocks * blockSize * strideKv;
            int64_t kvOffset = layerOffset + blockTableId * blockSize * strideKv;

            for (uint32_t splitIdx = 0; splitIdx < headSplitLoop; ++splitIdx) {
                uint32_t head_num_move =
                    (splitIdx == (headSplitLoop - 1)) ? curHeadNum - headSplitNum * splitIdx : headSplitNum;
                if (nIdx == 0 && splitIdx == 0) {
                    if (embd % BLOCK_SIZE == 0) {
                        copy_gm_to_cbuf((__cbuf__ half *)l1qBufAddr, (__gm__ half *)qGm + qOffset,
                            0, 1, roundK * curHeadNum / BLOCK_SIZE, 0, 0, PAD_NONE);
                    } else {
                        for (uint32_t copyIdx = 0; copyIdx < curHeadNum; copyIdx++) {
                            copy_gm_to_cbuf(
                                (__cbuf__ half *)l1qBufAddr + copyIdx * roundK,
                                (__gm__ half *)qGm + qOffset + copyIdx * embd,
                                0, 1, roundK / BLOCK_SIZE, 0, 0, PAD_NONE
                            );
                        }
                    }
                }
                wait_flag(PIPE_MTE1, PIPE_MTE2, l1PingPongFlag);
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    (__cbuf__ half *)l1kvBufAddr + l1Offset,
                    (__gm__ half *)kGm + kvOffset + (startHead + splitIdx * headSplitNum) / groupNum * embd,
                    0, 1, qkN, k * head_num_move, 0, strideKv, qkRoundN, 1, 0);
                set_flag(PIPE_MTE2, PIPE_MTE1, l1PingPongFlag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingPongFlag);

                for (uint32_t headdimIdx = 0; headdimIdx < head_num_move; headdimIdx++) {
                    wait_flag(PIPE_M, PIPE_MTE1, l0PingPongFlag);
                    load_cbuf_to_ca((__ca__ half *)l0aBuf + l0Offset,
                        (__cbuf__ half *)l1qBufAddr + splitIdx * headSplitNum * roundK + headdimIdx * roundK,
                        0, (roundK + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    load_cbuf_to_cb((__cb__ half *)l0bBuf,
                        (__cbuf__ half *)l1kvBufAddr + l1Offset + headdimIdx * roundK * qkRoundN, 0,
                        roundK * qkRoundN / CUBE_MATRIX_SIZE, 1, 0, 0, false, inc);
                    if (headdimIdx == head_num_move -1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, l1PingPongFlag);
                    }
                    set_flag(PIPE_MTE1, PIPE_M, l0PingPongFlag);
                    wait_flag(PIPE_MTE1, PIPE_M, l0PingPongFlag);
                    wait_flag(PIPE_FIX, PIPE_M, l0PingPongFlag);
                    mad((__cc__ float *)l0cBuf + l0Offset, (__ca__ half *)l0aBuf + l0Offset, (__cb__ half *)l0bBuf,
                        1, k, qkN, 0, 0, 0, 1);
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    set_flag(PIPE_M, PIPE_MTE1, l0PingPongFlag);
                    set_flag(PIPE_M, PIPE_FIX, l0PingPongFlag);
                    wait_flag(PIPE_M, PIPE_FIX, l0PingPongFlag);

                    copy_matrix_cc_to_gm(
                        (__gm__ float *)sGm + (uint64_t)block_idx * TMP_SIZE +
                            splitIdx * headSplitNum * qkRoundN + headdimIdx * qkRoundN,
                        (__cc__ float*)l0cBuf + l0Offset, 0, qkRoundN, 1, qkRoundN, 16, 0, NoQuant,
                        0, false, true);
                    set_flag(PIPE_FIX, PIPE_M, l0PingPongFlag);
                    l0PingPongFlag = 1 - l0PingPongFlag;
                    l0Offset = l0PingPongFlag * L0AB_HALF_BUF_SIZE;
                }
                l1PingPongFlag = 1 - l1PingPongFlag;
                l1Offset = l1PingPongFlag * L1_HALF_BUF_SIZE_LLM;
            }
            ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY << BIT_SHIFT_LLM));
            for (uint32_t splitIdx = 0; splitIdx < headSplitLoop; splitIdx++) {
                uint32_t head_num_move =
                    (splitIdx == (headSplitLoop - 1)) ? curHeadNum - headSplitNum * splitIdx : headSplitNum;
                    wait_flag(PIPE_MTE1, PIPE_MTE2, l1PingPongFlag);
                    copy_gm_to_cbuf_multi_nd2nz_b16(
                        (__cbuf__ half *)l1kvBufAddr + l1Offset,
                        (__gm__ half *)vGm + kvOffset + (startHead + splitIdx * headSplitNum) / groupNum * embd,
                        0, 1, qkN, k * head_num_move, 0, strideKv, qkRoundN, 1, 0);
                    set_flag(PIPE_MTE2, PIPE_MTE1, l1PingPongFlag);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingPongFlag);
                    for (uint32_t headdimIdx = 0; headdimIdx < head_num_move; ++headdimIdx) {
                        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                        if (qkRoundN <= roundK) {
                            for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < qkRoundN / BLOCK_SIZE; ++l0bLoadIdx) {
                                load_cbuf_to_cb(
                                    (__cb__ half *)l0bBuf + l0bLoadIdx * roundK * BLOCK_SIZE,
                                    (__cbuf__ half *)l1kvBufAddr + l1Offset + headdimIdx * roundK * qkRoundN /
                                        groupNum + l0bLoadIdx * CUBE_MATRIX_SIZE,
                                    0, roundK / BLOCK_SIZE, qkRoundN / BLOCK_SIZE, 0, 0, true, inc);
                            }
                        } else {
                            for (uint32_t l0bLoadIdx = 0; l0bLoadIdx < roundK / BLOCK_SIZE; ++l0bLoadIdx) {
                                load_cbuf_to_cb(
                                (__cb__ half *)l0bBuf + l0bLoadIdx * CUBE_MATRIX_SIZE,
                                (__cbuf__ half *)l1kvBufAddr + l1Offset +
                                    headdimIdx * roundK * qkRoundN / groupNum +
                                    l0bLoadIdx * qkRoundN * BLOCK_SIZE,
                                0, qkRoundN / BLOCK_SIZE, 1, roundK / BLOCK_SIZE - 1,
                                0, true, inc);
                            }
                        }
                        if (splitIdx == 0  && headdimIdx == 0) {
                            wait_flag_dev(SOFTMAX_READY);
                            copy_gm_to_cbuf((__cbuf__ half *)l1pBufAddr,
                                (__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE,
                                0, 1, qkRoundN * curHeadNum / BLOCK_SIZE, 0, 0, PAD_NONE);
                        }
                        set_flag(PIPE_MTE2, PIPE_MTE1, l0PingPongFlag);
                        wait_flag(PIPE_MTE2, PIPE_MTE1, l0PingPongFlag);
                        wait_flag(PIPE_M, PIPE_MTE1, l0PingPongFlag);
                        load_cbuf_to_ca((__ca__ half *)l0aBuf + l0Offset,
                            (__cbuf__ half *)l1pBufAddr + splitIdx * qkRoundN * headSplitNum +
                                headdimIdx * qkRoundN,
                            0, (qkRoundN + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, 1, 0, 0,
                            false, inc);
                        if (headdimIdx == head_num_move - 1) {
                            set_flag(PIPE_MTE1, PIPE_MTE2, l1PingPongFlag);
                        }
                        set_flag(PIPE_MTE1, PIPE_M, l0PingPongFlag);
                        wait_flag(PIPE_MTE1, PIPE_M, l0PingPongFlag);
                        wait_flag(PIPE_FIX, PIPE_M, l0PingPongFlag);
                        mad((__cc__ float *)l0cBuf + l0Offset, (__ca__ half *)l0aBuf + l0Offset,
                            (__cb__ half *)l0bBuf, 1, qkN, k, 0, 0, 0, 1);
                        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                        set_flag(PIPE_M, PIPE_MTE1, l0PingPongFlag);
                        set_flag(PIPE_M, PIPE_FIX, l0PingPongFlag);
                        wait_flag(PIPE_M, PIPE_FIX, l0PingPongFlag);

                        copy_matrix_cc_to_gm(
                            (__gm__ float *)oTmpGm + (uint64_t)block_idx * TMP_SIZE +
                                splitIdx * roundK * headSplitNum + headdimIdx * roundK,\
                            (__cc__ float *)l0cBuf + l0Offset,
                            0, roundK, 1, roundK, 16, 0, NoQuant, 0, false, true
                        );
                        set_flag(PIPE_FIX, PIPE_M, l0PingPongFlag);
                        l0PingPongFlag = 1 - l0PingPongFlag;
                        l0Offset = l0PingPongFlag * L0AB_HALF_BUF_SIZE;
                    }
                    l1PingPongFlag = 1 - l1PingPongFlag;
                    l1Offset = l1PingPongFlag * L1_HALF_BUF_SIZE_LLM;
            }
            ffts_cross_core_sync(PIPE_FIX, 0x21 + (UPDATE_READY_LLM << BIT_SHIFT_LLM));
        }
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % MASK_LOW_LLM;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }
        if (len == MASK_HIGH_LLM) {
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        } else if (len >= MASK_LOW_LLM) {
            set_vector_mask(mask, (uint64_t)-1);
        } else {
            set_vector_mask(0x0, mask);
        }
    }

    __aicore__ inline void RunAiv(uint32_t curBatch, uint32_t headIdx, uint32_t curHeadNum, uint32_t offsetTiling,
        int32_t subBlockIdx, uint32_t maxSeqlen, uint32_t embd, float tor, uint32_t batchMask, uint32_t maskStride,
        uint32_t k, uint32_t roundK, uint32_t kvSeqlen, uint32_t ppNScalar, uint32_t addrOHigh32, uint32_t addrOLoww32,
        __gm__ uint8_t* maskGm, __gm__ uint8_t* oGm, __gm__ uint8_t* sGm, __gm__ uint8_t* pGm, __gm__ uint8_t* oTmpGm)
    {
        uint32_t goFlagScalar{1};

        __ubuf__ uint8_t *lsUbuf = (__ubuf__ uint8_t *)get_imm(0);
        __ubuf__ uint8_t *lpUbuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE_DECODER);
        __ubuf__ uint8_t *loUbuf = (__ubuf__ uint8_t *)get_imm(3 * UB_UINT8_BLOCK_SIZE_DECODER);
        __ubuf__ uint8_t *lmUbuf = (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER);
        __ubuf__ uint8_t *hmUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 1 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *gmUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 2 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *dmUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 4 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *llUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 5 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *glUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 7 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *tvUbuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER + 10 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t *goUbuf =
            (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE_DECODER);

        uint64_t addrOScalar = (uint64_t)((uint64_t)addrOHigh32 << 32 | addrOLoww32);
        uint32_t nLoop = (kvSeqlen + ppNScalar - 1) / ppNScalar;
        uint64_t oOffset= addrOScalar + headIdx * embd;
        uint64_t maskBatchOffset = 0;

        uint32_t qkN = ppNScalar;
        uint32_t qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        uint32_t subM = (subBlockIdx == 1) ? (curHeadNum - curHeadNum / 2) : curHeadNum / 2;
        uint32_t subMD128 = (subM + HALF_VECTOR_SIZE_LLM - 1) / HALF_VECTOR_SIZE_LLM;
        uint32_t subMD64 = (subM + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        uint32_t roundSubM = (subM + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;

        for (uint32_t nIdx = 0; nIdx < nLoop; nIdx++) {
            if (nIdx == (nLoop - 1)) {
                qkN = (kvSeqlen - nIdx * ppNScalar);
                qkRoundN = (qkN + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
            }
            wait_flag_dev(QK_READY);
            if (subM > 0) {
                copy_gm_to_ubuf((__ubuf__ float *)lsUbuf,
                    (__gm__ float*)sGm + (uint64_t)block_idx * TMP_SIZE + (uint64_t)subBlockIdx * curHeadNum /
                        2 * qkRoundN,
                    0, 1, subM * qkRoundN / FLOAT_BLOCK_SIZE, 0, 0);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                vmuls((__ubuf__ float *)lsUbuf, (__ubuf__ float *)lsUbuf, tor,
                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                uint64_t maskOffset = maskBatchOffset + nIdx * ppNScalar;
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                if (maskGm != nullptr) {
                    copy_gm_to_ubuf((__ubuf__ half *)lpUbuf, (__gm__ half *)maskGm + maskOffset,
                        0, 1, qkRoundN / BLOCK_SIZE, 0, 0);
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (maskGm != nullptr) {
                    vconv_f162f32((__ubuf__ float *)lpUbuf + 256, (__ubuf__ half *)lpUbuf,
                        (qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, 1, 1, 8, 4);
                }
                pipe_barrier(PIPE_V);
                if (maskGm != nullptr) {
                    for (uint32_t vaddIdx = 0; vaddIdx < qkN / FLOAT_VECTOR_SIZE; ++vaddIdx) {
                        VecAddLlm((__ubuf__ float *)lsUbuf + vaddIdx * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)lsUbuf + vaddIdx * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)lpUbuf + 256 + vaddIdx * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                    }

                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                        SetMask(qkN % FLOAT_VECTOR_SIZE);
                        VecAddLlm((__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)lpUbuf + 256 + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                        set_vector_mask((uint64_t) - 1, (uint64_t) - 1);
                    }
                }
                pipe_barrier(PIPE_V);
                if (qkN <= FLOAT_VECTOR_SIZE) {
                    SetMask(qkN);
                    vcmax((__ubuf__ float *)lmUbuf, (__ubuf__ float *)lsUbuf,
                        subM, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, ONLY_VALUE);
                    set_vector_mask((uint64_t) - 1, (uint64_t) - 1);
                } else {
                    CopyUbufToUbufLlm((__ubuf__ float *)lpUbuf, (__ubuf__ float *)lsUbuf,
                        0, subM, 8, (qkRoundN - FLOAT_VECTOR_SIZE) / FLOAT_BLOCK_SIZE, 0);
                    pipe_barrier(PIPE_V);
                    for (uint32_t rowmax_idx = 1; rowmax_idx < qkN / FLOAT_VECTOR_SIZE; ++rowmax_idx) {
                        VecMaxLlm((__ubuf__ float *)lpUbuf,
                            (__ubuf__ float *)lpUbuf,
                            (__ubuf__ float *)lsUbuf + rowmax_idx * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, 8, 8, qkRoundN / FLOAT_BLOCK_SIZE);
                    }
                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                        SetMask(qkN % FLOAT_VECTOR_SIZE);
                        VecMaxLlm((__ubuf__ float *)lpUbuf,
                            (__ubuf__ float *)lpUbuf,
                            (__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1, 8, 8, qkRoundN / FLOAT_BLOCK_SIZE);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcmax((__ubuf__ float *)lmUbuf, (__ubuf__ float *)lpUbuf, subM, 1, 1, 8, ONLY_VALUE);
                }
                pipe_barrier(PIPE_V);
                if (nIdx != 0) {
                    VecMaxLlm((__ubuf__ float *)hmUbuf, (__ubuf__ float *)lmUbuf,
                        (__ubuf__ float *)gmUbuf, subMD64, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                    VecSubLlm((__ubuf__ float *)dmUbuf, (__ubuf__ float *)gmUbuf, (__ubuf__ float *)hmUbuf,
                        subMD64, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                } else {
                    CopyUbufToUbufLlm((__ubuf__ float *)hmUbuf,
                        (__ubuf__ float *)lmUbuf, 0, 1, roundSubM / FLOAT_BLOCK_SIZE, 0, 0);
                    pipe_barrier(PIPE_V);
                }
                CopyUbufToUbufLlm((__ubuf__ float *)gmUbuf,
                    (__ubuf__ float *)hmUbuf, 0, 1, roundSubM / FLOAT_BLOCK_SIZE, 0, 0);
                pipe_barrier(PIPE_V);
                vbrcb((__ubuf__ uint32_t *)tvUbuf, (__ubuf__ uint32_t *)hmUbuf,
                    1, 8, roundSubM / FLOAT_BLOCK_SIZE);
                pipe_barrier(PIPE_V);
                for (uint32_t vsubIdx=0; vsubIdx < qkN / FLOAT_VECTOR_SIZE; ++vsubIdx) {
                    VecSubLlm((__ubuf__ float *)lsUbuf + vsubIdx * FLOAT_VECTOR_SIZE,
                        (__ubuf__ float *)lsUbuf + vsubIdx * FLOAT_VECTOR_SIZE,
                        (__ubuf__ float *)tvUbuf, subM,
                        1, 1, 0,                              // src1BlockStride
                        qkRoundN / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                        qkRoundN / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                        1);
                }
                if (qkN % FLOAT_VECTOR_SIZE > 0) {
                    SetMask(qkN % FLOAT_VECTOR_SIZE);
                    VecSubLlm((__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                        (__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                        (__ubuf__ float *)tvUbuf, subM, 1, 1, 0,                              // src1BlockStride
                        qkRoundN / FLOAT_BLOCK_SIZE,  // dstRepeatStride
                        qkRoundN / FLOAT_BLOCK_SIZE,  // src0RepeatStride
                        1                               // src1RepeatStride
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                pipe_barrier(PIPE_V);
                // *** ls = exp(ls)
                VecExpLlm((__ubuf__ float *)lsUbuf, (__ubuf__ float *)lsUbuf,
                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
                    1, 1,                                                                // srcBlockStride
                    8, 8                                                                 // srcRepeatStride
                );

                pipe_barrier(PIPE_V);
                // *** lp = castfp32to16(ls)
                vconv_f322f16((__ubuf__ half *)lpUbuf, (__ubuf__ float *)lsUbuf,
                    (subM * qkRoundN + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE, // repeat
                    1, 1,                                                                // srcBlockStride
                    4,                                                                // dstRepeatStride
                    8                                                                 // srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                copy_ubuf_to_gm((__gm__ half *)pGm + (uint64_t)block_idx * TMP_SIZE +
                    (uint64_t)subBlockIdx * curHeadNum / 2 * qkRoundN,
                    (__ubuf__ half *)lpUbuf, 0, 1,                                  // nBurst
                    subM * qkRoundN / BLOCK_SIZE, 0, 0);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                // *** ll = rowsum(ls32)
                if (qkN <= FLOAT_VECTOR_SIZE) {
                    SetMask(qkN);
                    vcadd((__ubuf__ float *)llUbuf, (__ubuf__ float *)lsUbuf,
                        subM, 1, 1,                                  // srcBlockStride
                        qkRoundN / FLOAT_BLOCK_SIZE, 0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    for (uint32_t rowsumIdx = 1; rowsumIdx < qkN / FLOAT_VECTOR_SIZE; ++rowsumIdx) {
                        VecAddLlm((__ubuf__ float *)lsUbuf, (__ubuf__ float *)lsUbuf,
                            (__ubuf__ float *)lsUbuf + rowsumIdx * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1,                                              // src1BlockStride
                            qkRoundN / FLOAT_BLOCK_SIZE,                  // dstRepeatStride
                            qkRoundN / FLOAT_BLOCK_SIZE,                  // src0RepeatStride
                            qkRoundN / FLOAT_BLOCK_SIZE                   // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                    }
                    if (qkN % FLOAT_VECTOR_SIZE > 0) {
                        SetMask(qkN % FLOAT_VECTOR_SIZE);
                        VecAddLlm((__ubuf__ float *)lsUbuf, (__ubuf__ float *)lsUbuf,
                            (__ubuf__ float *)lsUbuf + qkN / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            subM, 1, 1, 1,                                              // src1BlockStride
                            qkRoundN / FLOAT_BLOCK_SIZE,                  // dstRepeatStride
                            qkRoundN / FLOAT_BLOCK_SIZE,                  // src0RepeatStride
                            qkRoundN / FLOAT_BLOCK_SIZE                   // src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcadd((__ubuf__ float *)llUbuf, (__ubuf__ float *)lsUbuf,
                        subM, 1, 1, qkRoundN / FLOAT_BLOCK_SIZE, 0);
                }
                pipe_barrier(PIPE_V);
            }
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY << BIT_SHIFT_LLM)); // 2
            wait_flag_dev(UPDATE_READY_LLM); // 4
            if (subM > 0) {
                copy_gm_to_ubuf(
                    (__ubuf__ float *)loUbuf,
                    (__gm__ float *)oTmpGm + (uint64_t)block_idx * TMP_SIZE +
                        (uint64_t)subBlockIdx * curHeadNum / 2 * roundK,
                    0, 1, subM * roundK / FLOAT_BLOCK_SIZE, 0, 0);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                // *** 更新 L 和 O
                if (nIdx != 0) {
                        // *** dm = exp(dm)
                        VecExpLlm((__ubuf__ float*)dmUbuf, (__ubuf__ float*)dmUbuf,
                            subMD64, 1, 1,                  // srcBlockStride
                            8,                  // dstRepeatStride
                            8                   // srcRepeatStride
                        );
                    pipe_barrier(PIPE_V);
                    // *** gl = dm * gl
                    VecMulLlm((__ubuf__ float *)glUbuf, (__ubuf__ float *)dmUbuf,
                        (__ubuf__ float *)glUbuf, subMD64,          // repeat
                        1, 1, 1,                  // src1BlockStride
                        8, 8, 8                   // src1RepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** gl = ll + gl
                    VecAddLlm((__ubuf__ float *)glUbuf, (__ubuf__ float *)glUbuf, (__ubuf__ float *)llUbuf,
                        subMD64, 1, 1, 1,                      // src1BlockStride
                        8, 8, 8                       // src1RepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // ***  dm_block = expand_to_block(dm), 存放于 tv
                    vbrcb((__ubuf__ uint32_t *)tvUbuf, (__ubuf__ uint32_t *)dmUbuf,
                        1, 8, roundSubM / FLOAT_BLOCK_SIZE      // repeat
                    );
                    pipe_barrier(PIPE_V);
                    if (goFlagScalar == 1) {
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        goFlagScalar = 0;
                    }
                    // *** go = go * dm_block
                    for (uint32_t vmulIdx = 0; vmulIdx < k / FLOAT_VECTOR_SIZE; ++vmulIdx) {
                        VecMulLlm((__ubuf__ float *)goUbuf + vmulIdx * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)goUbuf + vmulIdx * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)tvUbuf,
                            subM, 1, 1, 0,                                      // src1BlockStride
                            roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1);
                    }
                    if (k % FLOAT_VECTOR_SIZE > 0) {
                        SetMask(k % FLOAT_VECTOR_SIZE);
                        VecMulLlm((__ubuf__ float *)goUbuf + k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)goUbuf + k / FLOAT_VECTOR_SIZE * FLOAT_VECTOR_SIZE,
                            (__ubuf__ float *)tvUbuf, subM,                                  // repeat
                            1, 1, 0, roundK / FLOAT_BLOCK_SIZE, roundK / FLOAT_BLOCK_SIZE, 1);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    // *** go = lo + go
                    VecAddLlm((__ubuf__ float *)goUbuf,
                        (__ubuf__ float *)goUbuf,
                        (__ubuf__ float *)loUbuf,
                        (subM * roundK + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE,      // repeat
                        1, 1, 1,                                                                  // src1BlockStride
                        8, 8, 8);
                    pipe_barrier(PIPE_V);
                } else {
                    // *** gl = ll
                    CopyUbufToUbufLlm((__ubuf__ float *)glUbuf, (__ubuf__ float *)llUbuf,
                        0, 1, roundSubM / FLOAT_BLOCK_SIZE, 0, 0);
                    pipe_barrier(PIPE_V);
                    if (goFlagScalar == 1) {
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        goFlagScalar = 0;
                    }
                    // *** go = lo
                    CopyUbufToUbufLlm((__ubuf__ float *)goUbuf, (__ubuf__ float *)loUbuf,
                        0, 1, subM * roundK /FLOAT_BLOCK_SIZE, 0, 0);
                    pipe_barrier(PIPE_V);
                }
                if (nIdx == nLoop - 1) {
                    // *** gl = castfp322to16(gl)
                    vconv_f322f16((__ubuf__ half *)glUbuf, (__ubuf__ float *)glUbuf, subMD64,      // repeat
                        1,              // dstBlockStride
                        1,              // srcBlockStride
                        4,              // dstRepeatStride
                        8               // srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** go = castfp32to16(go)
                    vconv_f322f16((__ubuf__ half *)goUbuf, (__ubuf__ float *)goUbuf,
                        (subM * roundK + FLOAT_VECTOR_SIZE -1) / FLOAT_VECTOR_SIZE,     // repeat
                        1,                                                                // dstBlockStride
                        1,                                                                // srcBlockStride
                        4,                                                                // dstRepeatStride
                        8                                                                 // srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** gl_block = expand_to_block(gl), 存放于 tv
                    vbrcb((__ubuf__ uint16_t *)tvUbuf, (__ubuf__ uint16_t *)glUbuf,
                        1,                              // dstBlockStride
                        8,                              // dstRepeatStride
                        roundSubM / FLOAT_BLOCK_SIZE  // repeat
                    );
                    pipe_barrier(PIPE_V);
                    // *** go = go / gl_block
                    for (uint32_t vdivIdx = 0; vdivIdx < k / HALF_VECTOR_SIZE_LLM; ++vdivIdx) {
                        VecDivLlm((__ubuf__ half *)goUbuf + vdivIdx * HALF_VECTOR_SIZE_LLM,
                            (__ubuf__ half *)goUbuf + vdivIdx * HALF_VECTOR_SIZE_LLM, (__ubuf__ half *)tvUbuf, subM,
                            1, 1, 0,                                          // src1BlockStride
                            roundK / BLOCK_SIZE,                       // dstRepeatStride
                            roundK / BLOCK_SIZE,                       // src0RepeatStride
                            1                                           // src1RepeatStride
                        );
                    }
                    if (k % HALF_VECTOR_SIZE_LLM > 0) {
                        SetMask(k % HALF_VECTOR_SIZE_LLM);
                        VecDivLlm((__ubuf__ half *)goUbuf + k / HALF_VECTOR_SIZE_LLM * HALF_VECTOR_SIZE_LLM,
                            (__ubuf__ half *)goUbuf + k / HALF_VECTOR_SIZE_LLM * HALF_VECTOR_SIZE_LLM,
                            (__ubuf__ half *)tvUbuf, subM,
                            1,                                          // dstBlockStride
                            1,                                          // src0BlockStride
                            0,                                          // src1BlockStride
                            roundK / BLOCK_SIZE,                       // dstRepeatStride
                            roundK / BLOCK_SIZE,                       // src0RepeatStride
                            1                                           // src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    // **************** move O to GM ************
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    copy_ubuf_to_gm_align_b16((__gm__ half *)oGm + oOffset, (__ubuf__ half *)goUbuf,
                        0, subM, k * 2,    // lenBurst
                        0, 0, 0, 0);
                    if (goFlagScalar == 0) {
                        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        goFlagScalar = 1;
                    }
                }
            }
        }
    }
    __aicore__ void ExpandToBlockHalf(__ubuf__ half *dst, __ubuf__ half *src, int32_t len)
    {
        for (int32_t vaddsIdxllm = 0; vaddsIdxllm < 2; ++vaddsIdxllm) {
            vadds((__ubuf__ half *)dst + vaddsIdxllm * 8 * BLOCK_SIZE, (__ubuf__ half *)src, (half)0.0,
                len / BLOCK_SIZE, 1,                  // dstBlockStride
                0,                  // srcBlockStride
                16,                 // dstRepeatStride
                1                   // srcRepeatStride
            );
        }
        pipe_barrier(PIPE_V);
        for (int32_t vtransIdx = 0; vtransIdx < len / BLOCK_SIZE; ++vtransIdx) {
            vtranspose((__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE,
                (__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE
            );
        }
        pipe_barrier(PIPE_V);
    }

    template <typename T>
    __aicore__ inline void VecMaxLlm(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecAddLlm(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecSubLlm(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vsub(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecMulLlm(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmul(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecDivLlm(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecExpLlm(T *dst, T *src, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToUbufLlm(T *dst, T *src, uint16_t sid, uint16_t nBurst,
        uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
    {
        copy_ubuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap);
    }
};

extern "C" __global__ __aicore__ void unpad_paged_attention_mix_llm(GM_ADDR q, GM_ADDR kCache, GM_ADDR vCache,
    GM_ADDR qSeqlen, GM_ADDR kvSeqlen, GM_ADDR blockTableGm, GM_ADDR blockInfo, GM_ADDR layerId, GM_ADDR attentionMask,
    GM_ADDR outputO, GM_ADDR outputS, GM_ADDR outputP, GM_ADDR otmp, GM_ADDR usrWorkspace, GM_ADDR tiling)
{
    UnpadPagedAttentionMixLlm op;
    uint32_t *tilingParaGm = nullptr;

#ifdef __DAV_C220_CUBE__
    GET_TILING_DATA(tilingData, tiling);
    tilingParaGm = const_cast<uint32_t *>(tilingData.UnpadPagedAttentionMixLlmTilingParam);
    if (TILING_KEY_IS(0)) {
        op.unpad_flash_attention_decoder_mix_aic(q, kCache, vCache, blockTableGm, layerId, outputS, outputP, otmp,
            tiling, tilingParaGm);
    }
    if (TILING_KEY_IS(1)) {
        op.unpad_flashattention_encoder_mix_aic(q, kCache, vCache, attentionMask, blockTableGm, layerId, outputO,
            outputS, outputP, otmp, tiling, tilingParaGm);
    }
#elif __DAV_C220_VEC__
    GET_TILING_DATA(tilingData, tiling);
    tilingParaGm = const_cast<uint32_t *>(tilingData.UnpadPagedAttentionMixLlmTilingParam);
    if (TILING_KEY_IS(0)) {
        op.unpad_flash_attention_decoder_mix_aiv(nullptr, outputO, outputS, outputP, otmp, tiling, tilingParaGm);
    }
    if (TILING_KEY_IS(1)) {
        op.unpad_flashattention_encoder_mix_aiv(q, kCache, vCache, attentionMask, outputO, outputS, outputP, otmp,
            tiling, tilingParaGm);
    }
#endif
}
} // namespace AscendC