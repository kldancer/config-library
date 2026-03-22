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
constexpr uint32_t MAX_PROCESS_NUM = 65280;
constexpr uint32_t SEQ_LEN_ROUND_UP_FACTOR = 16;
constexpr uint32_t UB_INT32_ALIGN_NUM = 8;
constexpr uint32_t BATCHCORE = 8;
constexpr uint32_t MAX_TILINGBATCHCORE = 64;
constexpr uint32_t MAXSEQLEN_MAX = 1024;
constexpr uint32_t MAXSEQLEN_MIN = 1;
constexpr uint32_t ALIGN = 64;

class GenSeqLen {
public:
    __aicore__ inline GenSeqLen(uint32_t tilingBathCore, uint32_t tilingBatchCoreOffset, uint32_t tilingMaxSeqLen)
        : batchCore(tilingBathCore), batchCoreOffset(tilingBatchCoreOffset), maxSeqLen(tilingMaxSeqLen)
    {
    }

    __aicore__ inline void Init(GM_ADDR attnMask, GM_ADDR seqLen, GM_ADDR seqLenOri)
    {
        uint32_t coreSizeOffset = batchCoreOffset * maxSeqLen;
        gmAttnMask.SetGlobalBuffer((__gm__ int32_t *)attnMask + block_idx * coreSizeOffset, batchCore * maxSeqLen);
        gmSeqLen.SetGlobalBuffer((__gm__ int32_t *)seqLen + block_idx * batchCoreOffset, batchCore);
        gmSeqLenOri.SetGlobalBuffer((__gm__ int32_t *)seqLenOri + block_idx * batchCoreOffset, batchCore);

        pipe.InitBuffer(inQueue, 1, MAX_PROCESS_NUM * sizeof(int32_t));
        batchCoreAlign = (batchCore + UB_INT32_ALIGN_NUM - 1) / UB_INT32_ALIGN_NUM * UB_INT32_ALIGN_NUM;
        pipe.InitBuffer(outQueue, 2, batchCoreAlign * sizeof(int32_t)); // 2 outputs

        maxSeqLenAlign = (maxSeqLen + UB_INT32_ALIGN_NUM - 1) / UB_INT32_ALIGN_NUM * UB_INT32_ALIGN_NUM;
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < batchCore; ++i) {
            CopyIn(i);
            pipe_barrier(PIPE_ALL);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t pIndex)
    {
        AscendC::LocalTensor<int32_t> ubAttnMask = inQueue.AllocTensor<int32_t>();
        AscendC::DataCopy(ubAttnMask, gmAttnMask[pIndex * maxSeqLen], maxSeqLenAlign);
        inQueue.EnQue<int32_t>(ubAttnMask);
    }

    __aicore__ inline void Compute(int32_t pIndex)
    {
        AscendC::LocalTensor<int32_t> ubAttnMask = inQueue.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> ubSeqLen = outQueue.AllocTensor<int32_t>();
        AscendC::LocalTensor<int32_t> ubSeqLenOri = outQueue.AllocTensor<int32_t>();
        
        int32_t l = 0;
        int32_t r = maxSeqLen;
        while (l < r) {
            int32_t mid = (l + r) / 2; // 2分法
            if (ubAttnMask.GetValue(mid) == 1) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }
        int32_t roundedR = (r + SEQ_LEN_ROUND_UP_FACTOR - 1) / SEQ_LEN_ROUND_UP_FACTOR * SEQ_LEN_ROUND_UP_FACTOR;
        ubSeqLenOri.SetValue(pIndex, r);
        ubSeqLen.SetValue(pIndex, roundedR);

        outQueue.EnQue<int32_t>(ubSeqLen);
        outQueue.EnQue<int32_t>(ubSeqLenOri);
        inQueue.FreeTensor(ubAttnMask);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<int32_t> ubSeqLen = outQueue.DeQue<int32_t>();
        AscendC::LocalTensor<int32_t> ubSeqLenOri = outQueue.DeQue<int32_t>();
        AscendC::DataCopy(gmSeqLen, ubSeqLen, batchCoreAlign);
        AscendC::DataCopy(gmSeqLenOri, ubSeqLenOri, batchCoreAlign);
        outQueue.FreeTensor(ubSeqLen);
        outQueue.FreeTensor(ubSeqLenOri);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 2> outQueue; // 2 outputs
    AscendC::GlobalTensor<int32_t> gmAttnMask, gmSeqLen, gmSeqLenOri;
    uint32_t batchCore = 0, batchCoreAlign = 0, batchCoreOffset = 0, maxSeqLen = 0, maxSeqLenAlign = 0;
};
}

extern "C" __global__ __aicore__ void gen_seq_len(GM_ADDR attnMask, GM_ADDR seqLen, GM_ADDR seqLenOri,
    GM_ADDR workspace, GM_ADDR tilingData)
{
    GET_TILING_DATA(tiling, tilingData);
    uint32_t batchThisCore;
    if (block_idx == static_cast<int64_t>(tiling.blkDim - 1)) {
        batchThisCore = tiling.batchLastCore;
    } else {
        batchThisCore = tiling.batchCore;
    }
    ::GenSeqLen op(batchThisCore, tiling.batchCore, tiling.maxSeqLen);
    op.Init(attnMask, seqLen, seqLenOri);
    op.Process();
}