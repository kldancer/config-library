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

constexpr uint32_t MAX_X_BUF = 224 * 1024; // 224KB
constexpr uint32_t MAX_IDX_BUF = 24 * 1024; // 24KB
constexpr uint32_t MAX_STATE_BUF = 8 * 1024; // 8KB
constexpr uint32_t UB_ALIGN_BUF = 32;
constexpr uint32_t UB_ALIGN_X16_NUM = 16;
constexpr uint32_t UB_ALIGN_X32_NUM = 8;
constexpr uint32_t UB_ALIGN_X64_NUM = 4;
constexpr uint32_t TILING_NUM = 128;
constexpr uint64_t MAX_UINT32 = 4294967295;
constexpr uint32_t MAX_DATA = 200000;

/*
 * Restrictions:
 *   1) x ndims = 1;
 * Notes:
 *   对于dim或者index中的非法数据，算子直接忽略
 */
template<typename T>
class IndexFill {
public:
    __aicore__ inline IndexFill(uint32_t tIdxLen, uint32_t tNDims, uint32_t d0)
        : idxLen(tIdxLen), nDims(tNDims)
    {
        ubAlignTNum = UB_ALIGN_BUF / sizeof(T);
        dims[0] = d0;
    }
    
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dim, GM_ADDR index, GM_ADDR value, GM_ADDR y)
    {
        if (x == nullptr || dim == nullptr || index == nullptr || value == nullptr || y == nullptr) {
            return;
        }
        idxLenAlign = AlignUp(idxLen, UB_ALIGN_X64_NUM);
        uint32_t maxIdxNum = MAX_IDX_BUF / sizeof(int64_t);
        pIN = idxLenAlign < maxIdxNum ? idxLenAlign : maxIdxNum;
        pIT = idxLen / pIN;
        tailIN = idxLenAlign - pIN * pIT;
        pipe.InitBuffer(idxBuf, pIN * sizeof(int64_t));

        AscendC::LocalTensor<int32_t> ubDim = idxBuf.Get<int32_t>(UB_ALIGN_X32_NUM);
        gmDim.SetGlobalBuffer((__gm__ int32_t *)dim, UB_ALIGN_X32_NUM);
        AscendC::DataCopy(ubDim, gmDim, UB_ALIGN_X32_NUM);
        pipe_barrier(PIPE_ALL);
        int32_t fillDim = ubDim.GetValue(0);
        if (fillDim < 0) {
            fillDim += static_cast<int32_t>(nDims);
        }
        if (fillDim < 0) {
            initSuccess = false;
            return;
        }

        AscendC::LocalTensor<T> ubValue = idxBuf.Get<T>(ubAlignTNum);
        gmValue.SetGlobalBuffer((__gm__ T *)value, ubAlignTNum);
        AscendC::DataCopy(ubValue, gmValue, ubAlignTNum);
        pipe_barrier(PIPE_ALL);
        fillValue = ubValue.GetValue(0);

        if (iters > MAX_UINT32 || segLen > MAX_UINT32) {
            return;
        }
        for (uint32_t i = 0; i < nDims; ++i) {
            if (i < fillDim) {
                iters *= dims[i];
            } else if (i == fillDim) {
                segNum = dims[i];
            } else {
                segLen *= dims[i];
            }
            if (iters > MAX_UINT32 || segLen > MAX_UINT32) {
                return;
            }
        }
        if (iters * segNum > MAX_UINT32) {
            return;
        } else if (iters * segNum * segLen > MAX_UINT32) {
            return;
        } else {
            ntotal = iters * segNum * segLen;
        }
        gmX.SetGlobalBuffer((__gm__ T *)x, ntotal);
        gmY.SetGlobalBuffer((__gm__ T *)y, ntotal);
        gmIdx.SetGlobalBuffer((__gm__ int64_t *)index, idxLen);
        
        segLenAlign = AlignUp(segLen, ubAlignTNum);
        // Single process number (pN); process times (pT)
        uint32_t maxXNum = MAX_X_BUF / sizeof(T);
        pN = segLenAlign < maxXNum ? segLenAlign : maxXNum;
        pT = segLen / pN;
        tailN = segLenAlign - pN * pT;
        pipe.InitBuffer(xQ, 1, pN * sizeof(T));

        segNumAlign = AlignUp(segNum, UB_ALIGN_X16_NUM);
        uint32_t maxSNum = MAX_STATE_BUF / sizeof(int16_t);
        pSN = segNumAlign < maxSNum ? segNumAlign : maxSNum;
        pST = segNum / pSN;
        tailSN = segNumAlign - pSN * pST;
        pipe.InitBuffer(fillStateBuf, pSN * sizeof(int16_t));
    }

    __aicore__ inline void Process()
    {
        if (!initSuccess) {
            return;
        }

        AscendC::LocalTensor<int16_t> ubFillState = fillStateBuf.Get<int16_t>();
        AscendC::LocalTensor<int64_t> ubIdx = idxBuf.Get<int64_t>();
        uint32_t xOffset = 0;
        for (uint32_t i = 0; i < iters; ++i) {
            uint32_t segOffset = 0;
            for (uint32_t j = 0; j < pST; ++j) {
                UpdateFillStates(ubFillState, ubIdx, segOffset, pSN);
                for (uint32_t k = 0; k < pSN; ++k) {
                    bool doFill = ubFillState.GetValue(k) == 1;
                    ProcessSeg(xOffset, doFill);
                    xOffset += segLen;
                }
                segOffset += pSN;
            }
            if (tailSN > 0) {
                uint32_t realTailSN = segNum - segOffset;
                UpdateFillStates(ubFillState, ubIdx, segOffset, realTailSN);
                for (uint32_t k = 0; k < realTailSN; ++k) {
                    bool doFill = ubFillState.GetValue(k) == 1;
                    ProcessSeg(xOffset, doFill);
                    xOffset += segLen;
                }
            }
        }
    }

private:
    __aicore__ inline void ProcessSeg(uint32_t xOffset, bool doFill)
    {
        for (uint32_t i = 0; i < pT; ++i) {
            pipe_barrier(PIPE_ALL);
            CopyIn(xOffset, pN, doFill);
            pipe_barrier(PIPE_ALL);
            CopyOut(xOffset, pN);
            xOffset += pN;
        }
        if (tailN > 0) {
            pipe_barrier(PIPE_ALL);
            CopyIn(xOffset, tailN, doFill);
            pipe_barrier(PIPE_ALL);
            CopyOut(xOffset, tailN);
        }
    }

    __aicore__ inline void UpdateFillStates(AscendC::LocalTensor<int16_t>& ubFillState,
        AscendC::LocalTensor<int64_t>& ubIdx, uint32_t segOffset, uint32_t cnt)
    {
        pipe_barrier(PIPE_ALL);
        AscendC::Duplicate(ubFillState, (int16_t)0, pSN);
        uint32_t idxOffset = 0;
        for (uint32_t i = 0; i < pIT; ++i) {
            AscendC::DataCopy(ubIdx, gmIdx[idxOffset], pIN);
            pipe_barrier(PIPE_ALL);
            for (uint32_t j = 0; j < pIN; ++j) {
                int64_t fillIdx = ubIdx.GetValue(j);
                if (fillIdx < 0) {
                    fillIdx += segNum;
                }
                fillIdx -= segOffset;
                if (fillIdx >= 0 && fillIdx < cnt) {
                    ubFillState.SetValue(fillIdx, 1);
                }
            }
            idxOffset += pIN;
        }
        if (tailIN > 0) {
            AscendC::DataCopy(ubIdx, gmIdx[idxOffset], tailIN);
            pipe_barrier(PIPE_ALL);
            uint32_t realTailIN = idxLen - idxOffset;
            for (uint32_t j = 0; j < realTailIN; ++j) {
                int64_t fillIdx = ubIdx.GetValue(j);
                if (fillIdx < 0) {
                    fillIdx += segNum;
                }
                fillIdx -= segOffset;
                if (fillIdx >= 0 && fillIdx < cnt) {
                    ubFillState.SetValue(fillIdx, 1);
                }
            }
        }
    }

    __aicore__ inline uint32_t AlignUp(uint32_t x, uint32_t alignNum)
    {
        return (x + alignNum - 1) / alignNum * alignNum;
    }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t n, bool doFill)
    {
        AscendC::LocalTensor<T> ubX = xQ.AllocTensor<T>();
        if (doFill) {
            pipe_barrier(PIPE_ALL);
            AscendC::Duplicate(ubX, fillValue, n);
        } else {
            AscendC::DataCopy(ubX, gmX[offset], n);
        }
        xQ.EnQue<T>(ubX);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t n)
    {
        AscendC::LocalTensor<T> ubX = xQ.DeQue<T>();
        AscendC::DataCopy(gmY[offset], ubX, n);
        xQ.FreeTensor<T>(ubX);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> xQ;
    AscendC::TBuf<AscendC::TPosition::VECCALC> idxBuf, fillStateBuf;
    AscendC::GlobalTensor<T> gmX, gmValue, gmY;
    AscendC::GlobalTensor<int32_t> gmDim;
    AscendC::GlobalTensor<int64_t> gmIdx;
    uint64_t iters{1}, segNum{0}, segLen{1}, ntotal;
    uint32_t idxLen, nDims, ubAlignTNum, segLenAlign{0};
    uint32_t pIN{0}, pIT{0}, tailIN{0}, pSN{0}, pST{0}, tailSN{0}, pN{0}, pT{0}, tailN{0};
    uint32_t idxLenAlign{0}, segNumAlign{0};
    uint32_t dims[1]; // 最多支持1维
    bool initSuccess{true};
    T fillValue{0};
    bool valid = true;
};
}

extern "C" __global__ __aicore__ void index_fill(GM_ADDR x, GM_ADDR dim, GM_ADDR index,
    GM_ADDR value, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(t, tiling);
    if (t.typeBytes == 2) {
        ::IndexFill<uint16_t> op(t.idxLen, t.nDims, t.d0);
        op.Init(x, dim, index, value, y);
        op.Process();
    } else if (t.typeBytes == 4) {
        ::IndexFill<uint32_t> op(t.idxLen, t.nDims, t.d0);
        op.Init(x, dim, index, value, y);
        op.Process();
    }
}