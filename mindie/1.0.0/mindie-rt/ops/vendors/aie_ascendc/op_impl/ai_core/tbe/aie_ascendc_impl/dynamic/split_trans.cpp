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
constexpr size_t MAX_NUM = (192 - 42) * 1024;  // limited by UB Memory
constexpr size_t BLOCK_SIZE = 32;              // 32 bytes
constexpr size_t ALIGN = 16;                   // 32 bytes / sizeof(half) = 16
constexpr size_t SHAPE_NUM = 8;                // 32 bytes / sizeof(uint32_t) = 8
constexpr size_t SPLIT_NUM = 3;                // 3 is the split number

}

namespace {
class SplitTrans {
public:
    __aicore__ inline SplitTrans(uint32_t batch, uint32_t seqLen, uint32_t hiddenSize,
        uint32_t coreNum)
    {
        Check(seqLen, hiddenSize, coreNum);
        if (!valid) {
            return;
        }
        this->batch = batch;
        this->seqLen = seqLen;
        this->hiddenSize = hiddenSize;
        this->coreNum = coreNum;  // block_dim
    }

    __aicore__ inline void Init(__gm__ uint8_t *X, __gm__ uint8_t *Q, __gm__ uint8_t *K, __gm__ uint8_t *V,
                                __gm__ uint8_t *Shape)
    {
        if (!valid) {
            return;
        }

        uint32_t blockX = block_idx;
        uint32_t splitHiddenSize = hiddenSize / SPLIT_NUM;
        uint32_t coreTask = hiddenSize / coreNum;
        uint32_t curCoreTask = (blockX == (coreNum - 1)) ? (hiddenSize - (coreNum - 1) * coreTask) : coreTask;
        uint32_t splitCoreTask = curCoreTask / SPLIT_NUM;

        uint32_t colsBase = (MAX_NUM / sizeof(half) / seqLen / ALIGN) * ALIGN;
        uint32_t colsBaseH = colsBase / SPLIT_NUM;
        uint32_t repeat = (splitCoreTask + colsBaseH - 1) / colsBaseH;
        uint32_t remain = splitCoreTask % colsBaseH;
        uint32_t stride = (hiddenSize - ALIGN) / ALIGN;

        pipe.InitBuffer(cacheQ, 1, (MAX_NUM / SPLIT_NUM));
        LocalTensor<half> cacheQMemory = cacheQ.AllocTensor<half>();
        commonUbufQ = (__ubuf__ half *)cacheQMemory.GetPhyAddr();

        pipe.InitBuffer(cacheK, 1, (MAX_NUM / SPLIT_NUM));
        LocalTensor<half> cacheKMemory = cacheK.AllocTensor<half>();
        commonUbufK = (__ubuf__ half *)cacheKMemory.GetPhyAddr();

        pipe.InitBuffer(cacheV, 1, (MAX_NUM / SPLIT_NUM));
        LocalTensor<half> cacheVMemory = cacheV.AllocTensor<half>();
        commonUbufV = (__ubuf__ half *)cacheVMemory.GetPhyAddr();

        if (blockX == 1) {
            ShapeGm.SetGlobalBuffer((__gm__ int32_t *)Shape);
            pipe.InitBuffer(cacheShape, 1, BLOCK_SIZE);
            LocalTensor<int32_t> commonUbufShape = cacheShape.AllocTensor<int32_t>();
            Duplicate(commonUbufShape, 0, SHAPE_NUM);
            commonUbufShape.SetValue(0, batch);            // 0 is index
            commonUbufShape.SetValue(1, seqLen);           // 1 is index
            commonUbufShape.SetValue(2, splitHiddenSize);  // 2 is index
            DataCopy(ShapeGm, commonUbufShape, SHAPE_NUM);
        }

        for (uint32_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            XGm = (__gm__ half *)X + batch_idx * hiddenSize * seqLen + blockX * splitCoreTask;
            QGm = (__gm__ half *)Q + batch_idx * splitHiddenSize * seqLen + blockX * splitCoreTask * seqLen;
            KGm = (__gm__ half *)K + batch_idx * splitHiddenSize * seqLen + blockX * splitCoreTask * seqLen;
            VGm = (__gm__ half *)V + batch_idx * splitHiddenSize * seqLen + blockX * splitCoreTask * seqLen;

            for (uint32_t i = 0; i < repeat; ++i) {
                uint32_t curCols = ((remain != 0) && (i == repeat - 1)) ? remain : colsBaseH;
                for (uint32_t j = 0; j < (curCols / ALIGN); ++j) {
                    copy_gm_to_ubuf(commonUbufQ + i * colsBaseH * seqLen + j * ALIGN * seqLen,
                                    XGm + i * colsBaseH + j * ALIGN,
                                    0,
                                    seqLen,
                                    1,
                                    stride,
                                    0);
                    pipe_barrier(PIPE_ALL);
                    copy_gm_to_ubuf(commonUbufK + i * colsBaseH * seqLen + j * ALIGN * seqLen,
                                    XGm + splitHiddenSize + i * colsBaseH + j * ALIGN,
                                    0,
                                    seqLen,
                                    1,
                                    stride,
                                    0);
                    pipe_barrier(PIPE_ALL);
                    copy_gm_to_ubuf(commonUbufV + i * colsBaseH * seqLen + j * ALIGN * seqLen,
                                    XGm + splitHiddenSize * 2 + i * colsBaseH + j * ALIGN,
                                    0,
                                    seqLen,
                                    1,
                                    stride,
                                    0);
                }
                
                uint32_t blockNum = curCols * seqLen / ALIGN;
                copy_ubuf_to_gm(QGm + i * colsBaseH * seqLen,
                                commonUbufQ + i * colsBaseH * seqLen,
                                0,
                                1,
                                blockNum,
                                0,
                                0);
                pipe_barrier(PIPE_ALL);
                copy_ubuf_to_gm(KGm + i * colsBaseH * seqLen,
                                commonUbufK + i * colsBaseH * seqLen,
                                0,
                                1,
                                blockNum,
                                0,
                                0);
                pipe_barrier(PIPE_ALL);
                copy_ubuf_to_gm(VGm + i * colsBaseH * seqLen,
                                commonUbufV + i * colsBaseH * seqLen,
                                0,
                                1,
                                blockNum,
                                0,
                                0);
            }
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline void Check(uint32_t seqLen, uint32_t hiddenSize, uint32_t coreNum)
    {
        if (hiddenSize % ALIGN != 0 || hiddenSize % SPLIT_NUM != 0 ||
            seqLen * ALIGN * sizeof(half) * SPLIT_NUM > MAX_NUM) {
            valid = false;
        }
    }

private:
    __gm__ half *XGm;
    __gm__ half *QGm;
    __gm__ half *KGm;
    __gm__ half *VGm;
    GlobalTensor<int32_t> ShapeGm;

    uint32_t batch = 0;
    uint32_t seqLen = 0;
    uint32_t hiddenSize = 0;
    uint32_t coreNum = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> cacheQ;
    TQue<QuePosition::VECIN, 1> cacheK;
    TQue<QuePosition::VECIN, 1> cacheV;
    TQue<QuePosition::VECIN, 1> cacheShape;
    __ubuf__ half *commonUbufQ;
    __ubuf__ half *commonUbufK;
    __ubuf__ half *commonUbufV;
    __ubuf__ int32_t *commonUbufShape;
    bool valid = true;
};
}

namespace {
extern "C" __global__ __aicore__ void split_trans(GM_ADDR X, GM_ADDR Q, GM_ADDR K, GM_ADDR V,
                                                  GM_ADDR Shape, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    SplitTrans op(tiling_data.batch, tiling_data.seqLen, tiling_data.hiddenSize, tiling_data.coreNum);

#if defined(__DAV_M200__) || defined(__DAV_C220_VEC__)
    op.Init(X, Q, K, V, Shape);
#endif
}
}