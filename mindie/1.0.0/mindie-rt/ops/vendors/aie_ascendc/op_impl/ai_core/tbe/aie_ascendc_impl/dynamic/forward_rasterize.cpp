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

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"

namespace ForwardRasterize {
using namespace AscendC;

constexpr int32_t TRIANGLE_SIZE = 3;
constexpr int32_t FACE_SIZE = 9;
constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t B32_BLOCK_SIZE = 8;
constexpr int32_t B32_REPEAT_SIZE = 64;
constexpr int32_t CONSTANT_TWO = 2;
constexpr int32_t CONSTANT_THREE = 3;
constexpr int32_t CONSTANT_FOUR = 4;
constexpr int32_t CONSTANT_FIVE = 5;
constexpr int32_t CONSTANT_SIX = 6;
constexpr int32_t CONSTANT_SEVEN = 7;
constexpr int32_t CONSTANT_EIGHT = 8;
constexpr float FLOAT_CONSTANT_ZERO = 0.0;
constexpr float FLOAT_CONSTANT_ONE = 1.0;
constexpr float FLOAT_CONSTANT_TWO = 2.0;
constexpr float FLOAT_CONSTANT_THREE = 3.0;
constexpr float FLOAT_CONSTANT_FOUR = 4.0;
constexpr float FLOAT_CONSTANT_FIVE = 5.0;
constexpr float FLOAT_CONSTANT_SIX = 6.0;
constexpr float FLOAT_CONSTANT_SEVEN = 7.0;
constexpr float DEPTH_INIT_VALUE = 1000000.0;
constexpr float TRIANGLE_INIT_VALUE = -1.0;
constexpr float BARYW_INIT_VALUE = 0.0;
constexpr float MAX_FP32_VALUE = 3.4028235e+38;

class ForwardRasterize {
public:
// only support B = 1
__aicore__ inline ForwardRasterize(){};
__aicore__ inline void Init(GM_ADDR faceVertices,
                            GM_ADDR outputBuffer,
                            GM_ADDR workspace,
                            const ForwardRasterizeTilingData* tilingData,
                            TPipe* tPipe) {
    B_ = tilingData->bSize;
    N_ = tilingData->nSize;
    H_ = tilingData->hSize;
    W_ = tilingData->wSize;

    blockNum_ = tilingData->blockNum;
    blockFormerH_ = tilingData->blockFormerH;
    blockTailH_ = tilingData->blockTailH;
    ubFormerH_ = tilingData->ubFormerH;
    ubLoopOfFormerBlockH_ = tilingData->ubLoopOfFormerBlockH;
    ubLoopOfTailBlockH_ = tilingData->ubLoopOfTailBlockH;
    ubTailOfFormerBlockH_ = tilingData->ubTailOfFormerBlockH;
    ubTailOfTailBlockH_ = tilingData->ubTailOfTailBlockH;
    ubFormerNtri_ = tilingData->ubFormerNtri;
    ubTailNtri_ = tilingData->ubTailNtri;
    ubLoopOfNtri_ = tilingData->ubLoopOfNtri;

    pipePtr = tPipe;
    blockIdx = GetBlockIdx();
    // init gm tensors
    int32_t curBlockH = (blockIdx != blockNum_- 1) ? blockFormerH_ : blockTailH_;
    int32_t formerOutBlockLengthH = blockFormerH_ * W_;
    int32_t curOutBlockLength = curBlockH * W_;

    int32_t gmOutputOffset = H_ * W_ * B_;

    fVInGM.SetGlobalBuffer((__gm__ float*)faceVertices, B_ * N_ * FACE_SIZE);
    depthOutGM.SetGlobalBuffer((__gm__ float*)outputBuffer + formerOutBlockLengthH * blockIdx, curOutBlockLength);
    triangleOutGM.SetGlobalBuffer((__gm__ float*)outputBuffer + formerOutBlockLengthH * blockIdx + gmOutputOffset,
                                curOutBlockLength);
    barywOutGM.SetGlobalBuffer((__gm__ float*)outputBuffer + formerOutBlockLengthH * blockIdx + gmOutputOffset * 2,
                               H_ * W_ * CONSTANT_TWO + curOutBlockLength);

    // init queues
    pipePtr->InitBuffer(queueFV, 1,
        (ubFormerNtri_ * sizeof(float) * FACE_SIZE + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES);
    bufferElems_ = ubFormerH_ * W_;
    int32_t bufferSize_ = bufferElems_ * sizeof(float);
    pipePtr->InitBuffer(queueD, 1, bufferSize_);
    pipePtr->InitBuffer(queueT, 1, bufferSize_);
    pipePtr->InitBuffer(queueB, 1, bufferSize_ * TRIANGLE_SIZE);
    pipePtr->InitBuffer(tmpTensor, bufferSize_ * CONSTANT_EIGHT);
}

__aicore__ inline void Process() {
    if (blockIdx >= blockNum_) {
        return;
    }
    int32_t ubLoopOfH = (blockIdx == blockNum_ - 1) ? ubLoopOfTailBlockH_ : ubLoopOfFormerBlockH_;
    int32_t tailH = (blockIdx == blockNum_ - 1) ? ubTailOfTailBlockH_ : ubTailOfFormerBlockH_;

    for (int32_t i = 0; i < ubLoopOfH - 1; i++) {
        PreCompute(i, ubFormerH_);
        for (int32_t j = 0; j < ubLoopOfNtri_ - 1; j++) {
            CopyInFV(j, ubFormerNtri_);
            Compute(i, j, ubFormerH_, ubFormerNtri_);
        }
        CopyInFV(ubLoopOfNtri_ - 1, ubTailNtri_);
        Compute(i, ubLoopOfNtri_ - 1, ubFormerH_, ubTailNtri_);
        // outputs are aggregated
        CopyOutDepth(i, ubFormerH_);
        CopyOutTriangle(i, ubFormerH_);
        CopyOutBaryw(i, ubFormerH_);
    }
    PreCompute(ubLoopOfH - 1, tailH);
    for (int32_t j = 0; j < ubLoopOfNtri_ - 1; j++) {
        CopyInFV(j, ubFormerNtri_);
        Compute(ubLoopOfH - 1, j, tailH, ubFormerNtri_);
    }
    CopyInFV(ubLoopOfNtri_ - 1, ubTailNtri_);
    Compute(ubLoopOfH - 1, ubLoopOfNtri_ - 1, tailH, ubTailNtri_);
    // outputs are aggregated
    CopyOutDepth(ubLoopOfH - 1, tailH);
    CopyOutTriangle(ubLoopOfH - 1, tailH);
    CopyOutBaryw(ubLoopOfH - 1, tailH);
}

private:
__aicore__ inline void CopyInFV(const int32_t ubIdxN,
                                const int32_t curN) {
    bufferFV = queueFV.AllocTensor<float>();
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curN * FACE_SIZE * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    int count = curN * FACE_SIZE;
    DataCopy(bufferFV, fVInGM[ubFormerNtri_ * FACE_SIZE * ubIdxN], count);

    queueFV.EnQue(bufferFV);
}

__aicore__ inline void PreCompute(const int32_t ubIdxH,
                                  const int32_t curH) {
    // init output
    bufferD = queueD.AllocTensor<float>();
    Duplicate(bufferD, DEPTH_INIT_VALUE, ubFormerH_ * W_);
    bufferT = queueT.AllocTensor<float>();
    Duplicate(bufferT, TRIANGLE_INIT_VALUE, ubFormerH_ * W_);
    bufferB = queueB.AllocTensor<float>();
    Duplicate(bufferB, BARYW_INIT_VALUE, ubFormerH_ * W_ * TRIANGLE_SIZE);

    PipeBarrier<PIPE_V>();
    LocalTensor<float> totalTmpBuffer = tmpTensor.Get<float>();
    tmpBuffer0 = totalTmpBuffer[0];
    tmpBuffer1 = totalTmpBuffer[bufferElems_];
    tmpBuffer2 = totalTmpBuffer[bufferElems_ * CONSTANT_TWO];
    tmpBuffer3 = totalTmpBuffer[bufferElems_ * CONSTANT_THREE];
    tmpBuffer4 = totalTmpBuffer[bufferElems_ * CONSTANT_FOUR];
    tmpBuffer5 = totalTmpBuffer[bufferElems_ * CONSTANT_FIVE];
    tmpBuffer6 = totalTmpBuffer[bufferElems_ * CONSTANT_SIX];
    tmpBuffer7 = totalTmpBuffer[bufferElems_ * CONSTANT_SEVEN];
    InitIndexW(tmpBuffer0);
    InitIndexH(tmpBuffer1, ubIdxH, curH);
}

__aicore__ inline void Compute(const int32_t ubIdxH,
                               const int32_t ubIdxN,
                               const int32_t curH,
                               const int32_t curN) {
    bufferFV = queueFV.DeQue<float>();
    int32_t curLoopMinH = blockFormerH_ * blockIdx + ubIdxH * ubFormerH_;
    int32_t curLoopMaxH = curLoopMinH + curH;
    auto eventIDMTE2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIDMTE2S);
    WaitFlag<HardEvent::MTE2_S>(eventIDMTE2S);
    auto eventIDSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    for (int32_t i = 0; i < curN; i++) {
        float p0W = bufferFV.GetValue(i * FACE_SIZE);
        float p0H = bufferFV.GetValue(i * FACE_SIZE + 1);
        float p0D = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_TWO);
        float p1W = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_THREE);
        float p1H = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_FOUR);
        float p1D = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_FIVE);
        float p2W = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_SIX);
        float p2H = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_SEVEN);
        float p2D = bufferFV.GetValue(i * FACE_SIZE + CONSTANT_EIGHT);
        float v0W = p2W - p0W;
        float v0H = p2H - p0H;
        float v1W = p1W - p0W;
        float v1H = p1H - p0H;
        float dot00 = v0W * v0W + v0H * v0H;
        float dot01 = v0W * v1W + v0H * v1H;
        float dot11 = v1W * v1W + v1H * v1H;
        float tmpScalar = dot00 * dot11 - dot01 * dot01;
        float inverDeno = tmpScalar == 0 ? 0 : 1 / tmpScalar;
        int32_t minW = max(static_cast<int32_t>(conv_f322s32c(min(p0W, min(p1W, p2W)))), 0);
        int32_t maxW = min(static_cast<int32_t>(conv_f322s32f(max(p0W, max(p1W, p2W)))), W_ - 1);
        int32_t minH = max(static_cast<int32_t>(conv_f322s32c(min(p0H, min(p1H, p2H)))), curLoopMinH);
        int32_t maxH = min(static_cast<int32_t>(conv_f322s32f(max(p0H, max(p1H, p2H)))), curLoopMaxH - 1);
        if (maxW + 1 - minW <= 0) {
            continue;
        }

        int32_t windowSizeH = maxH + 1 - minH;
        if (windowSizeH <= 0) {
            continue;
        }
        
        int32_t minWAlign = minW / B32_BLOCK_SIZE * B32_BLOCK_SIZE;
        int32_t maxWAlign = (maxW + 1 + B32_BLOCK_SIZE - 1) / B32_BLOCK_SIZE * B32_BLOCK_SIZE;
        if (unlikely(maxWAlign == minWAlign)) {
            maxWAlign += B32_BLOCK_SIZE;
        }
        int32_t windowSizeW = maxWAlign - minWAlign;
        int32_t windowTotalSize = windowSizeH * windowSizeW;
        int32_t startIndex = (minH - curLoopMinH) * W_ + minWAlign;
        SetFlag<HardEvent::S_V>(eventIDSV);
        WaitFlag<HardEvent::S_V>(eventIDSV);
        // pW
        ConcatWindow(tmpBuffer2, tmpBuffer0, minWAlign, 1, windowSizeW);
        PipeBarrier<PIPE_V>();
        // pH
        ConcatWindow(tmpBuffer3, tmpBuffer1, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // v2W = pW - p0W
        Adds(tmpBuffer2, tmpBuffer2, -p0W, windowSizeW);
        // v2H = pH - p0H
        Adds(tmpBuffer3, tmpBuffer3, -p0H, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // v0H * v2H
        Muls(tmpBuffer4, tmpBuffer3, v0H, windowTotalSize);
        // v0W * v2W
        Muls(tmpBuffer5, tmpBuffer2, v0W, windowSizeW);
        PipeBarrier<PIPE_V>();
        // dot02 = v0H * v2H + v0W * v2W
        AddWithInlinedNLastBrcFP32(tmpBuffer4, tmpBuffer4, tmpBuffer5, curH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // v1H * v2H
        Muls(tmpBuffer5, tmpBuffer3, v1H, windowTotalSize);
        // v1W * v2W
        Muls(tmpBuffer6, tmpBuffer2, v1W, windowSizeW);
        PipeBarrier<PIPE_V>();
        // dot12 = v1H * v2H + v1W * v2W, v2W and v2H can be freed
        AddWithInlinedNLastBrcFP32(tmpBuffer5, tmpBuffer5, tmpBuffer6, curH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // dot11*dot02
        Muls(tmpBuffer2, tmpBuffer4, dot11, windowTotalSize);
        // dot01*dot12
        Muls(tmpBuffer3, tmpBuffer5, dot01, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // dot11*dot02 - dot01*dot1
        Sub(tmpBuffer2, tmpBuffer2, tmpBuffer3, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // u = (dot11*dot02 - dot01*dot12)*inverDeno;
        Muls(tmpBuffer2, tmpBuffer2, inverDeno, windowTotalSize);
        // dot00*dot12
        Muls(tmpBuffer6, tmpBuffer5, dot00, windowTotalSize);
        // dot01*dot02
        Muls(tmpBuffer7, tmpBuffer4, dot01, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // dot00*dot12 - dot01*dot02
        Sub(tmpBuffer3, tmpBuffer6, tmpBuffer7, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // v = (dot00*dot12 - dot01*dot02)*inverDeno
        Muls(tmpBuffer3, tmpBuffer3, inverDeno, windowTotalSize);
        PipeBarrier<PIPE_V>();
        // -u
        Muls(tmpBuffer4, tmpBuffer2, static_cast<float>(-1.0), windowTotalSize);
        // -v
        Muls(tmpBuffer5, tmpBuffer3, static_cast<float>(-1.0), windowTotalSize);
        PipeBarrier<PIPE_V>();
        // 1-u
        Adds(tmpBuffer4, tmpBuffer4, static_cast<float>(1.0), windowTotalSize);
        PipeBarrier<PIPE_V>();
        // 1-u-v
        Add(tmpBuffer4, tmpBuffer4, tmpBuffer5, windowTotalSize);
        uint32_t calCount = (windowTotalSize + B32_REPEAT_SIZE - 1) / B32_REPEAT_SIZE * B32_REPEAT_SIZE;
        // u >= 0
        CompareScalar(tmpBuffer5.ReinterpretCast<uint8_t>(), tmpBuffer2, static_cast<float>(0.0),
            CMPMODE::GE, calCount);
        // v >= 0
        CompareScalar(tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer3, static_cast<float>(0.0),
            CMPMODE::GE, calCount);
        // 1-u-v > 0
        CompareScalar(tmpBuffer7.ReinterpretCast<uint8_t>(), tmpBuffer4, static_cast<float>(0.0),
            CMPMODE::GT, calCount);
        int32_t calCountB16 = windowTotalSize / 2;
        PipeBarrier<PIPE_V>();
        And(tmpBuffer5.ReinterpretCast<uint16_t>(), tmpBuffer5.ReinterpretCast<uint16_t>(),
            tmpBuffer6.ReinterpretCast<uint16_t>(), calCountB16);
        PipeBarrier<PIPE_V>();
        And(tmpBuffer5.ReinterpretCast<uint16_t>(), tmpBuffer5.ReinterpretCast<uint16_t>(),
            tmpBuffer7.ReinterpretCast<uint16_t>(), calCountB16);
        PipeBarrier<PIPE_V>();
        if (p0D != 0) {
            // 1-u-v / face[2]
            Muls(tmpBuffer7, tmpBuffer4, static_cast<float>(1.0) / p0D, windowTotalSize);
        }
        if (p1D != 0) {
            // v / face[5]
            Muls(tmpBuffer6, tmpBuffer3, static_cast<float>(1.0) / p1D, windowTotalSize);
        }
        PipeBarrier<PIPE_V>();
        // 1-u-v / face[2] + v / face[5]
        Add(tmpBuffer7, tmpBuffer7, tmpBuffer6, windowTotalSize);
        PipeBarrier<PIPE_V>();
        if (p2D != 0) {
            // u / face[8]
            Muls(tmpBuffer6, tmpBuffer2, static_cast<float>(1.0) / p2D, windowTotalSize);
        }
        PipeBarrier<PIPE_V>();
        // 1-u-v / face[2] + v / face[5] + u / face[8]
        Add(tmpBuffer7, tmpBuffer7, tmpBuffer6, windowTotalSize);
        PipeBarrier<PIPE_V>();
        Duplicate(tmpBuffer6, static_cast<float>(1.0), windowTotalSize);
        PipeBarrier<PIPE_V>();
        // zp
        Div(tmpBuffer6, tmpBuffer6, tmpBuffer7, windowTotalSize);
        PipeBarrier<PIPE_V>();
        Select(tmpBuffer7, tmpBuffer5.ReinterpretCast<uint8_t>(), tmpBuffer6, MAX_FP32_VALUE,
               SELMODE::VSEL_TENSOR_SCALAR_MODE, calCount);
        PipeBarrier<PIPE_V>();
        // depth buffer window
        ConcatWindow(tmpBuffer5, bufferD, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        DuplicateWindowMask(tmpBuffer7, windowSizeH, windowSizeW, minW, minWAlign, maxW, maxWAlign);
        PipeBarrier<PIPE_V>();
        Min(tmpBuffer5, tmpBuffer5, tmpBuffer7, windowTotalSize);
        PipeBarrier<PIPE_V>();
        SplitWindow(bufferD, tmpBuffer5, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // zp can be freed and gen select mask
        Compare(tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer7, tmpBuffer5, CMPMODE::NE, calCount);
        PipeBarrier<PIPE_V>();
        // triangle buffer window
        ConcatWindow(tmpBuffer5, bufferT, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        Select(tmpBuffer7, tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer5,
            static_cast<float>(i + ubIdxN * ubFormerNtri_), SELMODE::VSEL_TENSOR_SCALAR_MODE, calCount);
        PipeBarrier<PIPE_V>();
        SplitWindow(bufferT, tmpBuffer7, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // baryw buffer 0
        ConcatWindow(tmpBuffer5, bufferB, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        Select(tmpBuffer7, tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer5, tmpBuffer4,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
        PipeBarrier<PIPE_V>();
        SplitWindow(bufferB, tmpBuffer7, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // baryw buffer 1
        ConcatWindow(tmpBuffer5, bufferB[bufferElems_], startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        Select(tmpBuffer7, tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer5, tmpBuffer3,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
        PipeBarrier<PIPE_V>();
        SplitWindow(bufferB[bufferElems_], tmpBuffer7, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        // baryw buffer 2
        ConcatWindow(tmpBuffer5, bufferB[bufferElems_ * CONSTANT_TWO], startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
        Select(tmpBuffer7, tmpBuffer6.ReinterpretCast<uint8_t>(), tmpBuffer5, tmpBuffer2,
               SELMODE::VSEL_TENSOR_TENSOR_MODE, calCount);
        PipeBarrier<PIPE_V>();
        SplitWindow(bufferB[bufferElems_ * CONSTANT_TWO], tmpBuffer7, startIndex, windowSizeH, windowSizeW);
        PipeBarrier<PIPE_V>();
    }
    queueFV.FreeTensor(bufferFV);
}

__aicore__ inline void CopyOutDepth(const int32_t ubIdxH,
                                    const int32_t curH) {
    queueD.EnQue(bufferD);
    bufferD = queueD.DeQue<float>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curH * W_ * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    int calCount = curH * W_;
    DataCopy(depthOutGM[bufferElems_ * ubIdxH], bufferD, calCount);
    queueD.FreeTensor(bufferD);
}

__aicore__ inline void CopyOutTriangle(const int32_t ubIdxH,
                                       const int32_t curH) {
    queueT.EnQue(bufferT);
    bufferT = queueT.DeQue<float>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curH * W_ * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    int calCount = curH*W_;
    DataCopy(triangleOutGM[bufferElems_ * ubIdxH], bufferT, calCount);
    queueT.FreeTensor(bufferT);
}

__aicore__ inline void CopyOutBaryw(const int32_t ubIdxH,
                                    const int32_t curH) {
    queueB.EnQue(bufferB);
    bufferB = queueB.DeQue<float>();
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = curH * W_ * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    int calCount = curH * W_;
    DataCopy(barywOutGM[bufferElems_ * ubIdxH], bufferB, calCount);
    DataCopy(barywOutGM[bufferElems_ * ubIdxH + W_ * H_], bufferB[bufferElems_], calCount);
    DataCopy(barywOutGM[bufferElems_ * ubIdxH + W_ * H_ * CONSTANT_TWO],
             bufferB[bufferElems_ * CONSTANT_TWO], calCount);
    queueB.FreeTensor(bufferB);
}

__aicore__ inline void InitIndexH(const LocalTensor<float>& dst,
                                  const int32_t ubIdxH,
                                  const int32_t curH) {
    float startValue = static_cast<float>(blockFormerH_ * blockIdx + ubIdxH * ubFormerH_);
    Duplicate(dst, startValue, W_);
    PipeBarrier<PIPE_V>();
    for (int32_t i = 1; i < curH; i++) {
        Adds(dst[i * W_], dst, static_cast<float>(i), W_);
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void InitIndexW(const LocalTensor<float>& dst) {
    // only 1 row needs to be generated, such as 0, 1, 2, ... , W - 1
    float firstValue = 0.0;
    dst.SetValue(0, FLOAT_CONSTANT_ZERO);
    dst.SetValue(1, FLOAT_CONSTANT_ONE);
    dst.SetValue(CONSTANT_TWO, FLOAT_CONSTANT_TWO);
    dst.SetValue(CONSTANT_THREE, FLOAT_CONSTANT_THREE);
    dst.SetValue(CONSTANT_FOUR, FLOAT_CONSTANT_FOUR);
    dst.SetValue(CONSTANT_FIVE, FLOAT_CONSTANT_FIVE);
    dst.SetValue(CONSTANT_SIX, FLOAT_CONSTANT_SIX);
    dst.SetValue(CONSTANT_SEVEN, FLOAT_CONSTANT_SEVEN);
    auto eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventID);
    WaitFlag<HardEvent::S_V>(eventID);
    int32_t offset;
    for (int32_t i = 1; i < B32_BLOCK_SIZE; i++) {
        offset = i * B32_BLOCK_SIZE;
        Adds(dst[offset], dst, static_cast<float>(offset), B32_BLOCK_SIZE, 1, {1, 1, 1, 1});
    }
    PipeBarrier<PIPE_V>();
    for (int32_t i = 1; i < W_ / B32_REPEAT_SIZE; i++) {
        offset = i * B32_REPEAT_SIZE;
        Adds(dst[offset], dst, static_cast<float>(offset), B32_REPEAT_SIZE, 1, {1, 1, 1, 1});
    }
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void ConcatWindow(const LocalTensor<float>& dst,
                                    const LocalTensor<float>& src,
                                    const int32_t startIndex,
                                    const int32_t windowSizeH,
                                    const int32_t windowSizeW) {
    DataCopyParams intriParams;
    intriParams.blockCount = windowSizeH;
    intriParams.blockLen = windowSizeW * sizeof(float) / BLOCK_BYTES;
    intriParams.srcStride = (W_ - windowSizeW) * sizeof(float) / BLOCK_BYTES;
    intriParams.dstStride = 0;
    DataCopy(dst, src[startIndex], intriParams);
}

__aicore__ inline void SplitWindow(const LocalTensor<float>& dst,
                                   const LocalTensor<float>& src,
                                   const int32_t startIndex,
                                   const int32_t windowSizeH,
                                   const int32_t windowSizeW) {
    DataCopyParams intriParams;
    intriParams.blockCount = windowSizeH;
    intriParams.blockLen = windowSizeW * sizeof(float) / BLOCK_BYTES;
    intriParams.srcStride = 0;
    intriParams.dstStride = (W_ - windowSizeW) * sizeof(float) / BLOCK_BYTES;
    DataCopy(dst[startIndex], src, intriParams);
}

__aicore__ inline void AddWithInlinedNLastBrcFP32(const LocalTensor<float>& dst,
                                                  const LocalTensor<float>& src0,
                                                  const LocalTensor<float>& src1,
                                                  const int64_t curH,
                                                  const int32_t windowSizeW) {
    // src1 need to do inline broadcast
    for (int64_t i = 0; i < curH; i++) {
        int64_t formerOffset = i * windowSizeW;
        Add(dst[formerOffset], src0[formerOffset], src1, windowSizeW);
    }
}

__aicore__ inline void DuplicateWindowMask(const LocalTensor<float>& dst,
                                           const int32_t windowSizeH,
                                           const int32_t windowSizeW,
                                           const int32_t minW,
                                           const int32_t minWAlign,
                                           const int32_t maxW,
                                           const int32_t maxWAlign) {
    int32_t leftDiff = minW - minWAlign;
    if (leftDiff > 0) {
        Duplicate(dst, MAX_FP32_VALUE, static_cast<uint64_t>(leftDiff), windowSizeH, 1,
                  windowSizeW / CONSTANT_EIGHT);
    }
    int32_t rightDiff = maxWAlign - 1 - maxW;
    PipeBarrier<PIPE_V>();
    if (rightDiff > 0) {
        uint64_t mask[2] = {(((uint64_t)1 << rightDiff) - 1) << (B32_BLOCK_SIZE - rightDiff), 0};
        Duplicate(dst[windowSizeW - B32_BLOCK_SIZE], MAX_FP32_VALUE, mask, windowSizeH, 1,
                  windowSizeW / CONSTANT_EIGHT);
    }
}

private:
TPipe* pipePtr;

TQue<QuePosition::VECIN, 1> queueFV;
TQue<QuePosition::VECOUT, 1> queueD;
TQue<QuePosition::VECOUT, 1> queueT;
TQue<QuePosition::VECOUT, 1> queueB;
TBuf<> tmpTensor;

GlobalTensor<float> fVInGM;
GlobalTensor<float> depthOutGM;
GlobalTensor<float> triangleOutGM;
GlobalTensor<float> barywOutGM;
GlobalTensor<float> workspaceGM;

LocalTensor<float> bufferFV;
LocalTensor<float> bufferD;
LocalTensor<float> bufferT;
LocalTensor<float> bufferB;
LocalTensor<float> tmpBuffer0;
LocalTensor<float> tmpBuffer1;
LocalTensor<float> tmpBuffer2;
LocalTensor<float> tmpBuffer3;
LocalTensor<float> tmpBuffer4;
LocalTensor<float> tmpBuffer5;
LocalTensor<float> tmpBuffer6;
LocalTensor<float> tmpBuffer7;
int32_t blockIdx;
int32_t blockNum_;
int32_t B_;
int32_t N_;
int32_t H_;
int32_t W_;
int32_t blockFormerH_;
int32_t blockTailH_;
int32_t ubFormerH_;
int32_t ubLoopOfFormerBlockH_;
int32_t ubLoopOfTailBlockH_;
int32_t ubTailOfFormerBlockH_;
int32_t ubTailOfTailBlockH_;
int32_t ubFormerNtri_;
int32_t ubTailNtri_;
int32_t ubLoopOfNtri_;
int32_t bufferElems_;
};  // namespace ForwardRasterize
}

extern "C" __global__ __aicore__ void forward_rasterize(GM_ADDR face_vertices, GM_ADDR out_shape,
    GM_ADDR output_buffer, GM_ADDR workspace, GM_ADDR tiling) {
    if (g_coreType == AscendC::AIC) {
        return;
    }
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    AscendC::TPipe stepTwoPipe;
    if (TILING_KEY_IS(0)) {
        ForwardRasterize::ForwardRasterize op;
        op.Init(face_vertices, output_buffer,
                usrWorkspace, &tilingData, &stepTwoPipe);
        op.Process();
    }
    return;
}