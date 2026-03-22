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
#include "yulu_tiling.h"

using namespace AscendC;
template <uint8_t bufferNum = 2>
struct YuLuKernel {
static constexpr float beta = -1.0;
public:
    // 空的构造函数，使用 inline 关键字声明，在每个使用点展开，以避免函数调用的开销
    __aicore__ inline YuLuKernel(GM_ADDR tiling, TPipe& pipe) : pipe_(pipe)
    {
        tiling_.GetTilingAndOffset(tiling, sizeof(DTYPE_X));
    }
    // 析构函数用于在对象生命周期结束时进行清理工作
    __aicore__ inline ~YuLuKernel() = default;
    __aicore__ inline void Process(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR zGm, GM_ADDR outGm)
    {
        Init(xGm, yGm, zGm, outGm);
        Process();
    }

private:
    __aicore__ inline void Init(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR zGm, GM_ADDR outGm)
    {
        // get start index for current core, core parallel
        this->xGm_.SetGlobalBuffer((__gm__ DTYPE_X*)xGm + tiling_.gmOffset, tiling_.blockLength);
        this->yGm_.SetGlobalBuffer((__gm__ DTYPE_Y*)yGm + tiling_.gmOffset, tiling_.blockLength);
        this->zGm_.SetGlobalBuffer((__gm__ DTYPE_Z*)zGm + tiling_.gmOffset, tiling_.blockLength);
        this->outGm_.SetGlobalBuffer((__gm__ DTYPE_OUT*)outGm + tiling_.gmOffset, tiling_.blockLength);

        // pipe alloc memory to queue, the unit is Bytes
        pipe_.InitBuffer(inQueueX_, bufferNum, tiling_.tileLength * sizeof(DTYPE_X));
        pipe_.InitBuffer(inQueueY_, bufferNum, tiling_.tileLength * sizeof(DTYPE_Y));
        pipe_.InitBuffer(inQueueZ_, bufferNum, tiling_.tileLength * sizeof(DTYPE_Z));
        pipe_.InitBuffer(outQueue_, bufferNum, tiling_.tileLength * sizeof(DTYPE_OUT));

        pipe_.InitBuffer(inputTempBuffer_, tiling_.tileLength * sizeof(float));
        pipe_.InitBuffer(outputTempBuffer_, tiling_.tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint64_t i = 0; i < tiling_.tileLoopNum; i++) {
            CopyIn(i * tiling_.tileLength, tiling_.tileLength);
            this->Compute(tiling_.tileLength);
            CopyOut(i * tiling_.tileLength, tiling_.tileLength);
        }
        if (tiling_.tailTileLen > 0) {
            CopyIn(tiling_.tileLoopNum * tiling_.tileLength, tiling_.tailTileLen);
            this->Compute(tiling_.tailTileLen);
            CopyOut(tiling_.tileLoopNum * tiling_.tileLength, tiling_.tailTileLen);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint64_t curTileLen)
    {
        // Copy X
        auto xLocal = this->inQueueX_.template AllocTensor<DTYPE_X>();
        ::DataCopy(xLocal, this->xGm_[offset], curTileLen);
        this->inQueueX_.EnQue(xLocal);
        // Copy Y
        auto yLocal = this->inQueueY_.template AllocTensor<DTYPE_Y>();
        ::DataCopy(yLocal, this->yGm_[offset], curTileLen);
        this->inQueueY_.EnQue(yLocal);
        // Copy Z
        auto zLocal = this->inQueueZ_.template AllocTensor<DTYPE_Z>();
        ::DataCopy(zLocal, this->zGm_[offset], curTileLen);
        this->inQueueZ_.EnQue(zLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint64_t curTileLen)
    {
        // deque output tensor from VECOUT queue
        LocalTensor<DTYPE_OUT> outLocal = this->outQueue_.template DeQue<DTYPE_OUT>();
        // copy progress_th tile from local tensor to global tensor
        ::DataCopy(this->outGm_[offset], outLocal, curTileLen);
        // free output tensor for reuse
        this->outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void Compute(uint64_t curTileLen)
    {
        auto xLocal = inQueueX_.template DeQue<DTYPE_X>();
        auto yLocal = inQueueY_.template DeQue<DTYPE_Y>();
        auto zLocal = inQueueZ_.template DeQue<DTYPE_Z>();
        ::MulAddDst(zLocal, xLocal, yLocal, curTileLen);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
        pipe_barrier(PIPE_V);

        LocalTensor<float> inputTemp = inputTempBuffer_.Get<float>();
        LocalTensor<float> outputTemp = outputTempBuffer_.Get<float>();
        ::Cast(inputTemp, zLocal, RoundMode::CAST_NONE, curTileLen);
        inQueueZ_.FreeTensor(zLocal);
        pipe_barrier(PIPE_V);

        // 调用Silu计算
        Silu(outputTemp, inputTemp, curTileLen);
        pipe_barrier(PIPE_V);

        // 将计算结果存入输出队列
        auto outLocal = outQueue_.template AllocTensor<DTYPE_OUT>();
        ::Cast(outLocal, outputTemp, RoundMode::CAST_NONE, curTileLen);
        outQueue_.EnQue(outLocal);
    }

    __aicore__ inline void Silu(const LocalTensor<float>& dst, const LocalTensor<float>& src, uint64_t curTileLen)
    {
        ::Muls(dst, src, beta, curTileLen);
        pipe_barrier(PIPE_V);

        // exp(-x)
        ::Exp(dst, dst, curTileLen);
        pipe_barrier(PIPE_V);

        // 1 + exp(-x)
        ::Adds(dst, dst, (float) (1.0), curTileLen);
        pipe_barrier(PIPE_V);

        // x / (1 + exp(-x))
        ::Div(dst, src, dst, curTileLen);
    }

private:
    // 将数据从GM搬运
    TQue<QuePosition::VECIN, bufferNum> inQueueX_;
    TQue<QuePosition::VECIN, bufferNum> inQueueY_;
    TQue<QuePosition::VECIN, bufferNum> inQueueZ_;
    TQue<QuePosition::VECOUT, bufferNum> outQueue_;

    // 临时存储空间
    TBuf<TPosition::VECCALC> inputTempBuffer_;
    TBuf<TPosition::VECCALC> outputTempBuffer_;

    GlobalTensor <DTYPE_X> xGm_;
    GlobalTensor <DTYPE_Y> yGm_;
    GlobalTensor <DTYPE_Z> zGm_;
    GlobalTensor <DTYPE_OUT> outGm_;

    YuluTilingKernel tiling_;
    TPipe& pipe_;
};

extern "C" __global__ __aicore__ void yulu(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR zGm, GM_ADDR outGm,
                                           GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    GET_TILING_DATA(tempTilingGm, tiling);
    if (TILING_KEY_IS(1)) {
        if (tempTilingGm.isDoubleBuffer == 1) {
            YuLuKernel<2> op(tiling, pipe);
            op.Process(xGm, yGm, zGm, outGm);
        } else {
            YuLuKernel<1> op(tiling, pipe);
            op.Process(xGm, yGm, zGm, outGm);
        }
    }
}

