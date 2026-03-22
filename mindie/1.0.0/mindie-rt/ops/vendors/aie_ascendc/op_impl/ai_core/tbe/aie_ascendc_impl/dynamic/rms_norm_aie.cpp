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
constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BUF_FACTOR = 4;       // 1(g) + 1(sqx) + 1(sum) + 1(workspace) = 4
constexpr uint32_t OFFSET_GAMMA = 0;     // the offset of gamma is 0
constexpr uint32_t OFFSET_SQX = 1;       // the offset of sqx is 1
constexpr uint32_t OFFSET_SUM = 2;       // the offset of sum is 2
constexpr uint32_t OFFSET_WORKSPACE = 3; // the offset of workspace is 3

class RmsNormShort {
public:
    __aicore__ inline RmsNormShort(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
        uint32_t numCore, uint32_t numCol, uint32_t numRow, uint32_t avgFactor, uint32_t epsilon,
        uint32_t sliceSize)
        : numCore_(numCore), numCol_(numCol), avgFactor_(*reinterpret_cast<float *>(&avgFactor)),
        epsilon_(*reinterpret_cast<float *>(&epsilon)), sliceSize_(sliceSize)
    {
        uint32_t rowWork = numRow / numCore;
        if (block_idx != numCore - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore - 1) * rowWork;
        }
        gmOffset_ = rowWork * numCol_;
        gmX_.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gmOffset_);
        gmG_.SetGlobalBuffer((__gm__ half *)g);
        gmY_.SetGlobalBuffer((__gm__ half *)y + AscendC::GetBlockIdx() * gmOffset_);
        pipe.InitBuffer(fp16XQue_, BUFFER_NUM, numCol_ * sizeof(half));
        pipe.InitBuffer(fp16YQue_, BUFFER_NUM, numCol_ * sizeof(half));
        pipe.InitBuffer(fp32XYBuf_, numCol_ * sizeof(float));
        pipe.InitBuffer(calcBuf_, BUF_FACTOR * numCol_ * sizeof(float));
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<half> fp16G = fp32XYBuf_.Get<half>(numCol_);
        AscendC::LocalTensor<float> fp32G = calcBuf_.Get<float>(numCol_);
        DataCopy(fp16G, gmG_, numCol_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Cast(fp32G, fp16G, AscendC::RoundMode::CAST_NONE, numCol_);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        int32_t pid = 0;
        while (pid < rowWork_) {
            uint32_t offset = pid * numCol_;
            CopyIn(offset, numCol_);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            Compute();

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            CopyOut(offset, numCol_);
            ++pid;
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16X = fp16XQue_.AllocTensor<half>();
        DataCopy(fp16X, gmX_[offset], numel);
        fp16XQue_.EnQue(fp16X);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> fp16X = fp16XQue_.DeQue<half>();
        AscendC::LocalTensor<float> fp32XY = fp32XYBuf_.Get<float>();
        AscendC::LocalTensor<half> fp16Y = fp16YQue_.AllocTensor<half>();
        pipe_barrier(PIPE_V);
        Cast(fp32XY, fp16X, AscendC::RoundMode::CAST_NONE, numCol_);
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * numCol_];
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * numCol_];
        AscendC::LocalTensor<float> sum = buf[OFFSET_SUM * numCol_];
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * numCol_];

        pipe_barrier(PIPE_V);
        Mul(sqx, fp32XY, fp32XY, numCol_);

        pipe_barrier(PIPE_V);
        ReduceSum(sum, sqx, work, numCol_);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float rms = sum.GetValue(0);
        sum.SetValue(0, rms * avgFactor_ + epsilon_);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        Sqrt(sum, sum, 1);
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);

        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        float factor = sum.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_V);
        Duplicate(work, factor, numCol_);

        pipe_barrier(PIPE_V);
        Div(fp32XY, fp32XY, work, numCol_);

        pipe_barrier(PIPE_V);
        Mul(fp32XY, fp32XY, g, numCol_);

        pipe_barrier(PIPE_V);
        Cast(fp16Y, fp32XY, AscendC::RoundMode::CAST_NONE, numCol_);

        fp16YQue_.EnQue(fp16Y);
        fp16XQue_.FreeTensor(fp16X);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16Y = fp16YQue_.DeQue<half>();
        DataCopy(gmY_[offset], fp16Y, numel);
        fp16YQue_.FreeTensor(fp16Y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16XQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16YQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32XYBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::GlobalTensor<half> gmX_;
    AscendC::GlobalTensor<half> gmG_;
    AscendC::GlobalTensor<half> gmY_;
    uint32_t numCore_{0};   // 一共激活多少AICORE
    uint32_t numCol_{0};    // 输入的列数
    uint32_t rowWork_{0};   // 需要计算多少行
    uint32_t rowStep_{0};   // 除最后一次，每次搬入多少行
    uint32_t rowTail_{0};   // 最后一次搬入多少行数据
    uint32_t gmOffset_{0};  // GM数据起始位置偏移量
    uint32_t sliceSize_{0}; // 每一行切分的大小
    float avgFactor_{1.0f}; // numCol_的倒数
    float epsilon_{1e-12f};  // norm平滑参数
};

class RmsNormLong {
public:
    __aicore__ inline RmsNormLong(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *y,
        uint32_t numCore, uint32_t numCol, uint32_t numRow, uint32_t avgFactor, uint32_t epsilon,
        uint32_t sliceSize)
        : numCore_(numCore), numCol_(numCol), avgFactor_(*reinterpret_cast<float *>(&avgFactor)),
        epsilon_(*reinterpret_cast<float *>(&epsilon)), sliceSize_(sliceSize)
    {
        uint32_t rowWork = numRow / numCore;
        if (block_idx != numCore - 1) {
            rowWork_ = rowWork;
        } else {
            rowWork_ = numRow - (numCore - 1) * rowWork;
        }
        numSlice_ = (numCol_ + sliceSize_ - 1) / sliceSize_;
        tailSize_ = numCol_ - (numSlice_ - 1) * sliceSize_;
        gmOffset_ = rowWork * numCol_;
        gmX_.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gmOffset_);
        gmG_.SetGlobalBuffer((__gm__ half *)g);
        gmY_.SetGlobalBuffer((__gm__ half *)y + AscendC::GetBlockIdx() * gmOffset_);
        pipe.InitBuffer(fp16XQue_, BUFFER_NUM, sliceSize_ * sizeof(half));
        pipe.InitBuffer(fp16YQue_, BUFFER_NUM, sliceSize_ * sizeof(half));
        pipe.InitBuffer(fp32XYBuf_, sliceSize_ * sizeof(float));
        pipe.InitBuffer(calcBuf_, BUF_FACTOR * sliceSize_ * sizeof(float));
    }

    __aicore__ inline void Launch()
    {
        int32_t pid = 0;
        while (pid < rowWork_) {
            uint32_t row_offset = pid * numCol_;
            int32_t num_ele = sliceSize_;
            squareSum_ = 0.0f;
            for (uint32_t sid = 0; sid < numSlice_; sid++) {
                uint32_t col_offset = row_offset + sid * sliceSize_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    num_ele = tailSize_;
                }
                CopyInX(col_offset, num_ele);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                squareSum_ += ComputeSquareSum(num_ele);
            }
            num_ele = sliceSize_;
            float factor = avgFactor_ * squareSum_ + epsilon_;
            for (uint32_t sid = 0; sid < numSlice_; sid++) {
                uint32_t col_offset = row_offset + sid * sliceSize_;
                if ((sid == (numSlice_ - 1)) && (tailSize_ != 0)) {
                    num_ele = tailSize_;
                }
                CopyInX(col_offset, num_ele);
                CopyInG(sid * sliceSize_, num_ele);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                ComputeNorm(factor, num_ele);
                pipe_barrier(PIPE_V);
                CopyOut(col_offset, num_ele);
            }
            pid++;
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void CopyInX(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16X = fp16XQue_.AllocTensor<half>();
        DataCopy(fp16X, gmX_[offset], numel);
        fp16XQue_.EnQue(fp16X);
    }

    __aicore__ inline void CopyInG(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16G = fp32XYBuf_.Get<half>();
        AscendC::LocalTensor<float> fp32G = calcBuf_.Get<float>();
        
        DataCopy(fp16G, gmG_[offset], numel);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        Cast(fp32G, fp16G, AscendC::RoundMode::CAST_NONE, numel);
    }

    __aicore__ inline float ComputeSquareSum(uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16X = fp16XQue_.DeQue<half>();
        AscendC::LocalTensor<float> fp32XY = fp32XYBuf_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * sliceSize_];
        AscendC::LocalTensor<float> sum = buf[OFFSET_SUM * sliceSize_];
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * sliceSize_];

        Cast(fp32XY, fp16X, AscendC::RoundMode::CAST_NONE, numel);
        pipe_barrier(PIPE_V);
        Mul(sqx, fp32XY, fp32XY, numel);
        pipe_barrier(PIPE_V);
        ReduceSum(sum, sqx, work, numel);
        pipe_barrier(PIPE_V);

        fp16XQue_.FreeTensor(fp16X);

        return sum.GetValue(0);
    }

    __aicore__ inline void ComputeNorm(float sqs, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16X = fp16XQue_.DeQue<half>();
        AscendC::LocalTensor<half> fp16Y = fp16YQue_.AllocTensor<half>();
        AscendC::LocalTensor<float> fp32XY = fp32XYBuf_.Get<float>();
        AscendC::LocalTensor<float> buf = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * sliceSize_];
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * sliceSize_];
        AscendC::LocalTensor<float> sum = buf[OFFSET_SUM * sliceSize_];
        AscendC::LocalTensor<float> work = buf[OFFSET_WORKSPACE * sliceSize_];

        Cast(fp32XY, fp16X, AscendC::RoundMode::CAST_NONE, numel);
        pipe_barrier(PIPE_V);
        Duplicate(sum, sqs, numel);
        pipe_barrier(PIPE_V);
        Sqrt(work, sum, numel);
        pipe_barrier(PIPE_V);
        Mul(fp32XY, fp32XY, g, numel);
        pipe_barrier(PIPE_V);
        Div(fp32XY, fp32XY, work, numel);
        pipe_barrier(PIPE_V);
        Cast(fp16Y, fp32XY, AscendC::RoundMode::CAST_NONE, numel);

        fp16XQue_.FreeTensor(fp16X);
        fp16YQue_.EnQue(fp16Y);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<half> fp16Y = fp16YQue_.DeQue<half>();
        DataCopy(gmY_[offset], fp16Y, numel);
        fp16YQue_.FreeTensor(fp16Y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16XQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> fp16YQue_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32XYBuf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calcBuf_;
    AscendC::GlobalTensor<half> gmX_;
    AscendC::GlobalTensor<half> gmG_;
    AscendC::GlobalTensor<half> gmY_;
    uint32_t numCore_{0};   // 一共激活多少AICORE
    uint32_t numCol_{0};    // 输入的列数
    uint32_t rowWork_{0};   // 需要计算多少行
    uint32_t rowStep_{0};   // 除最后一次，每次搬入多少行
    uint32_t rowTail_{0};   // 最后一次搬入多少行数据
    uint32_t gmOffset_{0};  // GM数据起始位置偏移量
    uint32_t sliceSize_{0}; // 每一行切分的大小
    int32_t numSlice_{0};
    int32_t tailSize_{0};
    float avgFactor_{1.0f}; // numCol_的倒数
    float epsilon_{1e-12f};  // norm平滑参数
    float squareSum_{0.0f};
};
}

namespace {
extern "C" __global__ __aicore__ void rms_norm_aie(GM_ADDR x, GM_ADDR g, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (tiling_data.numCol > tiling_data.sliceSize) {
        RmsNormLong kernel(x, g, y, tiling_data.numCore, tiling_data.numCol, tiling_data.numRow,
                        tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    } else {
        RmsNormShort kernel(x, g, y, tiling_data.numCore, tiling_data.numCol, tiling_data.numRow,
                        tiling_data.avgFactor, tiling_data.epsilon, tiling_data.sliceSize);
        kernel.Launch();
    }
}
}