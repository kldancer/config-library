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

namespace {
static constexpr uint32_t BUFFER_NUM = 1;       // split the UB to 2 equal part to enable ping-pong techniques
static constexpr uint32_t BUF_FACTOR = 4;       // 1(g) + 1(sqx) + 1(sum) + 1(workspace) = 4
static constexpr uint32_t OFFSET_GAMMA = 0;     // the offset of gamma is 0
static constexpr uint32_t OFFSET_SQX = 1;       // the offset of sqx is 1
static constexpr uint32_t OFFSET_SUM = 2;       // the offset of sum is 2
static constexpr uint32_t OFFSET_WORKSPACE = 3; // the offset of workspace is 3

static constexpr uint32_t BLOCK_SIZE = 32;

class RmsNormQuant {
public:
    __aicore__ inline RmsNormQuant(__gm__ uint8_t *x, __gm__ uint8_t *g, __gm__ uint8_t *b, __gm__ uint8_t *y,
        uint32_t numCore, uint32_t numCol, uint32_t numRow, float avgFactor,
        float inputScale, int32_t inputOffset, float epsilon, float offsetFp,
        uint32_t ascendType, float quantMin)
        : numCore_(numCore), numCol_(numCol), avgFactor_(*reinterpret_cast<float *>(&avgFactor)),
        inputScale_(*reinterpret_cast<float *>(&inputScale)),
        inputOffset_(*reinterpret_cast<float *>(&inputOffset)), epsilon_(*reinterpret_cast<float *>(&epsilon)),
        ascendType_(*reinterpret_cast<int32_t *>(&ascendType))
    {
        uint32_t row_work = (numRow + numCore - 1) / numCore;
        if (block_idx != numCore - 1) {
            rowWork_ = row_work;
        } else {
            rowWork_ = numRow - (numCore - 1) * row_work;
        }
        if (ascendType_ == 0) {
            cast_mode_enum = AscendC::RoundMode::CAST_NONE;
        } else if (ascendType_ == 2) {
            cast_mode_enum = AscendC::RoundMode::CAST_RINT;
        }
        gmOffset_ = row_work * numCol_;
        inputOffsetFP_ = *(reinterpret_cast<float *>(&offsetFp));
        quantMin_ = quantMin;

        gm_x_.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gmOffset_);
        gm_g_.SetGlobalBuffer((__gm__ half *)g);
        gm_b_.SetGlobalBuffer((__gm__ half *)b);
        gm_y_.SetGlobalBuffer((__gm__ int8_t *)y + AscendC::GetBlockIdx() * gmOffset_);

        pipe.InitBuffer(fp16_x_que_, BUFFER_NUM, numCol_ * sizeof(half));
        pipe.InitBuffer(int8_y_que_, BUFFER_NUM, numCol_ * sizeof(int8_t));
        pipe.InitBuffer(fp32_xy_buf_, numCol_ * sizeof(float));
        pipe.InitBuffer(fp16_buf_, numCol_ * sizeof(half));

        pipe.InitBuffer(calc_buf_, BUF_FACTOR * numCol_ * sizeof(float) + 32); // 32 for sum
    }

    __aicore__ inline void UpdateScaleAndOffset(__gm__ uint8_t *s, __gm__ uint8_t* o)
    {
        AscendC::GlobalTensor<half> gm_s;
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_s.SetGlobalBuffer((__gm__ half *)s);
        gm_o.SetGlobalBuffer((__gm__ int8_t *)o);

        AscendC::LocalTensor<half> scale_buffer;
        AscendC::LocalTensor<int8_t> offset_buffer;

        scale_buffer.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
        scale_buffer.InitBuffer(0, BLOCK_SIZE);

        offset_buffer.address_.logicPos = static_cast<uint8_t>(AscendC::TPosition::VECCALC);
        scale_buffer.InitBuffer(BLOCK_SIZE, BLOCK_SIZE);

        DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(half));
        DataCopy(offset_buffer, gm_o, BLOCK_SIZE / sizeof(int8_t));
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        inputScale_ = static_cast<float>(scale_buffer.GetValue(0));
        inputOffsetFP_ = static_cast<float>(offset_buffer.GetValue(0));
    }

    __aicore__ inline void Launch()
    {
        AscendC::LocalTensor<half> fp16_g = fp32_xy_buf_.Get<half>(numCol_);
        AscendC::LocalTensor<half> fp16_buffer = fp16_buf_.Get<half>(numCol_);

        AscendC::LocalTensor<float> fp32_g = calc_buf_.Get<float>(numCol_);
        DataCopy(fp16_g, gm_g_, numCol_);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        Cast(fp32_g[OFFSET_GAMMA * numCol_], fp16_g, AscendC::RoundMode::CAST_NONE, numCol_);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

        DataCopy(fp16_buffer, gm_b_, numCol_);
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
        AscendC::LocalTensor<half> fp16_x = fp16_x_que_.AllocTensor<half>();
        DataCopy(fp16_x, gm_x_[offset], numel);
        fp16_x_que_.EnQue(fp16_x);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<half> fp16_x = fp16_x_que_.DeQue<half>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
        pipe_barrier(PIPE_V);
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, numCol_);
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * numCol_];
        AscendC::LocalTensor<half> b = fp16_buf_.Get<half>(numCol_);
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * numCol_];
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * numCol_];
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * numCol_];

        pipe_barrier(PIPE_V);
        Mul(sqx, fp32_xy, fp32_xy, numCol_);
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
        float factor = 1 / sum.GetValue(0);
        set_flag(PIPE_S, PIPE_V, EVENT_ID0);

        wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
        Muls(fp32_xy, fp32_xy, factor, numCol_);
        
        pipe_barrier(PIPE_V);
        Mul(fp32_xy, fp32_xy, g, numCol_);
        
        pipe_barrier(PIPE_V);
        Cast(fp16_x, fp32_xy, AscendC::RoundMode::CAST_NONE, numCol_);

        pipe_barrier(PIPE_V);
        Add(fp16_x, fp16_x, b, numCol_);

        pipe_barrier(PIPE_V);
        Muls(fp16_x, fp16_x, (half)inputScale_, numCol_);

        pipe_barrier(PIPE_V);
        Adds(fp16_x, fp16_x, (half)inputOffsetFP_, numCol_);

        pipe_barrier(PIPE_V);
        Maxs(fp16_x, fp16_x, quantMin_, numCol_);

        pipe_barrier(PIPE_V);
        Cast(int8_y, fp16_x, cast_mode_enum, numCol_);

        int8_y_que_.EnQue(int8_y);
        fp16_x_que_.FreeTensor(fp16_x);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.DeQue<int8_t>();
        DataCopy(gm_y_[offset], int8_y, numel);
        int8_y_que_.FreeTensor(int8_y);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_x_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int8_y_que_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_buf_;

    AscendC::GlobalTensor<half> gm_x_;
    AscendC::GlobalTensor<half> gm_g_;
    AscendC::GlobalTensor<half> gm_b_;
    AscendC::GlobalTensor<int8_t> gm_y_;

    uint32_t numCore_{0};   // 一共激活多少AICORE
    uint32_t numCol_{0};    // 输入的列数
    uint32_t rowWork_{0};   // 需要计算多少行
    uint32_t rowStep_{0};   // 除最后一次，每次搬入多少行
    uint32_t rowTail_{0};   // 最后一次搬入多少行数据
    uint32_t gmOffset_{0};  // GM数据起始位置偏移量
    float avgFactor_{1.0f}; // numCol_的倒数
    float inputScale_{0};  // 非对称量化系数
    float inputOffset_{0};  // 非对称量化偏移
    float inputOffsetFP_{0};  // 非对称量化偏移适配高精度
    float epsilon_{1e-12f};  // norm平滑参数
    int32_t ascendType_{0}; // ascend type
    half quantMin_ = -128;
    half quantMax_ = 127;
    AscendC::RoundMode cast_mode_enum{0}; // 取整参数
};
}

namespace {
extern "C" __global__ __aicore__ void rms_norm_quant(GM_ADDR x, GM_ADDR g, GM_ADDR b, GM_ADDR s, GM_ADDR o, GM_ADDR y,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    RmsNormQuant kernel(x, g, b, y, tiling_data.numCore, tiling_data.numCol, tiling_data.numRow,
                        tiling_data.avgFactor, tiling_data.inputScale, tiling_data.inputOffset,
                        tiling_data.epsilon, tiling_data.offsetFp, tiling_data.ascendType,
                        tiling_data.quantMin);
    kernel.Launch();
}
}