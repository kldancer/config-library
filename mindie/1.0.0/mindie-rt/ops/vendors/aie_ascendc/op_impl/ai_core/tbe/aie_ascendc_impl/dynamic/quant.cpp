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
static constexpr int32_t BUFFER_NUM = 1; // tensor num for each queue
static constexpr uint32_t DATA_BYTE = 2;
static constexpr uint32_t BLOCK_NUMEL = 16;
static constexpr half QUANT_MAX = 127;

static constexpr uint32_t BLOCK_SIZE = 32;

namespace {
class Quant {
public:
    __aicore__ inline Quant() {}

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y == 0) {
            return UINT32_MAX;
        }
        return (x + y - 1) / y;
    }

    __aicore__ inline uint32_t ROUND_UP(uint32_t x)
    {
        return (x + BLOCK_NUMEL - 1) / BLOCK_NUMEL * BLOCK_NUMEL;
    }

    __aicore__ inline uint32_t MIN(uint32_t x, uint32_t y)
    {
        return x < y ? x : y;
    }
    __aicore__ inline uint32_t MAX(uint32_t x, uint32_t y)
    {
        return x > y ? x : y;
    }

    __aicore__ inline void Init(__gm__ uint8_t *x,
        __gm__ uint8_t *z, uint32_t num_core_, uint32_t num_Last_dim_, uint32_t num_first_dim_,
        uint32_t nl_first_dim_per_core_, uint32_t l_first_dim_per_core_, uint32_t first_dim_per_times_, float scale_,
        float offset_, uint32_t ascend_type_, int32_t quant_min_)
    {
        // 一次搬入多行
        num_core = num_core_;
        num_last_dim = num_Last_dim_;
        num_first_dim = num_first_dim_;
        nl_first_dim_per_core = nl_first_dim_per_core_;
        l_first_dim_per_core = l_first_dim_per_core_;
        first_dim_per_times = first_dim_per_times_;
        quant_min = static_cast<half>(quant_min_);

        inputScale1 = *reinterpret_cast<float *>(&scale_);
        inputOffsetFp = *reinterpret_cast<float *>(&offset_);
        ascend_type = *reinterpret_cast<int32_t *>(&ascend_type_);

        if (block_idx != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }
        if (ascend_type == 0) {
            cast_mode_enum = AscendC::RoundMode::CAST_NONE;
        } else if (ascend_type == 2) {
            cast_mode_enum = AscendC::RoundMode::CAST_RINT;
        }
        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);
        gm_offset_ = nl_first_dim_per_core * num_last_dim;
        x_gm.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP(num_last_dim) * sizeof(int8_t));
    }

    __aicore__ inline void Process()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step); // 一个核需要做多少次
        for (uint32_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                CopyIn(i, row_step * num_last_dim);

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

                Compute(row_step);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);

                CopyOut(i, row_step * num_last_dim);
            } else {
                CopyIn(i, row_tail_ * num_last_dim);

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

                Compute(row_tail_);

                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);

                CopyOut(i, row_tail_ * num_last_dim);
            }
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t proc_id, int32_t size)
    {
        // alloc tensor from queue memory
        LocalTensor<half> x_local = x_que.AllocTensor<half>();
        uint32_t offset = proc_id * row_step * num_last_dim;
        DataCopy(x_local, x_gm[offset], size);
        x_que.EnQue(x_local);
    }

    __aicore__ inline void Compute(int32_t nums)
    {
        LocalTensor<half> x_local = x_que.DeQue<half>();
        LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();

        for (int32_t rid = 0; rid < nums; ++rid) {
            pipe_barrier(PIPE_V);
            Muls(x_local[rid * num_last_dim], x_local[rid * num_last_dim], (half)inputScale1, num_last_dim);
            pipe_barrier(PIPE_V);
            Adds(x_local[rid * num_last_dim], x_local[rid * num_last_dim], (half)inputOffsetFp, num_last_dim);
            pipe_barrier(PIPE_V);
            Maxs(x_local[rid * num_last_dim], x_local[rid * num_last_dim], quant_min, num_last_dim);
            pipe_barrier(PIPE_V);
            Mins(x_local[rid * num_last_dim], x_local[rid * num_last_dim], quant_max, num_last_dim);
            pipe_barrier(PIPE_V);
            Cast(z_local[rid * num_last_dim], x_local[rid * num_last_dim], cast_mode_enum, num_last_dim);
        }

        z_que.EnQue(z_local);
        x_que.FreeTensor(x_local);
    }

    __aicore__ inline void CopyOut(uint32_t proc_id, int32_t size)
    {
        LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        uint32_t offset = proc_id * row_step * num_last_dim; // 单核一次总共做了多少。
        DataCopy(z_gm[offset], z, size);
        z_que.FreeTensor(z);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z_que;
    GlobalTensor<half> x_gm;
    GlobalTensor<int8_t> z_gm;
    float inputScale1 { 1.0 };
    float inputOffsetFp { 0 };
    int32_t input_offset;
    uint32_t num_core;      // 一共激活多少AICORE
    uint32_t num_first_dim; // 输入的列数
    uint32_t num_last_dim;  // 输入的列数
    uint32_t row_work;      // 每个AICORE需要计算多少行
    uint32_t row_step;      // 除最后一次，每次搬入多少行
    uint32_t row_tail_;     // 最后一次搬入多少行数据
    uint32_t gm_offset_;    // GM数据起始位置偏移量
    uint32_t nl_first_dim_per_core;
    uint32_t l_first_dim_per_core;
    uint32_t first_dim_per_times;
    int32_t ascend_type { 0 };
    half quant_min = -128;
    half quant_max = 127;
    RoundMode cast_mode_enum { 0 };
};
}

namespace {
extern "C" __global__ __aicore__ void quant(GM_ADDR x, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    Quant op;
    op.Init(x, z, tiling_data.numCore, tiling_data.numLastDim, tiling_data.numFirstDim, tiling_data.nlFirstdimPerCore,
        tiling_data.lFirstdimPerCore, tiling_data.firstDimPerTimes, tiling_data.inputScale, tiling_data.inputOffset,
        tiling_data.ascendType, tiling_data.quantMin);
    op.Process();
}
}