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
#include "kernel_utils.h"

using namespace AscendC;

namespace {
enum class DataFormatEnum {
    ND,
    NZ
};
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_FP16 = 16384; // 32 KB
constexpr uint32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 32768; // 32 KB
constexpr uint32_t BLOCK_SIZE_16 = 16;
constexpr uint32_t BLOCK_SIZE_32 = 32;
constexpr uint32_t CUBE_MATRIX_SIZE_256 = 256;           // 16 * 16
constexpr uint32_t CUBE_MATRIX_SIZE_512 = 16 * 32;       // 16 * 23
constexpr uint64_t L1_PINGPONG_BUFFER_LEN_FP16 = 131072; // 256 KB
constexpr uint64_t L1_PINGPONG_BUFFER_LEN_INT8 = 262144; // 256 KB
constexpr uint64_t SCALE_L1_LEN = 4096;
constexpr uint64_t BIAS_L1_LEN = 2048;
constexpr uint64_t CONST_2 = 2;
constexpr uint64_t CONST_4 = 4;
constexpr uint64_t CONST_8 = 8;
constexpr uint64_t CONST_32 = 32;
constexpr uint64_t CONST_64 = 64;
constexpr uint64_t CONST_128 = 128;
constexpr uint64_t ND2NZ_STRIDE_LIMIT = 65536;

__aicore__ inline __attribute__((always_inline)) uint32_t CeilDiv64(const uint32_t val)
{
    return (val + CONST_64 - 1) / CONST_64;
}

__aicore__ inline __attribute__((always_inline)) uint32_t RoundUp512(const uint32_t val)
{
    return (val + CUBE_MATRIX_SIZE_512 - 1) / CUBE_MATRIX_SIZE_512 * CUBE_MATRIX_SIZE_512;
}

__aicore__ inline __attribute__((always_inline)) uint32_t RoundUp32(const uint32_t val)
{
    return (val + BLOCK_SIZE_32 - 1) / BLOCK_SIZE_32 * BLOCK_SIZE_32;
}

__aicore__ inline __attribute__((always_inline)) uint32_t RoundUp16(const uint32_t val)
{
    return (val + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16 * BLOCK_SIZE_16;
}


template <typename Dtype, DataFormatEnum srcDataFormatEnum, DataFormatEnum dstDataFormatEnum>
__aicore__ inline __attribute__((always_inline)) void CopyGmToL1(__cbuf__ Dtype *dst, __gm__ Dtype *src, uint64_t row,
    uint64_t col, uint64_t tile_row, uint64_t tile_col, uint32_t dstNzC0Stride)
{}


template <>
__aicore__ inline __attribute__((always_inline)) void CopyGmToL1<int8_t, DataFormatEnum::ND, DataFormatEnum::NZ>(
    __cbuf__ int8_t *dst, __gm__ int8_t *src, uint64_t row, uint64_t col, uint64_t tile_row, uint64_t tile_col,
    uint32_t dstNzC0Stride)
{
    if (col < ND2NZ_STRIDE_LIMIT) {
        copy_gm_to_cbuf_multi_nd2nz_b8(dst, // dst
            src,                            // src
            0,                              // sid
            1,                              // ndNum
            tile_row,                       // nValue
            tile_col,                       // dValue
            0,                              // srcNdMatrixStride, unused
            col,                            // srcDValue
            dstNzC0Stride,                  // dstNzC0Stride
            1,                              // dstNzNStride
            0                               // dstNzMatrixStride, unused
        );
    } else {
        for (uint64_t i = 0; i < tile_row; i++) {
            copy_gm_to_cbuf_multi_nd2nz_b8(dst + i * CONST_32, // dst
                src + i * col,                                 // src
                0,                                             // sid
                1,                                             // ndNum
                1,                                             // nValue
                tile_col,                                      // dValue
                0,                                             // srcNdMatrixStride, unused
                0,                                             // srcDValue, unused
                dstNzC0Stride,                                 // dstNzC0Stride
                0,                                             // dstNzNStride, unused
                0                                              // dstNzMatrixStride, unused
            );
        }
    }
}
}


namespace AscendC {
template <bool TA, bool TB, bool SPLIT_K = false, bool HAVE_BIAS = false, bool IS_INT8 = false, typename InDtype = half,
    typename OutDtype = half, typename BiasDtype = float, typename ScaleDtype = float>
class PpMatmulI8 {
public:
    __aicore__ inline PpMatmulI8() {};

    __aicore__ inline void SetArgs(__gm__ uint8_t *__restrict__ a, __gm__ uint8_t *__restrict__ b,
        __gm__ uint8_t *__restrict__ c, __gm__ uint8_t *__restrict__ bias, __gm__ uint8_t *__restrict__ descale,
        const PpMatmulTilingData &tilingData)
    {
        gmBias = reinterpret_cast<__gm__ BiasDtype *>(bias);
        gmDescale = reinterpret_cast<__gm__ ScaleDtype *>(descale);
        gmA = reinterpret_cast<__gm__ InDtype *>(a);
        gmB = reinterpret_cast<__gm__ InDtype *>(b);
        gmC = reinterpret_cast<__gm__ OutDtype *>(c);
        batch_size = tilingData.batch;
        m = tilingData.m;
        k = tilingData.k;
        n = tilingData.n;
        m0 = tilingData.m0;
        k0 = tilingData.k0;
        n0 = tilingData.n0;
        m_loop = tilingData.mLoop;
        k_loop = tilingData.kLoop;
        n_loop = tilingData.nLoop;
        core_loop = tilingData.coreLoop;
        swizzle_cnt = tilingData.swizzlCount;
        swizzlDirect = tilingData.swizzlDirect;
        en_shuffle_k = tilingData.enShuffleK;
        batchSizeB = 1;

        block_size = BLOCK_SIZE_32;
        cube_matrix_size = CUBE_MATRIX_SIZE_512;
        uint32_t a_l1_size = RoundUp512((m0 * k0));
        L1_PINGPONG_BUFFER_LEN = a_l1_size + RoundUp512((n0 * k0));
        L0AB_PINGPONG_BUFFER_LEN = L0AB_PINGPONG_BUFFER_LEN_INT8;
        l1_base_b = reinterpret_cast<__cbuf__ InDtype *>((uintptr_t)a_l1_size + l1_base_a);
        core_num = get_block_num();
        core_idx = get_block_idx();
        ping_flag = 1;
    }

    __aicore__ inline __attribute__((always_inline)) void GetBlockIdx(uint32_t index, uint32_t &m_idx, uint32_t &n_idx)
    {
        uint32_t in_batch_idx = index % (m_loop * n_loop);
        if (swizzlDirect == 0) { // Zn
            uint32_t tile_block_loop = (m_loop + swizzle_cnt - 1) / swizzle_cnt;
            uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt * n_loop);
            uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * n_loop);

            uint32_t n_row = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_row = m_loop - swizzle_cnt * tile_block_idx;
            }
            m_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_row;
            n_idx = in_tile_block_idx / n_row;
            if (tile_block_idx % CONST_2 != 0) {
                n_idx = n_loop - n_idx - 1;
            }
        } else { // Nz
            uint32_t tile_block_loop = (n_loop + swizzle_cnt - 1) / swizzle_cnt;
            uint32_t tile_block_idx = in_batch_idx / (swizzle_cnt * m_loop);
            uint32_t in_tile_block_idx = in_batch_idx % (swizzle_cnt * m_loop);

            uint32_t n_col = swizzle_cnt;
            if (tile_block_idx == tile_block_loop - 1) {
                n_col = n_loop - swizzle_cnt * tile_block_idx;
            }
            m_idx = in_tile_block_idx / n_col;
            n_idx = tile_block_idx * swizzle_cnt + in_tile_block_idx % n_col;
            if (tile_block_idx % CONST_2 != 0) {
                m_idx = m_loop - m_idx - 1;
            }
        }
    }

    __aicore__ inline void run()
    {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        uint32_t m_idx = 0;
        uint32_t n_idx = 0;
        for (uint32_t loop_idx = core_idx; loop_idx < core_loop; loop_idx += core_num) {
            GetBlockIdx(loop_idx, m_idx, n_idx);
            uint64_t batch_idx = loop_idx / n_loop / m_loop;
            uint64_t batchBIdx = batch_idx % batchSizeB;
            uint64_t offset_a;
            uint64_t offset_b;
            uint64_t offset_bias;
            uint64_t offset_scalar;
            uint64_t offset_a_next;
            uint64_t offset_b_next;
            uint64_t offset_c = batch_idx * m * n + m_idx * m0 * n + n_idx * n0;
            uint32_t m_actual = (m_idx == (m_loop - 1)) ? (m - m_idx * m0) : m0;
            uint32_t n_actual = (n_idx == (n_loop - 1)) ? (n - n_idx * n0) : n0;
            uint32_t m_round = 0;
            uint32_t n_round = 0;
            uint64_t shuffle_k = en_shuffle_k ? core_idx % k_loop : 0;
            if (TA) {
                m_round = RoundUp32(m_actual);
            } else {
                m_round = RoundUp16(m_actual);
            }
            if (TB) {
                n_round = RoundUp16(n_actual);
            } else {
                n_round = RoundUp32(n_actual);
            }

            uint32_t mn_max = m_round > n_round ? m_round : n_round;
            uint32_t k_part_len = 0;
            k_part_len = L0AB_PINGPONG_BUFFER_LEN_INT8 / mn_max / BLOCK_SIZE_32 * BLOCK_SIZE_32;

            if (TA) {
                offset_a = batch_idx * m * k + m_idx * m0 + shuffle_k * k0;
            } else {
                offset_a = batch_idx * m * k + m_idx * m0 * k + shuffle_k * k0;
            }
            if (TB) {
                offset_b = batchBIdx * k * n + n_idx * n0 * k + shuffle_k * k0;
            } else {
                offset_b = batchBIdx * k * n + n_idx * n0 + shuffle_k * k0 * n;
            }
            offset_bias = batchBIdx * n + n_idx * n0;
            offset_scalar = batchBIdx * n + n_idx * n0;

            uint32_t k_actual = (shuffle_k == k_loop - 1) ? k - shuffle_k * k0 : k0;
            uint32_t k_round = (k_actual + block_size - 1) / block_size * block_size; // int8 ：32 fp16 ：16

            auto l1_buf_a = ping_flag ? l1_base_a : l1_base_a + L1_PINGPONG_BUFFER_LEN;
            auto l1_buf_b = ping_flag ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
            auto l0a_buf = ping_flag ? l0a_base : l0a_base + L0AB_PINGPONG_BUFFER_LEN;
            auto l0b_buf = ping_flag ? l0b_base : l0b_base + L0AB_PINGPONG_BUFFER_LEN;
            auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;
            if (HAVE_BIAS) {
                wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
                copy_gm_to_cbuf(bias_l1,  // dst
                    gmBias + offset_bias, // src
                    0,                    // sid
                    1,                    // nBurst
                    n_round / CONST_8,    // lenBurst
                    0,                    // srcGap
                    0,                    // dstGap
                    PAD_NONE);            // padMode
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
            }

            wait_flag(PIPE_MTE1, PIPE_MTE2, event_id);
            // *** load matrix A to L1
            if ((m == 1) || (m_actual == 1 && !TA)) {
                copy_gm_to_cbuf(l1_buf_a, // dst
                    gmA + offset_a,       // src
                    0,                    // sid
                    1,                    // nBurst
                    k_round / block_size, // lenBurst
                    0,                    // srcGap
                    0,                    // dstGap
                    PAD_NONE);            // padMode
            } else {
                if (TA) {
                    CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_a, gmA + offset_a, k, m,
                        k_actual, m_actual, k_round);
                } else {
                    CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_a, gmA + offset_a, m, k,
                        m_actual, k_actual, m_round);
                }
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, event_id);

            // *** load matrix B to L1
            wait_flag(PIPE_MTE1, PIPE_MTE2, event_id + CONST_2);
            if (TB) {
                CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_b, gmB + offset_b, n, k, n_actual,
                    k_actual, n_round);
            } else {
                CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_b, gmB + offset_b, k, n, k_actual,
                    n_actual, k_round);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, event_id + CONST_2);

            wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
            copy_gm_to_cbuf(scale_l1,                                     // dst
                gmDescale + offset_scalar,                                // src
                0,                                                        // sid
                1,                                                        // nBurst
                (n_actual * CONST_4 * CONST_2 + CONST_32 - 1) / CONST_32, // lenBurst
                0, 0, PAD_NONE);
            set_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_FIX, EVENT_ID0);
            copy_cbuf_to_fbuf(((__fbuf__ ScaleDtype *)scale_FB), ((__cbuf__ ScaleDtype *)scale_l1), 1,
                (n_actual * CONST_4 * CONST_2 + CONST_128 - 1) / CONST_128, 0, 0);
            // when move scalar form L1 to fifpipe end, can move A/B from gm to L1
            set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);

            for (uint64_t k_idx = 0; k_idx < k_loop; k_idx++) {
                shuffle_k = en_shuffle_k ? (k_idx + core_idx) % k_loop : k_idx;
                uint32_t k_actual = (shuffle_k == (k_loop - 1)) ? (k - shuffle_k * k0) : k0;
                uint32_t k_round = (k_actual + block_size - 1) / block_size * block_size;
                uint32_t k_part_loop = (k_actual + k_part_len - 1) / k_part_len;

                __cbuf__ InDtype *l1_buf_a = ping_flag ? l1_base_a : l1_base_a + L1_PINGPONG_BUFFER_LEN;
                __cbuf__ InDtype *l1_buf_b = ping_flag ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
                auto event_id = ping_flag ? EVENT_ID0 : EVENT_ID1;

                if (k_idx < k_loop - 1) {
                    uint64_t shuffle_k_next = en_shuffle_k ? (core_idx + k_idx + 1) % k_loop : k_idx + 1;
                    if (TA) {
                        offset_a_next = batch_idx * m * k + m_idx * m0 + shuffle_k_next * k0 * m;
                    } else {
                        offset_a_next = batch_idx * m * k + m_idx * m0 * k + shuffle_k_next * k0;
                    }

                    if (TB) {
                        offset_b_next = batchBIdx * k * n + n_idx * n0 * k + shuffle_k_next * k0;
                    } else {
                        offset_b_next = batchBIdx * k * n + n_idx * n0 + shuffle_k_next * k0 * n;
                    }

                    uint32_t k_actual_next = (shuffle_k_next == (k_loop - 1)) ? (k - shuffle_k_next * k0) : k0;
                    uint32_t k_round_next = (k_actual_next + block_size - 1) / block_size * block_size;

                    __cbuf__ InDtype *l1_buf_a_next = (1 - ping_flag) ? l1_base_a : l1_base_a + L1_PINGPONG_BUFFER_LEN;
                    __cbuf__ InDtype *l1_buf_b_next = (1 - ping_flag) ? l1_base_b : l1_base_b + L1_PINGPONG_BUFFER_LEN;
                    auto event_id_next = (1 - ping_flag) ? EVENT_ID0 : EVENT_ID1;

                    wait_flag(PIPE_MTE1, PIPE_MTE2, event_id_next);
                    // *** load matrix A to L1
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        copy_gm_to_cbuf(l1_buf_a_next, gmA + offset_a_next,
                            0,                         // sid
                            1,                         // nBurst
                            k_round_next / block_size, // lenBurst
                            0,                         // srcGap
                            0,                         // dstGap
                            PAD_NONE);                 // padMode
                    } else {
                        if (TA) {
                            CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_a_next,
                                gmA + offset_a_next, k, m, k_actual_next, m_actual, k_round_next);
                        } else {
                            CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_a_next,
                                gmA + offset_a_next, m, k, m_actual, k_actual_next, m_round);
                        }
                    }
                    set_flag(PIPE_MTE2, PIPE_MTE1, event_id_next);

                    // *** load matrix B to L1
                    wait_flag(PIPE_MTE1, PIPE_MTE2, event_id_next + CONST_2);
                    if (TB) {
                        CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_b_next, gmB + offset_b_next,
                            n, k, n_actual, k_actual_next, n_round);
                    } else {
                        CopyGmToL1<InDtype, DataFormatEnum::ND, DataFormatEnum::NZ>(l1_buf_b_next, gmB + offset_b_next,
                            k, n, k_actual_next, n_actual, k_round_next);
                    }
                    set_flag(PIPE_MTE2, PIPE_MTE1, event_id_next + CONST_2);
                }

                for (int k_part_idx = 0; k_part_idx < k_part_loop; k_part_idx++) {
                    uint32_t k0_round = (k_part_idx < k_part_loop - 1) ? k_part_len : k_round - k_part_idx * k_part_len;
                    uint32_t k0_actual =
                        (k_part_idx < k_part_loop - 1) ? k_part_len : k_actual - k_part_idx * k_part_len;

                    auto mte1_mad_ping_flag = 1 - k_part_idx % CONST_2;
                    auto mte1_mad_event_id = mte1_mad_ping_flag ? EVENT_ID0 : EVENT_ID1;
                    auto l0a_buf = l0a_base + (k_part_idx % CONST_2) * L0AB_PINGPONG_BUFFER_LEN;
                    auto l0b_buf = l0b_base + (k_part_idx % CONST_2) * L0AB_PINGPONG_BUFFER_LEN;

                    // *** load matrix A from L1 to L0A
                    if (k_part_idx == 0) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, event_id);
                    }
                    wait_flag(PIPE_M, PIPE_MTE1, mte1_mad_event_id);
                    if ((m == 1) || (m_actual == 1 && !TA)) {
                        load_cbuf_to_ca(l0a_buf, l1_buf_a + k_part_idx * k_part_len,
                            0,                                                    // baseIdx
                            (k0_round + cube_matrix_size - 1) / cube_matrix_size, // repeat
                            1,                                                    // srcStride
                            0,                                                    // dstStride
                            0,                                                    // sid
                            false,                                                // transpose
                            inc);                                                 // addr_cal_mode_t
                    } else {
                        if (TA) {
                            for (uint64_t i = 0; i < m_round / BLOCK_SIZE_32; i++) {
                                load_cbuf_to_ca_transpose(l0a_buf + i * k0_round * BLOCK_SIZE_32,
                                    l1_buf_a + k_part_idx * k_part_len * BLOCK_SIZE_32 + i * k_round * BLOCK_SIZE_32,
                                    0,                             // baseIdx
                                    k0_round / BLOCK_SIZE_32,      // repeat
                                    1,                             // srcStride
                                    0,                             // dstStride
                                    0,                             // addrmode
                                    k0_round / BLOCK_SIZE_32 - 1); // dstFracStride
                            }
                        } else {
                            for (uint64_t i = 0; i < m_round / BLOCK_SIZE_16; i++) {
                                load_cbuf_to_ca(l0a_buf + i * k0_round * BLOCK_SIZE_16,
                                    l1_buf_a + k_part_idx * k_part_len * m_round + i * cube_matrix_size,
                                    0,                       // baseIdx
                                    k0_round / block_size,   // repeat
                                    m_round / BLOCK_SIZE_16, // srcStride
                                    0,                       // dstStride
                                    0,                       // sid
                                    false,                   // transpose
                                    inc);                    // addr_cal_mode_t
                            }
                        }
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, event_id);
                    }

                    // *** load matrix B from L1 to L0B
                    if (k_part_idx == 0) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, event_id + CONST_2);
                    }
                    if (TB) {
                        load_cbuf_to_cb(l0b_buf, l1_buf_b + k_part_idx * k_part_len * n_round,
                            0,                                     // baseIdx
                            k0_round * n_round / cube_matrix_size, // repeat
                            1,                                     // srcStride
                            0,                                     // dstStride
                            0,                                     // sid
                            false,                                 // transpose
                            inc);                                  // addr_cal_mode_t
                    } else {
                        for (uint64_t i = 0; i < k0_round / BLOCK_SIZE_32; i++) {
                            load_cbuf_to_cb_transpose(l0b_buf + i * n_round * BLOCK_SIZE_32,
                                l1_buf_b + (k_part_idx * k_part_len + i * BLOCK_SIZE_32) * BLOCK_SIZE_32,
                                0,                       // baseIdx
                                n_round / BLOCK_SIZE_32, // repeat
                                k_round / BLOCK_SIZE_32, // srcStride
                                1,                       // dstStride
                                0,                       // addrmode
                                0);                      // dstFracStride
                        }
                    }
                    if (k_part_idx == k_part_loop - 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, event_id + CONST_2);
                    }

                    set_flag(PIPE_MTE1, PIPE_M, mte1_mad_event_id);
                    wait_flag(PIPE_MTE1, PIPE_M, mte1_mad_event_id);

                    bool init_c = (k_idx == 0 && k_part_idx == 0);
                    bool sp_flag = (m != 1 && m_actual == 1 && TA);
                    if (init_c) {
                        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                    }
                    if (init_c) {
                        if (HAVE_BIAS) {
                            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID6);
                            copy_cbuf_to_bt(bias_bt,           // dst
                                bias_l1,                       // src
                                (uint16_t)0ULL,                // convControl
                                1,                             // nBurst
                                CeilDiv64(n_actual * CONST_4), // lenBurst
                                0,                             // srcGap
                                0);                            // dstGap

                            set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7); // bias ready, mte2 can begin move A/B or scale
                            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);    // bias ready, mmad can begin
                            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID7);   // wait move bias fron L1 to BT
                            mad(l0c_buf, l0a_buf, l0b_buf, ((uint64_t)bias_bt),
                                sp_flag ? m_round : m_actual, // m
                                k0_actual,                    // k
                                n_actual,                     // n
                                0,                            // unitFlag
                                0,                            // kDirectionAlign
                                1,                            // cmatrixSource add C from BT
                                0);                           // cmatrixInitVal
                        } else {
                            mad(l0c_buf, l0a_buf, l0b_buf,
                                sp_flag ? m_round : m_actual, // m
                                k0_actual,                    // k
                                n_actual,                     // n
                                0,                            // unitFlag
                                0,                            // kDirectionAlign
                                0,                            // cmatrixSource
                                1);                           // cmatrixInitVal
                        }
                    } else {
                        mad(l0c_buf, l0a_buf, l0b_buf,
                            sp_flag ? m_round : m_actual, // m
                            k0_actual,                    // k
                            n_actual,                     // n
                            0,                            // unitFlag
                            0,                            // kDirectionAlign
                            0,                            // cmatrixSource
                            0);                           // cmatrixInitVal
                    }
                    pipe_barrier(PIPE_M);
                    set_flag(PIPE_M, PIPE_MTE1, mte1_mad_event_id);
                }

                ping_flag = 1 - ping_flag;
            }
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            pipe_barrier(PIPE_FIX);
            // copy from L0C to gm
            set_fpc(((int64_t)0 |
                ((int64_t)(((((uint64_t)((__fbuf__ uint64_t *)scale_FB)) & (uint64_t)65535ULL) >> (uint64_t)7ULL) <<
                (uint64_t)8ULL))));
            copy_matrix_cc_to_gm(gmC + (int64_t)offset_c, // dst
                (__cc__ int32_t *)l0c_buf,                // src
                0,                                        // sid
                n_actual,                                 // NSize
                m_actual,                                 // MSize
                n,                                        // dstStride_dst_D
                RoundUp16(m_actual),                      // srcStride
                0,                                        // UnitFlagMode
                VDEQF16,                                  // QuantPRE VDEQF16,  NoQuant
                0,                                        // ReLUPRE
                false,                                    // channelSplit
                true);                                    // NZ2ND_EN
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID7);
        pipe_barrier(PIPE_ALL);
    }

private:
    __gm__ InDtype *__restrict__ gmA { nullptr };
    __gm__ InDtype *__restrict__ gmB { nullptr };
    __gm__ BiasDtype *__restrict__ gmBias { nullptr };
    __gm__ ScaleDtype *__restrict__ gmDescale { nullptr };
    __gm__ OutDtype *__restrict__ gmC { nullptr };

    __cbuf__ InDtype *l1_base_a = reinterpret_cast<__cbuf__ InDtype *>((uintptr_t)(SCALE_L1_LEN + BIAS_L1_LEN));
    __cbuf__ InDtype *l1_base_b = reinterpret_cast<__cbuf__ InDtype *>((uintptr_t)(L1_PINGPONG_BUFFER_LEN_FP16));

    __ca__ InDtype *l0a_base = reinterpret_cast<__ca__ InDtype *>((uintptr_t)0);
    __cb__ InDtype *l0b_base = reinterpret_cast<__cb__ InDtype *>((uintptr_t)0);

    __cc__ BiasDtype *l0c_buf = reinterpret_cast<__cc__ BiasDtype *>((uintptr_t)0);

    __cbuf__ ScaleDtype *scale_l1 = reinterpret_cast<__cbuf__ ScaleDtype *>((uintptr_t)BIAS_L1_LEN);
    __fbuf__ ScaleDtype *scale_FB = (__fbuf__ ScaleDtype *)get_imm(0);

    __cbuf__ BiasDtype *bias_l1 = reinterpret_cast<__cbuf__ BiasDtype *>((uintptr_t)0);
    uint64_t bias_bt { 0 };

    uint32_t core_num { 0 };

    uint32_t batch_size { 0 };
    uint32_t m { 0 };
    uint32_t k { 0 };
    uint32_t n { 0 };

    uint32_t m0 { 0 };
    uint32_t k0 { 0 };
    uint32_t n0 { 0 };

    uint32_t m_loop { 0 };
    uint32_t n_loop { 0 };
    uint32_t k_loop { 0 };
    uint32_t core_loop { 0 };
    uint32_t core_idx { 0 };
    uint32_t ping_flag { 0 };
    uint32_t block_size { 0 };
    uint32_t cube_matrix_size { 0 };
    uint32_t swizzle_cnt { 1 };
    uint32_t en_shuffle_k { 0 };
    uint32_t swizzlDirect { 0 };

    int32_t batchSizeB { 0 };

    uint64_t L1_PINGPONG_BUFFER_LEN { 0 };
    uint32_t L0AB_PINGPONG_BUFFER_LEN { 0 };
};
}


namespace {
extern "C" __global__ __aicore__ void pp_matmul_i8(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBias, GM_ADDR gmDescale,
    GM_ADDR gmC, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    // int8 + int32_bias + u64_descale
    AscendC::PpMatmulI8<false, false, false, true, true, int8_t, half, int32_t, uint64_t> matmul_int8;
    AscendC::PpMatmulI8<true, false, false, true, true, int8_t, half, int32_t, uint64_t> matmul_int8_ta;
    AscendC::PpMatmulI8<false, true, false, true, true, int8_t, half, int32_t, uint64_t> matmul_int8_tb;
    AscendC::PpMatmulI8<true, true, false, true, true, int8_t, half, int32_t, uint64_t> matmul_int8_tatb;

    // int8  + u64_descale
    AscendC::PpMatmulI8<false, false, false, false, true, int8_t, half, int32_t, uint64_t> matmul_int8_nobias;
    AscendC::PpMatmulI8<true, false, false, false, true, int8_t, half, int32_t, uint64_t> matmul_int8_ta_nobias;
    AscendC::PpMatmulI8<false, true, false, false, true, int8_t, half, int32_t, uint64_t> matmul_int8_tb_nobias;
    AscendC::PpMatmulI8<true, true, false, false, true, int8_t, half, int32_t, uint64_t> matmul_int8_tatb_nobias;
#ifdef __DAV_C220_CUBE__
    set_padding(uint16_t(0));
    set_atomic_none();
    uint64_t config = 0x1;
    set_nd_para(config);

    switch (tilingData.tilingKey) {
        case 0b000110:
        case 0b100110:
            matmul_int8.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8.run();

            break;
        case 0b010110:
        case 0b110110:
            matmul_int8_ta.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_ta.run();

            break;
        case 0b001110:
        case 0b101110:
            matmul_int8_tb.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_tb.run();

            break;
        case 0b011110:
        case 0b111110:
            matmul_int8_tatb.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_tatb.run();

            break;
        case 0b000100:
        case 0b100100:
            matmul_int8_nobias.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_nobias.run();

            break;
        case 0b010100:
        case 0b110100:
            matmul_int8_ta_nobias.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_ta_nobias.run();

            break;
        case 0b001100:
        case 0b101100:
            matmul_int8_tb_nobias.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_tb_nobias.run();

            break;
        case 0b011100:
        case 0b111100:
            matmul_int8_tatb_nobias.SetArgs(gmA, gmB, gmC, gmBias, gmDescale, tilingData);
            matmul_int8_tatb_nobias.run();

            break;
        default:
            break;
    }
#endif
}
}
