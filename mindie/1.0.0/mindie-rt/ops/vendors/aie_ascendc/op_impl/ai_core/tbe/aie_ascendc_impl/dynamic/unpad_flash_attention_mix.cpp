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
constexpr int32_t TILING_PARA_SIZE = 24;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;    // 128 * 128
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t CUBE_MATRIX_SIZE = 256;        // 16 * 16
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072;
constexpr int64_t L0AB_UINT8_BLOCK_SIZE = 32768; // 128 * 128 * 2B
constexpr int64_t TMP_SIZE = 65536;              // 256 * 256
constexpr int32_t UB_HALF_BUF_SIZE = 8192;       // 64 * 128
constexpr int32_t VECTOR_SIZE = 128;
constexpr int64_t UB_UINT8_BLOCK_SIZE = 16384;   // 64 * 128 * 2B
constexpr int64_t UB_UINT8_LINE_SIZE = 512;      // 64 * 4B, 申请两倍空间防踩踏
constexpr int64_t UB_FLOAT_LINE_SIZE = 128;      // 64, 申请两倍空间防踩踏
constexpr int64_t UB_HALF_LINE_SIZE = 256;       // UB_FLOAT_LINE_SIZE * 2
constexpr uint32_t Q_SEQLEN_MIN = 1024;
constexpr uint32_t Q_SEQLEN_MAX = 4096;
constexpr uint32_t KV_SEQLEN_MIN = 1024;
constexpr uint32_t KV_SEQLEN_MAX = 4096;
constexpr uint32_t HEADS_MIN = 5;
constexpr uint32_t HEADS_MAX = 10;
constexpr uint32_t BATCH_MIN = 1;
constexpr uint32_t BATCH_MAX = 2;
constexpr uint32_t EMBED_1 = 40;
constexpr uint32_t EMBED_2 = 64;
}

namespace AscendC {
class UnpadFlashAttentionMix {
public:
    __aicore__ inline UnpadFlashAttentionMix() {}

    __aicore__ inline void unpad_flashattention_mix_aic(
        __gm__ uint8_t *__restrict__ q_gm_aic,
        __gm__ uint8_t *__restrict__ k_gm_aic,
        __gm__ uint8_t *__restrict__ v_gm_aic,
        __gm__ uint8_t *__restrict__ mask_gm_aic,
        __gm__ uint8_t *__restrict__ o_gm_aic,
        __gm__ uint8_t *__restrict__ s_gm_aic,
        __gm__ uint8_t *__restrict__ p_gm_aic,
        __gm__ uint8_t *__restrict__ o_tmp_gm_aic,
        uint32_t * tiling_para_gm)
    {
        set_padding(0);
        set_atomic_none();
        uint64_t config = 0x1;
        set_nd_para(config);
        set_mask_norm();

        __cbuf__ uint8_t *l1q_buf_addr = (__cbuf__ uint8_t *)get_imm(0);                         // 2 block
        __cbuf__ uint8_t *l1k_buf_addr = (__cbuf__ uint8_t *)get_imm(2 * L0AB_UINT8_BLOCK_SIZE); // 2 block
        __cbuf__ uint8_t *l1p_buf_addr = (__cbuf__ uint8_t *)get_imm(4 * L0AB_UINT8_BLOCK_SIZE); // 2 block
        __cbuf__ uint8_t *l1v_buf_addr = (__cbuf__ uint8_t *)get_imm(6 * L0AB_UINT8_BLOCK_SIZE); // 2 block
        __ca__ uint8_t *l0a_buf = (__ca__ uint8_t *)get_imm(0);
        __cb__ uint8_t *l0b_buf = (__cb__ uint8_t *)get_imm(0);
        __cc__ uint8_t *l0c_buf = (__cc__ uint8_t *)get_imm(0);

        uint32_t batchSize = (uint32_t)(*((int32_t *)tiling_para_gm + 20));
        int32_t currHeads = (int32_t)(*((int32_t *)tiling_para_gm + 2));

        // 根据block_idx计算当前逻辑核处理的batch_idx和q_blk_idx
        int32_t curBatch = 0;
        int32_t curTotalQblk = 0;
        int32_t curQblkId = 0;
        ComputeCurBatch(curBatch, curTotalQblk, curQblkId, batchSize, currHeads, tiling_para_gm, TILING_PARA_SIZE);

        int32_t q_seqlen;
        int32_t kv_seqlen;
        int32_t heads;
        int32_t embd;
        int32_t pp_m_scalar;
        int32_t pp_n_scalar;
        uint32_t addr_q_high32;
        uint32_t addr_q_loww32;
        int64_t addr_q_scalar;
        uint32_t addr_k_high32;
        uint32_t addr_k_loww32;
        int64_t addr_k_scalar;
        uint32_t addr_v_high32;
        uint32_t addr_v_loww32;
        int64_t addr_v_scalar;
        uint32_t addr_s_high32;
        uint32_t addr_s_loww32;
        int64_t addr_s_scalar;
        uint32_t addr_p_high32;
        uint32_t addr_p_loww32;
        int64_t addr_p_scalar;
        uint32_t addr_o_high32;
        uint32_t addr_o_loww32;
        int64_t addr_o_scalar;

        // get tiling args
        q_seqlen = (int32_t)(*((int32_t *)tiling_para_gm + TILING_PARA_SIZE * curBatch));
        kv_seqlen = (int32_t)(*((int32_t *)tiling_para_gm + 1 + TILING_PARA_SIZE * curBatch));
        heads = (int32_t)(*((int32_t *)tiling_para_gm + 2 + TILING_PARA_SIZE * curBatch));
        embd = (int32_t)(*((int32_t *)tiling_para_gm + 3 + TILING_PARA_SIZE * curBatch));
        int32_t maxSeqLen = (int32_t)(*((int32_t *)tiling_para_gm + 4 + TILING_PARA_SIZE * curBatch));
        pp_m_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 6 + TILING_PARA_SIZE * curBatch));
        pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 7 + TILING_PARA_SIZE * curBatch));

        addr_q_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 8 + TILING_PARA_SIZE * curBatch));
        addr_q_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 9 + TILING_PARA_SIZE * curBatch));
        addr_q_scalar = (int64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);

        addr_k_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 10 + TILING_PARA_SIZE * curBatch));
        addr_k_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 11 + TILING_PARA_SIZE * curBatch));
        addr_k_scalar = (int64_t)(((uint64_t)addr_k_high32) << 32 | addr_k_loww32);

        addr_v_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 12 + TILING_PARA_SIZE * curBatch));
        addr_v_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + TILING_PARA_SIZE * curBatch));
        addr_v_scalar = (int64_t)(((uint64_t)addr_v_high32) << 32 | addr_v_loww32);

        addr_s_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 14 + TILING_PARA_SIZE * curBatch));
        addr_s_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 15 + TILING_PARA_SIZE * curBatch));
        addr_s_scalar = (int64_t)(((uint64_t)addr_s_high32) << 32 | addr_s_loww32);

        addr_p_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 16 + TILING_PARA_SIZE * curBatch));
        addr_p_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 17 + TILING_PARA_SIZE * curBatch));
        addr_p_scalar = (int64_t)(((uint64_t)addr_p_high32) << 32 | addr_p_loww32);

        addr_o_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 18 + TILING_PARA_SIZE * curBatch));
        addr_o_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 19 + TILING_PARA_SIZE * curBatch));
        addr_o_scalar = (int64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

        int32_t curQblk = (int32_t)(*((int32_t *)tiling_para_gm + 21 + TILING_PARA_SIZE * curBatch));

        int32_t m_loop = 0;
        if (pp_m_scalar != 0) {
            m_loop = (q_seqlen + pp_m_scalar - 1) / pp_m_scalar;
        }

        int32_t n_loop = 0;
        if (pp_n_scalar != 0) {
            n_loop = (kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;
        }

        int32_t start = curQblkId * n_loop;
        int32_t end = start + n_loop;

        int32_t stride_qkvo = heads * embd;
        int32_t stride_sp = kv_seqlen;

        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

        for (int32_t loop_idx = start; loop_idx < end; loop_idx += 2) {
            int32_t head_idx0 = 0;
            int32_t m_idx0 = 0;
            int32_t n_idx0 = 0;
            if (m_loop != 0 && n_loop != 0) {
                head_idx0 = loop_idx / (m_loop * n_loop);
                m_idx0 = loop_idx % (m_loop * n_loop) / n_loop;
                n_idx0 = loop_idx % (m_loop * n_loop) % n_loop;
            }

            int64_t q_offset0 = addr_q_scalar + head_idx0 * embd + m_idx0 * pp_m_scalar * stride_qkvo;
            int64_t k_offset0 = addr_k_scalar + head_idx0 * embd + n_idx0 * pp_n_scalar * stride_qkvo;
            int64_t v_offset0 = addr_v_scalar + head_idx0 * embd + n_idx0 * pp_n_scalar * stride_qkvo;
            int64_t s_offset0 = addr_s_scalar + head_idx0 * q_seqlen * stride_sp
                + m_idx0 * pp_m_scalar * stride_sp + n_idx0 * pp_n_scalar;
            int64_t p_offset0 = addr_p_scalar + head_idx0 * q_seqlen * stride_sp
                + m_idx0 * pp_m_scalar * stride_sp + n_idx0 * pp_n_scalar;
            int64_t o_offset0 = addr_o_scalar + head_idx0 * embd + m_idx0 * pp_m_scalar * stride_qkvo;
            int32_t __m0 = (m_idx0 == (m_loop - 1)) ? (q_seqlen - m_idx0 * pp_m_scalar) : pp_m_scalar;
            int32_t __n0 = (n_idx0 == (n_loop - 1)) ? (kv_seqlen - n_idx0 * pp_n_scalar) : pp_n_scalar;
            int32_t __k0 = embd;
            int32_t round_m0 = (__m0 + 15) / 16 * 16;
            int32_t round_n0 = (__n0 + 15) / 16 * 16;
            int32_t round_k0 = (__k0 + 15) / 16 * 16;

            int32_t head_idx1 = 0;
            int32_t m_idx1 = 0;
            int32_t n_idx1 = 0;
            if (m_loop != 0 && n_loop != 0) {
                head_idx1 = (loop_idx + 1) / (m_loop * n_loop);
                m_idx1 = (loop_idx + 1) % (m_loop * n_loop) / n_loop;
                n_idx1 = (loop_idx + 1) % (m_loop * n_loop) % n_loop;
            }

            int64_t q_offset1 = addr_q_scalar + head_idx1 * embd + m_idx1 * pp_m_scalar * stride_qkvo;
            int64_t k_offset1 = addr_k_scalar + head_idx1 * embd + n_idx1 * pp_n_scalar * stride_qkvo;
            int64_t v_offset1 = addr_v_scalar + head_idx1 * embd + n_idx1 * pp_n_scalar * stride_qkvo;
            int64_t s_offset1 = addr_s_scalar + head_idx1 * q_seqlen * stride_sp
                + m_idx1 * pp_m_scalar * stride_sp + n_idx1 * pp_n_scalar;
            int64_t p_offset1 = addr_p_scalar + head_idx1 * q_seqlen * stride_sp
                + m_idx1 * pp_m_scalar * stride_sp + n_idx1 * pp_n_scalar;
            int64_t o_offset1 = addr_o_scalar + head_idx1 * embd + m_idx1 * pp_m_scalar * stride_qkvo;
            int32_t __m1 = (m_idx1 == (m_loop - 1)) ? (q_seqlen - m_idx1 * pp_m_scalar) : pp_m_scalar;
            int32_t __n1 = (n_idx1 == (n_loop - 1)) ? (kv_seqlen - n_idx1 * pp_n_scalar) : pp_n_scalar;
            int32_t __k1 = embd;
            int32_t round_m1 = (__m1 + 15) / 16 * 16;
            int32_t round_n1 = (__n1 + 15) / 16 * 16;
            int32_t round_k1 = (__k1 + 15) / 16 * 16;

            /* ************ stage 0-0  ************* */
            __cbuf__ uint8_t *l1q_buf0 = l1q_buf_addr;
            __cbuf__ uint8_t *l1k_buf0 = l1k_buf_addr;
            // *** Prepare Q to L1
            if (__m0 == 1) {
                copy_gm_to_cbuf(
                    (__cbuf__ half *)l1q_buf0,
                    (__gm__ half *)q_gm_aic + (int64_t)q_offset0,
                    0,             // sid
                    1,             // nBurst
                    round_k0 / 16, // lenBurst
                    0,             // srcGap
                    0,             // dstGap
                    PAD_NONE       // padMode
                );
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    (__cbuf__ half *)l1q_buf0,
                    (__gm__ half *)q_gm_aic + (int64_t)q_offset0,
                    0,           // sid
                    1,           // ndNum
                    __m0,        // nValue
                    __k0,        // dValue
                    0,           // srcNdMatrixStride, unused
                    stride_qkvo, // srcDValue
                    round_m0,    // dstNzC0Stride
                    1,           // dstNzNStride
                    0            // dstNzMatrixStride, unused
                );
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            if (__m0 == 1) {
                load_cbuf_to_ca(
                    (__ca__ half *)l0a_buf,
                    (__cbuf__ half *)l1q_buf0,
                    0,                                                    // baseIdx
                    (round_k0 + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, // repeat
                    1,                                                    // srcStride
                    0,                                                    // dstStride
                    0,                                                    // sid
                    false,                                                // transpose
                    inc                                                   // addr_cal_mode_t
                );
            } else if (round_m0 <= round_k0) {
                for (int32_t l0a_load_idx = 0; l0a_load_idx < round_m0 / BLOCK_SIZE; ++l0a_load_idx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + l0a_load_idx * round_k0 * BLOCK_SIZE,
                        (__cbuf__ half *)l1q_buf0 + l0a_load_idx * CUBE_MATRIX_SIZE,
                        0,                     // baseIdx
                        round_k0 / BLOCK_SIZE, // repeat
                        round_m0 / BLOCK_SIZE, // srcStride
                        0,                     // dstStride
                        0,                     // sid
                        false,                 // transpose
                        inc                    // addr_cal_mode_t
                    );
                }
            } else {
                for (int32_t l0a_load_idx = 0; l0a_load_idx < round_k0 / BLOCK_SIZE; ++l0a_load_idx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + l0a_load_idx * CUBE_MATRIX_SIZE,
                        (__cbuf__ half *)l1q_buf0 + l0a_load_idx * round_m0 * BLOCK_SIZE,
                        0,                         // baseIdx
                        round_m0 / BLOCK_SIZE,     // repeat
                        1,                         // srcStride
                        round_k0 / BLOCK_SIZE - 1, // dstStride
                        0,                         // sid
                        false,                     // transpose
                        inc                        // addr_cal_mode_t
                    );
                }
            }
            // *** Prepare K to L1
            CopyGmToCbufMultiNd2NzB16((__cbuf__ half *)l1k_buf0, (__gm__ half *)k_gm_aic + (int64_t)k_offset0, 0, 1,
                __n0, __k0, 0, stride_qkvo, round_n0, 1, 0);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            load_cbuf_to_cb(
                (__cb__ half *)l0b_buf,
                (__cbuf__ half *)l1k_buf0,
                0,                                      // baseIdx
                round_k0 * round_n0 / CUBE_MATRIX_SIZE, // repeat
                1,                                      // srcStride
                0,                                      // dstStride
                0,                                      // sid
                false,                                  // transpose
                inc                                     // addr_cal_mode_t
            );
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            mad((__cc__ float *)l0c_buf,
                (__ca__ half *)l0a_buf,
                (__cb__ half *)l0b_buf,
                __m0, // m
                __k0, // k
                __n0, // n
                0,    // unitFlag
                0,    // kDirectionAlign
                0,    // cmatrixSource
                1     // cmatrixInitVal
            );
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            // copy S to gm
            copy_matrix_cc_to_gm(
                (__gm__ half *)s_gm_aic + (int32_t)block_idx * TMP_SIZE,
                (__cc__ float *)l0c_buf,
                0,         // sid
                round_n0,      // NSize
                __m0,      // MSize
                round_n0, // dstStride_dst_D
                round_m0,  // srcStride
                0,         // UnitFlagMode
                F322F16,   // QuantPRE
                0,         // ReLUPRE
                false,     // channelSplit
                true       // NZ2ND_EN
            );
            ffts_cross_core_sync(PIPE_FIX, 33); // mode=2 id=0 10 0001

            if ((loop_idx + 1) < end) {
                /* ************ stage 1-0 ************** */
                __cbuf__ uint8_t *l1q_buf1 = l1q_buf_addr + 2 * L1_UINT8_BLOCK_SIZE;
                __cbuf__ uint8_t *l1k_buf1 = l1k_buf_addr + 2 * L1_UINT8_BLOCK_SIZE;
                // *** Prepare Q to L1
                if (__m1 == 1) {
                    copy_gm_to_cbuf(
                        (__cbuf__ half *)l1q_buf1,
                        (__gm__ half *)q_gm_aic + (int64_t)q_offset1,
                        0,             // sid
                        1,             // nBurst
                        round_k1 / 16, // lenBurst
                        0,             // srcGap
                        0,             // dstGap
                        PAD_NONE       // padMode
                    );
                } else {
                    copy_gm_to_cbuf_multi_nd2nz_b16(
                        (__cbuf__ half *)l1q_buf1,
                        (__gm__ half *)q_gm_aic + (int64_t)q_offset1,
                        0,           // sid
                        1,           // ndNum
                        __m1,        // nValue
                        __k1,        // dValue
                        0,           // srcNdMatrixStride, unused
                        stride_qkvo, // srcDValue
                        round_m1,    // dstNzC0Stride
                        1,           // dstNzNStride
                        0            // dstNzMatrixStride, unused
                    );
                }
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                if (__m1 == 1) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE,
                        (__cbuf__ half *)l1q_buf1,
                        0,                                                    // baseIdx
                        (round_k1 + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, // repeat
                        1,                                                    // srcStride
                        0,                                                    // dstStride
                        0,                                                    // sid
                        false,                                                // transpose
                        inc                                                   // addr_cal_mode_t
                    );
                } else if (round_m1 <= round_k1) {
                    for (int32_t l0a_load_idx = 0; l0a_load_idx < round_m1 / BLOCK_SIZE; ++l0a_load_idx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE + l0a_load_idx * round_k1 * BLOCK_SIZE,
                            (__cbuf__ half *)l1q_buf1 + l0a_load_idx * CUBE_MATRIX_SIZE,
                            0,                     // baseIdx
                            round_k1 / BLOCK_SIZE, // repeat
                            round_m1 / BLOCK_SIZE, // srcStride
                            0,                     // dstStride
                            0,                     // sid
                            false,                 // transpose
                            inc                    // addr_cal_mode_t
                        );
                    }
                } else {
                    for (int32_t l0a_load_idx = 0; l0a_load_idx < round_k1 / BLOCK_SIZE; ++l0a_load_idx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE + l0a_load_idx * CUBE_MATRIX_SIZE,
                            (__cbuf__ half *)l1q_buf1 + l0a_load_idx * round_m1 * BLOCK_SIZE,
                            0,                         // baseIdx
                            round_m1 / BLOCK_SIZE,     // repeat
                            1,                         // srcStride
                            round_k1 / BLOCK_SIZE - 1, // dstStride
                            0,                         // sid
                            false,                     // transpose
                            inc                        // addr_cal_mode_t
                        );
                    }
                }
                // *** Prepare K to L1
                CopyGmToCbufMultiNd2NzB16((__cbuf__ half *)l1k_buf1, (__gm__ half *)k_gm_aic + (int64_t)k_offset1,
                    0, 1, __n1, __k1, 0, stride_qkvo, round_n1, 1, 0);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                load_cbuf_to_cb(
                    (__cb__ half *)l0b_buf + L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1k_buf1,
                    0,                                      // baseIdx
                    round_k1 * round_n1 / CUBE_MATRIX_SIZE, // repeat
                    1,                                      // srcStride
                    0,                                      // dstStride
                    0,                                      // sid
                    false,                                  // transpose
                    inc                                     // addr_cal_mode_t
                );
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                mad((__cc__ float *)l0c_buf + L0AB_HALF_BUF_SIZE,
                    (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE,
                    (__cb__ half *)l0b_buf + L0AB_HALF_BUF_SIZE,
                    __m1, // m
                    __k1, // k
                    __n1, // n
                    0,    // unitFlag
                    0,    // kDirectionAlign
                    0,    // cmatrixSource
                    1     // cmatrixInitVal
                );
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                // copy S to gm
                copy_matrix_cc_to_gm(
                    (__gm__ half *)s_gm_aic + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2,
                    (__cc__ float *)l0c_buf + L0AB_HALF_BUF_SIZE,
                    0,         // sid
                    round_n1,      // NSize
                    __m1,      // MSize
                    round_n1, // dstStride_dst_D
                    round_m1,  // srcStride
                    0,         // UnitFlagMode
                    F322F16,   // QuantPRE
                    0,         // ReLUPRE
                    false,     // channelSplit
                    true       // NZ2ND_EN
                );
                ffts_cross_core_sync(PIPE_FIX, 289); // mode=2 id=1 1 0010 0001
            }

            /* ************ stage 0-1 *************** */
            wait_flag_dev(2);
            __cbuf__ uint8_t *l1p_buf0 = l1p_buf_addr;
            __cbuf__ uint8_t *l1v_buf0 = l1v_buf_addr;
            // *** Prepare P to L1
            if (__m0 == 1) {
                copy_gm_to_cbuf(
                    (__cbuf__ half *)l1p_buf0,
                    (__gm__ half *)p_gm_aic + (int32_t)block_idx * TMP_SIZE,
                    0,             // sid
                    1,             // nBurst
                    round_n0 / 16, // lenBurst
                    0,             // srcGap
                    0,             // dstGap
                    PAD_NONE       // padMode
                );
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    (__cbuf__ half *)l1p_buf0,
                    (__gm__ half *)p_gm_aic + (int32_t)block_idx * TMP_SIZE,
                    0,         // sid
                    1,         // ndNum
                    __m0,      // nValue
                    __n0,      // dValue
                    0,         // srcNdMatrixStride, unused
                    round_n0, // srcDValue
                    round_m0,  // dstNzC0Stride
                    1,         // dstNzNStride
                    0          // dstNzMatrixStride, unused
                );
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            if (__m0 == 1) {
                load_cbuf_to_ca(
                    (__ca__ half *)l0a_buf,
                    (__cbuf__ half *)l1p_buf0,
                    0,                                                    // baseIdx
                    (round_n0 + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, // repeat
                    1,                                                    // srcStride
                    0,                                                    // dstStride
                    0,                                                    // sid
                    false,                                                // transpose
                    inc                                                   // addr_cal_mode_t
                );
            } else if (round_m0 <= round_n0) {
                for (int32_t l0a_load_idx = 0; l0a_load_idx < round_m0 / BLOCK_SIZE; ++l0a_load_idx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + l0a_load_idx * round_n0 * BLOCK_SIZE,
                        (__cbuf__ half *)l1p_buf0 + l0a_load_idx * CUBE_MATRIX_SIZE,
                        0,                     // baseIdx
                        round_n0 / BLOCK_SIZE, // repeat
                        round_m0 / BLOCK_SIZE, // srcStride
                        0,                     // dstStride
                        0,                     // sid
                        false,                 // transpose
                        inc                    // addr_cal_mode_t
                    );
                }
            } else {
                for (int32_t l0a_load_idx = 0; l0a_load_idx < round_n0 / BLOCK_SIZE; ++l0a_load_idx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + l0a_load_idx * CUBE_MATRIX_SIZE,
                        (__cbuf__ half *)l1p_buf0 + l0a_load_idx * round_m0 * BLOCK_SIZE,
                        0,                         // baseIdx
                        round_m0 / BLOCK_SIZE,     // repeat
                        1,                         // srcStride
                        round_n0 / BLOCK_SIZE - 1, // dstStride
                        0,                         // sid
                        false,                     // transpose
                        inc                        // addr_cal_mode_t
                    );
                }
            }
            // *** Prepare V to L1
            CopyGmToCbufMultiNd2NzB16((__cbuf__ half *)l1v_buf0, (__gm__ half *)v_gm_aic + (int64_t)v_offset0,
                0, 1, __n0, __k0, 0, stride_qkvo, round_n0, 1, 0);
            set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
            if (round_n0 <= round_k0) {
                for (int32_t l0b_load_idx = 0; l0b_load_idx < round_n0 / BLOCK_SIZE; ++l0b_load_idx) {
                    load_cbuf_to_cb(
                        (__cb__ half *)l0b_buf + l0b_load_idx * round_k0 * BLOCK_SIZE,
                        (__cbuf__ half *)l1v_buf0 + l0b_load_idx * CUBE_MATRIX_SIZE,
                        0,                     // baseIdx
                        round_k0 / BLOCK_SIZE, // repeat
                        round_n0 / BLOCK_SIZE, // srcStride
                        0,                     // dstStride
                        0,                     // sid
                        true,                  // transpose
                        inc                    // addr_cal_mode_t
                    );
                }
            } else {
                for (int32_t l0b_load_idx = 0; l0b_load_idx < round_k0 / BLOCK_SIZE; ++l0b_load_idx) {
                    load_cbuf_to_cb(
                        (__cb__ half *)l0b_buf + l0b_load_idx * CUBE_MATRIX_SIZE,
                        (__cbuf__ half *)l1v_buf0 + l0b_load_idx * round_n0 * BLOCK_SIZE,
                        0,                         // baseIdx
                        round_n0 / BLOCK_SIZE,     // repeat
                        1,                         // srcStride
                        round_k0 / BLOCK_SIZE - 1, // dstStride
                        0,                         // sid
                        true,                      // transpose
                        inc                        // addr_cal_mode_t
                    );
                }
            }
            set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
            mad((__cc__ float *)l0c_buf,
                (__ca__ half *)l0a_buf,
                (__cb__ half *)l0b_buf,
                __m0, // m
                __n0, // k
                __k0, // n
                0,    // unitFlag
                0,    // kDirectionAlign
                0,    // cmatrixSource
                1     // cmatrixInitVal
            );
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            // copy O to gm
            copy_matrix_cc_to_gm(
                (__gm__ half *)o_tmp_gm_aic + (int32_t)block_idx * TMP_SIZE,
                (__cc__ float *)l0c_buf,
                0,            // sid
                round_k0,         // NSize
                __m0,         // MSize
                round_k0, // dstStride_dst_D
                round_m0,     // srcStride
                0,            // UnitFlagMode
                F322F16,      // QuantPRE
                0,            // ReLUPRE
                false,        // channelSplit
                true          // NZ2ND_EN
            );
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
            ffts_cross_core_sync(PIPE_FIX, 1057); // mode=2 id=4 100 0010 0001

            if ((loop_idx + 1) < end) {
                /* ************* stage 1-1 **************** */
                wait_flag_dev(3);
                __cbuf__ uint8_t *l1p_buf1 = l1p_buf_addr + 2 * L1_UINT8_BLOCK_SIZE;
                __cbuf__ uint8_t *l1v_buf1 = l1v_buf_addr + 2 * L1_UINT8_BLOCK_SIZE;
                // *** Prepare P to L1
                if (__m1 == 1) {
                    copy_gm_to_cbuf(
                        (__cbuf__ half *)l1p_buf1,
                        (__gm__ half *)p_gm_aic + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2,
                        0,             // sid
                        1,             // nBurst
                        round_n1 / 16, // lenBurst
                        0,             // srcGap
                        0,             // dstGap
                        PAD_NONE       // padMode
                    );
                } else {
                    copy_gm_to_cbuf_multi_nd2nz_b16(
                        (__cbuf__ half *)l1p_buf1,
                        (__gm__ half *)p_gm_aic + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2,
                        0,         // sid
                        1,         // ndNum
                        __m1,      // nValue
                        __n1,      // dValue
                        0,         // srcNdMatrixStride, unused
                        round_n1, // srcDValue
                        round_m1,  // dstNzC0Stride
                        1,         // dstNzNStride
                        0          // dstNzMatrixStride, unused
                    );
                }
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                if (__m1 == 1) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE,
                        (__cbuf__ half *)l1p_buf1,
                        0,                                                    // baseIdx
                        (round_n1 + CUBE_MATRIX_SIZE - 1) / CUBE_MATRIX_SIZE, // repeat
                        1,                                                    // srcStride
                        0,                                                    // dstStride
                        0,                                                    // sid
                        false,                                                // transpose
                        inc                                                   // addr_cal_mode_t
                    );
                } else if (round_m1 <= round_n1) {
                    for (int32_t l0a_load_idx = 0; l0a_load_idx < round_m1 / BLOCK_SIZE; ++l0a_load_idx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE + l0a_load_idx * round_n1 * BLOCK_SIZE,
                            (__cbuf__ half *)l1p_buf1 + l0a_load_idx * CUBE_MATRIX_SIZE,
                            0,                     // baseIdx
                            round_n1 / BLOCK_SIZE, // repeat
                            round_m1 / BLOCK_SIZE, // srcStride
                            0,                     // dstStride
                            0,                     // sid
                            false,                 // transpose
                            inc                    // addr_cal_mode_t
                        );
                    }
                } else {
                    for (int32_t l0a_load_idx = 0; l0a_load_idx < round_n1 / BLOCK_SIZE; ++l0a_load_idx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE + l0a_load_idx * CUBE_MATRIX_SIZE,
                            (__cbuf__ half *)l1p_buf1 + l0a_load_idx * round_m1 * BLOCK_SIZE,
                            0,                         // baseIdx
                            round_m1 / BLOCK_SIZE,     // repeat
                            1,                         // srcStride
                            round_n1 / BLOCK_SIZE - 1, // dstStride
                            0,                         // sid
                            false,                     // transpose
                            inc                        // addr_cal_mode_t
                        );
                    }
                }
                // *** Prepare V to L1
                CopyGmToCbufMultiNd2NzB16((__cbuf__ half *)l1v_buf1, (__gm__ half *)v_gm_aic + (int64_t)v_offset1, 0, 1,
                    __n1, __k1, 0, stride_qkvo, round_n1, 1, 0);
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
                if (round_n1 <= round_k1) {
                    for (int32_t l0b_load_idx = 0; l0b_load_idx < round_n1 / BLOCK_SIZE; ++l0b_load_idx) {
                        load_cbuf_to_cb(
                            (__cb__ half *)l0b_buf + L0AB_HALF_BUF_SIZE + l0b_load_idx * round_k1 * BLOCK_SIZE,
                            (__cbuf__ half *)l1v_buf1 + l0b_load_idx * CUBE_MATRIX_SIZE,
                            0,                     // baseIdx
                            round_k1 / BLOCK_SIZE, // repeat
                            round_n1 / BLOCK_SIZE, // srcStride
                            0,                     // dstStride
                            0,                     // sid
                            true,                  // transpose
                            inc                    // addr_cal_mode_t
                        );
                    }
                } else {
                    for (int32_t l0b_load_idx = 0; l0b_load_idx < round_k1 / BLOCK_SIZE; ++l0b_load_idx) {
                        load_cbuf_to_cb(
                            (__cb__ half *)l0b_buf + L0AB_HALF_BUF_SIZE + l0b_load_idx * CUBE_MATRIX_SIZE,
                            (__cbuf__ half *)l1v_buf1 + l0b_load_idx * round_n1 * BLOCK_SIZE,
                            0,                         // baseIdx
                            round_n1 / BLOCK_SIZE,     // repeat
                            1,                         // srcStride
                            round_k1 / BLOCK_SIZE - 1, // dstStride
                            0,                         // sid
                            true,                      // transpose
                            inc                        // addr_cal_mode_t
                        );
                    }
                }
                set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
                mad((__cc__ float *)l0c_buf + L0AB_HALF_BUF_SIZE,
                    (__ca__ half *)l0a_buf + L0AB_HALF_BUF_SIZE,
                    (__cb__ half *)l0b_buf + L0AB_HALF_BUF_SIZE,
                    __m1, // m
                    __n1, // k
                    __k1, // n
                    0,    // unitFlag
                    0,    // kDirectionAlign
                    0,    // cmatrixSource
                    1     // cmatrixInitVal
                );
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
                // copy O to gm
                copy_matrix_cc_to_gm(
                    (__gm__ half *)o_tmp_gm_aic + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2,
                    (__cc__ float *)l0c_buf + L0AB_HALF_BUF_SIZE,
                    0,            // sid
                    round_k1,         // NSize
                    __m1,         // MSize
                    round_k1, // dstStride_dst_D
                    round_m1,     // srcStride
                    0,            // UnitFlagMode
                    F322F16,      // QuantPRE
                    0,            // ReLUPRE
                    false,        // channelSplit
                    true          // NZ2ND_EN
                );
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                ffts_cross_core_sync(PIPE_FIX, 1313); // mode=2 id=5 101 0010 0001
            }
        }
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flashattention_mix_aiv(
        __gm__ uint8_t *__restrict__ q_gm_aiv,
        __gm__ uint8_t *__restrict__ k_gm_aiv,
        __gm__ uint8_t *__restrict__ v_gm_aiv,
        __gm__ uint8_t *__restrict__ mask_gm_aiv,
        __gm__ uint8_t *__restrict__ o_gm_aiv,
        __gm__ uint8_t *__restrict__ s_gm_aiv,
        __gm__ uint8_t *__restrict__ p_gm_aiv,
        __gm__ uint8_t *__restrict__ o_tmp_gm_aiv,
        uint32_t * tiling_para_gm)
    {
        int32_t sub_block_idx = get_subblockid();
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        __ubuf__ uint8_t* ls_ubuf = (__ubuf__ uint8_t *)get_imm(0);
        __ubuf__ uint8_t* lp_ubuf = (__ubuf__ uint8_t *)get_imm(0);
        __ubuf__ uint8_t* ls32_ubuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        __ubuf__ uint8_t* lo_ubuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE);
        __ubuf__ uint8_t* lm_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE);
        __ubuf__ uint8_t* hm_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t* gm_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t* dm_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE + 5 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t* ll_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t* gl_ubuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE);
        __ubuf__ uint8_t* tv_ubuf = (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE);
        __ubuf__ uint8_t* go_ubuf = (__ubuf__ uint8_t *)get_imm(8 * UB_UINT8_BLOCK_SIZE);
        int32_t go_flag_scalar = 1;

        uint32_t batchSize = (uint32_t)(*((int32_t *)tiling_para_gm + 20));
        int64_t currHeads = (int32_t)(*((int32_t *)tiling_para_gm + 2));

        int32_t curBatch = 0;
        int32_t curTotalQblk = 0;
        int32_t curQblkId = 0;
        ComputeCurBatch(curBatch, curTotalQblk, curQblkId, batchSize, currHeads, tiling_para_gm, TILING_PARA_SIZE);

        // get tiling args
        int32_t q_seqlen = (int32_t)(*((int32_t *)tiling_para_gm + TILING_PARA_SIZE * curBatch));
        int32_t kv_seqlen = (int32_t)(*((int32_t *)tiling_para_gm + 1 + TILING_PARA_SIZE * curBatch));
        int32_t heads = (int32_t)(*((int32_t *)tiling_para_gm + 2 + TILING_PARA_SIZE * curBatch));
        int32_t embd = (int32_t)(*((int32_t *)tiling_para_gm + 3 + TILING_PARA_SIZE * curBatch));

        int32_t max_seqlen = (int32_t)(*((int32_t *)tiling_para_gm + 4 + TILING_PARA_SIZE * curBatch));
        half tor = (half)(*((float *)tiling_para_gm + 5 + TILING_PARA_SIZE * curBatch));
        int32_t pp_m_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 6 + TILING_PARA_SIZE * curBatch));
        int32_t pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 7 + TILING_PARA_SIZE * curBatch));

        uint32_t addr_q_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 8 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_q_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 9 + TILING_PARA_SIZE * curBatch));
        int64_t addr_q_scalar = (int64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32);

        uint32_t addr_k_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 10 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_k_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 11 + TILING_PARA_SIZE * curBatch));
        int64_t addr_k_scalar = (int64_t)(((uint64_t)addr_k_high32) << 32 | addr_k_loww32);

        uint32_t addr_v_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 12 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_v_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + TILING_PARA_SIZE * curBatch));
        int64_t addr_v_scalar = (int64_t)(((uint64_t)addr_v_high32) << 32 | addr_v_loww32);

        uint32_t addr_s_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 14 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_s_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 15 + TILING_PARA_SIZE * curBatch));
        int64_t addr_s_scalar = (int64_t)(((uint64_t)addr_s_high32) << 32 | addr_s_loww32);

        uint32_t addr_p_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 16 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_p_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 17 + TILING_PARA_SIZE * curBatch));
        int64_t addr_p_scalar = (int64_t)(((uint64_t)addr_p_high32) << 32 | addr_p_loww32);

        uint32_t addr_o_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 18 + TILING_PARA_SIZE * curBatch));
        uint32_t addr_o_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 19 + TILING_PARA_SIZE * curBatch));
        int64_t addr_o_scalar = (int64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32);

        int32_t curQblk = (int32_t)(*((int32_t *)tiling_para_gm + 21 + TILING_PARA_SIZE * curBatch));

        int32_t m_loop = 0;
        if (pp_m_scalar != 0) {
            m_loop = (q_seqlen + pp_m_scalar - 1) / pp_m_scalar;
        }
        int32_t n_loop = 0;
        if (pp_n_scalar != 0) {
            n_loop = (kv_seqlen + pp_n_scalar - 1) / pp_n_scalar;
        }

        int32_t start = curQblkId * n_loop;
        int32_t end = start + n_loop;

        int32_t stride_qkvo = heads * embd;
        int32_t stride_sp = kv_seqlen;

        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        if (((int32_t)sub_block_idx) < 2) {
            for (int32_t loop_idx = start; loop_idx < end; loop_idx += 2) {
                int32_t head_idx0 = 0;
                int32_t m_idx0 = 0;
                int32_t n_idx0 = 0;
                if (m_loop != 0 && n_loop != 0) {
                    head_idx0 = loop_idx / (m_loop * n_loop);
                    m_idx0 = loop_idx % (m_loop * n_loop) / n_loop;
                    n_idx0 = loop_idx % (m_loop * n_loop) % n_loop;
                }

                int64_t s_offset0 = addr_s_scalar + head_idx0 * q_seqlen * stride_sp + m_idx0 * pp_m_scalar * stride_sp
                    + n_idx0 * pp_n_scalar;
                int64_t p_offset0 = addr_p_scalar + head_idx0 * q_seqlen * stride_sp + m_idx0 * pp_m_scalar * stride_sp
                    + n_idx0 * pp_n_scalar;
                int64_t o_offset0 = addr_o_scalar + head_idx0 * embd + m_idx0 * pp_m_scalar * stride_qkvo;
                int64_t mask_offset0 = m_idx0 * pp_m_scalar * max_seqlen + n_idx0 * pp_n_scalar;
                int32_t last_n_loop0 = (n_idx0 == (n_loop - 1)) ? 1 : 0;
                int32_t wrap_o0 = last_n_loop0;
                int32_t init_g0 = (n_idx0 == 0) ? 1 : 0;
                int32_t __m0 = (m_idx0 == (m_loop - 1)) ? (q_seqlen - m_idx0 * pp_m_scalar) : pp_m_scalar;
                int32_t __n0 = (n_idx0 == (n_loop - 1)) ? (kv_seqlen - n_idx0 * pp_n_scalar) : pp_n_scalar;
                int32_t __k0 = embd;
                int32_t sub_m0 = (sub_block_idx == 1) ? (__m0 - __m0 / 2) : __m0 / 2;
                int32_t sub_m0_d128 = (sub_m0 + 127) / 128;   // up aligned to 128
                int32_t sub_m0_d64 = (sub_m0 + 63) / 64;   // up aligned to 128
                int32_t round_sub_m0 = (sub_m0 + 15) / 16 * 16;
                int32_t round_n0 = (__n0 + 15) / 16 * 16;
                int32_t round_k0 = (__k0 + 15) / 16 * 16;
                int32_t p_size0 = round_sub_m0 * round_n0;
                int32_t o_size0 = round_sub_m0 * round_k0;

                int32_t head_idx1 = 0;
                int32_t m_idx1 = 0;
                int32_t n_idx1 = 0;
                if (m_loop != 0 && n_loop != 0) {
                    head_idx1 = (loop_idx + 1) / (m_loop * n_loop);
                    m_idx1 = (loop_idx + 1) % (m_loop * n_loop) / n_loop;
                    n_idx1 = (loop_idx + 1) % (m_loop * n_loop) % n_loop;
                }

                int64_t s_offset1 = addr_s_scalar + head_idx1 * q_seqlen * stride_sp + m_idx1 * pp_m_scalar * stride_sp
                    + n_idx1 * pp_n_scalar;
                int64_t p_offset1 = addr_p_scalar + head_idx1 * q_seqlen * stride_sp + m_idx1 * pp_m_scalar * stride_sp
                    + n_idx1 * pp_n_scalar;
                int64_t o_offset1 = addr_o_scalar + head_idx1 * embd + m_idx1 * pp_m_scalar * stride_qkvo;
                int64_t mask_offset1 = m_idx1 * pp_m_scalar * max_seqlen + n_idx1 * pp_n_scalar;
                int32_t last_n_loop1 = (n_idx1 == (n_loop - 1)) ? 1 : 0;
                int32_t wrap_o1 = last_n_loop1;
                int32_t init_g1 = (n_idx1 == 0) ? 1 : 0;
                int32_t __m1 = (m_idx1 == (m_loop - 1)) ? (q_seqlen - m_idx1 * pp_m_scalar) : pp_m_scalar;
                int32_t __n1 = (n_idx1 == (n_loop - 1)) ? (kv_seqlen - n_idx1 * pp_n_scalar) : pp_n_scalar;
                int32_t __k1 = embd;
                int32_t sub_m1 = (sub_block_idx == 1) ? (__m1 - __m1 / 2) : __m1 / 2;
                int32_t sub_m1_d128 = (sub_m1 + 127) / 128;   // up aligned to 128
                int32_t sub_m1_d64 = (sub_m1 + 63) / 64;   // up aligned to 128
                int32_t round_sub_m1 = (sub_m1 + 15) / 16 * 16;
                int32_t round_n1 = (__n1 + 15) / 16 * 16;
                int32_t round_k1 = (__k1 + 15) / 16 * 16;
                int32_t p_size1 = round_sub_m1 * round_n1;
                int32_t o_size1 = round_sub_m1 * round_k1;

                /* ************ stage 0-0  ************* */
                wait_flag_dev(0);
                if (sub_m0 > 0) {
                    copy_gm_to_ubuf(
                        (__ubuf__ half *)ls_ubuf,
                        (__gm__ half *)s_gm_aiv + (int32_t)block_idx * TMP_SIZE
                            + (int32_t)sub_block_idx * __m0 / 2 * round_n0,
                        0,                      // sid
                        1,                      // nBurst
                        sub_m0 * round_n0 / 16, // lenBurst
                        0,                      // srcGap
                        0                       // dstGap
                    );
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    // *** ls = tor * ls
                    vmuls((__ubuf__ half *)ls_ubuf,
                        (__ubuf__ half *)ls_ubuf,
                        tor,
                        p_size0 / VECTOR_SIZE, // repeat
                        1,                     // dstBlockStride
                        1,                     // srcBlockStride
                        8,                     // dstRepeatStride
                        8                      // srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    if (mask_gm_aiv != nullptr) {
                        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                        copy_gm_to_ubuf_align_b16(
                            (__ubuf__ half *)lo_ubuf,
                            (__gm__ half *)mask_gm_aiv + (int64_t)mask_offset0
                                + (int32_t)sub_block_idx * __m0 / 2 * max_seqlen,
                            0,                       // sid
                            sub_m0,                  // nBurst
                            __n0 * 2,                // lenBurst
                            0,                       // leftPaddingNum
                            0,                       // rightPaddingNum
                            (max_seqlen - __n0) * 2, // srcGap
                            0                        // dstGap
                        );
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        // *** ls = ls + mask
                        vadd((__ubuf__ half *)ls_ubuf,
                            (__ubuf__ half *)ls_ubuf,
                            (__ubuf__ half *)lo_ubuf,
                            p_size0 / VECTOR_SIZE, // repeat
                            1,                     // dstBlockStride
                            1,                     // src0BlockStride
                            1,                     // src1BlockStride
                            8,                     // dstRepeatStride
                            8,                     // src0RepeatStride
                            8                      // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    }
                    // *** lm = rowmax(ls)
                    if (__n0 <= VECTOR_SIZE) {
                        set_mask(__n0);
                        vcmax((__ubuf__ half *)lm_ubuf,
                            (__ubuf__ half *)ls_ubuf, // (sub_m0, __n0)
                            sub_m0,                // repeat
                            1,                     // dstRepeatStride
                            1,                     // srcBlockStride
                            round_n0 / BLOCK_SIZE, // srcRepeatStride
                            ONLY_VALUE             // order
                        );
                    } else {
                        copy_ubuf_to_ubuf(
                            (__ubuf__ half *)tv_ubuf,
                            (__ubuf__ half *)ls_ubuf,
                            0,                                     // sid
                            sub_m0,                                // nBurst
                            VECTOR_SIZE / BLOCK_SIZE,              // lenBurst
                            (round_n0 - VECTOR_SIZE) / BLOCK_SIZE, // srcGap
                            0                                      // dstGap
                        );
                        pipe_barrier(PIPE_V);
                        set_mask(__n0 - 128);
                        VecMax((__ubuf__ half *)tv_ubuf, (__ubuf__ half *)tv_ubuf,
                               (__ubuf__ half *)ls_ubuf + VECTOR_SIZE, sub_m0, 1, 1, 1, 8, 8, round_n0 / BLOCK_SIZE);
                        pipe_barrier(PIPE_V);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        vcmax((__ubuf__ half *)lm_ubuf,
                            (__ubuf__ half *)tv_ubuf,
                            sub_m0,    // repeat
                            1,         // dstRepeatStride
                            1,         // srcBlockStride
                            8,         // srcRepeatStride
                            ONLY_VALUE // order
                        );
                    }
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    if (init_g0 == 0) {
                        // *** hm = vmax(lm, gm)
                        vmax((__ubuf__ half *)hm_ubuf,
                            (__ubuf__ half *)lm_ubuf,
                            (__ubuf__ half *)gm_ubuf,
                            sub_m0_d128, // repeat
                            1,           // dstBlockStride
                            1,           // src0BlockStride
                            1,           // src1BlockStride
                            8,           // dstRepeatStride
                            8,           // src0RepeatStride
                            8            // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** dm = gm - hm
                        VecSub((__ubuf__ half *)dm_ubuf, (__ubuf__ half *)gm_ubuf, (__ubuf__ half *)hm_ubuf,
                               sub_m0_d128, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    } else {
                        // *** hm = lm
                        copy_ubuf_to_ubuf(
                            (__ubuf__ half *)hm_ubuf,
                            (__ubuf__ half *)lm_ubuf,
                            0,                         // sid
                            1,                         // nBurst
                            round_sub_m0 / BLOCK_SIZE, // lenBurst
                            0,                         // srcGap
                            0                          // dstGap
                        );
                        pipe_barrier(PIPE_V);
                    }
                    // *** gm = hm
                    copy_ubuf_to_ubuf(
                        (__ubuf__ half *)gm_ubuf,
                        (__ubuf__ half *)hm_ubuf,
                        0,                         // sid
                        1,                         // nBurst
                        round_sub_m0 / BLOCK_SIZE, // lenBurst
                        0,                         // srcGap
                        0                          // dstGap
                    );
                    pipe_barrier(PIPE_V);
                    // *** hm_block = expand_to_block(hm), 存放于 tv
                    vbrcb(
                        (__ubuf__ uint16_t *)tv_ubuf,
                        (__ubuf__ uint16_t *)hm_ubuf,
                        1,  // dstBlockStride
                        8,  // dstRepeatStride
                        round_sub_m0 / 8  // repeat
                    );  // (mi,) -> (mi, 16)
                    pipe_barrier(PIPE_V);
                    // *** ls = ls - hm_block
                    for (int32_t vsub_idx = 0; vsub_idx < round_n0 / BLOCK_SIZE; ++vsub_idx) {
                        vsub((__ubuf__ half *)ls_ubuf + vsub_idx * BLOCK_SIZE,
                            (__ubuf__ half *)ls_ubuf + vsub_idx * BLOCK_SIZE,
                            (__ubuf__ half *)tv_ubuf,
                            round_sub_m0 * BLOCK_SIZE / 128, // repeat
                            round_n0 / BLOCK_SIZE,           // dstBlockStride
                            round_n0 / BLOCK_SIZE,           // src0BlockStride
                            1,                               // src1BlockStride
                            8 * round_n0 / BLOCK_SIZE,       // dstRepeatStride
                            8 * round_n0 / BLOCK_SIZE,       // src0RepeatStride
                            8                                // src1RepeatStride
                        );
                    }
                    pipe_barrier(PIPE_V);
                    // *** ls = castfp16to32(ls)
                    vexp((__ubuf__ half *)ls_ubuf,
                        (__ubuf__ half *)ls_ubuf,
                        p_size0 / VECTOR_SIZE, // repeat
                        1,                     // dstBlockStride
                        1,                     // srcBlockStride
                        8,                     // dstRepeatStride
                        8                      // srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    // *** ll = rowsum(lp)
                    if (__n0 <= VECTOR_SIZE) {
                        set_mask(__n0);
                        vcadd((__ubuf__ half *)ll_ubuf,
                            (__ubuf__ half *)lp_ubuf,
                            sub_m0,                // repeat
                            1,                     // dstRepeatStride
                            1,                     // srcBlockStride
                            round_n0 / BLOCK_SIZE, // srcRepeatStride
                            0                      // mode
                        );
                    } else {
                        copy_ubuf_to_ubuf(
                            (__ubuf__ half *)tv_ubuf,
                            (__ubuf__ half *)lp_ubuf,
                            0,                                     // sid
                            sub_m0,                                // nBurst
                            VECTOR_SIZE / BLOCK_SIZE,              // lenBurst
                            (round_n0 - VECTOR_SIZE) / BLOCK_SIZE, // srcGap
                            0                                      // dstGap
                        );
                        pipe_barrier(PIPE_V);
                        set_mask(__n0 - 128);
                        VecAdd((__ubuf__ half *)tv_ubuf, (__ubuf__ half *)tv_ubuf,
                               (__ubuf__ half *)lp_ubuf + VECTOR_SIZE, sub_m0, 1, 1, 1, 8, 8, round_n0 / BLOCK_SIZE);
                        pipe_barrier(PIPE_V);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        vcadd((__ubuf__ half *)ll_ubuf,
                            (__ubuf__ half *)tv_ubuf,
                            sub_m0, // repeat
                            1,      // dstRepeatStride
                            1,      // srcBlockStride
                            8,      // srcRepeatStride
                            0       // order
                        );
                    }
                    pipe_barrier(PIPE_V);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    copy_ubuf_to_gm(
                        (__gm__ half *)p_gm_aiv + (int32_t)block_idx * TMP_SIZE
                            + (int32_t)sub_block_idx * __m0 / 2 * round_n0,
                        (__ubuf__ half *)lp_ubuf,
                        0,                      // sid
                        1,                      // nBurst
                        sub_m0 * round_n0 / 16, // lenBurst
                        0,                      // srcGap
                        0                       // dstGap
                    );
                }
                ffts_cross_core_sync(PIPE_MTE3, 545); // mode=2 id=2 10 0010 0001

                if ((loop_idx + 1) < end) {
                    /* ************ stage 1-0  ************* */
                    wait_flag_dev(1);
                    if (sub_m1 > 0) {
                        copy_gm_to_ubuf(
                            (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                            (__gm__ half *)s_gm_aiv + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2
                                + (int32_t)sub_block_idx * __m1 / 2 * round_n1,
                            0,                      // sid
                            1,                      // nBurst
                            sub_m1 * round_n1 / 16, // lenBurst
                            0,                      // srcGap
                            0                       // dstGap
                        );
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        // *** ls = tor * ls
                        vmuls((__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                            (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                            tor,
                            p_size1 / VECTOR_SIZE, // repeat
                            1,                     // dstBlockStride
                            1,                     // srcBlockStride
                            8,                     // dstRepeatStride
                            8                      // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        if (mask_gm_aiv != nullptr) {
                            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                            copy_gm_to_ubuf_align_b16(
                                (__ubuf__ half *)lo_ubuf + UB_HALF_BUF_SIZE,
                                (__gm__ half *)mask_gm_aiv + (int64_t)mask_offset1
                                    + (int32_t)sub_block_idx * __m1 / 2 * max_seqlen,
                                0,                       // sid
                                sub_m1,                  // nBurst
                                __n1 * 2,                // lenBurst
                                0,                       // leftPaddingNum
                                0,                       // rightPaddingNum
                                (max_seqlen - __n1) * 2, // srcGap
                                0                        // dstGap
                            );
                            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                            // *** ls = ls + mask
                            vadd((__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                                (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                                (__ubuf__ half *)lo_ubuf + UB_HALF_BUF_SIZE,
                                p_size1 / VECTOR_SIZE, // repeat
                                1,                     // dstBlockStride
                                1,                     // src0BlockStride
                                1,                     // src1BlockStride
                                8,                     // dstRepeatStride
                                8,                     // src0RepeatStride
                                8                      // src1RepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                        }
                        // *** lm = rowmax(ls)
                        if (__n1 <= VECTOR_SIZE) {
                            set_mask(__n1);
                            vcmax((__ubuf__ half *)lm_ubuf,
                                (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                                sub_m1,                // repeat
                                1,                     // dstRepeatStride
                                1,                     // srcBlockStride
                                round_n1 / BLOCK_SIZE, // srcRepeatStride
                                ONLY_VALUE             // order
                            );
                        } else {
                            copy_ubuf_to_ubuf(
                                (__ubuf__ half *)tv_ubuf,
                                (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                                0,                                     // sid
                                sub_m1,                                // nBurst
                                VECTOR_SIZE / BLOCK_SIZE,              // lenBurst
                                (round_n1 - VECTOR_SIZE) / BLOCK_SIZE, // srcGap
                                0                                      // dstGap
                            );
                            pipe_barrier(PIPE_V);
                            set_mask(__n1 - 128);
                            VecMax((__ubuf__ half *)tv_ubuf, (__ubuf__ half *)tv_ubuf,
                                   (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE + VECTOR_SIZE,
                                   sub_m1, 1, 1, 1, 8, 8, round_n1 / BLOCK_SIZE);
                            pipe_barrier(PIPE_V);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            vcmax((__ubuf__ half *)lm_ubuf,
                                (__ubuf__ half *)tv_ubuf,
                                sub_m1,    // repeat
                                1,         // dstRepeatStride
                                1,         // srcBlockStride
                                8,         // srcRepeatStride
                                ONLY_VALUE // order
                            );
                        }
                        pipe_barrier(PIPE_V);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        if (init_g1 == 0) {
                            // *** hm = vmax(lm, gm)
                            vmax((__ubuf__ half *)hm_ubuf + UB_HALF_LINE_SIZE,
                                (__ubuf__ half *)lm_ubuf,
                                (__ubuf__ half *)gm_ubuf,
                                sub_m1_d128, // repeat
                                1,           // dstBlockStride
                                1,           // src0BlockStride
                                1,           // src1BlockStride
                                8,           // dstRepeatStride
                                8,           // src0RepeatStride
                                8            // src1RepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            // *** dm = gm - hm
                            VecSub((__ubuf__ half *)dm_ubuf + UB_HALF_LINE_SIZE, (__ubuf__ half *)gm_ubuf,
                                   (__ubuf__ half *)hm_ubuf + UB_HALF_LINE_SIZE, sub_m1_d128, 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                        } else {
                            // *** hm = lm
                            copy_ubuf_to_ubuf(
                                (__ubuf__ half *)hm_ubuf + UB_HALF_LINE_SIZE,
                                (__ubuf__ half *)lm_ubuf,
                                0,                         // sid
                                1,                         // nBurst
                                round_sub_m1 / BLOCK_SIZE, // lenBurst
                                0,                         // srcGap
                                0                          // dstGap
                            );
                            pipe_barrier(PIPE_V);
                        }
                        // *** gm = hm
                        copy_ubuf_to_ubuf(
                            (__ubuf__ half *)gm_ubuf,
                            (__ubuf__ half *)hm_ubuf + UB_HALF_LINE_SIZE,
                            0,                         // sid
                            1,                         // nBurst
                            round_sub_m1 / BLOCK_SIZE, // lenBurst
                            0,                         // srcGap
                            0                          // dstGap
                        );
                        pipe_barrier(PIPE_V);
                        // *** hm_block = expand_to_block(hm), 存放于 tv
                        vbrcb(
                            (__ubuf__ uint16_t *)tv_ubuf,
                            (__ubuf__ uint16_t *)hm_ubuf + UB_HALF_LINE_SIZE,
                            1,  // dstBlockStride
                            8,  // dstRepeatStride
                            round_sub_m1 / 8  // repeat
                        );
                        pipe_barrier(PIPE_V);
                        // *** ls = ls - hm_block
                        for (int32_t vsub_idx = 0; vsub_idx < round_n1 / BLOCK_SIZE; ++vsub_idx) {
                            vsub((__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE + vsub_idx * BLOCK_SIZE,
                                (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE + vsub_idx * BLOCK_SIZE,
                                (__ubuf__ half *)tv_ubuf,
                                round_sub_m1 * BLOCK_SIZE / 128, // repeat
                                round_n1 / BLOCK_SIZE,           // dstBlockStride
                                round_n1 / BLOCK_SIZE,           // src0BlockStride
                                1,                               // src1BlockStride
                                8 * round_n1 / BLOCK_SIZE,       // dstRepeatStride
                                8 * round_n1 / BLOCK_SIZE,       // src0RepeatStride
                                8                                // src1RepeatStride
                            );
                        }
                        pipe_barrier(PIPE_V);
                        // *** ls = castfp16to32(ls)
                        vexp((__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                            (__ubuf__ half *)ls_ubuf + UB_HALF_BUF_SIZE,
                            p_size1 / VECTOR_SIZE, // repeat
                            1,            // dstBlockStride
                            1,            // srcBlockStride
                            8,            // dstRepeatStride
                            8             // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                        // *** ll = rowsum(lp)
                        if (__n1 <= VECTOR_SIZE) {
                            set_mask(__n1);
                            vcadd((__ubuf__ half *)ll_ubuf + UB_HALF_LINE_SIZE,
                                (__ubuf__ half *)lp_ubuf + UB_HALF_BUF_SIZE,
                                sub_m1,                // repeat
                                1,                     // dstRepeatStride
                                1,                     // srcBlockStride
                                round_n1 / BLOCK_SIZE, // srcRepeatStride
                                0                      // mode
                            );
                        } else {
                            copy_ubuf_to_ubuf(
                                (__ubuf__ half *)tv_ubuf,
                                (__ubuf__ half *)lp_ubuf + UB_HALF_BUF_SIZE,
                                0,                                     // sid
                                sub_m1,                                // nBurst
                                VECTOR_SIZE / BLOCK_SIZE,              // lenBurst
                                (round_n1 - VECTOR_SIZE) / BLOCK_SIZE, // srcGap
                                0                                      // dstGap
                            );
                            pipe_barrier(PIPE_V);
                            set_mask(__n1 - 128);
                            VecAdd((__ubuf__ half *)tv_ubuf, (__ubuf__ half *)tv_ubuf,
                                   (__ubuf__ half *)lp_ubuf + UB_HALF_BUF_SIZE + VECTOR_SIZE,
                                   sub_m1, 1, 1, 1, 8, 8, round_n1 / BLOCK_SIZE);
                            pipe_barrier(PIPE_V);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            vcadd((__ubuf__ half *)ll_ubuf + UB_HALF_LINE_SIZE,
                                (__ubuf__ half *)tv_ubuf,
                                sub_m1, // repeat
                                1,      // dstRepeatStride
                                1,      // srcBlockStride
                                8,      // srcRepeatStride
                                0       // order
                            );
                        }
                        pipe_barrier(PIPE_V);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                        copy_ubuf_to_gm(
                            (__gm__ half *)p_gm_aiv + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2
                                + (int32_t)sub_block_idx * __m1 / 2 * round_n1,
                            (__ubuf__ half *)lp_ubuf + UB_HALF_BUF_SIZE,
                            0,                      // sid
                            1,                      // nBurst
                            sub_m1 * round_n1 / 16, // lenBurst
                            0,                      // srcGap
                            0                       // dstGap
                        );
                    }
                    ffts_cross_core_sync(PIPE_MTE3, 801);  // mode=2 id=3 11 0010 0001
                }

                /* ************ stage 0-1  ************* */
                wait_flag_dev(4);
                if (sub_m0 > 0) {
                    if (mask_gm_aiv != nullptr) {
                        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    }
                    copy_gm_to_ubuf(
                        (__ubuf__ half *)lo_ubuf,
                        (__gm__ half *)o_tmp_gm_aiv + (int32_t)block_idx * TMP_SIZE +
                            (int32_t)sub_block_idx * __m0 / 2 * round_k0,
                        0,                         // sid
                        1,                         // nBurst
                        sub_m0 * round_k0 / 16,    // lenBurst
                        0,                         // srcGap
                        0                          // dstGap
                    );
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    // *** 更新 L 和 O
                    if (init_g0 == 0) {
                        // *** dm32 = castfp16to32(dm), 存放于 tv
                        vexp((__ubuf__ half *)dm_ubuf,
                            (__ubuf__ half *)dm_ubuf,
                            sub_m0_d64, // repeat
                            1,          // dstBlockStride
                            1,          // srcBlockStride
                            8,          // dstRepeatStride
                            8           // srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        // *** gl = dm * gl
                        VecMul((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)dm_ubuf, (__ubuf__ half *)gl_ubuf,
                               sub_m0_d128, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                        // *** gl = ll + gl
                        VecAdd((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)gl_ubuf, (__ubuf__ half *)ll_ubuf,
                               sub_m0_d128, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                        // *** dm_block = expand_to_block(dm), 存放于 tv
                        vbrcb(
                            (__ubuf__ uint16_t *)tv_ubuf,
                            (__ubuf__ uint16_t *)dm_ubuf,
                            1,  // dstBlockStride
                            8,  // dstRepeatStride
                            round_sub_m0 / 8  // repeat
                        );
                        pipe_barrier(PIPE_V);
                        if (go_flag_scalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 0;
                        }
                        // *** go = go * dm_block
                        for (int32_t vmul_idx = 0; vmul_idx < round_k0 / BLOCK_SIZE; ++vmul_idx) {
                            VecMul((__ubuf__ half *)go_ubuf + vmul_idx * BLOCK_SIZE,
                                   (__ubuf__ half *)go_ubuf + vmul_idx * BLOCK_SIZE,
                                   (__ubuf__ half *)tv_ubuf, round_sub_m0 * BLOCK_SIZE / 128, round_k0 / BLOCK_SIZE,
                                   round_k0 / BLOCK_SIZE, 1, 8 * round_k0 / BLOCK_SIZE, 8 * round_k0 / BLOCK_SIZE, 8);
                        }
                        pipe_barrier(PIPE_V);
                        // *** go = lo + go
                        VecAdd((__ubuf__ half *)go_ubuf, (__ubuf__ half *)go_ubuf, (__ubuf__ half *)lo_ubuf,
                               o_size0 / 128, 1, 1, 1, 8, 8, 8);
                        pipe_barrier(PIPE_V);
                    } else {
                        // *** gl = ll
                        CopyUbufToUbuf((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)ll_ubuf, 0, 1,
                                       round_sub_m0 / BLOCK_SIZE, 0, 0);
                        pipe_barrier(PIPE_V);
                        if (go_flag_scalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 0;
                        }
                        // *** go = lo
                        copy_ubuf_to_ubuf(
                            (__ubuf__ half *)go_ubuf,
                            (__ubuf__ half *)lo_ubuf,
                            0,                    // sid
                            1,                    // nBurst
                            o_size0 / BLOCK_SIZE, // lenBurst
                            0,                    // srcGap
                            0                     // dstGap
                        );
                        pipe_barrier(PIPE_V);
                    }
                    if (mask_gm_aiv != nullptr) {
                        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                    }
                    if (wrap_o0 == 1) {
                        // *** gl_block = expand_to_block(gl), 存放于 tv
                        vbrcb(
                            (__ubuf__ uint16_t *)tv_ubuf,
                            (__ubuf__ uint16_t *)gl_ubuf,
                            1,  // dstBlockStride
                            8,  // dstRepeatStride
                            round_sub_m0 / 8  // repeat
                        );
                        pipe_barrier(PIPE_V);
                        // *** go = go / gl_block
                        for (int32_t vdiv_idx = 0; vdiv_idx < round_k0 / BLOCK_SIZE; ++vdiv_idx) {
                            VecDiv((__ubuf__ half *)go_ubuf + vdiv_idx * BLOCK_SIZE,
                                   (__ubuf__ half *)go_ubuf + vdiv_idx * BLOCK_SIZE,
                                   (__ubuf__ half *)tv_ubuf, round_sub_m0 * BLOCK_SIZE / 128, round_k0 / BLOCK_SIZE,
                                   round_k0 / BLOCK_SIZE, 1, 8 * round_k0 / BLOCK_SIZE, 8 * round_k0 / BLOCK_SIZE, 8);
                        }
                        // ********************* move O to GM *********************
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        copy_ubuf_to_gm_align_b16(
                            (__gm__ half *)o_gm_aiv + (int64_t)o_offset0 +
                                (int32_t)sub_block_idx * __m0 / 2 * stride_qkvo,
                            (__ubuf__ half *)go_ubuf,
                            0,                       // sid
                            sub_m0,                  // nBurst
                            __k0 * 2,                // lenBurst
                            0,                       // leftPaddingNum
                            0,                       // rightPaddingNum
                            0,                       // srcGap
                            (stride_qkvo - __k0) * 2 // dstGap
                        );
                        if (go_flag_scalar == 0) {
                            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 1;
                        }
                    }
                }

                if ((loop_idx + 1) < end) {
                    /* ************ stage 1-1  ************* */
                    wait_flag_dev(5);
                    if (sub_m1 > 0) {
                        if (mask_gm_aiv != nullptr) {
                            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                        }
                        copy_gm_to_ubuf(
                            (__ubuf__ half *)lo_ubuf + UB_HALF_BUF_SIZE,
                            (__gm__ half *)o_tmp_gm_aiv + (int32_t)block_idx * TMP_SIZE + TMP_SIZE / 2 +
                                (int32_t)sub_block_idx * __m1 / 2 * round_k1,
                            0,                         // sid
                            1,                         // nBurst
                            sub_m1 * round_k1 / 16,    // lenBurst
                            0,                         // srcGap
                            0                          // dstGap
                        );
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        // *** 更新 L 和 O
                        if (init_g1 == 0) {
                            // *** dm32 = castfp16to32(dm), 存放于 tv
                            vexp((__ubuf__ half *)dm_ubuf + UB_HALF_LINE_SIZE,
                                (__ubuf__ half *)dm_ubuf + UB_HALF_LINE_SIZE,
                                sub_m1_d64, // repeat
                                1,          // dstBlockStride
                                1,          // srcBlockStride
                                8,          // dstRepeatStride
                                8           // srcRepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            // *** gl = dm * gl
                            VecMul((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)dm_ubuf + UB_HALF_LINE_SIZE,
                                   (__ubuf__ half *)gl_ubuf, sub_m1_d128, 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                            // *** gl = ll + gl
                            VecAdd((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)gl_ubuf,
                                   (__ubuf__ half *)ll_ubuf + UB_HALF_LINE_SIZE, sub_m1_d128, 1, 1, 1, 8, 8, 8);
                            pipe_barrier(PIPE_V);
                            // *** dm_block = expand_to_block(dm), 存放于 tv
                            vbrcb(
                                (__ubuf__ uint16_t *)tv_ubuf,
                                (__ubuf__ uint16_t *)dm_ubuf + UB_HALF_LINE_SIZE,
                                1,  // dstBlockStride
                                8,  // dstRepeatStride
                                round_sub_m1 / 8  // repeat
                            );
                            pipe_barrier(PIPE_V);
                            if (go_flag_scalar == 1) {
                                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                                go_flag_scalar = 0;
                            }
                            // *** go = go * dm_block
                            for (int32_t vmul_idx = 0; vmul_idx < round_k1 / BLOCK_SIZE; ++vmul_idx) {
                                VecMul((__ubuf__ half *)go_ubuf + vmul_idx * BLOCK_SIZE,
                                       (__ubuf__ half *)go_ubuf + vmul_idx * BLOCK_SIZE, (__ubuf__ half *)tv_ubuf,
                                       round_sub_m1 * BLOCK_SIZE / 128, round_k1 / BLOCK_SIZE, round_k1 / BLOCK_SIZE,
                                       1, 8 * round_k1 / BLOCK_SIZE, 8 * round_k1 / BLOCK_SIZE, 8);
                            }
                            pipe_barrier(PIPE_V);
                            // *** go = lo + go
                            vadd((__ubuf__ half *)go_ubuf,
                                (__ubuf__ half *)go_ubuf,
                                (__ubuf__ half *)lo_ubuf + UB_HALF_BUF_SIZE,
                                o_size1 / 128, // repeat
                                1,             // dstBlockStride
                                1,             // src0BlockStride
                                1,             // src1BlockStride
                                8,             // dstRepeatStride
                                8,             // src0RepeatStride
                                8);            // src1RepeatStride
                            pipe_barrier(PIPE_V);
                        } else {
                            // *** gl = ll
                            CopyUbufToUbuf((__ubuf__ half *)gl_ubuf, (__ubuf__ half *)ll_ubuf + UB_HALF_LINE_SIZE,
                                           0, 1, round_sub_m1 / BLOCK_SIZE, 0, 0);
                            pipe_barrier(PIPE_V);
                            if (go_flag_scalar == 1) {
                                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                                go_flag_scalar = 0;
                            }
                            // *** go = lo
                            copy_ubuf_to_ubuf(
                                (__ubuf__ half *)go_ubuf,
                                (__ubuf__ half *)lo_ubuf + UB_HALF_BUF_SIZE,
                                0,                    // sid
                                1,                    // nBurst
                                o_size1 / BLOCK_SIZE, // lenBurst
                                0,                    // srcGap
                                0                     // dstGap
                            );
                            pipe_barrier(PIPE_V);
                        }
                        if (mask_gm_aiv != nullptr) {
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                        }
                        if (wrap_o1 == 1) {
                            // *** gl_block = expand_to_block(gl), 存放于 tv
                            vbrcb(
                                (__ubuf__ uint16_t *)tv_ubuf,
                                (__ubuf__ uint16_t *)gl_ubuf,
                                1,  // dstBlockStride
                                8,  // dstRepeatStride
                                round_sub_m1 / 8  // repeat
                            );
                            pipe_barrier(PIPE_V);
                            // *** go = go / gl_block
                            for (int32_t vdiv_idx = 0; vdiv_idx < round_k1 / BLOCK_SIZE; ++vdiv_idx) {
                                VecDiv((__ubuf__ half *)go_ubuf + vdiv_idx * BLOCK_SIZE,
                                       (__ubuf__ half *)go_ubuf + vdiv_idx * BLOCK_SIZE,
                                       (__ubuf__ half *)tv_ubuf, round_sub_m1 * BLOCK_SIZE / 128,
                                       round_k1 / BLOCK_SIZE, round_k1 / BLOCK_SIZE, 1,
                                       8 * round_k1 / BLOCK_SIZE, 8 * round_k1 / BLOCK_SIZE, 8);
                            }
                            // ********************* move O to GM *********************
                            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                            copy_ubuf_to_gm_align_b16(
                                (__gm__ half *)o_gm_aiv + (int64_t)o_offset1
                                    + (int32_t)sub_block_idx * __m1 / 2 * stride_qkvo,
                                (__ubuf__ half *)go_ubuf,
                                0,                       // sid
                                sub_m1,                  // nBurst
                                __k1 * 2,                // lenBurst
                                0,                       // leftPaddingNum
                                0,                       // rightPaddingNum
                                0,                       // srcGap
                                (stride_qkvo - __k1) * 2 // dstGap
                            );
                            if (go_flag_scalar == 0) {
                                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                                go_flag_scalar = 1;
                            }
                        }
                    }
                }
            }
        }
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void CopyGmToCbufMultiNd2NzB16(__cbuf__ half* dst, __gm__ half * src, uint16_t sid,
    uint16_t ndNum, uint16_t nValue, uint16_t dValue, uint16_t srcNdMatrixStride, uint16_t srcDValue,
    uint16_t dstNzC0Stride, uint16_t dstNzNStride, uint16_t dstNzMatrixStride)
    {
        copy_gm_to_cbuf_multi_nd2nz_b16(dst, src, sid, ndNum, nValue, dValue, srcNdMatrixStride, srcDValue,
            dstNzC0Stride, dstNzNStride, dstNzMatrixStride);
    }

    template <typename T>
    __aicore__ inline void VecMax(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecAdd(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecSub(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vsub(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecMul(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmul(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecDiv(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToUbuf(T *dst, T *src, uint16_t sid, uint16_t nBurst,
        uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
    {
        copy_ubuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap);
    }

    __aicore__ inline void ComputeCurBatch(int32_t &curBatch, int32_t &curTotalQblk, int32_t &curQblkId,
        uint32_t batchSize, int32_t currHeads, uint32_t *tiling_para_gm, int32_t TILING_PARA_SIZE)
    {
        for (int32_t i = 0; i < batchSize; ++i) {
            int32_t curQblk = (int32_t)(*((int32_t *)tiling_para_gm + TILING_PARA_SIZE * i + 21));
            curTotalQblk += (curQblk * currHeads);
            if (curTotalQblk > block_idx) {
                curBatch = i;
                curQblkId = block_idx - (curTotalQblk - (curQblk * currHeads));
                break;
            }
        }
    }

    __aicore__ void set_mask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % 64;
        uint64_t bar = 128;
        uint64_t lowerBound = 64;
        for (int64_t i = 0; i < temp; ++i) {
            mask |= one << i;
        }

        if (len == bar) {
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        } else if (len >= lowerBound) {
            set_vector_mask(mask, (uint64_t)-1);
        } else {
            set_vector_mask(0x0, mask);
        }
    }

    __aicore__ void expand_to_block_half(__ubuf__ half *dst, __ubuf__ half *src, int32_t len)
    {
        constexpr int32_t BLOCK_SIZE = 16;
        constexpr int32_t CUBE_MATRIX_SIZE = 256;
        for (int32_t vadds_idx = 0; vadds_idx < 2; ++vadds_idx) {
            vadds((__ubuf__ half *)dst + vadds_idx * 8 * BLOCK_SIZE,
                (__ubuf__ half *)src,
                (half)0.0,
                len / BLOCK_SIZE, // repeat
                1,                // dstBlockStride
                0,                // srcBlockStride
                16,               // dstRepeatStride
                1                 // srcRepeatStride
            );
        }
        pipe_barrier(PIPE_V);
        for (int32_t vtrans_idx = 0; vtrans_idx < len / BLOCK_SIZE; ++vtrans_idx) {
            vtranspose((__ubuf__ uint16_t *)dst + vtrans_idx * CUBE_MATRIX_SIZE,
                (__ubuf__ uint16_t *)dst + vtrans_idx * CUBE_MATRIX_SIZE);
        }
        pipe_barrier(PIPE_V);
    }
};
}

extern "C" __global__ __aicore__ void unpad_flash_attention_mix(GM_ADDR Q, GM_ADDR Kcache, GM_ADDR Vcache,
                                                                GM_ADDR AttentionMask, GM_ADDR OutputO, GM_ADDR OutputS,
                                                                GM_ADDR OutputP, GM_ADDR Otmp, GM_ADDR usrWorkspace,
                                                                GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    UnpadFlashAttentionMix op;
    uint32_t *tilingPara = const_cast<uint32_t *>(tiling_data.UnpadFlashAttentionMixTilingParam);

#ifdef __DAV_C220_CUBE__
    op.unpad_flashattention_mix_aic(Q, Kcache, Vcache, AttentionMask, OutputO, OutputS, OutputP, Otmp, tilingPara);
#elif __DAV_C220_VEC__
    op.unpad_flashattention_mix_aiv(Q, Kcache, Vcache, AttentionMask, OutputO, OutputS, OutputP, Otmp, tilingPara);
#endif
}