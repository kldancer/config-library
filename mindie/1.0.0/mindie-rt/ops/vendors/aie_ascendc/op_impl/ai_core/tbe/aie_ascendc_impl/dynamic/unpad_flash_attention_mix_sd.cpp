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

#define TILING_PART_ARGS(curIdx) \
    if (block_idx == (curIdx)) { \
        q_seqlen = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + offsetTiling)); \
        kv_seqlen = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 1 + offsetTiling)); \
        heads = cur_heads; \
        embd = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 3 + offsetTiling)); \
        max_seqlen = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 4 + offsetTiling)); \
        tor = (half)(*((__gm__ float *)tiling_para_gm + 5 + offsetTiling)); \
        pp_m_scalar = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 6 + offsetTiling)); \
        pp_n_scalar = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 7 + offsetTiling)); \
        addr_q_high32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 8 + offsetTiling)); \
        addr_q_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 9 + offsetTiling)); \
        addr_q_scalar = (int64_t)(((uint64_t)addr_q_high32) << 32 | addr_q_loww32); \
        addr_k_high32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 10 + offsetTiling)); \
        addr_k_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 11 + offsetTiling)); \
        addr_k_scalar = (int64_t)(((uint64_t)addr_k_high32) << 32 | addr_k_loww32); \
        addr_v_high32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 12 +offsetTiling)); \
        addr_v_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 13 + offsetTiling)); \
        addr_v_scalar = (int64_t)(((uint64_t)addr_v_high32) << 32 | addr_v_loww32); \
        addr_o_high32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 18 + offsetTiling)); \
        addr_o_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 19 + offsetTiling)); \
        addr_o_scalar = (int64_t)(((uint64_t)addr_o_high32) << 32 | addr_o_loww32); \
    }

#define TILING_QBLOCK(curIdx) \
    if (block_idx == (curIdx)) { \
        cur_q_block_num = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + TILING_PARA_SIZE_FA * i + 21)); \
    }
#define TILING_BASE_ARGS(curIdx) \
    if (block_idx == (curIdx)) { \
        batch_size = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 20)); \
        max_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling_para_gm + 4)); \
        cur_heads = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 2)); \
        total_q_block_num = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 14)); \
        kv_heads_num = (int32_t)(*((__gm__ int32_t *)tiling_para_gm + 22)); \
    }

using namespace AscendC;

namespace {
// FFTS Flag
constexpr int32_t QK_READY_FA = 0;
constexpr int32_t SOFTMAX_READY_FA = 1;
constexpr int32_t UPDATE_READY_FA = 2;
constexpr int32_t TILING_HEAD_SIZE_FA = 16;
constexpr int32_t BIT_SHIFT_FA = 8;

constexpr int32_t L0AB_HALF_BUF_SIZE_FA = 16384;     // 128 * 128
constexpr int32_t L1_HALF_BUF_SIZE_FA = 65536;       // 256 * 256
constexpr int32_t BLOCK_SIZE_FA = 16;
constexpr int32_t CUBE_MATRIX_SIZE_FA = 256;         // 16 * 16
constexpr int32_t L0AB_UINT8_BLOCK_SIZE_FA = 32768;  // 128 * 128 * 2B
constexpr int32_t TMP_SIZE_FA = 32768;               // 128 * 256
constexpr int32_t FLOAT_BLOCK_SIZE_FA = 8;
constexpr int32_t HALF_VECTOR_SIZE_FA = 128;
constexpr int32_t FLOAT_VECTOR_SIZE_FA = 64;
constexpr int32_t MASK_LOW_FA = 64;
constexpr int32_t MASK_HIGH_FA = 128;
constexpr int32_t UB_UINT8_BLOCK_SIZE_DECODER_FA = 24576;  // 96 * 128 * 2B
constexpr int32_t UB_UINT8_LINE_SIZE_FA = 512;             // 128 * 4B

constexpr int32_t TILING_PARA_SIZE_FA = 26;
constexpr int64_t L1_UINT8_BLOCK_SIZE_FA = 131072;         // 128K, 910B L1 512K
constexpr int32_t UB_HALF_BUF_SIZE_FA = 8192;              // 64 * 128
constexpr int64_t UB_UINT8_BLOCK_SIZE_PREFILL_FA = 16384;  // 64 * 128 * 2B
constexpr int64_t UB_FLOAT_LINE_SIZE_FA = 128;             // 64，申请两倍空间防踩踏。
constexpr int64_t UB_HALF_LINE_SIZE_FA = 256;              // UB_FLOAT_LINE_SIZE_FA * 2

constexpr int32_t VECTOR_SIZE = 128;
}
namespace AscendC {
class UnpadFlashAttentionMixSd {
public:
    __aicore__ inline UnpadFlashAttentionMixSd() {}

    __aicore__ inline void unpad_flash_attention_decoder_mix_aic(
        __gm__ uint8_t *__restrict__ q_gm,
        __gm__ uint8_t *__restrict__ k_gm,
        __gm__ uint8_t *__restrict__ v_gm,
        __gm__ uint8_t *__restrict__ layerId_gm,
        __gm__ uint8_t *__restrict__ s_gm,
        __gm__ uint8_t *__restrict__ p_gm,
        __gm__ uint8_t *__restrict__ o_tmp_gm,
        __gm__ uint8_t *tiling,
        uint32_t *tiling_para_gm)
    {
        set_padding(0);
        set_atomic_none();
        uint64_t config = 0x1;
        set_nd_para(config);
        set_mask_norm();

        uint32_t batch_size_fa;
        uint32_t max_seqlen_fa;
        uint32_t q_heads_fa;
        uint32_t embdFa;
        uint32_t kv_heads_fa;
        uint32_t batch_mask_fa;
        uint32_t former_batch_fa;
        uint32_t former_head_split_fa;
        uint32_t tail_batch_fa;
        uint32_t tail_head_split_fa;
        uint32_t head_split_num_fa;

        if (tiling == nullptr) {
            batch_size_fa = (uint32_t)(*((int32_t *)tiling_para_gm));
            max_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1));
            q_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2));
            embdFa = (uint32_t)(*((int32_t *)tiling_para_gm + 3));
            kv_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 4));
            former_batch_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 7));
            former_head_split_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 8));
            tail_batch_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 9));
            tail_head_split_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 10));
            head_split_num_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 12));

        } else {
            batch_size_fa = (uint32_t)(*((__gm__ int32_t *)tiling));
            max_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            q_heads_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embdFa = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            kv_heads_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 4));
            former_batch_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 7));
            former_head_split_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            tail_batch_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            tail_head_split_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 10));
            head_split_num_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 12));
        }

        uint32_t group_num = q_heads_fa / kv_heads_fa;
        uint64_t stride_kv = kv_heads_fa * embdFa;
        uint64_t batch_stride_kv = batch_size_fa * max_seqlen_fa * stride_kv;
        // 指针地址
        uint64_t kCacheAddr  = *((__gm__ uint64_t *)k_gm);
        uint64_t vCacheAddr  = *((__gm__ uint64_t *)v_gm);
        __gm__ half *kCacheGm = (__gm__ half *)kCacheAddr;
        __gm__ half *vCacheGm = (__gm__ half *)vCacheAddr;
        // 指针地址
        uint32_t layer_id = *((__gm__ uint32_t *)layerId_gm);
        kCacheGm = kCacheGm + layer_id * batch_stride_kv;
        vCacheGm = vCacheGm + layer_id * batch_stride_kv;

        uint32_t __k = embdFa;
        uint32_t round_k = (__k + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        uint32_t core_per_batch = (q_heads_fa + former_head_split_fa - 1) / former_head_split_fa;
        uint32_t process_num = former_batch_fa * core_per_batch;

        uint32_t kv_seqlen;
        uint32_t pp_n_scalar;
        uint32_t addr_q_high32;
        uint32_t addr_q_loww32;
        uint32_t addr_k_high32;
        uint32_t addr_k_loww32;
        uint32_t addr_v_high32;
        uint32_t addr_v_loww32;

        for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
            uint32_t cur_batch = process / core_per_batch;
            uint32_t cur_core = process % core_per_batch;
            uint32_t cur_head_num = former_head_split_fa;
            if (cur_core == (core_per_batch - 1)) {
                cur_head_num = q_heads_fa - cur_core * former_head_split_fa;
            }
            uint32_t head_split_loop = (cur_head_num + head_split_num_fa - 1) / head_split_num_fa;
            uint32_t start_head = (process % core_per_batch) * former_head_split_fa;

            uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;

            if (tiling == nullptr) {
                kv_seqlen = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                addr_q_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 4 + offset_tiling));
                addr_q_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 5 + offset_tiling));
                addr_k_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 6 + offset_tiling));
                addr_k_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 7 + offset_tiling));
                addr_v_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 8 + offset_tiling));
                addr_v_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 9 + offset_tiling));

            } else {
                kv_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                pp_n_scalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                addr_q_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offset_tiling));
                addr_q_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offset_tiling));
                addr_k_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offset_tiling));
                addr_k_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offset_tiling));
                addr_v_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offset_tiling));
                addr_v_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offset_tiling));
            }

            RunAic(cur_batch, start_head, cur_head_num, head_split_loop, offset_tiling, embdFa,
                   head_split_num_fa, group_num, stride_kv, __k, round_k,
                   kv_seqlen, pp_n_scalar, addr_q_high32, addr_q_loww32, addr_k_high32,
                   addr_k_loww32, addr_v_high32, addr_v_loww32,
                   q_gm, kCacheGm, vCacheGm, s_gm, p_gm, o_tmp_gm);
        }

        if (tail_batch_fa > 0) {
            core_per_batch = (q_heads_fa + tail_head_split_fa - 1) / tail_head_split_fa;
            process_num = tail_batch_fa * core_per_batch;
            for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
                uint32_t cur_batch = process / core_per_batch + former_batch_fa;
                uint32_t cur_core = process % core_per_batch;
                uint32_t cur_head_num = tail_head_split_fa;
                if (cur_core == (core_per_batch - 1)) {
                    cur_head_num = q_heads_fa - cur_core * tail_head_split_fa;
                }
                uint32_t head_split_loop = (cur_head_num + head_split_num_fa - 1) / head_split_num_fa;
                uint32_t start_head = (process % core_per_batch) * tail_head_split_fa;

                uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;

                if (tiling == nullptr) {
                    kv_seqlen = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                    pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                    addr_q_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 4 + offset_tiling));
                    addr_q_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 5 + offset_tiling));
                    addr_k_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 6 + offset_tiling));
                    addr_k_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 7 + offset_tiling));
                    addr_v_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 8 + offset_tiling));
                    addr_v_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 9 + offset_tiling));
                } else {
                    kv_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                    pp_n_scalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                    addr_q_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offset_tiling));
                    addr_q_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offset_tiling));
                    addr_k_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offset_tiling));
                    addr_k_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offset_tiling));
                    addr_v_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offset_tiling));
                    addr_v_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offset_tiling));
                }

                RunAic(cur_batch, start_head, cur_head_num, head_split_loop, offset_tiling, embdFa,
                    head_split_num_fa, group_num, stride_kv, __k, round_k,
                    kv_seqlen, pp_n_scalar, addr_q_high32, addr_q_loww32, addr_k_high32,
                    addr_k_loww32, addr_v_high32, addr_v_loww32,
                    q_gm, kCacheGm, vCacheGm, s_gm, p_gm, o_tmp_gm);
            }
        }

        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flash_attention_decoder_mix_aiv(
        __gm__ uint8_t *__restrict__ mask_gm,
        __gm__ uint8_t *__restrict__ o_gm,
        __gm__ uint8_t *__restrict__ s_gm,
        __gm__ uint8_t *__restrict__ p_gm,
        __gm__ uint8_t *__restrict__ o_tmp_gm,
        __gm__ uint8_t *tiling,
        uint32_t *tiling_para_gm)
    {
        int32_t sub_block_idx = get_subblockid();
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        uint32_t batch_size_fa;
        uint32_t max_seqlen_fa;
        uint32_t q_heads_fa;
        uint32_t embdFa;
        float torFa;
        uint32_t batch_mask_fa;
        uint32_t former_batch_fa;
        uint32_t former_head_split_fa;
        uint32_t tail_batch_fa;
        uint32_t tail_head_split_fa;
        uint32_t mask_stride_fa;

        __ubuf__ int32_t *tiling_para_ub;

        if (tiling == nullptr) {
            batch_size_fa = (uint32_t)(*((int32_t *)tiling_para_gm));
            max_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1));
            q_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2));
            embdFa = (uint32_t)(*((int32_t *)tiling_para_gm + 3));
            torFa = (half)(*((float *)tiling_para_gm + 5));
            batch_mask_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 6));
            former_batch_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 7));
            former_head_split_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 8));
            tail_batch_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 9));
            tail_head_split_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 10));
            mask_stride_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 11));
        } else {
            batch_size_fa = (uint32_t)(*((__gm__ int32_t *)tiling));
            max_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            q_heads_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embdFa = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            torFa = (half)(*((__gm__ float *)tiling + 5));
            batch_mask_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 6));
            former_batch_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 7));
            former_head_split_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            tail_batch_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            tail_head_split_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 10));
            mask_stride_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 11));
        }

        uint32_t __k = embdFa;
        uint32_t round_k = (__k + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

        uint32_t core_per_batch = (q_heads_fa + former_head_split_fa - 1) / former_head_split_fa;
        uint32_t process_num = former_batch_fa * core_per_batch;

        uint32_t kv_seqlen;
        uint32_t pp_n_scalar;
        uint32_t addr_o_high32;
        uint32_t addr_o_loww32;

        for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
            uint32_t cur_batch = process / core_per_batch;
            uint32_t cur_core = process % core_per_batch;
            uint32_t cur_head_num = former_head_split_fa;
            if (cur_core == (core_per_batch - 1)) {
                cur_head_num = q_heads_fa - cur_core * former_head_split_fa;
            }
            uint32_t start_head = (process % core_per_batch) * former_head_split_fa;
            uint32_t head_idx = start_head + sub_block_idx * cur_head_num / 2;

            // get tiling args
            uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;
            if (tiling == nullptr) {
                kv_seqlen = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                addr_o_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 10 + offset_tiling));
                addr_o_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 11 + offset_tiling));
            } else {
                kv_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                pp_n_scalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                addr_o_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offset_tiling));
                addr_o_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offset_tiling));
            }

            RunAiv(cur_batch, head_idx, cur_head_num, offset_tiling, sub_block_idx, max_seqlen_fa,
                   embdFa, torFa, batch_mask_fa, mask_stride_fa, __k, round_k,
                   kv_seqlen, pp_n_scalar, addr_o_high32, addr_o_loww32,
                   mask_gm, o_gm, s_gm, p_gm, o_tmp_gm);
        }
        if (tail_batch_fa > 0) {
            core_per_batch = (q_heads_fa + tail_head_split_fa - 1) / tail_head_split_fa;
            process_num = tail_batch_fa * core_per_batch;
            for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
                uint32_t cur_batch = process / core_per_batch + former_batch_fa;
                uint32_t cur_core = process % core_per_batch;
                uint32_t cur_head_num = tail_head_split_fa;
                if (cur_core == (core_per_batch - 1)) {
                    cur_head_num = q_heads_fa - cur_core * tail_head_split_fa;
                }
                uint32_t start_head = (process % core_per_batch) * tail_head_split_fa;
                uint32_t head_idx = start_head + sub_block_idx * cur_head_num / 2;

                // get tiling args
                uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;
                if (tiling == nullptr) {
                    kv_seqlen = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                    pp_n_scalar = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                    addr_o_high32 = (uint32_t)(*((int32_t *)tiling_para_gm + 10 + offset_tiling));
                    addr_o_loww32 = (uint32_t)(*((int32_t *)tiling_para_gm + 11 + offset_tiling));
                } else {

                    kv_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                    pp_n_scalar = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                    addr_o_high32 = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offset_tiling));
                    addr_o_loww32 = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offset_tiling));
                }

                RunAiv(cur_batch, head_idx, cur_head_num, offset_tiling, sub_block_idx, max_seqlen_fa,
                       embdFa, torFa, batch_mask_fa, mask_stride_fa, __k, round_k,
                       kv_seqlen, pp_n_scalar, addr_o_high32, addr_o_loww32,
                       mask_gm, o_gm, s_gm, p_gm, o_tmp_gm);
            }
        }

        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flashattention_encoder_mix_aic(
        __gm__ uint8_t *__restrict__ q_gm,
        __gm__ uint8_t *__restrict__ k_gm,
        __gm__ uint8_t *__restrict__ v_gm,
        __gm__ uint8_t *__restrict__ layerId_gm,
        __gm__ uint8_t *__restrict__ mask_gm,
        __gm__ uint8_t *__restrict__ o_gm,
        __gm__ uint8_t *__restrict__ s_gm,
        __gm__ uint8_t *__restrict__ p_gm,
        __gm__ uint8_t *__restrict__ o_tmp_gm,
        __gm__ uint8_t *tiling,
        uint32_t *tiling_para_gm)
    {
        set_padding(0);
        set_atomic_none();
        uint64_t config = 0x1;
        set_nd_para(config);
        set_mask_norm();

        __cbuf__ uint8_t *l1q_buf_addr = (__cbuf__ uint8_t *)get_imm(0);                             // 2 block
        __cbuf__ uint8_t *l1k_buf_addr = (__cbuf__ uint8_t *)get_imm(2 * L0AB_UINT8_BLOCK_SIZE_FA);  // 2 block
        __cbuf__ uint8_t *l1p_buf_addr = (__cbuf__ uint8_t *)get_imm(4 * L0AB_UINT8_BLOCK_SIZE_FA);  // 2 block
        __cbuf__ uint8_t *l1v_buf_addr = (__cbuf__ uint8_t *)get_imm(6 * L0AB_UINT8_BLOCK_SIZE_FA);  // 2 block
        __ca__ uint8_t *l0a_buf = (__ca__ uint8_t *)get_imm(0);
        __cb__ uint8_t *l0b_buf = (__cb__ uint8_t *)get_imm(0);
        __cc__ uint8_t *l0c_buf = (__cc__ uint8_t *)get_imm(0);

        uint32_t batch_size_fa;
        uint32_t max_seqlen_fa;
        uint32_t q_heads_fa;
        uint32_t embdFa;
        uint32_t kv_heads_fa;
        uint32_t is_triu_mask_fa;
        uint32_t total_q_blk_num_fa;

        if (tiling == nullptr) {
            batch_size_fa = (uint32_t)(*((int32_t *)tiling_para_gm));
            max_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1));
            q_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2));
            embdFa = (uint32_t)(*((int32_t *)tiling_para_gm + 3));
            kv_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 4));
            is_triu_mask_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 8));
            total_q_blk_num_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 9));
        } else {
            batch_size_fa = (uint32_t)(*((__gm__ int32_t *)tiling));
            max_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            q_heads_fa= (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embdFa = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            kv_heads_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 4));
            is_triu_mask_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            total_q_blk_num_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
        }

        uint32_t group_num = q_heads_fa / kv_heads_fa;
        uint64_t stride_qo = q_heads_fa * embdFa;
        uint64_t stride_kv = kv_heads_fa * embdFa;

        uint32_t __k = embdFa;
        uint32_t round_k = (__k + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);

        uint32_t cur_batch = 0;
        uint32_t pre_total_q_blk_num = 0;
        uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;
        uint32_t cur_total_q_blk_num;
        if (tiling == nullptr) {
            cur_total_q_blk_num = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + offset_tiling));
        } else {
            cur_total_q_blk_num = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offset_tiling));
        }
        uint32_t process_num = total_q_blk_num_fa * q_heads_fa;

        uint32_t layer_id = *((__gm__ uint32_t *)layerId_gm);
        uint64_t batch_stride_kv = batch_size_fa * max_seqlen_fa * stride_kv;
        // 指针地址
        uint64_t kCacheAddr  = *((__gm__ uint64_t *)k_gm);
        uint64_t vCacheAddr  = *((__gm__ uint64_t *)v_gm);
        __gm__ half *kCacheGm = (__gm__ half *)kCacheAddr;
        __gm__ half *vCacheGm = (__gm__ half *)vCacheAddr;
        // 指针地址
        kCacheGm = kCacheGm + layer_id * batch_stride_kv;
        vCacheGm = vCacheGm + layer_id * batch_stride_kv;

        for (uint32_t process = 0; process < process_num; process++) {
            if (process >= cur_total_q_blk_num * q_heads_fa) {
                while (cur_batch < batch_size_fa) {
                    cur_batch++;
                    pre_total_q_blk_num = cur_total_q_blk_num;
                    offset_tiling += TILING_PARA_SIZE_FA;
                    uint32_t q_seqlen;
                    if (tiling == nullptr) {
                        cur_total_q_blk_num = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + offset_tiling));
                        q_seqlen = (uint32_t)(*((int32_t *)tiling_para_gm + offset_tiling));
                    } else {
                        cur_total_q_blk_num = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offset_tiling));
                        q_seqlen = (uint32_t)(*((__gm__ int32_t *)tiling + offset_tiling));
                    }

                    if (q_seqlen != 0) {
                        break;
                    }
                }
            }
            uint32_t cur_core_idx = process % block_num;
            if (is_triu_mask_fa) {
                if ((process / block_num) % 2 == 1) {
                    cur_core_idx = block_num - process % block_num - 1;
                }
            }
            if (block_idx != cur_core_idx) {
                continue;
            }

            uint32_t q_seqlen_fa;
            uint32_t kv_seqlen_fa;
            uint32_t pp_m_scalar_fa;
            uint32_t pp_n_scalar_fa;
            uint32_t addr_q_high32_fa;
            uint32_t addr_q_loww32_fa;
            uint32_t addr_k_high32_fa;
            uint32_t addr_k_loww32_fa;
            uint32_t addr_v_high32_fa;
            uint32_t addr_v_loww32_fa;
            if (tiling == nullptr) {
                q_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + offset_tiling));
                kv_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                pp_m_scalar_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2 + offset_tiling));
                pp_n_scalar_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                addr_q_high32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 4 + offset_tiling));
                addr_q_loww32_fa= (uint32_t)(*((int32_t *)tiling_para_gm + 5 + offset_tiling));
                addr_k_high32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 6 + offset_tiling));
                addr_k_loww32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 7 + offset_tiling));
                addr_v_high32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 8 + offset_tiling));
                addr_v_loww32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 9 + offset_tiling));
            } else {
                q_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + offset_tiling));
                kv_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                pp_m_scalar_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 2 + offset_tiling));
                pp_n_scalar_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                addr_q_high32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 4 + offset_tiling));
                addr_q_loww32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 5 + offset_tiling));
                addr_k_high32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 6 + offset_tiling));
                addr_k_loww32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 7 + offset_tiling));
                addr_v_high32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 8 + offset_tiling));
                addr_v_loww32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 9 + offset_tiling));
            }
            uint64_t addr_q_scalar = (uint64_t)(((uint64_t)addr_q_high32_fa) << 32 | addr_q_loww32_fa);
            uint64_t addr_k_scalar = (uint64_t)(((uint64_t)addr_k_high32_fa) << 32 | addr_k_loww32_fa);
            uint64_t addr_v_scalar = (uint64_t)(((uint64_t)addr_v_high32_fa) << 32 | addr_v_loww32_fa);

            uint32_t process_idx = process - pre_total_q_blk_num * q_heads_fa;
            uint32_t m_idx = process_idx / q_heads_fa;
            uint32_t head_idx = process_idx % q_heads_fa;

            uint32_t m_loop = (q_seqlen_fa + pp_m_scalar_fa - 1) / pp_m_scalar_fa;
            uint32_t n_loop = (kv_seqlen_fa + pp_n_scalar_fa - 1) / pp_n_scalar_fa;

            uint32_t qk_m = (m_idx == (m_loop - 1)) ? (q_seqlen_fa - m_idx * pp_m_scalar_fa) : pp_m_scalar_fa;
            uint32_t qk_round_m = (qk_m + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

            uint64_t qk_index = 0;
            /**************** pre_load *****************/
            uint32_t qk_n = (qk_index == (n_loop - 1)) ? kv_seqlen_fa : pp_n_scalar_fa;
            uint32_t qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

            uint32_t pingpong_flag = 0;
            uint32_t offset = pingpong_flag * L0AB_HALF_BUF_SIZE_FA;

            uint32_t s_pingpong_flag = 0;
            uint32_t p_pingpong_flag = 0;
            uint32_t o_pingpong_flag = 0;

            uint64_t q_offset = addr_q_scalar + head_idx * embdFa + m_idx * pp_m_scalar_fa * stride_qo;
            uint64_t k_offset = addr_k_scalar + (head_idx / group_num) * embdFa;
            // Only need load Q once
            if (qk_m == 1) {
                copy_gm_to_cbuf((__cbuf__ half *)l1q_buf_addr, (__gm__ half *)q_gm + q_offset,
                    0, 1, round_k / BLOCK_SIZE_FA,  // sid nBurst lenBurst
                    0, 0, PAD_NONE                  // srcGap dstGap padMode
                );
            } else {
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1q_buf_addr, (__gm__ half *)q_gm + q_offset,
                    0, 1, qk_m, __k, 0,          // sid ndNum nValue dValue srcNdMatrixStride(unused)
                    stride_qo, qk_round_m, 1, 0  // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
                );
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
            wait_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
            if (qk_m == 1) {
                load_cbuf_to_ca((__ca__ half *)l0a_buf + offset, (__cbuf__ half *)l1q_buf_addr,
                    0, (round_k + CUBE_MATRIX_SIZE_FA - 1) / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                    1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                );
            } else {
                for (uint32_t l0a_load_idx = 0; l0a_load_idx < qk_round_m / BLOCK_SIZE_FA; ++l0a_load_idx) {
                    load_cbuf_to_ca(
                        (__ca__ half *)l0a_buf + offset + l0a_load_idx * round_k * BLOCK_SIZE_FA,
                        (__cbuf__ half *)l1q_buf_addr + l0a_load_idx * CUBE_MATRIX_SIZE_FA,
                        0, round_k / BLOCK_SIZE_FA, qk_round_m / BLOCK_SIZE_FA, // baseIdx repeat srcStride
                        0, 0, false, inc             // dstStride sid transpose addr_cal_mode_t
                    );
                }
            }
            // *** Prepare K to L1
            wait_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag);
            copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1k_buf_addr + offset,
                // (__gm__ half *)k_gm + k_offset,
                (__gm__ half *)kCacheGm + k_offset,
                // (__gm__ half *)kCacheGm + layer_id * batch_stride_kv + k_offset,
                0, 1, qk_n, __k, 0,          // sid ndNum nValue dValue srcNdMatrixStride(unused)
                stride_kv, qk_round_n, 1, 0  // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
            );
            set_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
            load_cbuf_to_cb((__cb__ half *)l0b_buf + offset, (__cbuf__ half *)l1k_buf_addr + offset,
                0, round_k * qk_round_n / CUBE_MATRIX_SIZE_FA, // baseIdx repeat
                1, 0, 0, false, inc                            // srcStride dstStride sid transpose addr_cal_mode_t
            );
            set_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag);
            set_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
            wait_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
            wait_flag(PIPE_FIX, PIPE_M, pingpong_flag);
            mad((__cc__ float *)l0c_buf + offset, (__ca__ half *)l0a_buf + offset,
                (__cb__ half *)l0b_buf + offset, qk_m, __k, qk_n,  // m k n
                0, 0, 0, 1  // unitFlag kDirectionAlign cmatrixSource cmatrixInitVal
            );
            set_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
            set_flag(PIPE_M, PIPE_FIX, pingpong_flag);
            wait_flag(PIPE_M, PIPE_FIX, pingpong_flag);
            // copy S to gm
            copy_matrix_cc_to_gm(
                (__gm__ half *)s_gm + (uint64_t)block_idx * TMP_SIZE_FA + s_pingpong_flag * TMP_SIZE_FA / 2,
                (__cc__ float *)l0c_buf + offset, 0, qk_round_n, qk_m,  // sid NSize MSize
                qk_round_n, qk_round_m,     // dstStride_dst_D srcStride
                0, F322F16, 0, false, true  // UnitFlagMode QuantPRE ReLUPRE channelSplit NZ2ND_EN
            );
            set_flag(PIPE_FIX, PIPE_M, pingpong_flag);
            pingpong_flag = 1 - pingpong_flag;
            offset = pingpong_flag * L0AB_HALF_BUF_SIZE_FA;
            ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY_FA << BIT_SHIFT_FA));  // 0
            s_pingpong_flag = 1 - s_pingpong_flag;
            qk_index++;

            uint32_t sv_n = pp_n_scalar_fa;
            uint32_t sv_round_n = (sv_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
            uint64_t v_offset = addr_v_scalar + (head_idx / group_num) * embdFa;
            uint32_t n_end = n_loop;
            if (is_triu_mask_fa) {
                n_end = m_idx + 1;
            }
            for (uint32_t n_idx = 0; n_idx < n_end; n_idx++) {
                if (qk_index < n_end) {
                    if (qk_index == (n_loop - 1)) {
                        qk_n = (kv_seqlen_fa - qk_index * pp_n_scalar_fa);
                        qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
                    }
                    k_offset += pp_n_scalar_fa * stride_kv;
                    wait_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
                    if (qk_m == 1) {
                        load_cbuf_to_ca((__ca__ half *)l0a_buf + offset, (__cbuf__ half *)l1q_buf_addr,
                            0, (round_k + CUBE_MATRIX_SIZE_FA - 1) / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                            1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                        );
                    } else {
                        for (uint32_t l0a_load_idx = 0; l0a_load_idx < qk_round_m / BLOCK_SIZE_FA; ++l0a_load_idx) {
                            load_cbuf_to_ca((__ca__ half *)l0a_buf + offset + l0a_load_idx * round_k * BLOCK_SIZE_FA,
                                (__cbuf__ half *)l1q_buf_addr + l0a_load_idx * CUBE_MATRIX_SIZE_FA,
                                0, round_k / BLOCK_SIZE_FA, qk_round_m / BLOCK_SIZE_FA,  // baseIdx repeat srcStride
                                0, 0, false, inc  // dstStride sid transpose addr_cal_mode_t
                            );
                        }
                    }
                    // *** Prepare K to L1
                    wait_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag);
                    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1k_buf_addr + offset,
                        (__gm__ half *)kCacheGm + k_offset, 0, 1, qk_n, __k,  // sid ndNum nValue dValue
                        0, stride_kv, qk_round_n,  // srcNdMatrixStride(unused) srcDValue dstNzC0Stride
                        1, 0                       // dstNzNStride dstNzMatrixStride(unused)
                    );
                    set_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                    load_cbuf_to_cb((__cb__ half *)l0b_buf + offset,
                        (__cbuf__ half *)l1k_buf_addr + offset,
                        0, round_k * qk_round_n / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                        1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                    );
                    set_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag);
                    set_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
                    wait_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
                    wait_flag(PIPE_FIX, PIPE_M, pingpong_flag);
                    mad((__cc__ float *)l0c_buf + offset, (__ca__ half *)l0a_buf + offset,
                        (__cb__ half *)l0b_buf + offset,
                        qk_m, __k, qk_n, 0, 0, 0, 1  // m k n unitFlag kDirectionAlign cmatrixSource cmatrixInitVal
                    );
                    set_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
                    set_flag(PIPE_M, PIPE_FIX, pingpong_flag);
                    wait_flag(PIPE_M, PIPE_FIX, pingpong_flag);
                    // copy S to gm
                    copy_matrix_cc_to_gm(
                        (__gm__ half *)s_gm + (uint64_t)block_idx * TMP_SIZE_FA + s_pingpong_flag * TMP_SIZE_FA / 2,
                        (__cc__ float *)l0c_buf + offset,
                        0, qk_round_n, qk_m, qk_round_n, qk_round_m,  // sid NSize MSize dstStride_dst_D srcStride
                        0, F322F16, 0, false, true  // UnitFlagMode QuantPRE ReLUPRE channelSplit NZ2ND_EN
                    );
                    set_flag(PIPE_FIX, PIPE_M, pingpong_flag);
                    pingpong_flag = 1 - pingpong_flag;
                    offset = pingpong_flag * L0AB_HALF_BUF_SIZE_FA;
                    ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY_FA << BIT_SHIFT_FA));  // 0
                    s_pingpong_flag = 1 - s_pingpong_flag;
                }

                if (n_idx == (n_loop - 1)) {
                    sv_n = (kv_seqlen_fa - n_idx * pp_n_scalar_fa);
                    sv_round_n = (sv_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
                }
                // *** Prepare V to L1
                wait_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag + 2);
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1v_buf_addr + offset,
                    (__gm__ half *)vCacheGm + v_offset,
                    0, 1, sv_n, __k, 0,          // sid ndNum nValue dValue srcNdMatrixStride(unused)
                    stride_kv, sv_round_n, 1, 0  // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
                );
                set_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                wait_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
                for (uint32_t l0b_load_idx = 0; l0b_load_idx < sv_round_n / BLOCK_SIZE_FA; ++l0b_load_idx) {
                    load_cbuf_to_cb((__cb__ half *)l0b_buf + offset + l0b_load_idx * round_k * BLOCK_SIZE_FA,
                        (__cbuf__ half *)l1v_buf_addr + offset + l0b_load_idx * CUBE_MATRIX_SIZE_FA,
                        0, round_k / BLOCK_SIZE_FA,  // baseIdx repeat
                        sv_round_n / BLOCK_SIZE_FA,  // srcStride
                        0, 0, true, inc              // dstStride sid transpose addr_cal_mode_t
                    );
                }
                v_offset += pp_n_scalar_fa * stride_kv;

                wait_flag_dev(SOFTMAX_READY_FA);  // 2
                // *** Prepare P to L1
                if (qk_m == 1) {
                    copy_gm_to_cbuf((__cbuf__ half *)l1p_buf_addr + offset,
                        (__gm__ half *)p_gm + (uint64_t)block_idx * TMP_SIZE_FA + p_pingpong_flag * TMP_SIZE_FA / 2,
                        0, 1, sv_round_n / BLOCK_SIZE_FA,  // sid nBurst lenBurst
                        0, 0, PAD_NONE                     // srcGap dstGap padMode
                    );
                } else {
                    copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1p_buf_addr + offset,
                        (__gm__ half *)p_gm + (uint64_t)block_idx * TMP_SIZE_FA + p_pingpong_flag * TMP_SIZE_FA / 2,
                        0, 1, qk_m, sv_n, 0,         // sid ndNum nValue dValue srcNdMatrixStride(unused)
                        sv_round_n, qk_round_m, 1, 0 // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
                    );
                }
                p_pingpong_flag = 1 - p_pingpong_flag;
                set_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, pingpong_flag);
                if (qk_m == 1) {
                    load_cbuf_to_ca((__ca__ half *)l0a_buf + offset,
                        (__cbuf__ half *)l1p_buf_addr + offset, 0,                     // baseIdx
                        (sv_round_n + CUBE_MATRIX_SIZE_FA - 1) / CUBE_MATRIX_SIZE_FA,  // repeat
                        1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                    );
                } else {
                    for (uint32_t l0a_load_idx = 0; l0a_load_idx < qk_round_m / BLOCK_SIZE_FA; ++l0a_load_idx) {
                        load_cbuf_to_ca(
                            (__ca__ half *)l0a_buf + offset + l0a_load_idx * sv_round_n * BLOCK_SIZE_FA,
                            (__cbuf__ half *)l1p_buf_addr + offset + l0a_load_idx * CUBE_MATRIX_SIZE_FA,
                            0, sv_round_n / BLOCK_SIZE_FA,  // baseIdx repeat
                            qk_round_m / BLOCK_SIZE_FA,     // srcStride
                            0, 0, false, inc                // dstStride sid transpose addr_cal_mode_t
                        );
                    }
                }
                set_flag(PIPE_MTE1, PIPE_MTE2, pingpong_flag + 2);
                set_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
                wait_flag(PIPE_MTE1, PIPE_M, pingpong_flag);
                wait_flag(PIPE_FIX, PIPE_M, pingpong_flag);
                mad((__cc__ float *)l0c_buf + offset, (__ca__ half *)l0a_buf + offset,
                    (__cb__ half *)l0b_buf + offset,
                    qk_m, sv_n, __k, 0, 0, 0, 1  // m k n unitFlag kDirectionAlign cmatrixSource cmatrixInitVal
                );
                set_flag(PIPE_M, PIPE_MTE1, pingpong_flag);
                set_flag(PIPE_M, PIPE_FIX, pingpong_flag);
                wait_flag(PIPE_M, PIPE_FIX, pingpong_flag);
                // copy O to gm
                copy_matrix_cc_to_gm(
                    (__gm__ float *)o_tmp_gm + (uint64_t)block_idx * TMP_SIZE_FA + o_pingpong_flag * TMP_SIZE_FA / 2,
                    (__cc__ float *)l0c_buf + offset,
                    0, round_k, qk_m, round_k, qk_round_m,  // sid NSize MSize dstStride_dst_D srcStride
                    0, NoQuant, 0, false, true              // UnitFlagMode QuantPRE ReLUPRE channelSplit NZ2ND_EN
                );
                set_flag(PIPE_FIX, PIPE_M, pingpong_flag);
                pingpong_flag = 1 - pingpong_flag;
                offset = pingpong_flag * L0AB_HALF_BUF_SIZE_FA;
                ffts_cross_core_sync(PIPE_FIX, 0x21 + (UPDATE_READY_FA << BIT_SHIFT_FA));  // 4
                o_pingpong_flag = 1 - o_pingpong_flag;
                qk_index++;
            }
        }
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void unpad_flashattention_encoder_mix_aiv(
        __gm__ uint8_t *__restrict__ q_gm,
        __gm__ uint8_t *__restrict__ k_gm,
        __gm__ uint8_t *__restrict__ v_gm,
        __gm__ uint8_t *__restrict__ mask_gm,
        __gm__ uint8_t *__restrict__ o_gm,
        __gm__ uint8_t *__restrict__ s_gm,
        __gm__ uint8_t *__restrict__ p_gm,
        __gm__ uint8_t *__restrict__ o_tmp_gm,
        __gm__ uint8_t *tiling,
        uint32_t *tiling_para_gm)
    {
        int32_t sub_block_idx = get_subblockid();
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);

        __ubuf__ uint8_t *ls_ubuf = (__ubuf__ uint8_t *)get_imm(0);  // 2块，存放 local S, fp16
        __ubuf__ uint8_t *lp_ubuf = (__ubuf__ uint8_t *)get_imm(0);  // 2块，存放 local P, fp16, 复用 local S fp16 空间
        __ubuf__ uint8_t *ls32_ubuf =
            (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE_PREFILL_FA);  // 1块，计算 exp 时，存放一块 S, fp32
        // 1块，存放 mask, fp16
        __ubuf__ uint8_t *mask_ubuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE_PREFILL_FA);
        // 1块，存放 local O, fp32
        __ubuf__ uint8_t *lo_ubuf = (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_PREFILL_FA);
        // 1块，存放 local m, fp16
        __ubuf__ uint8_t *lm_ubuf = (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA);
        __ubuf__ uint8_t *hm_ubuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA +
            1 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 hat m, fp16
        __ubuf__ uint8_t *gm_ubuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA +
            2 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 global m, fp32
        __ubuf__ uint8_t *dm_ubuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA +
            4 * UB_UINT8_LINE_SIZE_FA);  // 2块，存放 diff m, fp16
        __ubuf__ uint8_t *ll_ubuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA +
            6 * UB_UINT8_LINE_SIZE_FA);  // 2块，存放 local l, fp32
        __ubuf__ uint8_t *gl_ubuf =
            (__ubuf__ uint8_t *)get_imm(7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA +
            10 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 global l, fp32
        __ubuf__ uint8_t *tv_ubuf = (__ubuf__ uint8_t *)get_imm(
            7 * UB_UINT8_BLOCK_SIZE_PREFILL_FA + 11 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放临时 vector 计算变量
        __ubuf__ uint8_t *go_ubuf =
            (__ubuf__ uint8_t *)get_imm(8 * UB_UINT8_BLOCK_SIZE_PREFILL_FA);  // 1块，存放 global O, fp32
        uint32_t go_flag_scalar = 1;

        uint32_t batch_size_fa;
        uint32_t max_seqlen_fa;
        uint32_t q_heads_fa;
        uint32_t embdFa;
        half torFa;
        uint32_t mask_stride_fa;
        uint32_t is_triu_mask_fa;
        uint32_t total_q_blk_num_fa;
        uint32_t isClamp;
        half clampMin;
        half clampMax;
        uint32_t head_stride;

        if (tiling == nullptr) {
            batch_size_fa = (uint32_t)(*((int32_t *)tiling_para_gm));
            max_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1));
            q_heads_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2));
            embdFa = (uint32_t)(*((int32_t *)tiling_para_gm + 3));
            torFa = (half)(*((float *)tiling_para_gm + 5));
            mask_stride_fa = (uint32_t)(*((uint32_t *)tiling_para_gm + 7));
            is_triu_mask_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 8));
            total_q_blk_num_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 9));
            isClamp = (uint32_t)(*((uint32_t *)tiling_para_gm + 10));
            clampMin = (half)(*((float *)tiling_para_gm + 11));
            clampMax = (half)(*((float *)tiling_para_gm + 12));
            head_stride = (uint32_t)(*((uint32_t *)tiling_para_gm + 13));
        } else {
            batch_size_fa = (uint32_t)(*((__gm__ int32_t *)tiling));
            max_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1));
            q_heads_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 2));
            embdFa = (uint32_t)(*((__gm__ int32_t *)tiling + 3));
            torFa = (half)(*((__gm__ float *)tiling + 5));
            mask_stride_fa = (uint32_t)(*((__gm__ uint32_t *)tiling + 7));
            is_triu_mask_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 8));
            total_q_blk_num_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 9));
            isClamp = (uint32_t)(*((__gm__ uint32_t *)tiling + 10));
            clampMin = (half)(*((__gm__ float *)tiling + 11));
            clampMax = (half)(*((__gm__ float *)tiling + 12));
            head_stride = (uint32_t)(*((__gm__ uint32_t *)tiling + 13));
        }

        uint64_t stride_qo = q_heads_fa * embdFa;

        uint32_t __k = embdFa;
        uint32_t round_k = (__k + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);

        uint32_t cur_batch = 0;
        uint32_t pre_total_q_blk_num = 0;
        uint32_t offset_tiling = TILING_HEAD_SIZE_FA + TILING_PARA_SIZE_FA * cur_batch;
        uint32_t cur_total_q_blk_num;
        if (tiling == nullptr) {
            cur_total_q_blk_num = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + offset_tiling));
        } else {
            cur_total_q_blk_num = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offset_tiling));
        }
        uint32_t process_num = total_q_blk_num_fa * q_heads_fa;
        for (uint32_t process = 0; process < process_num; process++) {
            if (process >= cur_total_q_blk_num * q_heads_fa) {
                while (cur_batch < batch_size_fa) {
                    cur_batch++;
                    pre_total_q_blk_num = cur_total_q_blk_num;
                    offset_tiling += TILING_PARA_SIZE_FA;
                    uint32_t q_seqlen_fa;
                    if (tiling == nullptr) {
                        cur_total_q_blk_num = (uint32_t)(*((int32_t *)tiling_para_gm + 13 + offset_tiling));
                        q_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + offset_tiling));
                    } else {
                        cur_total_q_blk_num = (uint32_t)(*((__gm__ int32_t *)tiling + 13 + offset_tiling));
                        q_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + offset_tiling));
                    }

                    if (q_seqlen_fa != 0) {
                        break;
                    }
                }
            }
            uint32_t cur_core_idx = process % block_num;
            if (is_triu_mask_fa) {
                if ((process / block_num) % 2 == 1) {
                    cur_core_idx = block_num - process % block_num - 1;
                }
            }
            if (block_idx != cur_core_idx) {
                continue;
            }

            uint32_t q_seqlen_fa;
            uint32_t kv_seqlen_fa;
            uint32_t pp_m_scalar_fa;
            uint32_t pp_n_scalar_fa;
            uint32_t addr_o_high32_fa;
            uint32_t addr_o_loww32_fa;
            if (tiling == nullptr) {
                q_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + offset_tiling));
                kv_seqlen_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 1 + offset_tiling));
                pp_m_scalar_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 2 + offset_tiling));
                pp_n_scalar_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 3 + offset_tiling));
                addr_o_high32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 10 + offset_tiling));
                addr_o_loww32_fa = (uint32_t)(*((int32_t *)tiling_para_gm + 11 + offset_tiling));
            } else {
                q_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + offset_tiling));
                kv_seqlen_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 1 + offset_tiling));
                pp_m_scalar_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 2 + offset_tiling));
                pp_n_scalar_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 3 + offset_tiling));
                addr_o_high32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 10 + offset_tiling));
                addr_o_loww32_fa = (uint32_t)(*((__gm__ int32_t *)tiling + 11 + offset_tiling));
            }
            uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32_fa) << 32 | addr_o_loww32_fa);

            uint32_t process_idx = process - pre_total_q_blk_num * q_heads_fa;
            uint32_t m_idx = process_idx / q_heads_fa;
            uint32_t head_idx = process_idx % q_heads_fa;

            uint32_t m_loop = (q_seqlen_fa + pp_m_scalar_fa - 1) / pp_m_scalar_fa;
            uint32_t n_loop = (kv_seqlen_fa + pp_n_scalar_fa - 1) / pp_n_scalar_fa;

            uint32_t qk_m = (m_idx == (m_loop - 1)) ? (q_seqlen_fa - m_idx * pp_m_scalar_fa) : pp_m_scalar_fa;
            uint32_t sub_m = (sub_block_idx == 1) ? (qk_m - qk_m / 2) : qk_m / 2;
            uint32_t sub_m_d128 = (sub_m + VECTOR_SIZE - 1) / VECTOR_SIZE;                   // up aligned to 128
            uint32_t sub_m_d64 = (sub_m + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA;  // up aligned to 64
            uint32_t round_sub_m = (sub_m + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

            uint64_t qk_index = 0;
            /******** pre_load *******/
            uint32_t qk_n = (qk_index == (n_loop - 1)) ? kv_seqlen_fa : pp_n_scalar_fa;
            uint32_t qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

            uint32_t pingpong_flag = 0;
            uint32_t offset = pingpong_flag * UB_HALF_BUF_SIZE_FA;

            uint64_t mask_batch_offset = 0;
            uint64_t mask_head_offset = 0;
            uint64_t mask_offset = mask_batch_offset + mask_head_offset + m_idx * pp_m_scalar_fa * max_seqlen_fa;

            uint32_t s_pingpong_flag = 0;
            uint32_t p_pingpong_flag = 0;
            uint32_t o_pingpong_flag = 0;

            if (sub_m > 0 && mask_gm != nullptr) {
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                uint64_t offset_1 = sub_block_idx * qk_m / 2 * max_seqlen_fa;
                copy_gm_to_ubuf_align_b16(
                    (__ubuf__ half *)mask_ubuf, (__gm__ half *)mask_gm + mask_offset + offset_1,
                    0, sub_m, qk_n * 2, 0,            // sid nBurst lenBurst leftPaddingNum
                    0, (max_seqlen_fa - qk_n) * 2, 0  // rightPaddingNum srcGap dstGap
                );
            }
            wait_flag_dev(QK_READY_FA);
            if (sub_m > 0) {
                wait_flag(PIPE_MTE3, PIPE_MTE2, pingpong_flag);
                // input QK
                uint64_t offset_2 = sub_block_idx * qk_m / 2 * qk_round_n;
                copy_gm_to_ubuf((__ubuf__ half *)ls_ubuf + offset,
                    (__gm__ half *)s_gm + (uint64_t)block_idx * TMP_SIZE_FA + s_pingpong_flag * TMP_SIZE_FA / 2 +
                        offset_2,
                    0, 1, sub_m * qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                s_pingpong_flag = 1 - s_pingpong_flag;
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                vmuls((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset, torFa,
                    (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE,  // repeat
                    1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                if (mask_gm != nullptr) {
                    VecAddFormula((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset,
                        (__ubuf__ half *)mask_ubuf, (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE, // repeat
                        1, 1, 1, 8, 8, 8  // 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                }
                if (qk_n <= VECTOR_SIZE) {
                    SetMask(qk_n);
                    vcmax((__ubuf__ half *)lm_ubuf, (__ubuf__ half *)ls_ubuf + offset, sub_m, // repeat
                        1, 1, qk_round_n / BLOCK_SIZE_FA, ONLY_VALUE  // 1:dstRepeatStride srcRepeatStride order
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    CopyUbufToUbufFormula((__ubuf__ half *)ls32_ubuf, (__ubuf__ half *)ls_ubuf + offset,
                        0, sub_m, VECTOR_SIZE / BLOCK_SIZE_FA,         // sid nBurst lenBurst
                        (qk_round_n - VECTOR_SIZE) / BLOCK_SIZE_FA, 0  // srcGap dstGap
                    );
                    pipe_barrier(PIPE_V);
                    SetMask(qk_n - VECTOR_SIZE);
                    VecMaxFormula((__ubuf__ half *)ls32_ubuf, (__ubuf__ half *)ls32_ubuf,
                        (__ubuf__ half *)ls_ubuf + offset + VECTOR_SIZE, sub_m,  // repeat
                        1, 1, 1, 8, 8, qk_round_n / BLOCK_SIZE_FA // 1:dstBlockStride 8:dstRepeatStride src1RepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    vcmax((__ubuf__ half *)lm_ubuf, (__ubuf__ half *)ls32_ubuf,
                        sub_m, 1, 1, 8, ONLY_VALUE  // repeat dstBlockStride srcBlockStride srcRepeatStride order
                    );
                }
                pipe_barrier(PIPE_V);
                CopyUbufToUbufFormula((__ubuf__ half *)hm_ubuf, (__ubuf__ half *)lm_ubuf,
                    0, 1, round_sub_m / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                pipe_barrier(PIPE_V);
                CopyUbufToUbufFormula((__ubuf__ half *)gm_ubuf, (__ubuf__ half *)hm_ubuf,
                    0, 1, round_sub_m / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                pipe_barrier(PIPE_V);
                vbrcb((__ubuf__ uint16_t *)tv_ubuf, (__ubuf__ uint16_t *)hm_ubuf,
                    1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                );
                pipe_barrier(PIPE_V);
                for (uint32_t vsub_idx = 0; vsub_idx < qk_n / VECTOR_SIZE; ++vsub_idx) {
                    VecSubFormula((__ubuf__ half *)ls_ubuf + offset + vsub_idx * VECTOR_SIZE,
                        (__ubuf__ half *)ls_ubuf + offset + vsub_idx * VECTOR_SIZE, (__ubuf__ half *)tv_ubuf,
                        sub_m, 1, 1, 0,  // repeat dstBlockStride src0BlockStride src1BlockStride
                        qk_round_n / BLOCK_SIZE_FA, qk_round_n / BLOCK_SIZE_FA, // dstRepeatStride src0RepeatStride
                        1  // src1RepeatStride
                    );
                }
                if (qk_n % VECTOR_SIZE > 0) {
                    SetMask(qk_n % VECTOR_SIZE);
                    VecSubFormula((__ubuf__ half *)ls_ubuf + offset + qk_n / VECTOR_SIZE * VECTOR_SIZE,
                        (__ubuf__ half *)ls_ubuf + offset + qk_n / VECTOR_SIZE * VECTOR_SIZE,
                        (__ubuf__ half *)tv_ubuf, sub_m, 1, 1, 0,  // repeat 1:dstBlockStride 0:src1BlockStride
                        qk_round_n / BLOCK_SIZE_FA, qk_round_n / BLOCK_SIZE_FA,  // dstRepeatStride src0RepeatStride
                        1  // src1RepeatStride
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                pipe_barrier(PIPE_V);
                vconv_f162f32((__ubuf__ float *)ls32_ubuf, (__ubuf__ half *)ls_ubuf + offset,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 8, 4  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                VecExpFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                vconv_f322f16((__ubuf__ half *)lp_ubuf + offset, (__ubuf__ float *)ls32_ubuf,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                if (qk_n <= FLOAT_VECTOR_SIZE_FA) {
                    SetMask(qk_n);
                    vcadd((__ubuf__ float *)ll_ubuf, (__ubuf__ float *)ls32_ubuf, sub_m,  // repeat
                        1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA, 0  // 1:dstRepeatStride srcRepeatStride mode
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++rowsum_idx) {
                        VecAddFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                            (__ubuf__ float *)ls32_ubuf + rowsum_idx * FLOAT_VECTOR_SIZE_FA, sub_m,  // repeat
                            1, 1, 1,                           // 1:dstBlockStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                    }
                    if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                        SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                        VecAddFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                            (__ubuf__ float *)ls32_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1,                    // repeat 1:dstBlockStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcadd((__ubuf__ float *)ll_ubuf, (__ubuf__ float *)ls32_ubuf, sub_m,  // repeat
                        1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA, 0  // 1:dstRepeatStride srcRepeatStride order
                    );
                }
                pipe_barrier(PIPE_V);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                uint64_t offset_3 = sub_block_idx * qk_m / 2 * qk_round_n;
                copy_ubuf_to_gm(
                    (__gm__ half *)p_gm + (uint64_t)block_idx * TMP_SIZE_FA + p_pingpong_flag * TMP_SIZE_FA / 2 +
                        offset_3,
                    (__ubuf__ half *)lp_ubuf + offset,
                    0, 1, sub_m * qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                set_flag(PIPE_MTE3, PIPE_MTE2, pingpong_flag);
                pingpong_flag = 1 - pingpong_flag;
                offset = pingpong_flag * UB_HALF_BUF_SIZE_FA;
                p_pingpong_flag = 1 - p_pingpong_flag;
            }
            qk_index++;
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY_FA << BIT_SHIFT_FA));  // 2

            uint64_t o_offset = addr_o_scalar + head_idx * embdFa + m_idx * pp_m_scalar_fa * stride_qo;
            uint32_t n_end = n_loop;
            if (is_triu_mask_fa) {
                n_end = m_idx + 1;
            }
            for (uint32_t n_idx = 0; n_idx < n_end; n_idx++) {
                if (qk_index < n_end) {
                    if (qk_index == (n_loop - 1)) {
                        qk_n = (kv_seqlen_fa - qk_index * pp_n_scalar_fa);
                        qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
                    }
                    if (sub_m > 0 && mask_gm != nullptr) {
                        mask_offset += pp_n_scalar_fa;
                        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                        uint64_t offset_4 = sub_block_idx * qk_m / 2 * max_seqlen_fa;
                        copy_gm_to_ubuf_align_b16((__ubuf__ half *)mask_ubuf,
                            (__gm__ half *)mask_gm + mask_offset + offset_4,
                            0, sub_m, qk_n * 2, 0,            // sid nBurst lenBurst leftPaddingNum
                            0, (max_seqlen_fa - qk_n) * 2, 0  // rightPaddingNum srcGap dstGap
                        );
                    }
                    wait_flag_dev(QK_READY_FA);
                    if (sub_m > 0) {
                        wait_flag(PIPE_MTE3, PIPE_MTE2, pingpong_flag);
                        // input QK
                        uint64_t offset_5 = sub_block_idx * qk_m / 2 * qk_round_n;
                        copy_gm_to_ubuf((__ubuf__ half *)ls_ubuf + offset,
                            (__gm__ half *)s_gm +
                            (uint64_t)block_idx * TMP_SIZE_FA + s_pingpong_flag * TMP_SIZE_FA / 2 + offset_5,
                            0, 1, sub_m * qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                        );
                        s_pingpong_flag = 1 - s_pingpong_flag;
                        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                        vmuls((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset,
                            torFa, (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE,  // repeat
                            1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);

                        if (isClamp == 1) {
                            vmaxs((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset,
                                clampMin, (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE,  // repeat
                                1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                            );
                            pipe_barrier(PIPE_V);

                            vmins((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset,
                                clampMax, (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE,  // repeat
                                1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                            );
                            pipe_barrier(PIPE_V);
                        }

                        if (mask_gm != nullptr) {
                            VecAddFormula((__ubuf__ half *)ls_ubuf + offset, (__ubuf__ half *)ls_ubuf + offset,
                                (__ubuf__ half *)mask_ubuf, (sub_m * qk_round_n + VECTOR_SIZE - 1) / VECTOR_SIZE,
                                1, 1, 1, 8, 8, 8  // 1:dstBlockStride 8:dstRepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
                        }
                        if (qk_n <= VECTOR_SIZE) {
                            SetMask(qk_n);
                            vcmax((__ubuf__ half *)lm_ubuf, (__ubuf__ half *)ls_ubuf + offset, sub_m,  // repeat
                                1, 1, qk_round_n / BLOCK_SIZE_FA, ONLY_VALUE // dstRepeatStride srcRepeatStride order
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        } else {
                            CopyUbufToUbufFormula((__ubuf__ half *)ls32_ubuf, (__ubuf__ half *)ls_ubuf + offset,
                                0, sub_m, VECTOR_SIZE / BLOCK_SIZE_FA,         // sid nBurst lenBurst
                                (qk_round_n - VECTOR_SIZE) / BLOCK_SIZE_FA, 0  // srcGap dstGap
                            );
                            pipe_barrier(PIPE_V);
                            SetMask(qk_n - VECTOR_SIZE);
                            VecMaxFormula((__ubuf__ half *)ls32_ubuf, (__ubuf__ half *)ls32_ubuf,
                                (__ubuf__ half *)ls_ubuf + offset + VECTOR_SIZE, sub_m,  // repeat
                                1, 1, 1, 8, 8,              // 1:dstBlockStride 8:dstRepeatStride
                                qk_round_n / BLOCK_SIZE_FA  // src1RepeatStride
                            );
                            pipe_barrier(PIPE_V);
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            vcmax((__ubuf__ half *)lm_ubuf, (__ubuf__ half *)ls32_ubuf, sub_m,  // repeat
                                1, 1, 8, ONLY_VALUE  // dstRepeatStride srcBlockStride srcRepeatStride order
                            );
                        }
                        pipe_barrier(PIPE_V);
                        VecMaxFormula((__ubuf__ half *)hm_ubuf, (__ubuf__ half *)lm_ubuf, (__ubuf__ half *)gm_ubuf,
                            sub_m_d128, 1, 1, 1, 8, 8, 8  // repeat 1:dstBlockStride 8:dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        VecSubFormula((__ubuf__ half *)dm_ubuf + qk_index % 2 * UB_HALF_LINE_SIZE_FA,
                            (__ubuf__ half *)gm_ubuf, (__ubuf__ half *)hm_ubuf, sub_m_d128,  // repeat
                            1, 1, 1, 8, 8, 8  // 1:dstBlockStride 8:dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        CopyUbufToUbufFormula((__ubuf__ half *)gm_ubuf, (__ubuf__ half *)hm_ubuf,
                            0, 1, round_sub_m / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                        );
                        pipe_barrier(PIPE_V);
                        vbrcb((__ubuf__ uint16_t *)tv_ubuf, (__ubuf__ uint16_t *)hm_ubuf,
                            1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                        );
                        pipe_barrier(PIPE_V);
                        for (uint32_t vsub_idx = 0; vsub_idx < qk_n / VECTOR_SIZE; ++vsub_idx) {
                            VecSubFormula((__ubuf__ half *)ls_ubuf + offset + vsub_idx * VECTOR_SIZE,
                                (__ubuf__ half *)ls_ubuf + offset + vsub_idx * VECTOR_SIZE,
                                (__ubuf__ half *)tv_ubuf, sub_m, 1, 1, 0,  // repeat 1:dstBlockStride 0:src1BlockStride
                                qk_round_n / BLOCK_SIZE_FA,                // dstRepeatStride
                                qk_round_n / BLOCK_SIZE_FA, 1              // src0RepeatStride src1RepeatStride
                            );
                        }
                        if (qk_n % VECTOR_SIZE > 0) {
                            SetMask(qk_n % VECTOR_SIZE);
                            VecSubFormula((__ubuf__ half *)ls_ubuf + offset + qk_n / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)ls_ubuf + offset + qk_n / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)tv_ubuf, sub_m, 1, 1, 0,  // repeat 1:dstBlockStride 0:src1BlockStride
                                qk_round_n / BLOCK_SIZE_FA,                // dstRepeatStride
                                qk_round_n / BLOCK_SIZE_FA, 1              // src0RepeatStride src1RepeatStride
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        pipe_barrier(PIPE_V);
                        vconv_f162f32((__ubuf__ float *)ls32_ubuf, (__ubuf__ half *)ls_ubuf + offset,
                            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                            1, 1, 8, 4  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        VecExpFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                            1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        vconv_f322f16((__ubuf__ half *)lp_ubuf + offset, (__ubuf__ float *)ls32_ubuf,
                            (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                            1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        if (qk_n <= FLOAT_VECTOR_SIZE_FA) {
                            SetMask(qk_n);
                            vcadd((__ubuf__ float *)ll_ubuf + qk_index % 2 * UB_FLOAT_LINE_SIZE_FA,
                                (__ubuf__ float *)ls32_ubuf, sub_m, 1, 1,  // repeat dstRepeatStride srcBlockStride
                                qk_round_n / FLOAT_BLOCK_SIZE_FA, 0        // srcRepeatStride mode
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        } else {
                            for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++rowsum_idx) {
                                VecAddFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                                    (__ubuf__ float *)ls32_ubuf + rowsum_idx * FLOAT_VECTOR_SIZE_FA,
                                    sub_m, 1, 1, 1,  // repeat dstBlockStride src0BlockStride src1BlockStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                                );
                                pipe_barrier(PIPE_V);
                            }
                            if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                                SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                                VecAddFormula((__ubuf__ float *)ls32_ubuf, (__ubuf__ float *)ls32_ubuf,
                                    (__ubuf__ float *)ls32_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                                    sub_m, 1, 1, 1,  // repeat dstBlockStride src0BlockStride src1BlockStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                                    qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                                );
                                set_vector_mask((uint64_t)-1, (uint64_t)-1);
                            }
                            pipe_barrier(PIPE_V);
                            vcadd((__ubuf__ float *)ll_ubuf + qk_index % 2 * UB_FLOAT_LINE_SIZE_FA,
                                (__ubuf__ float *)ls32_ubuf, sub_m, 1, 1,  // repeat dstRepeatStride srcBlockStride
                                qk_round_n / FLOAT_BLOCK_SIZE_FA, 0        // srcRepeatStride order
                            );
                        }
                        pipe_barrier(PIPE_V);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        uint64_t offset_6 = sub_block_idx * qk_m / 2 * qk_round_n;
                        copy_ubuf_to_gm((__gm__ half *)p_gm +
                            (uint64_t)block_idx * TMP_SIZE_FA + p_pingpong_flag * TMP_SIZE_FA / 2 + offset_6,
                            (__ubuf__ half *)lp_ubuf + offset,
                            0, 1, sub_m * qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                        );
                        set_flag(PIPE_MTE3, PIPE_MTE2, pingpong_flag);
                        pingpong_flag = 1 - pingpong_flag;
                        offset = pingpong_flag * UB_HALF_BUF_SIZE_FA;
                        p_pingpong_flag = 1 - p_pingpong_flag;
                    }
                    qk_index++;
                    ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY_FA << BIT_SHIFT_FA));  // 2
                }

                wait_flag_dev(UPDATE_READY_FA);  // 4
                if (sub_m > 0) {
                    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                    uint64_t offset_7 = sub_block_idx * qk_m / 2 * round_k;
                    copy_gm_to_ubuf((__ubuf__ float *)lo_ubuf,
                        (__gm__ float *)o_tmp_gm +
                        (uint64_t)block_idx * TMP_SIZE_FA + o_pingpong_flag * TMP_SIZE_FA / 2 + offset_7,
                        0, 1, sub_m * round_k / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                    );
                    o_pingpong_flag = 1 - o_pingpong_flag;
                    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                    // *** 更新 L 和 O
                    if (n_idx != 0) {
                        // *** dm32 = castfp16to32(dm), 存放于 tv
                        vconv_f162f32((__ubuf__ float *)tv_ubuf,
                            (__ubuf__ half *)dm_ubuf + n_idx % 2 * UB_HALF_LINE_SIZE_FA, sub_m_d64,  // repeat
                            1, 1, 8, 4  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        vbrcb((__ubuf__ uint32_t *)tv_ubuf + VECTOR_SIZE, (__ubuf__ uint32_t *)tv_ubuf,
                            1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                        );
                        pipe_barrier(PIPE_V);
                        VecExpFormula((__ubuf__ float *)tv_ubuf, (__ubuf__ float *)tv_ubuf, sub_m_d64,  // repeat
                            1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        VecMulFormula((__ubuf__ float *)gl_ubuf, (__ubuf__ float *)tv_ubuf,
                            (__ubuf__ float *)gl_ubuf, sub_m_d64,  // repeat
                            1, 1, 1, 8, 8, 8  // 1: dstBlockStride 8: dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        VecAddFormula((__ubuf__ float *)gl_ubuf, (__ubuf__ float *)gl_ubuf,
                            (__ubuf__ float *)ll_ubuf + n_idx % 2 * UB_FLOAT_LINE_SIZE_FA,
                            sub_m_d64, 1, 1, 1, 8, 8, 8  // repeat 1:dstBlockStride 8:dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        VecExpFormula((__ubuf__ float *)tv_ubuf + VECTOR_SIZE, (__ubuf__ float *)tv_ubuf + VECTOR_SIZE,
                            (sub_m * FLOAT_BLOCK_SIZE_FA + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA, // repeat
                            1, 1, 8, 8  // 1: dstBlockStride 8: dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        if (go_flag_scalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 0;
                        }
                        for (uint32_t vmul_idx = 0; vmul_idx < __k / FLOAT_VECTOR_SIZE_FA; ++vmul_idx) {
                            VecMulFormula((__ubuf__ float *)go_ubuf + vmul_idx * FLOAT_VECTOR_SIZE_FA,
                                (__ubuf__ float *)go_ubuf + vmul_idx * FLOAT_VECTOR_SIZE_FA,
                                (__ubuf__ float *)tv_ubuf + VECTOR_SIZE, sub_m,  // repeat
                                1, 1, 0,                          // dstBlockStride src0RepeatStride src1BlockStride
                                round_k / FLOAT_BLOCK_SIZE_FA,    // dstRepeatStride
                                round_k / FLOAT_BLOCK_SIZE_FA, 1  // src0RepeatStride src1RepeatStride
                            );
                        }
                        if (__k % FLOAT_VECTOR_SIZE_FA > 0) {
                            SetMask(__k % FLOAT_VECTOR_SIZE_FA);
                            VecMulFormula((__ubuf__ float *)go_ubuf + __k / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                                (__ubuf__ float *)go_ubuf + __k / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                                (__ubuf__ float *)tv_ubuf + VECTOR_SIZE, sub_m,  // repeat
                                1, 1, 0,                          // dstBlockStride src0BlockStride src1BlockStride
                                round_k / FLOAT_BLOCK_SIZE_FA,    // dstRepeatStride
                                round_k / FLOAT_BLOCK_SIZE_FA, 1  // src0RepeatStride src1RepeatStride
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        pipe_barrier(PIPE_V);
                        VecAddFormula((__ubuf__ float *)go_ubuf,
                            (__ubuf__ float *)go_ubuf, (__ubuf__ float *)lo_ubuf,
                            (sub_m * round_k + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                            1, 1, 1, 8, 8, 8  // 1: dstBlockStride 8: dstRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                    } else {
                        CopyUbufToUbufFormula((__ubuf__ float *)gl_ubuf,
                            (__ubuf__ float *)ll_ubuf + n_idx % 2 * UB_FLOAT_LINE_SIZE_FA,
                            0, 1, round_sub_m / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                        );
                        pipe_barrier(PIPE_V);
                        if (go_flag_scalar == 1) {
                            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 0;
                        }
                        CopyUbufToUbufFormula((__ubuf__ float *)go_ubuf, (__ubuf__ float *)lo_ubuf,
                            0, 1, sub_m * round_k / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                        );
                        pipe_barrier(PIPE_V);
                    }
                    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
                    if (n_idx == n_end - 1) {
                        vconv_f322f16((__ubuf__ half *)gl_ubuf, (__ubuf__ float *)gl_ubuf, sub_m_d64,  // repeat
                            1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        vconv_f322f16((__ubuf__ half *)go_ubuf, (__ubuf__ float *)go_ubuf,
                            (sub_m * round_k + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                            1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                        );
                        pipe_barrier(PIPE_V);
                        vbrcb((__ubuf__ uint16_t *)tv_ubuf, (__ubuf__ uint16_t *)gl_ubuf,
                            1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                        );
                        pipe_barrier(PIPE_V);
                        for (uint32_t vdiv_idx = 0; vdiv_idx < __k / VECTOR_SIZE; ++vdiv_idx) {
                            VecDivFormula((__ubuf__ half *)go_ubuf + vdiv_idx * VECTOR_SIZE,
                                (__ubuf__ half *)go_ubuf + vdiv_idx * VECTOR_SIZE, (__ubuf__ half *)tv_ubuf,
                                sub_m, 1, 1, 0,             // repeat dstBlockStride src0BlockStride src1BlockStride
                                round_k / BLOCK_SIZE_FA,    // dstRepeatStride
                                round_k / BLOCK_SIZE_FA, 1  // src0RepeatStride src1RepeatStride
                            );
                        }
                        if (__k % VECTOR_SIZE > 0) {
                            SetMask(__k % VECTOR_SIZE);
                            VecDivFormula((__ubuf__ half *)go_ubuf + __k / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)go_ubuf + __k / VECTOR_SIZE * VECTOR_SIZE,
                                (__ubuf__ half *)tv_ubuf, sub_m,  // repeat
                                1, 1, 0,                    // dstBlockStride src0BlockStride src1BlockStride
                                round_k / BLOCK_SIZE_FA,    // dstRepeatStride
                                round_k / BLOCK_SIZE_FA, 1  // src0RepeatStride src1RepeatStride
                            );
                            set_vector_mask((uint64_t)-1, (uint64_t)-1);
                        }
                        // ******************** move O to GM ***********************
                        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                        uint64_t offset_8 = sub_block_idx * qk_m / 2 * stride_qo;
                        copy_ubuf_to_gm_align_b16((__gm__ half *)o_gm + o_offset + offset_8,
                            (__ubuf__ half *)go_ubuf, 0, sub_m, __k * 2,  // sid nBurst lenBurst
                            0, 0, 0, (stride_qo - __k) * 2  // leftPaddingNum rightPaddingNum srcGap dstGap
                        );
                        if (go_flag_scalar == 0) {
                            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                            go_flag_scalar = 1;
                        }
                    }
                }
            }
        }
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void RunAic(uint32_t cur_batch_fa, uint32_t start_head_fa,
                                  uint32_t cur_head_num_fa, uint32_t head_split_loop_fa, uint32_t offset_tiling_fa,
                                  uint32_t embdFa, uint32_t head_split_num_fa, uint32_t group_num_fa,
                                  uint64_t stride_kv_fa,
                                  uint32_t __k_fa, uint32_t round_k_fa,
                                  uint32_t kv_seqlen_fa, uint32_t pp_n_scalar_fa, uint32_t addr_q_high32_fa,
                                  uint32_t addr_q_loww32_fa, uint32_t addr_k_high32_fa, uint32_t addr_k_loww32_fa,
                                  uint32_t addr_v_high32_fa, uint32_t addr_v_loww32_fa,
                                  __gm__ uint8_t* q_gm_fa,
                                  __gm__ half* k_gm_fa,
                                  __gm__ half* v_gm_fa,
                                  __gm__ uint8_t* s_gm_fa,
                                  __gm__ uint8_t* p_gm_fa,
                                  __gm__ uint8_t* o_tmp_gm_fa)
    {
        uint64_t addr_q_scalar = (uint64_t)(((uint64_t)addr_q_high32_fa) << 32 | addr_q_loww32_fa);
        uint64_t addr_k_scalar = (uint64_t)(((uint64_t)addr_k_high32_fa) << 32 | addr_k_loww32_fa);
        uint64_t addr_v_scalar = (uint64_t)(((uint64_t)addr_v_high32_fa) << 32 | addr_v_loww32_fa);

        __cbuf__ uint8_t *l1q_buf_addr = (__cbuf__ uint8_t *)get_imm(0);                             // 2 block
        __cbuf__ uint8_t *l1p_buf_addr = (__cbuf__ uint8_t *)get_imm(2 * L0AB_UINT8_BLOCK_SIZE_FA);  // 2 block
        __cbuf__ uint8_t *l1kv_buf_addr = (__cbuf__ uint8_t *)get_imm(4 * L0AB_UINT8_BLOCK_SIZE_FA); // 8 block
        __ca__ uint8_t *l0a_buf = (__ca__ uint8_t *)get_imm(0);
        __cb__ uint8_t *l0b_buf = (__cb__ uint8_t *)get_imm(0);
        __cc__ uint8_t *l0c_buf = (__cc__ uint8_t *)get_imm(0);

        uint32_t n_loop = (kv_seqlen_fa + pp_n_scalar_fa - 1) / pp_n_scalar_fa;

        uint64_t q_offset = addr_q_scalar + start_head_fa * embdFa;
        uint64_t k_offset = addr_k_scalar;
        uint64_t v_offset = addr_v_scalar;

        uint32_t qk_n = pp_n_scalar_fa;
        uint32_t qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        uint32_t l1_pingpong_flag = 0;
        uint32_t l0_pingpong_flag = 0;
        uint32_t l1_offset = l1_pingpong_flag * L1_HALF_BUF_SIZE_FA;
        uint32_t l0_offset = l0_pingpong_flag * L0AB_HALF_BUF_SIZE_FA;

        for (uint32_t n_idx = 0; n_idx < n_loop; n_idx++) {
            if (n_idx == (n_loop - 1)) {
                qk_n = (kv_seqlen_fa - n_idx * pp_n_scalar_fa);
                qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
            }
            for (uint32_t split_idx = 0; split_idx < head_split_loop_fa; ++split_idx) {
                // Only need load Q once
                uint32_t head_num_move =
                    (split_idx == (head_split_loop_fa - 1)) ?
                     cur_head_num_fa - head_split_num_fa * split_idx : head_split_num_fa;
                if (n_idx == 0 && split_idx == 0) {
                    if (embdFa % BLOCK_SIZE_FA == 0) {
                        copy_gm_to_cbuf((__cbuf__ half *)l1q_buf_addr,
                            (__gm__ half *)q_gm_fa + q_offset, 0, 1,       // sid nBurst
                            round_k_fa * cur_head_num_fa / BLOCK_SIZE_FA,  // lenBurst
                            0, 0, PAD_NONE                                 // srcGap dstGap padMode
                        );
                    } else {
                        for (uint32_t copy_idx = 0; copy_idx < cur_head_num_fa; copy_idx++) {
                            copy_gm_to_cbuf((__cbuf__ half *)l1q_buf_addr + copy_idx * round_k_fa,
                                (__gm__ half *)q_gm_fa + q_offset + copy_idx * embdFa,
                                0, 1, round_k_fa / BLOCK_SIZE_FA,  // sid nBurst lenBurst
                                0, 0, PAD_NONE                     // srcGap dstGap padMode
                            );
                        }
                    }
                }
                // *** Prepare K to L1
                wait_flag(PIPE_MTE1, PIPE_MTE2, l1_pingpong_flag);
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1kv_buf_addr + l1_offset,
                    (__gm__ half *)k_gm_fa + k_offset +
                    (start_head_fa + split_idx * head_split_num_fa) / group_num_fa * embdFa,
                    0, 1, qk_n, __k_fa * head_num_move, 0, // sid ndNum nValue dValue srcNdMatrixStride(unused)
                    stride_kv_fa, qk_round_n, 1, 0 // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
                );
                set_flag(PIPE_MTE2, PIPE_MTE1, l1_pingpong_flag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, l1_pingpong_flag);
                for (uint32_t headdim_idx = 0; headdim_idx < head_num_move; headdim_idx++) {
                    wait_flag(PIPE_M, PIPE_MTE1, l0_pingpong_flag);
                    load_cbuf_to_ca((__ca__ half *)l0a_buf + l0_offset,
                        (__cbuf__ half *)l1q_buf_addr + split_idx * head_split_num_fa * round_k_fa +
                        headdim_idx * round_k_fa,
                        0, (round_k_fa + CUBE_MATRIX_SIZE_FA - 1) / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                        1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                    );
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    load_cbuf_to_cb((__cb__ half *)l0b_buf,
                        (__cbuf__ half *)l1kv_buf_addr + l1_offset + headdim_idx * round_k_fa * qk_round_n,
                        0, round_k_fa * qk_round_n / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                        1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                    );
                    if (headdim_idx == head_num_move - 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, l1_pingpong_flag);
                    }
                    set_flag(PIPE_MTE1, PIPE_M, l0_pingpong_flag);
                    wait_flag(PIPE_MTE1, PIPE_M, l0_pingpong_flag);
                    wait_flag(PIPE_FIX, PIPE_M, l0_pingpong_flag);
                    mad((__cc__ float *)l0c_buf + l0_offset, (__ca__ half *)l0a_buf + l0_offset,
                        (__cb__ half *)l0b_buf,
                        1, __k_fa, qk_n, 0, 0, 0, 1  // m k n unitFlag kDirectionAlign cmatrixSource cmatrixInitVal
                    );
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    set_flag(PIPE_M, PIPE_MTE1, l0_pingpong_flag);
                    set_flag(PIPE_M, PIPE_FIX, l0_pingpong_flag);
                    wait_flag(PIPE_M, PIPE_FIX, l0_pingpong_flag);
                    // copy S to gm
                    copy_matrix_cc_to_gm((__gm__ float *)s_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA +
                            split_idx * head_split_num_fa * qk_round_n + headdim_idx * qk_round_n,
                        (__cc__ float *)l0c_buf + l0_offset,
                        0, qk_round_n, 1, qk_round_n, 16,  // sid NSize MSize dstStride_dst_D srcStride
                        0, NoQuant, 0, false, true         // UnitFlagMode QuantPRE ReLUPRE channelSplit NZ2ND_EN
                    );
                    set_flag(PIPE_FIX, PIPE_M, l0_pingpong_flag);
                    l0_pingpong_flag = 1 - l0_pingpong_flag;
                    l0_offset = l0_pingpong_flag * L0AB_HALF_BUF_SIZE_FA;
                }
                l1_pingpong_flag = 1 - l1_pingpong_flag;
                l1_offset = l1_pingpong_flag * L1_HALF_BUF_SIZE_FA;
            }
            k_offset += pp_n_scalar_fa * stride_kv_fa;

            ffts_cross_core_sync(PIPE_FIX, 0x21 + (QK_READY_FA << BIT_SHIFT_FA));  // 0

            for (uint32_t split_idx = 0; split_idx < head_split_loop_fa; split_idx++) {
                uint32_t head_num_move =
                    (split_idx == (head_split_loop_fa - 1)) ?
                    cur_head_num_fa - head_split_num_fa * split_idx : head_split_num_fa;
                // *** Prepare V to L1
                wait_flag(PIPE_MTE1, PIPE_MTE2, l1_pingpong_flag);
                copy_gm_to_cbuf_multi_nd2nz_b16((__cbuf__ half *)l1kv_buf_addr + l1_offset,
                    (__gm__ half *)v_gm_fa + v_offset +
                    (start_head_fa + split_idx * head_split_num_fa) / group_num_fa * embdFa,
                    0, 1, qk_n, __k_fa * head_num_move, 0,  // sid ndNum nValue dValue srcNdMatrixStride(unused)
                    stride_kv_fa, qk_round_n, 1, 0  // srcDValue dstNzC0Stride dstNzNStride dstNzMatrixStride(unused)
                );
                set_flag(PIPE_MTE2, PIPE_MTE1, l1_pingpong_flag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, l1_pingpong_flag);
                for (uint32_t headdim_idx = 0; headdim_idx < head_num_move; ++headdim_idx) {
                    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    if (qk_round_n <= round_k_fa) {
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < qk_round_n / BLOCK_SIZE_FA; ++l0b_load_idx) {
                            load_cbuf_to_cb ((__cb__ half *)l0b_buf + l0b_load_idx * round_k_fa * BLOCK_SIZE_FA,
                                (__cbuf__ half *)l1kv_buf_addr + l1_offset +
                                    headdim_idx * round_k_fa * qk_round_n / group_num_fa +
                                    l0b_load_idx * CUBE_MATRIX_SIZE_FA,
                                0, round_k_fa / BLOCK_SIZE_FA, qk_round_n / BLOCK_SIZE_FA, // baseIdx repeat srcStride
                                0, 0, true, inc  // dstStride sid transpose addr_cal_mode_t
                            );
                        }
                    } else {
                        for (uint32_t l0b_load_idx = 0; l0b_load_idx < round_k_fa / BLOCK_SIZE_FA; ++l0b_load_idx) {
                            load_cbuf_to_cb((__cb__ half *)l0b_buf + l0b_load_idx * CUBE_MATRIX_SIZE_FA,
                                (__cbuf__ half *)l1kv_buf_addr + l1_offset +
                                    headdim_idx * round_k_fa * qk_round_n / group_num_fa +
                                    l0b_load_idx * qk_round_n * BLOCK_SIZE_FA,
                                0, qk_round_n / BLOCK_SIZE_FA, 1,            // baseIdx repeat srcStride
                                round_k_fa / BLOCK_SIZE_FA - 1, 0, true, inc // dstStride sid transpose addr_cal_mode_t
                            );
                        }
                    }
                    if (split_idx == 0 && headdim_idx == 0) {
                        wait_flag_dev(SOFTMAX_READY_FA);  // 2
                        copy_gm_to_cbuf((__cbuf__ half *)l1p_buf_addr,
                            (__gm__ half *)p_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA, 0, 1,  // sid nBurst
                            qk_round_n * cur_head_num_fa / BLOCK_SIZE_FA,                      // lenBurst
                            0, 0, PAD_NONE  // srcGap dstGap padMode
                        );
                    }
                    set_flag(PIPE_MTE2, PIPE_MTE1, l0_pingpong_flag);
                    wait_flag(PIPE_MTE2, PIPE_MTE1, l0_pingpong_flag);
                    wait_flag(PIPE_M, PIPE_MTE1, l0_pingpong_flag);
                    load_cbuf_to_ca((__ca__ half *)l0a_buf + l0_offset,
                        (__cbuf__ half *)l1p_buf_addr + split_idx * qk_round_n * head_split_num_fa +
                            headdim_idx * qk_round_n,
                        0, (qk_round_n + CUBE_MATRIX_SIZE_FA - 1) / CUBE_MATRIX_SIZE_FA,  // baseIdx repeat
                        1, 0, 0, false, inc  // srcStride dstStride sid transpose addr_cal_mode_t
                    );
                    if (headdim_idx == head_num_move - 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, l1_pingpong_flag);
                    }
                    set_flag(PIPE_MTE1, PIPE_M, l0_pingpong_flag);
                    wait_flag(PIPE_MTE1, PIPE_M, l0_pingpong_flag);
                    wait_flag(PIPE_FIX, PIPE_M, l0_pingpong_flag);
                    mad((__cc__ float *)l0c_buf + l0_offset, (__ca__ half *)l0a_buf + l0_offset,
                        (__cb__ half *)l0b_buf,
                        1, qk_n, __k_fa, 0, 0, 0, 1  // m k n unitFlag kDirectionAlign cmatrixSource cmatrixInitVal
                    );
                    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
                    set_flag(PIPE_M, PIPE_MTE1, l0_pingpong_flag);
                    set_flag(PIPE_M, PIPE_FIX, l0_pingpong_flag);
                    wait_flag(PIPE_M, PIPE_FIX, l0_pingpong_flag);
                    // copy O to gm
                    copy_matrix_cc_to_gm((__gm__ float *)o_tmp_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA +
                            split_idx * round_k_fa * head_split_num_fa + headdim_idx * round_k_fa,
                        (__cc__ float *)l0c_buf + l0_offset,
                        0, round_k_fa, 1, round_k_fa, 16,  // sid NSize MSize dstStride_dst_D srcStride
                        0, NoQuant, 0, false, true         // UnitFlagMode QuantPRE ReLUPRE channelSplit NZ2ND_EN
                    );
                    set_flag(PIPE_FIX, PIPE_M, l0_pingpong_flag);
                    l0_pingpong_flag = 1 - l0_pingpong_flag;
                    l0_offset = l0_pingpong_flag * L0AB_HALF_BUF_SIZE_FA;
                }
                l1_pingpong_flag = 1 - l1_pingpong_flag;
                l1_offset = l1_pingpong_flag * L1_HALF_BUF_SIZE_FA;
            }
            v_offset += pp_n_scalar_fa * stride_kv_fa;

            ffts_cross_core_sync(PIPE_FIX, 0x21 + (UPDATE_READY_FA << BIT_SHIFT_FA));  // 4
        }
    }

    __aicore__ inline void SetMask(int32_t len)
    {
        uint64_t mask = 0;
        uint64_t one = 1;
        uint64_t temp = len % MASK_LOW_FA;
        for (int64_t i = 0; i < temp; i++) {
            mask |= one << i;
        }

        if (len == MASK_HIGH_FA) {
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        } else if (len >= MASK_LOW_FA) {
            set_vector_mask(mask, (uint64_t)-1);
        } else {
            set_vector_mask(0x0, mask);
        }
    }

    __aicore__ inline void RunAiv(uint32_t cur_batch_fa,
                                  uint32_t head_idx_fa, uint32_t cur_head_num_fa, uint32_t offset_tiling_fa,
                                  int32_t sub_block_idx_fa, uint32_t max_seqlen_fa, uint32_t embdFa, float torFa,
                                  uint32_t batch_mask_fa, uint32_t mask_stride_fa, uint32_t __k_fa,
                                  uint32_t round_k_fa,
                                  uint32_t kv_seqlen_fa, uint32_t pp_n_scalar_fa, uint32_t addr_o_high32_fa,
                                  uint32_t addr_o_loww32_fa,
                                  __gm__ uint8_t* mask_gm_fa,
                                  __gm__ uint8_t* o_gm_fa,
                                  __gm__ uint8_t* s_gm_fa,
                                  __gm__ uint8_t* p_gm_fa,
                                  __gm__ uint8_t* o_tmp_gm_fa)
    {
        uint32_t go_flag_scalar{1};

        __ubuf__ uint8_t *ls_ubuf = (__ubuf__ uint8_t *)get_imm(0);  // 1块，存放 local S, fp32
        __ubuf__ uint8_t *lp_ubuf = (__ubuf__ uint8_t *)get_imm(
            2 * UB_UINT8_BLOCK_SIZE_DECODER_FA);  // 1块，存放 local P, fp16
        __ubuf__ uint8_t *lo_ubuf = (__ubuf__ uint8_t *)get_imm(
            3 * UB_UINT8_BLOCK_SIZE_DECODER_FA);  // 1块，存放 local O, fp32
        __ubuf__ uint8_t *lm_ubuf = (__ubuf__ uint8_t *)get_imm(
            5 * UB_UINT8_BLOCK_SIZE_DECODER_FA);  // 1块，存放 local m, fp16
        __ubuf__ uint8_t *hm_ubuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER_FA +
            1 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 hat m, fp16
        __ubuf__ uint8_t *gm_ubuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER_FA +
            2 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 global m, fp32
        __ubuf__ uint8_t *dm_ubuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER_FA +
            4 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 diff m, fp16
        __ubuf__ uint8_t *ll_ubuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER_FA +
            5 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 local l, fp32
        __ubuf__ uint8_t *gl_ubuf =
            (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE_DECODER_FA +
            7 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放 global l, fp32
        __ubuf__ uint8_t *tv_ubuf = (__ubuf__ uint8_t *)get_imm(
            5 * UB_UINT8_BLOCK_SIZE_DECODER_FA + 10 * UB_UINT8_LINE_SIZE_FA);  // 1块，存放临时 vector 计算变量
        __ubuf__ uint8_t *go_ubuf = (__ubuf__ uint8_t *)get_imm(
            6 * UB_UINT8_BLOCK_SIZE_DECODER_FA);  // 1块，存放 global O, fp32

        uint64_t addr_o_scalar = (uint64_t)(((uint64_t)addr_o_high32_fa) << 32 | addr_o_loww32_fa);
        uint32_t n_loop = (kv_seqlen_fa + pp_n_scalar_fa - 1) / pp_n_scalar_fa;

        uint64_t o_offset = addr_o_scalar + head_idx_fa * embdFa;
        uint64_t mask_batch_offset = 0;

        uint32_t qk_n = pp_n_scalar_fa;
        uint32_t qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        uint32_t sub_m = (sub_block_idx_fa == 1) ? (cur_head_num_fa - cur_head_num_fa / 2) : cur_head_num_fa / 2;
        uint32_t sub_m_d128 = (sub_m + HALF_VECTOR_SIZE_FA - 1) / HALF_VECTOR_SIZE_FA;   // up aligned to 128
        uint32_t sub_m_d64 = (sub_m + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA;  // up aligned to 64
        uint32_t round_sub_m = (sub_m + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;

        for (uint32_t n_idx = 0; n_idx < n_loop; n_idx++) {
            if (n_idx == (n_loop - 1)) {
                qk_n = (kv_seqlen_fa - n_idx * pp_n_scalar_fa);
                qk_round_n = (qk_n + BLOCK_SIZE_FA - 1) / BLOCK_SIZE_FA * BLOCK_SIZE_FA;
            }

            wait_flag_dev(QK_READY_FA);
            if (sub_m > 0) {
                // input QK
                uint64_t offset_9 = sub_block_idx_fa * cur_head_num_fa / 2 * qk_round_n;
                copy_gm_to_ubuf((__ubuf__ float *)ls_ubuf,
                    (__gm__ float *)s_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA + offset_9,
                    0, 1, sub_m * qk_round_n / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                // *** ls = torFa * ls
                vmuls((__ubuf__ float *)ls_ubuf, (__ubuf__ float *)ls_ubuf, torFa,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                // *** ls = ls + mask
                uint64_t mask_offset = mask_batch_offset + n_idx * pp_n_scalar_fa;
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                if (mask_gm_fa != nullptr) {
                    copy_gm_to_ubuf((__ubuf__ half *)lp_ubuf, (__gm__ half *)mask_gm_fa + mask_offset,
                        0, 1, qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                    );
                }
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                if (mask_gm_fa != nullptr) {
                    vconv_f162f32((__ubuf__ float *)lp_ubuf + 256, (__ubuf__ half *)lp_ubuf,
                        (qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                        1, 1, 8, 4  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                    );
                }
                pipe_barrier(PIPE_V);
                if (mask_gm_fa != nullptr) {
                    for (uint32_t vadd_idx = 0; vadd_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++vadd_idx) {
                        VecAddFormula((__ubuf__ float *)ls_ubuf + vadd_idx * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)ls_ubuf + vadd_idx * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)lp_ubuf + 256 + vadd_idx * FLOAT_VECTOR_SIZE_FA, sub_m,  // repeat
                            1, 1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA,  // 1:dstBlockStride dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA, 0         // src0RepeatStride src1RepeatStride
                        );
                    }

                    if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                        SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                        VecAddFormula((__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)lp_ubuf + 256 + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1,                      // repeat 1:dstBlockStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,    // dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA, 0  // src0RepeatStride src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                }
                pipe_barrier(PIPE_V);
                if (qk_n <= FLOAT_VECTOR_SIZE_FA) {
                    SetMask(qk_n);
                    vcmax((__ubuf__ float *)lm_ubuf, (__ubuf__ float *)ls_ubuf, sub_m,  // repeat
                        1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA, ONLY_VALUE  // 1:dstRepeatStride srcRepeatStride order
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    CopyUbufToUbufFormula((__ubuf__ float *)lp_ubuf, (__ubuf__ float *)ls_ubuf,
                        0, sub_m, 8,                                                  // sid nBurst lenBurst
                        (qk_round_n - FLOAT_VECTOR_SIZE_FA) / FLOAT_BLOCK_SIZE_FA, 0  // srcGap dstGap
                    );
                    pipe_barrier(PIPE_V);
                    for (uint32_t rowmax_idx = 1; rowmax_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++rowmax_idx) {
                        VecMaxFormula((__ubuf__ float *)lp_ubuf, (__ubuf__ float *)lp_ubuf,
                            (__ubuf__ float *)ls_ubuf + rowmax_idx * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1, 8, 8,  // repeat 1:dstBlockStride 8:dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA  // src1RepeatStride
                        );
                    }
                    if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                        SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                        VecMaxFormula((__ubuf__ float *)lp_ubuf, (__ubuf__ float *)lp_ubuf,
                            (__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1, 8, 8,             // repeat 1:dstBlockStride 8:dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA  // src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcmax((__ubuf__ float *)lm_ubuf, (__ubuf__ float *)lp_ubuf,
                        sub_m, 1, 1, 8, ONLY_VALUE  // repeat dstRepeatStride srcBlockStride srcRepeatStride order
                    );
                }
                pipe_barrier(PIPE_V);
                if (n_idx != 0) {
                    // *** hm = vmax(lm, gm)
                    VecMaxFormula((__ubuf__ float *)hm_ubuf, (__ubuf__ float *)lm_ubuf, (__ubuf__ float *)gm_ubuf,
                        sub_m_d64, 1, 1, 1, 8, 8, 8  // repeat 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** dm = gm - hm
                    VecSubFormula((__ubuf__ float *)dm_ubuf, (__ubuf__ float *)gm_ubuf, (__ubuf__ float *)hm_ubuf,
                        sub_m_d64, 1, 1, 1, 8, 8, 8  // repeat 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                } else {
                    // *** hm = lm
                    CopyUbufToUbufFormula((__ubuf__ float *)hm_ubuf, (__ubuf__ float *)lm_ubuf,
                        0, 1, round_sub_m / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                    );
                    pipe_barrier(PIPE_V);
                }
                // *** gm = hm
                CopyUbufToUbufFormula((__ubuf__ float *)gm_ubuf, (__ubuf__ float *)hm_ubuf,
                    0, 1, round_sub_m / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                pipe_barrier(PIPE_V);
                // *** hm_block = expand_to_block(hm), 存放于 tv
                vbrcb((__ubuf__ uint32_t *)tv_ubuf, (__ubuf__ uint32_t *)hm_ubuf,
                    1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                );
                pipe_barrier(PIPE_V);
                for (uint32_t vsub_idx = 0; vsub_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++vsub_idx) {
                    VecSubFormula((__ubuf__ float *)ls_ubuf + vsub_idx * FLOAT_VECTOR_SIZE_FA,
                        (__ubuf__ float *)ls_ubuf + vsub_idx * FLOAT_VECTOR_SIZE_FA,
                        (__ubuf__ float *)tv_ubuf, sub_m, 1, 1, 0, // repeat 1:dstBlockStride 0:src1BlockStride
                        qk_round_n / FLOAT_BLOCK_SIZE_FA,          // dstRepeatStride
                        qk_round_n / FLOAT_BLOCK_SIZE_FA, 1        // src0RepeatStride src1RepeatStride
                    );
                }
                if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                    SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                    VecSubFormula((__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                        (__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                        (__ubuf__ float *)tv_ubuf, sub_m, 1, 1, 0,  // repeat 1:dstBlockStride 0:src1BlockStride
                        qk_round_n / FLOAT_BLOCK_SIZE_FA,           // dstRepeatStride
                        qk_round_n / FLOAT_BLOCK_SIZE_FA, 1         // src0RepeatStride src1RepeatStride
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
                pipe_barrier(PIPE_V);
                VecExpFormula((__ubuf__ float *)ls_ubuf, (__ubuf__ float *)ls_ubuf,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 8, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                vconv_f322f16((__ubuf__ half *)lp_ubuf, (__ubuf__ float *)ls_ubuf,
                    (sub_m * qk_round_n + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                    1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                );
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                uint64_t offset_10 = sub_block_idx_fa * cur_head_num_fa / 2 * qk_round_n;
                copy_ubuf_to_gm((__gm__ half *)p_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA + offset_10,
                    (__ubuf__ half *)lp_ubuf,
                    0, 1, sub_m * qk_round_n / BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                );
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                if (qk_n <= FLOAT_VECTOR_SIZE_FA) {
                    SetMask(qk_n);
                    vcadd((__ubuf__ float *)ll_ubuf, (__ubuf__ float *)ls_ubuf, sub_m,  // repeat
                        1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA, 0  // 1:dstRepeatStride srcRepeatStride mode
                    );
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                } else {
                    for (uint32_t rowsum_idx = 1; rowsum_idx < qk_n / FLOAT_VECTOR_SIZE_FA; ++rowsum_idx) {
                        VecAddFormula((__ubuf__ float *)ls_ubuf, (__ubuf__ float *)ls_ubuf,
                            (__ubuf__ float *)ls_ubuf + rowsum_idx * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1,  // repeat dstBlockStride src0BlockStride src1BlockStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                        );
                        pipe_barrier(PIPE_V);
                    }
                    if (qk_n % FLOAT_VECTOR_SIZE_FA > 0) {
                        SetMask(qk_n % FLOAT_VECTOR_SIZE_FA);
                        VecAddFormula((__ubuf__ float *)ls_ubuf, (__ubuf__ float *)ls_ubuf,
                            (__ubuf__ float *)ls_ubuf + qk_n / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            sub_m, 1, 1, 1,  // repeat dstBlockStride src0BlockStride src1BlockStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // dstRepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA,  // src0RepeatStride
                            qk_round_n / FLOAT_BLOCK_SIZE_FA   // src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    vcadd((__ubuf__ float *)ll_ubuf, (__ubuf__ float *)ls_ubuf, sub_m,  // repeat
                        1, 1, qk_round_n / FLOAT_BLOCK_SIZE_FA, 0  // 1:dstRepeatStride srcRepeatStride order
                    );
                }
                pipe_barrier(PIPE_V);
            }
            ffts_cross_core_sync(PIPE_MTE3, 0x21 + (SOFTMAX_READY_FA << BIT_SHIFT_FA));  // 2

            wait_flag_dev(UPDATE_READY_FA);  // 4
            if (sub_m > 0) {
                uint64_t offset_11 = sub_block_idx_fa * cur_head_num_fa / 2 * round_k_fa;
                copy_gm_to_ubuf((__ubuf__ float *)lo_ubuf,
                    (__gm__ float *)o_tmp_gm_fa + (uint64_t)block_idx * TMP_SIZE_FA + offset_11,
                    0, 1, sub_m * round_k_fa / FLOAT_BLOCK_SIZE_FA, 0, 0 // sid nBurst lenBurst srcGap dstGap
                );
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
                // *** 更新 L 和 O
                if (n_idx != 0) {
                    VecExpFormula((__ubuf__ float *)dm_ubuf, (__ubuf__ float *)dm_ubuf, sub_m_d64,  // repeat
                        1, 1, 8, 8  // 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    VecMulFormula((__ubuf__ float *)gl_ubuf, (__ubuf__ float *)dm_ubuf,
                        (__ubuf__ float *)gl_ubuf, sub_m_d64,  // repeat
                        1, 1, 1, 8, 8, 8                       // 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    VecAddFormula((__ubuf__ float *)gl_ubuf, (__ubuf__ float *)gl_ubuf,
                        (__ubuf__ float *)ll_ubuf, sub_m_d64,  // repeat
                        1, 1, 1, 8, 8, 8                       // 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    vbrcb((__ubuf__ uint32_t *)tv_ubuf, (__ubuf__ uint32_t *)dm_ubuf,
                        1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                    );
                    pipe_barrier(PIPE_V);
                    if (go_flag_scalar == 1) {
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        go_flag_scalar = 0;
                    }
                    // *** go = go * dm_block
                    for (uint32_t vmul_idx = 0; vmul_idx < __k_fa / FLOAT_VECTOR_SIZE_FA; ++vmul_idx) {
                        VecMulFormula((__ubuf__ float *)go_ubuf + vmul_idx * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)go_ubuf + vmul_idx * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)tv_ubuf, sub_m, 1, 1,  // repeat dstBlockStride
                            0, round_k_fa / FLOAT_BLOCK_SIZE_FA,     // src1BlockStride dstRepeatStride
                            round_k_fa / FLOAT_BLOCK_SIZE_FA, 1      // src0RepeatStride src1RepeatStride
                        );
                    }
                    if (__k_fa % FLOAT_VECTOR_SIZE_FA > 0) {
                        SetMask(__k_fa % FLOAT_VECTOR_SIZE_FA);
                        VecMulFormula((__ubuf__ float *)go_ubuf + __k_fa / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)go_ubuf + __k_fa / FLOAT_VECTOR_SIZE_FA * FLOAT_VECTOR_SIZE_FA,
                            (__ubuf__ float *)tv_ubuf, sub_m, 1, 1,  // repeat dstBlockStride
                            0, round_k_fa / FLOAT_BLOCK_SIZE_FA,     // src1BlockStride dstRepeatStride
                            round_k_fa / FLOAT_BLOCK_SIZE_FA, 1      // src0RepeatStride src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    pipe_barrier(PIPE_V);
                    // *** go = lo + go
                    VecAddFormula((__ubuf__ float *)go_ubuf, (__ubuf__ float *)go_ubuf,
                        (__ubuf__ float *)lo_ubuf,
                        (sub_m * round_k_fa + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                        1, 1, 1, 8, 8, 8  // 1:dstBlockStride 8:dstRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                } else {
                    // *** gl = ll
                    CopyUbufToUbufFormula((__ubuf__ float *)gl_ubuf, (__ubuf__ float *)ll_ubuf,
                        0, 1, round_sub_m / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                    );
                    pipe_barrier(PIPE_V);
                    if (go_flag_scalar == 1) {
                        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        go_flag_scalar = 0;
                    }
                    // *** go = lo
                    CopyUbufToUbufFormula((__ubuf__ float *)go_ubuf, (__ubuf__ float *)lo_ubuf,
                        0, 1, sub_m * round_k_fa / FLOAT_BLOCK_SIZE_FA, 0, 0  // sid nBurst lenBurst srcGap dstGap
                    );
                    pipe_barrier(PIPE_V);
                }
                if (n_idx == n_loop - 1) {
                    // *** gl = castfp32to16(gl)
                    vconv_f322f16((__ubuf__ half *)gl_ubuf, (__ubuf__ float *)gl_ubuf, sub_m_d64,  // repeat
                        1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** go = castfp32to16(go)
                    vconv_f322f16((__ubuf__ half *)go_ubuf, (__ubuf__ float *)go_ubuf,
                        (sub_m * round_k_fa + FLOAT_VECTOR_SIZE_FA - 1) / FLOAT_VECTOR_SIZE_FA,  // repeat
                        1, 1, 4, 8  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
                    );
                    pipe_barrier(PIPE_V);
                    // *** gl_block = expand_to_block(gl), 存放于 tv
                    vbrcb((__ubuf__ uint16_t *)tv_ubuf, (__ubuf__ uint16_t *)gl_ubuf,
                        1, 8, round_sub_m / FLOAT_BLOCK_SIZE_FA  // dstBlockStride dstRepeatStride repeat
                    );
                    pipe_barrier(PIPE_V);
                    // *** go = go / gl_block
                    for (uint32_t vdiv_idx = 0; vdiv_idx < __k_fa / HALF_VECTOR_SIZE_FA; ++vdiv_idx) {
                        VecDivFormula((__ubuf__ half *)go_ubuf + vdiv_idx * HALF_VECTOR_SIZE_FA,
                            (__ubuf__ half *)go_ubuf + vdiv_idx * HALF_VECTOR_SIZE_FA,
                            (__ubuf__ half *)tv_ubuf, sub_m,  // repeat
                            1, 1, 0,                          // 1:dstBlockStride 0:src1BlockStride
                            round_k_fa / BLOCK_SIZE_FA,       // dstRepeatStride
                            round_k_fa / BLOCK_SIZE_FA, 1     // src0RepeatStride src1RepeatStride
                        );
                    }
                    if (__k_fa % HALF_VECTOR_SIZE_FA > 0) {
                        SetMask(__k_fa % HALF_VECTOR_SIZE_FA);
                        VecDivFormula((__ubuf__ half *)go_ubuf + __k_fa / HALF_VECTOR_SIZE_FA * HALF_VECTOR_SIZE_FA,
                            (__ubuf__ half *)go_ubuf + __k_fa / HALF_VECTOR_SIZE_FA * HALF_VECTOR_SIZE_FA,
                            (__ubuf__ half *)tv_ubuf, sub_m,  // repeat
                            1, 1, 0,                          // 1:dstBlockStride 0:src1BlockStride
                            round_k_fa / BLOCK_SIZE_FA,       // dstRepeatStride
                            round_k_fa / BLOCK_SIZE_FA, 1     // src0RepeatStride src1RepeatStride
                        );
                        set_vector_mask((uint64_t)-1, (uint64_t)-1);
                    }
                    // ********************* move O to GM ************************
                    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
                    copy_ubuf_to_gm_align_b16((__gm__ half *)o_gm_fa + o_offset,
                        (__ubuf__ half *)go_ubuf, 0, sub_m, __k_fa * 2,  // sid nBurst lenBurst
                        0, 0, 0, 0             // leftPaddingNum rightPaddingNum srcGap dstGap
                    );
                    if (go_flag_scalar == 0) {
                        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
                        go_flag_scalar = 1;
                    }
                }
            }
        }
    }

    __aicore__ void ExpandToBlockHalf(__ubuf__ half *dst_fa, __ubuf__ half *src_fa, int32_t len_fa)
    {
        for (int32_t vadds_idx = 0; vadds_idx < 2; ++vadds_idx) {
            vadds((__ubuf__ half *)dst_fa + vadds_idx * 8 * BLOCK_SIZE_FA,
                (__ubuf__ half *)src_fa, (half)0.0, len_fa / BLOCK_SIZE_FA, // repeat
                1, 0, 16, 1  // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
            );
        }
        pipe_barrier(PIPE_V);
        for (int32_t vtrans_idx = 0; vtrans_idx < len_fa / BLOCK_SIZE_FA; ++vtrans_idx) {
            vtranspose((__ubuf__ uint16_t *)dst_fa + vtrans_idx * CUBE_MATRIX_SIZE_FA,
                (__ubuf__ uint16_t *)dst_fa + vtrans_idx * CUBE_MATRIX_SIZE_FA);
        }
        pipe_barrier(PIPE_V);
    }

    template <typename T>
    __aicore__ inline void VecMaxFormula(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecAddFormula(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecSubFormula(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vsub(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecMulFormula(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmul(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecDivFormula(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecExpFormula(T *dst, T *src, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t srcBlockStride, uint16_t dstRepeatStride, uint16_t srcRepeatStride)
    {
        vexp(dst, src, repeat, dstBlockStride, srcBlockStride, dstRepeatStride, srcRepeatStride);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToUbufFormula(T *dst, T *src, uint16_t sid, uint16_t nBurst,
        uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
    {
        copy_ubuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap);
    }
};
}

namespace {
extern "C" __global__ __aicore__ void unpad_flash_attention_mix_sd(GM_ADDR Q,
    GM_ADDR Kcache, GM_ADDR Vcache, GM_ADDR qSeqlen, GM_ADDR kvSeqlen, GM_ADDR kvSeqlenShape, GM_ADDR layerID,
    GM_ADDR AttentionMask, GM_ADDR OutputO, GM_ADDR OutputS, GM_ADDR OutputP, GM_ADDR Otmp,
    GM_ADDR usrWorkspace, GM_ADDR tiling)
{
    UnpadFlashAttentionMixSd op;
    uint32_t *tiling_para_gm = nullptr;

#ifdef __DAV_C220_CUBE__
    GET_TILING_DATA(tiling_data, tiling);
    tiling_para_gm = const_cast<uint32_t *>(tiling_data.UnpadFlashAttentionMixSdTilingParam);
    if (TILING_KEY_IS(0)) {
        op.unpad_flash_attention_decoder_mix_aic(Q, Kcache, Vcache, layerID, OutputS, OutputP, Otmp,
        tiling, tiling_para_gm);
    }
    if (TILING_KEY_IS(1)) {
        op.unpad_flashattention_encoder_mix_aic(Q, Kcache, Vcache, layerID, AttentionMask, OutputO,
        OutputS, OutputP, Otmp,
        tiling, tiling_para_gm);
    }

#elif __DAV_C220_VEC__

    GET_TILING_DATA(tiling_data, tiling);
    tiling_para_gm = const_cast<uint32_t *>(tiling_data.UnpadFlashAttentionMixSdTilingParam);
    if (TILING_KEY_IS(0)) {
        op.unpad_flash_attention_decoder_mix_aiv(nullptr, OutputO, OutputS, OutputP, Otmp, tiling, tiling_para_gm);
    }
    if (TILING_KEY_IS(1)) {
        op.unpad_flashattention_encoder_mix_aiv(Q, Kcache, Vcache, AttentionMask, OutputO, OutputS, OutputP, Otmp,
            tiling, tiling_para_gm);
    }
#endif
}
}