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

namespace {
constexpr int32_t TILING_PARA_HEAD_SIZE = 7;
constexpr int32_t TILING_PARA_ADDR_SIZE = 9;
constexpr int32_t ADDR_INFO_START_INDEX = 1;
constexpr int32_t TILING_CATE_NUM = 3;
constexpr int32_t TILING_PARA_PPPP_SIZE = TILING_CATE_NUM * 2;
constexpr int32_t TILING_PARA_INFO_SIZE = TILING_PARA_ADDR_SIZE + TILING_PARA_PPPP_SIZE * 2 + 2;
constexpr int32_t TILING_PARA_SIZE = (TILING_PARA_HEAD_SIZE + TILING_PARA_INFO_SIZE * 2 + 7) / 8 * 8;
constexpr int32_t PP_MM_NUM = 8;
constexpr int32_t PP_NN_NUM = 16;
constexpr int32_t PP_KK_NUM = 16;
constexpr int32_t CORE_NUM = 64;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384;
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t STRIDE_UPPER_BOUND = 65535;
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072;
constexpr int64_t UB_UINT8_BLOCK_SIZE = 32768;
constexpr int64_t UB_UINT8_LINE_SIZE = 1024;
constexpr int64_t UB_FLOAT_LINE_SIZE = 256;
constexpr int64_t UB_HALF_LINE_SIZE = 512;
constexpr uint16_t DEFAULT_REPEAT_STRIDE = 8;
}

namespace AscendC {
class FlashAttention {
public:
    __aicore__ inline FlashAttention(__gm__ uint8_t *__restrict__ gmSrcq,
                                     __gm__ uint8_t *__restrict__ gmSrck,
                                     __gm__ uint8_t *__restrict__ gmSrcv,
                                     __gm__ uint8_t *__restrict__ gmSrcm,
                                     __gm__ uint8_t *__restrict__ gmDsto,
                                     half tor) : gmSrcq(gmSrcq), gmSrck(gmSrck), gmSrcv(gmSrcv), gmSrcm(gmSrcm),
                                                 gmDsto(gmDsto), tor(tor) {}

    __aicore__ inline void Init(int32_t m, int32_t n, int32_t k, int32_t sp, int64_t srcqOffset, int64_t srckOffset,
                                int32_t srcvOffset, int64_t dstoOffset, int32_t initG, int32_t wrapO,
                                int32_t ntokens)
    {
        this->m = m;
        this->n = n;
        this->k = k;
        this->d = k;
        this->sp = sp;

        this->srcqOffset = srcqOffset;
        this->srckOffset = srckOffset;
        this->srcvOffset = srcvOffset;
        this->dstoOffset = dstoOffset;

        this->initG = initG;
        this->wrapO = wrapO;
        this->ntokens = ntokens;

        l1qBufAddr = (__cbuf__ uint8_t *)get_imm(0);
        l1kBufAddr = (__cbuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        l1pBufAddr = (__cbuf__ uint8_t *)get_imm(2 * L1_UINT8_BLOCK_SIZE);
        l1vBufAddr = (__cbuf__ uint8_t *)get_imm(2 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE));

        l0aBuf = (__ca__ uint8_t *)get_imm(0);
        l0bBuf = (__cb__ uint8_t *)get_imm(0);
        l0cBuf = (__cc__ uint8_t *)get_imm(0);

        lsUbuf = (__ubuf__ uint8_t *)get_imm(0);
        lpUbuf = (__ubuf__ uint8_t *)get_imm(0);
        // 2 for float32
        ls32Ubuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        // 2 for UB memory offset
        loUbuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        // 4 for UB memory offset
        lmUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE);
        // 4 for UB memory offset and 1 for local m memory offset(fp16)
        hmUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE + 1 * UB_UINT8_LINE_SIZE);
        // 4 for UB memory offset and 2 for hat m memory offset(fp16)
        gmUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE + 2 * UB_UINT8_LINE_SIZE);
        // 4 for UB memory offset and 3 for global m memory offset(fp16)
        dmUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE + 3 * UB_UINT8_LINE_SIZE);
        // 4 for UB memory offset and 5 for global m memory offset(fp32&fp16)
        llUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE + 5 * UB_UINT8_LINE_SIZE);
        // 4 for UB memory offset and 7 for local l memory offset(fp32)
        glUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE + 7 * UB_UINT8_LINE_SIZE);
        // 5 for UB memory offset, to save temp vector
        tvUbuf = (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE);
        // 6 for UB memory offset, to save global O(fp32)
        goUbuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE);
    }

    __aicore__ inline void RunPpNzCompute(int32_t m, int32_t n, int32_t k)
    {
        int32_t tn = n * max(m, k) > L0AB_HALF_BUF_SIZE ? 2 : 1;
        FlashAttentionNzCompute(m, n, k, tn);
    }

    template <typename T>
    __aicore__ inline void VecExp(T *dst, T *src, uint8_t expRepeat)
    {
        vexp(dst, src, expRepeat, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    template <typename T>
    __aicore__ inline void VecAdd(T *dst, T *src0, T *src1, uint8_t addRepeat)
    {
        vadd(dst, src0, src1, addRepeat, 1, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE,
             DEFAULT_REPEAT_STRIDE);
    }

    template <typename T>
    __aicore__ inline void VecSub(T *dst, T *src0, T *src1, uint8_t subRepeat)
    {
        vsub(dst, src0, src1, subRepeat, 1, 1, 1,
             DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    template <typename T>
    __aicore__ inline void VecMul(T *dst, T *src0, T *src1, uint8_t mulRepeat)
    {
        vmul(dst, src0, src1, mulRepeat, 1, 1, 1,
             DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    template <typename T>
    __aicore__ inline void VecDiv(T *dst, T *src0, T *src1, uint8_t divRepeat)
    {
        vdiv(dst, src0, src1, divRepeat, 1, 1, 1,
             DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    template <typename T>
    __aicore__ inline void VecMax(T *dst, T *src0, T *src1, uint8_t maxRepeat)
    {
        vmax(dst, src0, src1, maxRepeat, 1, 1, 1,
             DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    __aicore__ inline void CopyGmToL1(__cbuf__ half *dst, __gm__ half *src, uint16_t num, uint16_t len,
                                      uint16_t srcStride, uint16_t dstStride)
    {
        copy_gm_to_cbuf(dst, src, 0, num, len, srcStride, dstStride, PAD_NONE);
    }

    template <typename T>
    __aicore__ inline void CopyUbToUb(T *dst, T *src, uint16_t num, uint16_t len, uint16_t srcStride,
                                      uint16_t dstStride)
    {
        copy_ubuf_to_ubuf(dst, src, 0, num, len, srcStride, dstStride);
    }

    __aicore__ inline void CopyUbToGm(__gm__ half *dst, __ubuf__ half *src, uint16_t num, uint16_t len,
                                      uint16_t srcStride, uint16_t dstStride)
    {
        copy_ubuf_to_gm(dst, src, 0, num, len, srcStride, dstStride);
    }

private:
    __aicore__ inline void ExpandToBlockHalf(__ubuf__ half *dst, __ubuf__ half *src, int32_t len)
    {
        for (int32_t vaddsIdx = 0; vaddsIdx < 2; ++vaddsIdx) {
            vadds((__ubuf__ half *)dst + vaddsIdx * 8 * BLOCK_SIZE,
                  (__ubuf__ half *)src,
                  (half)(0.0),
                  len / BLOCK_SIZE,
                  1, 0, uint16_t(16), uint16_t(1));
        }
        pipe_barrier(PIPE_V);
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE); ++vtransIdx) {
            vtranspose((__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE,
                       (__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void FlashAttentionNzCompute(const int32_t fm, const int32_t fn, const int32_t fk,
                                                   const int32_t fnLoop)
    {
        if (fnLoop == 0) {
            return;
        }
        
        int32_t fn0 = fn / 16 / fnLoop * 16;
        int32_t fn0Tail = (fn / 16) % fnLoop * 16;
        int32_t fd = fk;

        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1qBuf = l1qBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1kBuf = l1kBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1vBuf = l1vBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1pBuf = l1pBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;

        wait_flag(PIPE_MTE1, PIPE_MTE2, l1PingpongFlag);

        int32_t oSize = fm * fd;
        int32_t mD64 = (fm + 63) / 64;
        int32_t mD128 = (fm + 127) / 128;
        int32_t initGg = (initG == 1) ? 1 : 0;

        // inner loop
        for (int32_t fnIdx = 0; fnIdx < fnLoop; ++fnIdx) {
            // 1. bmm1 part
            int32_t n0 = fn0 + fn0Tail * fnIdx;
            int32_t pSize = fm * n0;

            if (fnIdx == 0) {
                if (ntokens <= STRIDE_UPPER_BOUND + fm) {
                    CopyGmToL1((__cbuf__ half *)l1qBuf,
                               (__gm__ half *)gmSrcq + (int64_t)srcqOffset,
                               fk / BLOCK_SIZE,
                               fm,
                               ntokens - fm,
                               0);
                } else {
                    for (int32_t l1qBurstIdx = 0; l1qBurstIdx < (fk / BLOCK_SIZE); ++l1qBurstIdx) {
                        CopyGmToL1((__cbuf__ half *)l1qBuf + l1qBurstIdx * fm * BLOCK_SIZE,
                                   (__gm__ half *)gmSrcq + (int64_t)srcqOffset + l1qBurstIdx * ntokens * BLOCK_SIZE,
                                   1, fm, 0, 0);
                    }
                }

                set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            }
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);

            // 16 is blocksize in format zN
            if (fk == 16) {
                load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                (__cbuf__ half *)l1qBuf,
                                0, fm / BLOCK_SIZE, 1, 0, 0);
            } else {
                for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) {
                    load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                                        l0aLoadIdx * fk * BLOCK_SIZE,
                                    (__cbuf__ half *)l1qBuf + l0aLoadIdx * CUBE_MATRIX_SIZE,
                                    0,
                                    fk / BLOCK_SIZE,
                                    fm / BLOCK_SIZE,
                                    0,
                                    0);
                }
            }

            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            
            if (ntokens <= STRIDE_UPPER_BOUND + n0) {
                CopyGmToL1((__cbuf__ half *)l1kBuf + fnIdx * L0AB_HALF_BUF_SIZE,
                           (__gm__ half *)gmSrck + (int64_t)fnIdx * fn0 * BLOCK_SIZE + (int64_t)srckOffset,
                           fk / BLOCK_SIZE,
                           n0,
                           ntokens - n0,
                           0);
            } else {
                for (int32_t l1kBurstIdx = 0; l1kBurstIdx < (fk / BLOCK_SIZE); ++l1kBurstIdx) {
                    CopyGmToL1((__cbuf__ half *)l1kBuf + fnIdx * L0AB_HALF_BUF_SIZE + l1kBurstIdx * n0 * BLOCK_SIZE,
                               (__gm__ half *)gmSrck + (int64_t)fnIdx * fn0 * BLOCK_SIZE +
                                   (int64_t)srckOffset + l1kBurstIdx * ntokens * BLOCK_SIZE,
                               1, n0, 0, 0);
                }
            }

            set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);

            load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                            (__cbuf__ half *)l1kBuf + fnIdx * L0AB_HALF_BUF_SIZE,
                            0,
                            fk * n0 / CUBE_MATRIX_SIZE,
                            1,
                            0,
                            0);

            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            wait_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            
            mad((__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                fm,
                fk,
                n0,
                1);

            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);
            // 2 for double buffer
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            set_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_MTE3, PIPE_V, ibPingpongFlag);

            copy_matrix_cc_to_ubuf((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                   (__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                   0,
                                   1,
                                   pSize / CUBE_MATRIX_SIZE,
                                   0,
                                   0,
                                   CRMODE_F32toF16_NONE);

            set_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            pipe_barrier(PIPE_V);

            // 2. mask(attention score * tor)
            vmuls((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                  (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                  tor, pSize / 128, 1, 1, uint16_t(8), uint16_t(8));

            if (sp > 0 && fnIdx == (fnLoop - 1)) {
                set_flag(PIPE_V, PIPE_MTE2, ibPingpongFlag);
                wait_flag(PIPE_V, PIPE_MTE2, ibPingpongFlag);

                __ubuf__ half *lastBlockCol = (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                                               fm * (n0 - 16);

                for (int32_t mIdx = 0; mIdx < fm; ++mIdx) {
                    copy_gm_to_ubuf((__ubuf__ half *)loUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + mIdx * BLOCK_SIZE,
                                    (__gm__ half *)gmSrcm + (sp - 1) * BLOCK_SIZE,
                                    0,
                                    1,
                                    1,
                                    0,
                                    0);
                }
                set_flag(PIPE_MTE2, PIPE_V, ibPingpongFlag);
                wait_flag(PIPE_MTE2, PIPE_V, ibPingpongFlag);

                VecAdd((__ubuf__ half *)lastBlockCol,
                       (__ubuf__ half *)lastBlockCol,
                       (__ubuf__ half *)loUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                       fm * BLOCK_SIZE / 128);
                
                pipe_barrier(PIPE_V);
            } else {
                pipe_barrier(PIPE_V);
            }

            // 3. softmax part
            if (n0 / BLOCK_SIZE > 1) {
                VecMax((__ubuf__ half *)tvUbuf,
                       (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                       (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + fm * BLOCK_SIZE,
                       fm * BLOCK_SIZE / 128);

                pipe_barrier(PIPE_V);
            } else {
                CopyUbToUb((__ubuf__ half *)tvUbuf,
                           (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                           1,
                           fm,
                           0,
                           0);

                pipe_barrier(PIPE_V);
            }

            for (int32_t rowmaxIdx = 2; rowmaxIdx < (n0 / BLOCK_SIZE); ++rowmaxIdx) {
                VecMax((__ubuf__ half *)tvUbuf,
                       (__ubuf__ half *)tvUbuf,
                       (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + rowmaxIdx * fm * BLOCK_SIZE,
                       fm * BLOCK_SIZE / 128);

                pipe_barrier(PIPE_V);
            }

            vcgmax((__ubuf__ half *)lmUbuf,
                   (__ubuf__ half *)tvUbuf,
                   fm * BLOCK_SIZE / 128,
                   1, 1, 8);

            pipe_barrier(PIPE_V);

            if (initGg == 0) {
                VecMax((__ubuf__ half *)hmUbuf,
                       (__ubuf__ half *)lmUbuf,
                       (__ubuf__ half *)gmUbuf,
                       mD128);
                
                pipe_barrier(PIPE_V);

                VecSub((__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE,
                       (__ubuf__ half *)gmUbuf,
                       (__ubuf__ half *)hmUbuf,
                       mD128);
                
                pipe_barrier(PIPE_V);
            } else {
                CopyUbToUb((__ubuf__ half *)hmUbuf,
                           (__ubuf__ half *)lmUbuf,
                           1,
                           fm / BLOCK_SIZE,
                           0,
                           0);
                
                pipe_barrier(PIPE_V);
            }

            CopyUbToUb((__ubuf__ half *)gmUbuf,
                       (__ubuf__ half *)hmUbuf,
                       1,
                       fm / BLOCK_SIZE,
                       0,
                       0);
            
            pipe_barrier(PIPE_V);
            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)hmUbuf, fm);

            for (int32_t vsubIdx = 0; vsubIdx < (n0 / BLOCK_SIZE); ++vsubIdx) {
                VecSub((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                       (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                       (__ubuf__ half *)tvUbuf,
                       fm * BLOCK_SIZE / 128);
            }
            pipe_barrier(PIPE_V);
            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f162f32((__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2,
                              (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                              pSize / 2 / 64,
                              1, 1, uint16_t(8), uint16_t(4));
            }

            pipe_barrier(PIPE_V);

            // 2 for double buffer
            for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
                VecExp((__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                       (__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                       pSize / 2 / 64);
            }

            pipe_barrier(PIPE_V);

            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f322f16((__ubuf__ half *)lpUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                              (__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2,
                              pSize / 2 / 64,
                              1,
                              1,
                              4,
                              8);
            }
            pipe_barrier(PIPE_V);

            if (n0 / BLOCK_SIZE > 1) {
                VecAdd((__ubuf__ float *)tvUbuf,
                       (__ubuf__ float *)ls32Ubuf,
                       (__ubuf__ float *)ls32Ubuf + fm * BLOCK_SIZE,
                       fm * BLOCK_SIZE / 64);
                
                pipe_barrier(PIPE_V);
            } else {
                CopyUbToUb((__ubuf__ float *)tvUbuf,
                           (__ubuf__ float *)ls32Ubuf,
                           1,
                           fm * BLOCK_SIZE / 8,
                           0,
                           0);
                
                pipe_barrier(PIPE_V);
            }

            for (int32_t rowsumIdx = 2; rowsumIdx < (n0 / BLOCK_SIZE); ++rowsumIdx) {
                VecAdd((__ubuf__ float *)tvUbuf,
                       (__ubuf__ float *)tvUbuf,
                       (__ubuf__ float *)ls32Ubuf + rowsumIdx * fm * BLOCK_SIZE,
                       fm * BLOCK_SIZE / 64);

                pipe_barrier(PIPE_V);
            }

            set_vector_mask(0x0, 0xffff);

            vcadd((__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE,
                  (__ubuf__ float *)tvUbuf,
                  fm,
                  1,
                  1,
                  2);

            pipe_barrier(PIPE_V);
            set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);

            set_flag(PIPE_V, PIPE_MTE3, ibPingpongFlag);
            wait_flag(PIPE_V, PIPE_MTE3, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_MTE3, l1PingpongFlag);

            copy_ubuf_to_cbuf((__cbuf__ half *)l1pBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                              (__ubuf__ half *)lpUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                              0,
                              1,
                              pSize / BLOCK_SIZE,
                              0,
                              0);

            set_flag(PIPE_MTE3, PIPE_V, ibPingpongFlag);
            set_flag(PIPE_MTE3, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE3, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);

            // 16 is blocksize in format zN
            if (n0 == 16) {
                load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                (__cbuf__ half *)l1pBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                0,
                                fm / BLOCK_SIZE,
                                1,
                                0,
                                0);
            } else {
                for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) {
                    load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                                        l0aLoadIdx * n0 * BLOCK_SIZE,
                                    (__cbuf__ half *)l1pBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                                        l0aLoadIdx * CUBE_MATRIX_SIZE,
                                    0,
                                    n0 / BLOCK_SIZE,
                                    fm / BLOCK_SIZE,
                                    0,
                                    0);
                }
            }

            set_flag(PIPE_MTE1, PIPE_MTE3, l1PingpongFlag);
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);

            // 4. bmm2 part
            if (ntokens <= STRIDE_UPPER_BOUND + n0) {
                CopyGmToL1((__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE,
                           (__gm__ half *)gmSrcv + fnIdx * fn0 * BLOCK_SIZE + (int64_t)srcvOffset,
                           fd / BLOCK_SIZE,
                           n0,
                           ntokens - n0,
                           0);
            } else {
                for (int32_t l1vBurstIdx = 0; l1vBurstIdx < (fd / BLOCK_SIZE); ++l1vBurstIdx) {
                    CopyGmToL1((__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE + l1vBurstIdx * n0 * BLOCK_SIZE,
                               (__gm__ half *)gmSrcv + fnIdx * fn0 * BLOCK_SIZE + (int64_t)srcvOffset +
                                   l1vBurstIdx * ntokens * BLOCK_SIZE,
                               1, n0, 0, 0);
                }
            }

            set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);

            // 16 is blocksize in format zN
            if (fd == 16) {
                load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                (__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE,
                                0,
                                n0 / BLOCK_SIZE,
                                1,
                                0,
                                1);
            } else {
                for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (n0 / BLOCK_SIZE); ++l0bLoadIdx) {
                    load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                                        l0bLoadIdx * fd * BLOCK_SIZE,
                                    (__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE +
                                        l0bLoadIdx * CUBE_MATRIX_SIZE,
                                    0,
                                    fd / BLOCK_SIZE,
                                    n0 / BLOCK_SIZE,
                                    0,
                                    1);
                }
            }

            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            wait_flag(PIPE_V, PIPE_M, ibPingpongFlag);

            mad((__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ca__ __fp16 *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__cb__ __fp16 *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                fm,
                n0,
                fd,
                1);

            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);
            // 2 for double buffer
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            set_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_M, PIPE_V, ibPingpongFlag);

            copy_matrix_cc_to_ubuf((__ubuf__ float *)loUbuf,
                                   (__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                                   0,
                                   1,
                                   oSize / CUBE_MATRIX_SIZE,
                                   0,
                                   0,
                                   CRMODE_NONE);
            
            set_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            pipe_barrier(PIPE_V);

            // 5. update for outer loop
            if (initGg == 0) {
                vconv_f162f32((__ubuf__ float *)tvUbuf,
                              (__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE,
                              mD64,
                              1,
                              1,
                              uint16_t(8),
                              uint16_t(4));

                pipe_barrier(PIPE_V);
                
                VecExp((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf, mD64);

                pipe_barrier(PIPE_V);
                VecMul((__ubuf__ float *)glUbuf,
                       (__ubuf__ float *)tvUbuf,
                       (__ubuf__ float *)glUbuf,
                       mD64);
                pipe_barrier(PIPE_V);

                VecAdd((__ubuf__ float *)glUbuf,
                       (__ubuf__ float *)glUbuf,
                       (__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE,
                       mD64);
                pipe_barrier(PIPE_V);

                ExpandToBlockHalf((__ubuf__ half *)tvUbuf,
                                  (__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE,
                                  fm);

                vconv_f162f32((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                              (__ubuf__ half *)tvUbuf,
                              fm * BLOCK_SIZE / 64,
                              1,
                              1,
                              uint16_t(8),
                              uint16_t(4));
                pipe_barrier(PIPE_V);

                VecExp((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                       (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                       fm * BLOCK_SIZE / 64);
                
                pipe_barrier(PIPE_V);

                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }

                for (int32_t vmulIdx = 0; vmulIdx < (fd / BLOCK_SIZE); ++vmulIdx) {
                    VecMul((__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                           (__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                           (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                           fm * BLOCK_SIZE / 64);
                }
                pipe_barrier(PIPE_V);

                // 2 for double buffer
                for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) {
                    VecAdd((__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                           (__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                           (__ubuf__ float *)loUbuf + vaddIdx * oSize / 2,
                           oSize / 2 / 64);
                }

                pipe_barrier(PIPE_V);
            } else {
                CopyUbToUb((__ubuf__ float *)glUbuf,
                           (__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE,
                           1,
                           fm / 8,
                           0,
                           0);
                pipe_barrier(PIPE_V);
                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }

                CopyUbToUb((__ubuf__ float *)goUbuf,
                           (__ubuf__ float *)loUbuf,
                           1,
                           oSize / 8,
                           0,
                           0);
                pipe_barrier(PIPE_V);
            }

            initGg = 0;
            ibPingpongFlag = 1 - ibPingpongFlag;
        }

        if (wrapO == 1) {
            vconv_f322f16((__ubuf__ half *)glUbuf,
                          (__ubuf__ float *)glUbuf,
                          mD64,
                          1,
                          1,
                          uint16_t(4),
                          uint16_t(8));

            pipe_barrier(PIPE_V);
            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f322f16((__ubuf__ half *)goUbuf + vconvIdx * oSize / 2,
                              (__ubuf__ float *)goUbuf + vconvIdx * oSize / 2,
                              oSize / 2 / 64,
                              1,
                              1,
                              uint16_t(4),
                              uint16_t(8));
                pipe_barrier(PIPE_V);
            }

            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)glUbuf, fm);
            
            for (int32_t vdivIdx = 0; vdivIdx < (fd / BLOCK_SIZE); ++vdivIdx) {
                VecDiv((__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                       (__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                       (__ubuf__ half *)tvUbuf,
                       fm * BLOCK_SIZE / 128);
            }
            pipe_barrier(PIPE_V);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

            // move O to gm
            if (ntokens <= STRIDE_UPPER_BOUND + fm) {
                CopyUbToGm((__gm__ half *)gmDsto + (int64_t)dstoOffset,
                           (__ubuf__ half *)goUbuf,
                           fd / BLOCK_SIZE,
                           fm,
                           0,
                           ntokens - fm);
            } else {
                for (int32_t gmBurstIdx = 0; gmBurstIdx < (fd / BLOCK_SIZE); ++gmBurstIdx) {
                    CopyUbToGm((__gm__ half *)gmDsto + (int64_t)dstoOffset + gmBurstIdx * ntokens * BLOCK_SIZE,
                               (__ubuf__ half *)goUbuf + gmBurstIdx * fm * BLOCK_SIZE,
                               1,
                               fm,
                               0,
                               0);
                }
            }

            if (vmPingpongFlag == 0) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                vmPingpongFlag = 1;
            }
        }

        set_flag(PIPE_MTE1, PIPE_MTE2, l1PingpongFlag);
        l1PingpongFlag = 1 - l1PingpongFlag;
    }

private:
    int32_t l1PingpongFlag = 0;
    int32_t ibPingpongFlag = 0;
    int32_t vmPingpongFlag = 1;

    __cbuf__ uint8_t *l1qBufAddr;
    __cbuf__ uint8_t *l1kBufAddr;
    __cbuf__ uint8_t *l1pBufAddr;
    __cbuf__ uint8_t *l1vBufAddr;

    __ca__ uint8_t* l0aBuf;
    __cb__ uint8_t* l0bBuf;
    __cc__ uint8_t* l0cBuf;

    __ubuf__ uint8_t* lsUbuf;
    __ubuf__ uint8_t* lpUbuf;
    __ubuf__ uint8_t* ls32Ubuf;
    __ubuf__ uint8_t* loUbuf;
    __ubuf__ uint8_t* lmUbuf;
    __ubuf__ uint8_t* hmUbuf;
    __ubuf__ uint8_t* gmUbuf;
    __ubuf__ uint8_t* dmUbuf;
    __ubuf__ uint8_t* llUbuf;
    __ubuf__ uint8_t* glUbuf;
    __ubuf__ uint8_t* tvUbuf;
    __ubuf__ uint8_t* goUbuf;

    __gm__ uint8_t *__restrict__ gmSrcq;
    __gm__ uint8_t *__restrict__ gmSrck;
    __gm__ uint8_t *__restrict__ gmSrcv;
    __gm__ uint8_t *__restrict__ gmSrcm;
    __gm__ uint8_t *__restrict__ gmDsto;

    half tor = 0;

    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t d = 0;
    int32_t sp = 0;
    int32_t ntokens = 0;

    int64_t srcqOffset = 0;
    int64_t srckOffset = 0;
    int32_t srcvOffset = 0;
    int64_t dstoOffset = 0;

    int32_t initG = 0;
    int32_t wrapO = 0;
};
} // namespace ascendc

namespace {
extern "C" __global__ __aicore__ void flash_attention(GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR seqLen,
    GM_ADDR batch, GM_ADDR spMask, GM_ADDR attnOut, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_padding(uint16_t(0));
    set_atomic_none();

    int32_t ppMmScalararrayTemp[PP_MM_NUM];
    int32_t ppNnScalararrayTemp[PP_NN_NUM];

    int32_t mm1 = 0;
    int32_t nn1 = 0;

    if ((int32_t)block_idx < CORE_NUM) {
        uint32_t *tilingPara = const_cast<uint32_t *>(tilingData.tilingParam) + TILING_PARA_SIZE * (int32_t)block_idx;
        
        set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

        int32_t ntokens = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 0));
        int32_t batchStrideScalar = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 1));
        int32_t batchStrideqScalar = batchStrideScalar;
        int32_t batchStridekScalar = batchStrideScalar;
        int32_t batchStridevScalar = batchStrideScalar;
        int32_t batchStrideoScalar = batchStrideScalar;
        half torScalar = (half)(*(float *)((float *)tilingPara + 2));

        int32_t batchSizeScalar[2];
        int32_t mScalar[2];
        int32_t nScalar[2];
        int32_t kScalar[2];
        int32_t spScalar[2];
        int64_t addrqScalar[2];
        int32_t addrkScalar[2];
        int64_t addrvScalar[2];
        int64_t addroScalar[2];
        int32_t tilingParaScalararray[2][TILING_PARA_PPPP_SIZE * 2 + 2];

        // 2 for info offset
        for (int32_t infoIdx = 0; infoIdx < 2; ++infoIdx) {
            mScalar[infoIdx] = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 3));
            nScalar[infoIdx] = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 4));
            kScalar[infoIdx] = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 5));
            spScalar[infoIdx] = (int32_t)(*(int32_t *)((int32_t *)tilingPara + 6));

            int32_t infoOffset = TILING_PARA_HEAD_SIZE + infoIdx * TILING_PARA_INFO_SIZE;
            batchSizeScalar[infoIdx] = (int32_t)(*(int32_t *)((int32_t *)tilingPara + infoOffset));

            uint32_t addrqHigh32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX));
            uint32_t addrqLoww32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 1));
            addrqScalar[infoIdx] = (uint64_t)(((uint64_t)addrqHigh32) << 32 | addrqLoww32);

            uint32_t addrkHigh32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 2));
            uint32_t addrkLoww32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 3));
            addrkScalar[infoIdx] = (uint64_t)(((uint64_t)addrkHigh32) << 32 | addrkLoww32);

            uint32_t addrvHigh32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 4));
            uint32_t addrvLoww32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                     infoOffset + ADDR_INFO_START_INDEX + 5));
            addrvScalar[infoIdx] = (uint64_t)(((uint64_t)addrvHigh32) << 32 | addrvLoww32);

            uint32_t addroHigh32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 6));
            uint32_t addroLoww32 = (uint32_t)(*(uint32_t *)((uint32_t *)tilingPara +
                                   infoOffset + ADDR_INFO_START_INDEX + 7));
            addroScalar[infoIdx] = (uint64_t)(((uint64_t)addroHigh32) << 32 | addroLoww32);

            for (int32_t i = 0; i < (TILING_PARA_PPPP_SIZE * 2 + 2); ++i) {
                tilingParaScalararray[infoIdx][i] = (int32_t)(*((int32_t *)tilingPara +
                                                    infoOffset + TILING_PARA_ADDR_SIZE + i));
            }
        }

        AscendC::FlashAttention foo(query, key, value, spMask, attnOut, torScalar);

        int32_t ppMmScalararray[PP_MM_NUM] = {16, 32, 48, 64, 80, 96, 112, 128};
        int32_t ppNnScalararray[PP_NN_NUM] = {16, 32, 48, 64, 80, 96, 112, 128, 144, 160,
                                              176, 192, 208, 224, 240, 256};
        
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_V, PIPE_M, EVENT_ID0);
        set_flag(PIPE_V, PIPE_M, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);

        for (int32_t infoIdx = 0; infoIdx < 2; ++infoIdx) {
            if (batchSizeScalar[infoIdx] <= 0) {
                continue;
            }

            for (int32_t batchIdx = 0; batchIdx < batchSizeScalar[infoIdx]; ++batchIdx) {
                int32_t mOffset = 0;
                int32_t nOffset = 0;
                int64_t srcqOffset = 0;
                int64_t srckOffset = 0;
                int64_t srcvOffset = 0;
                int64_t dstoOffset = 0;

                for (int32_t ppIdm = 0; ppIdm < TILING_CATE_NUM; ++ppIdm) {
                    int32_t ppNummScalar = tilingParaScalararray[infoIdx][2 * ppIdm];
                    if (ppNummScalar == 0) {
                        break;
                    }

                    int32_t ppIdxmScalar = tilingParaScalararray[infoIdx][2 * ppIdm + 1];
                    for (int32_t mTid = 0; mTid < ppNummScalar; ++mTid) {
                        int32_t initg = 1;
                        for (int32_t ppIdn = 0; ppIdn < TILING_CATE_NUM; ++ppIdn) {
                            int32_t ppNumnScalar =
                                tilingParaScalararray[infoIdx][TILING_PARA_PPPP_SIZE + 2 * ppIdn];
                            if (ppNumnScalar == 0) {
                                break;
                            }

                            int32_t ppIdxnScalar =
                                tilingParaScalararray[infoIdx][TILING_PARA_PPPP_SIZE + 2 * ppIdn + 1];
                            int32_t ppNextNumnScalar =
                                tilingParaScalararray[infoIdx][TILING_PARA_PPPP_SIZE + 2 * ppIdn + 2];
                            int32_t lastnLoop = (ppIdn < TILING_CATE_NUM - 1 && ppNextNumnScalar != 0) ? 0 : 1;

                            for (int32_t nTid = 0; nTid < ppNumnScalar; ++nTid) {
                                int32_t lastnTile =
                                    (lastnLoop == 1 && nTid == ppNumnScalar - 1) ? 1 : 0;
                                
                                srcqOffset = (int64_t)mOffset * BLOCK_SIZE;
                                srckOffset = (int64_t)nOffset * BLOCK_SIZE;
                                srcvOffset = (int64_t)nOffset * BLOCK_SIZE;
                                dstoOffset = (int64_t)mOffset * BLOCK_SIZE;
                                int32_t warpo = lastnTile;

                                foo.Init(
                                    mScalar[infoIdx],
                                    nScalar[infoIdx],
                                    kScalar[infoIdx],
                                    lastnTile * spScalar[infoIdx],
                                    srcqOffset + addrqScalar[infoIdx] + batchIdx * batchStrideqScalar,
                                    srckOffset + addrkScalar[infoIdx] + batchIdx * batchStridekScalar,
                                    srcvOffset + addrvScalar[infoIdx] + batchIdx * batchStridevScalar,
                                    dstoOffset + addroScalar[infoIdx] + batchIdx * batchStrideoScalar,
                                    initg,
                                    warpo,
                                    ntokens);

                                for (int iii = 0; iii < PP_MM_NUM; ++iii) {
                                    ppMmScalararrayTemp[iii] = ppMmScalararray[iii];
                                }

                                for (int iii = 0; iii < PP_NN_NUM; ++iii) {
                                    ppNnScalararrayTemp[iii] = ppNnScalararray[iii];
                                }

                                mm1 = ppMmScalararrayTemp[ppIdxmScalar];
                                nn1 = ppNnScalararrayTemp[ppIdxnScalar];

                                foo.RunPpNzCompute(mm1, nn1, kScalar[infoIdx]);
                                initg = 0;
                                nOffset += nn1;
                            }
                        }

                        mOffset += mm1;
                        nOffset = 0;
                    }
                }
            }
        }

        wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_M, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    }

    pipe_barrier(PIPE_ALL);
}
}