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
constexpr int32_t BATCH_TILING_OFFSET = 128;
constexpr int32_t TILING_PARA_SIZE = 4;
constexpr int32_t LOWER_MASK = 64;
constexpr int32_t FLOAT_VECTOR_SIZE = 64;
constexpr int32_t VECTOR_SIZE = 128;
constexpr int32_t BLOCK_SIZE = 16;
constexpr int32_t L0AB_HALF_BUF_SIZE = 16384; // 128 * 128
constexpr int32_t CUBE_MATRIX_SIZE = 256;
constexpr int32_t STRIDE_UPPER_BOUND = 65535;
constexpr int64_t L1_UINT8_BLOCK_SIZE = 131072; // 128K
constexpr int64_t UB_UINT8_BLOCK_SIZE = 32768;  // 128 * 128 * 2 = 32K
constexpr int64_t UB_UINT8_LINE_SIZE = 1024;
constexpr int64_t UB_FLOAT_LINE_SIZE = 256;
constexpr int64_t UB_HALF_LINE_SIZE = 512;
constexpr int64_t MAX_TILING = 32768;            // 32k
}
#define TILING_CORE(i)                                                                                       \
    if (block_idx == (i)) {                                                                                    \
        startBatch = (int32_t)(*((int32_t *)tiling_para_ub_static + 11 + block_idx * 4 + 0));                 \
        endBatch = (int32_t)(*((int32_t *)tiling_para_ub_static + 11 + block_idx * 4 + 1));                   \
        startBlk = (int32_t)(*((int32_t *)tiling_para_ub_static + 11 + block_idx * 4 + 2));                   \
        endBlk = (int32_t)(*((int32_t *)tiling_para_ub_static + 11 + block_idx * 4 + 3));                     \
    }

namespace AscendC {
class FlashAttentionSoftmaxFp32 {
public:
    __aicore__ inline FlashAttentionSoftmaxFp32(__gm__ uint8_t *__restrict__ gmSrcq,
        __gm__ uint8_t *__restrict__ gmSrck, __gm__ uint8_t *__restrict__ gmSrcv,
        __gm__ uint8_t *__restrict__ gmSrcm, __gm__ uint8_t *__restrict__ gmDsto,
        half tor): gmSrcq(gmSrcq), gmSrck(gmSrck), gmSrcv(gmSrcv), gmSrcm(gmSrcm), gmDsto(gmDsto), tor(tor)
    {
        GetInitAddr();
        GetInitBuf();
    }

    __aicore__ inline void GetInitAddr()
    {
        l1qBufAddr = (__cbuf__ uint8_t *)get_imm(0);
        l1kBufAddr = (__cbuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        l1pBufAddr = (__cbuf__ uint8_t *)get_imm(2 * L1_UINT8_BLOCK_SIZE);
        l1vBufAddr = (__cbuf__ uint8_t *)get_imm(2 * (L1_UINT8_BLOCK_SIZE + UB_UINT8_BLOCK_SIZE));
        l1maxkBufAddr = (__cbuf__ uint8_t *)get_imm(4 * L1_UINT8_BLOCK_SIZE);
    }

    __aicore__ inline void GetInitBuf()
    {
        l0aBuf = (__ca__ uint8_t *)get_imm(0);
        l0bBuf = (__cb__ uint8_t *)get_imm(0);
        l0cBuf = (__cc__ uint8_t *)get_imm(0);

        lsUbuf = (__ubuf__ uint8_t *)get_imm(0); // 32K * 2
        lpUbuf = (__ubuf__ uint8_t *)get_imm(0);
        // 2 for float32
        ls32Ubuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE); // 32K * 4
        // 2 for UB memory offset
        loUbuf = (__ubuf__ uint8_t *)get_imm(2 * UB_UINT8_BLOCK_SIZE);
        // 4 for UB memory offset
        lmUbuf = (__ubuf__ uint8_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE);  // 32K * 5
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
        tvUbuf = (__ubuf__ uint8_t *)get_imm(5 * UB_UINT8_BLOCK_SIZE); // 32K * 6
        // 6 for UB memory offset, to save global O(fp32)
        goUbuf = (__ubuf__ uint8_t *)get_imm(6 * UB_UINT8_BLOCK_SIZE); // 32K * (7~8)
    }

    __aicore__ inline void Init(int32_t mReal, int32_t nReal, int32_t kReal,
                                int64_t srcqOffsetReal, int64_t srckOffsetReal, int32_t srcvOffsetReal,
                                int32_t srcmOffsetReal, int64_t dstoOffsetReal,
                                int32_t initGReal, int32_t wrapOReal, int32_t ntokensQReal, int32_t maskStrideReal,
                                int32_t qSeqLenAlign, int32_t kvSeqLenAlign)
    {
        m = mReal;
        n = nReal;
        k = kReal;
        d = kReal;

        srcqOffset = srcqOffsetReal;
        srckOffset = srckOffsetReal;
        srcvOffset = srcvOffsetReal;
        srcmOffset = srcmOffsetReal;
        dstoOffset = dstoOffsetReal;
        maskStride = maskStrideReal;

        initG = initGReal;
        wrapO = wrapOReal;
        ntokensQ = ntokensQReal;

        this->qSeqLenAlign = qSeqLenAlign;
        this->kvSeqLenAlign = kvSeqLenAlign;
    }

    __aicore__ inline void FlashAttentionNzPrefillCompute(const int32_t fm, const int32_t fn, const int32_t fk,
        const int32_t bm, const int32_t bn, const int32_t bk,
        const int32_t m0Value, const int32_t n0Value,
        const int32_t m1Value, const int32_t n1Value,
        const int32_t pp_n_scalar,
        const int32_t fnLoop = 1)
    {
        int32_t Pingflag = 0; // manual PingPong attempt
        int32_t Pongflag = 1;

        int32_t fn0 = fn / BLOCK_SIZE / fnLoop * BLOCK_SIZE; // kvseq aligned
        int32_t fn1 = bn / BLOCK_SIZE / fnLoop * BLOCK_SIZE; // kvseq aligned
        int32_t fd = fk;

        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1qBuf = l1qBufAddr;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1kPingBuf = l1kBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index
        __cbuf__ uint8_t *l1kPongBuf = l1kBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1vPingBuf = l1vBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index
        __cbuf__ uint8_t *l1vPongBuf = l1vBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1pPingBuf = l1pBufAddr + Pingflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index
        __cbuf__ uint8_t *l1pPongBuf = l1pBufAddr + Pongflag * 4 * L1_UINT8_BLOCK_SIZE; // 4 is index

        wait_flag(PIPE_MTE1, PIPE_MTE2, Pingflag);
        wait_flag(PIPE_MTE1, PIPE_MTE2, Pongflag);

        int32_t oSize = fm * fd;
        int32_t mD64 = (fm + FLOAT_VECTOR_SIZE - 1) / FLOAT_VECTOR_SIZE;
        int32_t mD128 = (fm + VECTOR_SIZE - 1) / VECTOR_SIZE;
        int32_t initGgDm = (initG == 1) ? 1 : 0;
        int32_t initGgO = (initG == 1) ? 1 : 0;
        int32_t initGgMask = (initG == 1) ? 1 : 0;

        int32_t n0 = fn0; // tailBlock
        int32_t pSize = fm * n0;

        int32_t n1 = fn1; // tailBlock
        int32_t pSize_b = fm * n1;
        // 1. ################ Bmm1 Ping Start #######################
        if (initGgO != 0) {
            if (m0Value == 1) {
                copy_gm_to_cbuf((__cbuf__ half *)l1qBuf, (__gm__ half *)gmSrcq + (int64_t)srcqOffset, 0,
                    fk / BLOCK_SIZE, 1, qSeqLenAlign - 1, 0, PAD_NONE);
            } else if (qSeqLenAlign <= STRIDE_UPPER_BOUND + fm) { // (fm, fk)
                copy_gm_to_cbuf((__cbuf__ half *)l1qBuf, (__gm__ half *)gmSrcq + (int64_t)srcqOffset, 0,
                    fk / BLOCK_SIZE, fm, qSeqLenAlign - fm, 0, PAD_NONE);
            } else {
                for (int32_t l1qBurstIdx = 0; l1qBurstIdx < (fk / BLOCK_SIZE); ++l1qBurstIdx) {
                    copy_gm_to_cbuf((__cbuf__ half *)l1qBuf + l1qBurstIdx * fm * BLOCK_SIZE,
                        (__gm__ half *)gmSrcq + (int64_t)srcqOffset + l1qBurstIdx * qSeqLenAlign * BLOCK_SIZE, 0, 1, fm,
                        0, 0, PAD_NONE);
                }
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
        }
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag);

        // 16 is blocksize in format zN
        if (m0Value == 1) {
            load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                            (__cbuf__ half *)l1qBuf, 0,
                            1, 1, 0, 0);
        } else if (fk == BLOCK_SIZE) {
            load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE, (__cbuf__ half *)l1qBuf, 0,
                fm / BLOCK_SIZE, 1, 0, 0);
        } else {
            for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, fk) Nz-> zZ
                load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE +
                                l0aLoadIdx * fk * BLOCK_SIZE,
                                (__cbuf__ half *)l1qBuf + l0aLoadIdx * CUBE_MATRIX_SIZE, 0,
                                fk / BLOCK_SIZE, // repeat
                                fm / BLOCK_SIZE, // srcStride
                                0, 0);
            }
        }

        set_flag(PIPE_MTE1, PIPE_M, Pingflag);
        wait_flag(PIPE_MTE1, PIPE_M, Pingflag);

        if (kvSeqLenAlign <= STRIDE_UPPER_BOUND + n0) {
            copy_gm_to_cbuf((__cbuf__ half *)l1kPingBuf, (__gm__ half *)gmSrck + (int64_t)srckOffset,
                            0, fk / BLOCK_SIZE, n0, kvSeqLenAlign - n0, 0, PAD_NONE);
        } else {
            for (int32_t l1kBurstIdx = 0; l1kBurstIdx < (fk / BLOCK_SIZE); ++l1kBurstIdx) {
                copy_gm_to_cbuf((__cbuf__ half *)l1kPingBuf + l1kBurstIdx * n0 * BLOCK_SIZE,
                    (__gm__ half *)gmSrck + (int64_t)srckOffset + l1kBurstIdx * kvSeqLenAlign * BLOCK_SIZE,
                    0, 1, n0, 0, 0, PAD_NONE);
            }
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
        wait_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
        // 2 for double buffer
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);

        load_cbuf_to_cb((__cb__ half *)l0bBuf + Pingflag * L0AB_HALF_BUF_SIZE, // Nz ->(transpose) nZ -> nZ
                (__cbuf__ half *)l1kPingBuf, 0, fk * n0 / CUBE_MATRIX_SIZE, 1, 0, 0);

        // 2 for double buffer
        set_flag(PIPE_MTE1, PIPE_M, Pingflag + 2);
        // 2 for double buffer
        wait_flag(PIPE_MTE1, PIPE_M, Pingflag + 2);
        wait_flag(PIPE_V, PIPE_M, Pingflag);

        mad((__cc__ float *)l0cBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__cb__ half *)l0bBuf + Pingflag * L0AB_HALF_BUF_SIZE, m0Value, fk, n0Value, 1);

        set_flag(PIPE_M, PIPE_MTE1, Pingflag);
        // 2 for L0B DMA Sync Flag
        set_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);   // 2 means ping-pong
        // 2 means ping-pong
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);  // Must Sync Right Now. Otherwise cause result error.
        set_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);   // 2 means ping-pong
        set_flag(PIPE_M, PIPE_V, Pingflag);
        wait_flag(PIPE_M, PIPE_V, Pingflag);
        wait_flag(PIPE_MTE3, PIPE_V, Pingflag);

        // 1. ################ Bmm1 Ping Ends #######################
        // 3. ################ Softmax Ping Starts #######################
        copy_matrix_cc_to_ubuf((__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__cc__ float *)l0cBuf + Pingflag * L0AB_HALF_BUF_SIZE, 0, 1, pSize / CUBE_MATRIX_SIZE, 0, 0,
                CRMODE_F32toF16_NONE); // fp32 -> fp16

        set_flag(PIPE_V, PIPE_M, Pingflag);
        pipe_barrier(PIPE_V);

        // 3.1. mask(attention score * tor)
        if (float(tor) != float(1.0)) {
            vmuls((__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE,
                  (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE, tor, pSize / 128, 1, 1, uint16_t(8),
                  uint16_t(8));
        }
        pipe_barrier(PIPE_V);

        //  2.1 decoder mask
        wait_flag(PIPE_MTE1, PIPE_MTE2, Pingflag + 2);
        if (gmSrcm != nullptr) {
            copy_gm_to_cbuf((__cbuf__ half *)l1maxkBufAddr + Pingflag * L0AB_HALF_BUF_SIZE,  // Nz load
                        (__gm__ half *)gmSrcm + srcmOffset, 0, n0 / BLOCK_SIZE, fm, maskStride - fm, 0, PAD_NONE);
        }
        set_flag(PIPE_MTE2, PIPE_MTE1, Pingflag + 5);    // 5 is offset
        wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, Pingflag + 5);   // 5 is offset
        if (gmSrcm != nullptr) {
            copy_cbuf_to_ubuf((__ubuf__ half *)loUbuf, (__cbuf__ half *)l1maxkBufAddr + Pingflag * L0AB_HALF_BUF_SIZE,
                0, 1, fm * n0 / BLOCK_SIZE, 0, 0);
        }
        set_flag(PIPE_MTE1, PIPE_V, Pingflag);
        wait_flag(PIPE_MTE1, PIPE_V, Pingflag);
        set_flag(PIPE_MTE1, PIPE_MTE2, Pingflag + 2);  // 2 means pingpong
        if (gmSrcm != nullptr) {
            VecAddFp32((__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)loUbuf, fm * n0 / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);

        // 3. softmax part
        if (n0Value / BLOCK_SIZE > 1) { // 前两个(fm, 16)求最大值
            VecMaxFp32((__ubuf__ half *)tvUbuf,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE, // Nz (fm, n0)
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + fm * BLOCK_SIZE,
                fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                1, 1, 1,               // dstBlockStride src0BlockStride src1BlockStride
                8, 8, 8);              // dstRepeatStride src0RepeatStride src1RepeatStride
            pipe_barrier(PIPE_V);
        } else {
            CopyUbufToUbufFp32((__ubuf__ half *)tvUbuf,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE,
                0, 1, fm, 0, 0); // dstStride
            pipe_barrier(PIPE_V);
        }
        for (int32_t rowmaxIdx = 2; rowmaxIdx < (n0Value / BLOCK_SIZE); ++rowmaxIdx) { // 循环比较(fm, 16)
            VecMaxFp32((__ubuf__ half *)tvUbuf, (__ubuf__ half *)tvUbuf,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + rowmaxIdx * fm * BLOCK_SIZE,
                fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        if (n0Value % BLOCK_SIZE > 0) {
            SetMaskFa(n0Value % BLOCK_SIZE);
            if (n0Value / BLOCK_SIZE > 0) {
                VecMaxFp32((__ubuf__ half *)tvUbuf, (__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + n0Value / BLOCK_SIZE * fm * BLOCK_SIZE,
                    fm, 1, 1, 1, 1, 1, 1);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }
        }
        if (n0Value < BLOCK_SIZE) {
            SetVcgMaskFa(n0Value);
        }
        vcgmax((__ubuf__ half *)lmUbuf, // 最后(fm, 16)求reduce max
            (__ubuf__ half *)tvUbuf,
            fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
            1, 1, 8);              // dstRepeatStride, srcBlockStride,
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        pipe_barrier(PIPE_V);

        if (initGgDm == 0) { // 需要update m_j
            VecMaxFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf,
                (__ubuf__ half *)gmUbuf, // m_j = max(m_j-1, m_cur)
                mD128, 1, 1, 1, 8, 8, 8);

            pipe_barrier(PIPE_V);

            VecSubFp32((__ubuf__ half *)dmUbuf + Pingflag * UB_HALF_LINE_SIZE, // dm = m_j-1 - m_j
                (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, mD128, 1, 1, 1, 8, 8, 8);

            pipe_barrier(PIPE_V);
        } else {
            CopyUbufToUbufFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf, 0, 1, fm / BLOCK_SIZE, 0, 0);

            pipe_barrier(PIPE_V);
        }

        CopyUbufToUbufFp32((__ubuf__ half *)gmUbuf, // 更新m_j
                (__ubuf__ half *)hmUbuf, 0, 1, fm / BLOCK_SIZE, 0, 0);
        initGgDm = 0;
        pipe_barrier(PIPE_V);
        ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)hmUbuf, fm); // (fm,) -> (fm, 16)
        for (int32_t vsubIdx = 0; vsubIdx < (n0 / BLOCK_SIZE); ++vsubIdx) { // (fm, n0)
            VecSubFp32((__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                (__ubuf__ half *)tvUbuf, fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
        }
        pipe_barrier(PIPE_V);
        // 2 for Repeatimes above 255
        for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
            vconv_f162f32((__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2,
                (__ubuf__ half *)lsUbuf + Pingflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                pSize / 2 / FLOAT_VECTOR_SIZE,            // 每次处理64个数*2=128字节, src 4 block, 对应dst 8个block;
                1, 1,                      // dstBlockStride srcBlockStride
                uint16_t(8), uint16_t(4)); // dstRepeatStride srcRepeatStride
        }
        pipe_barrier(PIPE_V);
        // 2. ################ Bmm1 Pong Starts #######################
        if (n1Value != -1) {
            if (kvSeqLenAlign <= STRIDE_UPPER_BOUND + n1) {
                copy_gm_to_cbuf((__cbuf__ half *)l1kPongBuf,
                                (__gm__ half *)gmSrck + (int64_t)srckOffset + Pongflag * pp_n_scalar * BLOCK_SIZE,
                                0, fk / BLOCK_SIZE, n1,                  // (n0, fk) double buffer
                                kvSeqLenAlign - n1, 0, PAD_NONE);
            } else {
                for (int32_t l1kBurstIdx = 0; l1kBurstIdx < (fk / BLOCK_SIZE); ++l1kBurstIdx) {
                    copy_gm_to_cbuf((__cbuf__ half *)l1kPongBuf + l1kBurstIdx * n1 * BLOCK_SIZE,
                        (__gm__ half *)gmSrck + (int64_t)srckOffset +
                        Pongflag * pp_n_scalar * BLOCK_SIZE + l1kBurstIdx * kvSeqLenAlign * BLOCK_SIZE,
                        0, 1, n1, 0, 0, PAD_NONE);
                }
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);    // // 2 means ping-pong
            load_cbuf_to_cb((__cb__ half *)l0bBuf + Pongflag * L0AB_HALF_BUF_SIZE, // Nz ->(transpose) nZ -> nZ
                    (__cbuf__ half *)l1kPongBuf, 0, fk * n1 / CUBE_MATRIX_SIZE, 1, 0, 0);

            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, Pongflag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, Pongflag + 2);
            wait_flag(PIPE_V, PIPE_M, Pongflag);

            mad((__cc__ float *)l0cBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                    (__cb__ half *)l0bBuf + Pongflag * L0AB_HALF_BUF_SIZE, m0Value, fk, n1Value, 1);

            set_flag(PIPE_M, PIPE_MTE1, Pongflag);
            // 2 for L0B Sync Flag
            set_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);
            set_flag(PIPE_M, PIPE_MTE1, Pongflag + 3);   // 3 means offset
            set_flag(PIPE_M, PIPE_V, Pongflag);
        }
        // 2. ################ Bmm1 Pong Ends #######################
        // 2 for Repeatimes above 255
        for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
            vexp((__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                (__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                pSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        // 2 for Repeatimes above 255
        for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
            vconv_f322f16((__ubuf__ half *)lpUbuf + Pingflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                (__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2, pSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);
        if (n0Value / BLOCK_SIZE > 1) {
            VecAddFp32((__ubuf__ float *)tvUbuf,
                (__ubuf__ float *)ls32Ubuf, // Nz (fm, n0)
                (__ubuf__ float *)ls32Ubuf + fm * BLOCK_SIZE, fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);

            pipe_barrier(PIPE_V);
        } else {
            CopyUbufToUbufFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)ls32Ubuf, 0, 1, fm * BLOCK_SIZE / 8, 0,
                0);
            pipe_barrier(PIPE_V);
        }
        for (int32_t rowsumIdx = 2; rowsumIdx < (n0Value / BLOCK_SIZE); ++rowsumIdx) {
            VecAddFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf,
                (__ubuf__ float *)ls32Ubuf + rowsumIdx * fm * BLOCK_SIZE,
                fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
        }
        set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
        if (n0Value % BLOCK_SIZE > 0) {
            SetMaskFa(n0Value % BLOCK_SIZE);
            if (n0Value / BLOCK_SIZE > 0) {
                VecAddFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf,
                    (__ubuf__ float *)ls32Ubuf + n0Value / BLOCK_SIZE * fm * BLOCK_SIZE, fm, 1, 1, 1, 2, 2, 2);
                pipe_barrier(PIPE_V);
                set_vector_mask(0x0, 0xffff);
            }
        } else {
            set_vector_mask(0x0, 0xffff);
        }
        vcadd((__ubuf__ float *)llUbuf + Pingflag * UB_FLOAT_LINE_SIZE,
            (__ubuf__ float *)tvUbuf, // (fm, 16) -> (fm, )
            fm, 1, 1, 2); // srcRepeatStride, fp32 2 block
        pipe_barrier(PIPE_V);
        set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
        set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);

        set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE3, Pingflag);
        wait_flag(PIPE_V, PIPE_MTE3, Pingflag);
        // 3. ################ Softmax Ping Ends #######################
        // 4. ################ Softmax Pong Starts #######################
        if (n1Value != -1) {
            wait_flag(PIPE_M, PIPE_V, Pongflag);
            wait_flag(PIPE_MTE3, PIPE_V, Pongflag);
            copy_matrix_cc_to_ubuf((__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__cc__ float *)l0cBuf + Pongflag * L0AB_HALF_BUF_SIZE, 0, 1, pSize / CUBE_MATRIX_SIZE, 0, 0,
                    CRMODE_F32toF16_NONE); // fp32 -> fp16

            set_flag(PIPE_V, PIPE_M, Pongflag);
            pipe_barrier(PIPE_V);

            // 3.1. mask(attention score * tor)
            if (float(tor) != float(1.0)) {
                vmuls((__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE, tor, pSize / 128, 1, 1, uint16_t(8),
                    uint16_t(8));
            }
            pipe_barrier(PIPE_V);
            //  2.1 decoder mask
            wait_flag(PIPE_MTE1, PIPE_MTE2, Pongflag + 2);
            if (gmSrcm != nullptr) {
                copy_gm_to_cbuf((__cbuf__ half *)l1maxkBufAddr + Pongflag * L0AB_HALF_BUF_SIZE,  // Nz load
                                (__gm__ half *)gmSrcm + srcmOffset + maskStride * pp_n_scalar,
                                0, n1 / BLOCK_SIZE, fm, maskStride - fm, 0, PAD_NONE);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
            if (gmSrcm != nullptr) {
                copy_cbuf_to_ubuf((__ubuf__ half *)loUbuf,  // Element-Wise load
                                (__cbuf__ half *)l1maxkBufAddr + Pongflag * L0AB_HALF_BUF_SIZE,
                                0, 1, fm * n1 / BLOCK_SIZE, 0, 0);
            }
            set_flag(PIPE_MTE1, PIPE_V, Pongflag);
            wait_flag(PIPE_MTE1, PIPE_V, Pongflag);
            set_flag(PIPE_MTE1, PIPE_MTE2, Pongflag + 2);    // 2 means offset
            if (gmSrcm != nullptr) {
                VecAddFp32((__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)loUbuf, fm * n1 / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
            // 3. softmax part
            if (n1Value / BLOCK_SIZE > 1) { // 前两个(fm, 16)求最大值
                VecMaxFp32((__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE, // Nz (fm, n0)
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE + fm * BLOCK_SIZE,
                    fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                    1, 1, 1,               // dstBlockStride src0BlockStride src1BlockStride
                    8, 8, 8);              // dstRepeatStride src0RepeatStride src1RepeatStride
                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32((__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    0, 1, fm, 0, 0); // dstStride
                pipe_barrier(PIPE_V);
            }
            for (int32_t rowmaxIdx = 2; rowmaxIdx < (n1Value / BLOCK_SIZE); ++rowmaxIdx) { // 循环比较(fm, 16)
                VecMaxFp32((__ubuf__ half *)tvUbuf, (__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE + rowmaxIdx * fm * BLOCK_SIZE,
                    fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            if (n1Value % BLOCK_SIZE > 0) {
                SetMaskFa(n1Value % BLOCK_SIZE);
                if (n1Value / BLOCK_SIZE > 0) {
                    VecMaxFp32((__ubuf__ half *)tvUbuf, (__ubuf__ half *)tvUbuf,
                        (__ubuf__ half *)lsUbuf +
                        Pongflag * L0AB_HALF_BUF_SIZE + n1Value / BLOCK_SIZE * fm * BLOCK_SIZE,
                        fm, 1, 1, 1, 1, 1, 1);
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }
            }
            if (n1Value < BLOCK_SIZE) {
                SetVcgMaskFa(n1Value);
            }
            vcgmax((__ubuf__ half *)lmUbuf, // 最后(fm, 16)求reduce max
                (__ubuf__ half *)tvUbuf,
                fm * BLOCK_SIZE / VECTOR_SIZE, // repeat
                1, 1, 8);              // dstRepeatStride, srcBlockStride,
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
            pipe_barrier(PIPE_V);

            if (initGgDm == 0) { // 需要update m_j
                VecMaxFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf,
                    (__ubuf__ half *)gmUbuf, // m_j = max(m_j-1, m_cur)
                    mD128, 1, 1, 1, 8, 8, 8);

                pipe_barrier(PIPE_V);

                VecSubFp32((__ubuf__ half *)dmUbuf + Pongflag * UB_HALF_LINE_SIZE, // dm = m_j-1 - m_j
                    (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, mD128, 1, 1, 1, 8, 8, 8);

                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf, 0, 1, fm / BLOCK_SIZE, 0, 0);

                pipe_barrier(PIPE_V);
            }

            CopyUbufToUbufFp32((__ubuf__ half *)gmUbuf, // 更新m_j
                    (__ubuf__ half *)hmUbuf, 0, 1, fm / BLOCK_SIZE, 0, 0);
            initGgDm = 0;
            pipe_barrier(PIPE_V);
            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)hmUbuf, fm); // (fm,) -> (fm, 16)
            for (int32_t vsubIdx = 0; vsubIdx < (n0 / BLOCK_SIZE); ++vsubIdx) { // (fm, n0)
                VecSubFp32((__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE + vsubIdx * fm * BLOCK_SIZE,
                    (__ubuf__ half *)tvUbuf, fm * BLOCK_SIZE / VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
            // 2 for Repeatimes above 255
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f162f32((__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2,
                    (__ubuf__ half *)lsUbuf + Pongflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                    pSize / 2 / FLOAT_VECTOR_SIZE,            // 每次处理64个数*2=128字节, src 4 block, 对应dst 8个block;
                    1, 1,                      // dstBlockStride srcBlockStride
                    uint16_t(8), uint16_t(4)); // dstRepeatStride srcRepeatStride
            }
            pipe_barrier(PIPE_V);
        }
        // 5. ################ Bmm2 Ping Starts #######################
        if (m0Value == 1) {
            copy_ubuf_to_cbuf((__cbuf__ half *)l1pPingBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)lpUbuf + Pingflag * L0AB_HALF_BUF_SIZE, 0,
                n0 / BLOCK_SIZE, 1, fm - 1, 0);  // Gap
        } else {
            copy_ubuf_to_cbuf((__cbuf__ half *)l1pPingBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)lpUbuf + Pingflag * L0AB_HALF_BUF_SIZE, 0,
                1, pSize / BLOCK_SIZE, 0, 0);
        }
        set_flag(PIPE_MTE3, PIPE_V, Pingflag);
        set_flag(PIPE_MTE3, PIPE_MTE1, Pingflag);
        wait_flag(PIPE_MTE3, PIPE_MTE1, Pingflag);
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag);
        if (n1Value != -1) {
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag + 3);    // 3 means offset
        }
        // 16 is blocksize in format zN
        if (m0Value == 1) {
            load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__cbuf__ half *)l1pPingBuf + Pingflag * L0AB_HALF_BUF_SIZE, 0, 1, 1, 0, 0);
        } else if (n0 == BLOCK_SIZE) {
            load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__cbuf__ half *)l1pPingBuf + Pingflag * L0AB_HALF_BUF_SIZE, 0, fm / BLOCK_SIZE, 1, 0, 0);
        } else {
            for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, n0)
                load_cbuf_to_ca((__ca__ half *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE +
                    l0aLoadIdx * n0 * BLOCK_SIZE,
                    (__cbuf__ half *)l1pPingBuf + Pingflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * CUBE_MATRIX_SIZE,
                    0, n0 / BLOCK_SIZE, fm / BLOCK_SIZE, 0, 0);
            }
        }
        set_flag(PIPE_MTE1, PIPE_M, Pingflag);
        wait_flag(PIPE_MTE1, PIPE_M, Pingflag);
        // 4. bmm2 part
        if (kvSeqLenAlign <= STRIDE_UPPER_BOUND + n0) {
            copy_gm_to_cbuf((__cbuf__ half *)l1vPingBuf, // load V Nz
                (__gm__ half *)gmSrcv + (int64_t)srcvOffset, 0, fd / BLOCK_SIZE, n0,
                kvSeqLenAlign - n0, 0, PAD_NONE);
        } else {
            for (int32_t l1vBurstIdx = 0; l1vBurstIdx < (fd / BLOCK_SIZE); ++l1vBurstIdx) {
                copy_gm_to_cbuf((__cbuf__ half *)l1vPingBuf + l1vBurstIdx * n0 * BLOCK_SIZE,
                    (__gm__ half *)gmSrcv + (int64_t)srcvOffset +
                    l1vBurstIdx * kvSeqLenAlign * BLOCK_SIZE,
                    0, 1, n0, 0, 0, PAD_NONE);
            }
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
        wait_flag(PIPE_MTE2, PIPE_MTE1, Pingflag);
        // 2 for double buffer
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);

        // 16 is blocksize in format zN
        if (fd == 16) {
            load_cbuf_to_cb((__cb__ half *)l0bBuf + Pingflag * L0AB_HALF_BUF_SIZE,
                (__cbuf__ half *)l1vPingBuf, 0, n0 / BLOCK_SIZE, 1, 0, 1);
        } else {
            for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (n0 / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                load_cbuf_to_cb((__cb__ half *)l0bBuf + Pingflag * L0AB_HALF_BUF_SIZE +
                    l0bLoadIdx * fd * BLOCK_SIZE,
                    (__cbuf__ half *)l1vPingBuf + l0bLoadIdx * CUBE_MATRIX_SIZE, 0,
                    fd / BLOCK_SIZE, n0 / BLOCK_SIZE, 0, 1); // transpose
            }
        }

        // 2 for double buffer
        set_flag(PIPE_MTE1, PIPE_M, Pingflag + 2);
        // 2 for double buffer
        wait_flag(PIPE_MTE1, PIPE_M, Pingflag + 2);
        wait_flag(PIPE_V, PIPE_M, Pingflag);

        mad((__cc__ float *)l0cBuf + Pingflag * L0AB_HALF_BUF_SIZE,
            (__ca__ __fp16 *)l0aBuf + Pingflag * L0AB_HALF_BUF_SIZE,
            (__cb__ __fp16 *)l0bBuf + Pingflag * L0AB_HALF_BUF_SIZE, m0Value, n0Value, fd, 1);

        set_flag(PIPE_M, PIPE_MTE1, Pingflag);
        // 2 for double buffer
        set_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);    // 2 is offset
        wait_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);   // 2 is offset
        set_flag(PIPE_M, PIPE_MTE1, Pingflag + 2);    // 2 is offset
        set_flag(PIPE_M, PIPE_V, Pingflag);
        wait_flag(PIPE_M, PIPE_V, Pingflag);
        // 5. ################ Bmm2 Ping Ends #######################
        if (n1Value != -1) {
            // 2 for Repeatimes above 255
            for (int32_t vexpIdx = 0; vexpIdx < 2; ++vexpIdx) {
                vexp((__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                    (__ubuf__ float *)ls32Ubuf + vexpIdx * pSize / 2,
                    pSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 8, 8);
            }
            pipe_barrier(PIPE_V);

            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f322f16((__ubuf__ half *)lpUbuf + Pongflag * L0AB_HALF_BUF_SIZE + vconvIdx * pSize / 2,
                    (__ubuf__ float *)ls32Ubuf + vconvIdx * pSize / 2, pSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 4, 8);
            }
            pipe_barrier(PIPE_V);
            if (n1Value / BLOCK_SIZE > 1) {
                VecAddFp32((__ubuf__ float *)tvUbuf,
                    (__ubuf__ float *)ls32Ubuf, // Nz (fm, n0)
                    (__ubuf__ float *)ls32Ubuf + fm * BLOCK_SIZE,
                    fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);

                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)ls32Ubuf, 0, 1, fm * BLOCK_SIZE / 8, 0,
                    0);
                pipe_barrier(PIPE_V);
            }
            for (int32_t rowsumIdx = 2; rowsumIdx < (n1Value / BLOCK_SIZE); ++rowsumIdx) {
                VecAddFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf,
                    (__ubuf__ float *)ls32Ubuf + rowsumIdx * fm * BLOCK_SIZE,
                    fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
            if (n1Value % BLOCK_SIZE > 0) {
                SetMaskFa(n1Value % BLOCK_SIZE);
                if (n1Value / BLOCK_SIZE > 0) {
                    VecAddFp32((__ubuf__ float *)tvUbuf, (__ubuf__ float *)tvUbuf,
                        (__ubuf__ float *)ls32Ubuf + n1Value / BLOCK_SIZE * fm * BLOCK_SIZE, fm, 1, 1, 1, 2, 2, 2);
                    pipe_barrier(PIPE_V);
                    set_vector_mask(0x0, 0xffff);
                }
            } else {
                set_vector_mask(0x0, 0xffff);
            }
            vcadd((__ubuf__ float *)llUbuf + Pongflag * UB_FLOAT_LINE_SIZE,
                (__ubuf__ float *)tvUbuf, // (fm, 16) -> (fm, )
                fm, 1, 1,  // srcBlockStride
                2); // srcRepeatStride, fp32 2 block
            pipe_barrier(PIPE_V);
            set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
            set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);

            set_flag(PIPE_V, PIPE_MTE3, Pongflag);
            wait_flag(PIPE_V, PIPE_MTE3, Pongflag);
        }
        // 4. ################ Softmax Pong Ends #######################
        // 7. ################ Update Ping Starts #######################
        copy_matrix_cc_to_ubuf((__ubuf__ float *)loUbuf,
            (__cc__ float *)l0cBuf + Pingflag * L0AB_HALF_BUF_SIZE, 0, 1, oSize / CUBE_MATRIX_SIZE, 0, 0,
            CRMODE_NONE);
        set_flag(PIPE_V, PIPE_M, Pingflag);
        pipe_barrier(PIPE_V);

        // 5. update for outer loop
        if (initGgO == 0) { // 需要更新O
            vconv_f162f32((__ubuf__ float *)tvUbuf, (__ubuf__ half *)dmUbuf + Pingflag * UB_HALF_LINE_SIZE,
                mD64, 1, 1, uint16_t(8), uint16_t(4));

            pipe_barrier(PIPE_V);

            vexp((__ubuf__ float *)tvUbuf, // e^(m_j-1 - m_j)
                (__ubuf__ float *)tvUbuf, mD64, 1, 1, uint16_t(8), uint16_t(8));
            pipe_barrier(PIPE_V);
            VecMulFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1
                (__ubuf__ float *)tvUbuf, (__ubuf__ float *)glUbuf, mD64, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            VecAddFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1 + row_sum(Pj)
                (__ubuf__ float *)glUbuf, (__ubuf__ float *)llUbuf + Pingflag * UB_FLOAT_LINE_SIZE, mD64, 1,
                1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, // broadcast(m_j-1 - m_j)
                (__ubuf__ half *)dmUbuf + Pingflag * UB_HALF_LINE_SIZE, fm);

            vconv_f162f32((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, (__ubuf__ half *)tvUbuf,
                fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, uint16_t(8), uint16_t(4));
            pipe_barrier(PIPE_V);

            vexp((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, // e^broadcast(m_j-1 - m_j)
                (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, uint16_t(8),
                uint16_t(8));
            pipe_barrier(PIPE_V);

            if (vmPingpongFlag == 1) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                vmPingpongFlag = 0;
            }

            for (int32_t vmulIdx = 0; vmulIdx < (fd / BLOCK_SIZE); ++vmulIdx) { // e^broadcast(m_j-1 - m_j) * Oj_1
                VecMulFp32((__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                    (__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                    (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                    fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);

            // 2 for double buffer
            for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) { // update Oj
                VecAddFp32((__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                (__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                (__ubuf__ float *)loUbuf + vaddIdx * oSize / 2, oSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);
        } else {
            CopyUbufToUbufFp32((__ubuf__ float *)glUbuf,
                (__ubuf__ float *)llUbuf + Pingflag * UB_FLOAT_LINE_SIZE, 0, 1, fm / 8, 0, 0);
            pipe_barrier(PIPE_V);
            if (vmPingpongFlag == 1) {
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                vmPingpongFlag = 0;
            }

            CopyUbufToUbufFp32((__ubuf__ float *)goUbuf, (__ubuf__ float *)loUbuf, 0, 1, oSize / 8, 0, 0);
            pipe_barrier(PIPE_V);
        }
        if (n1Value == -1) {
            set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
        }
        pipe_barrier(PIPE_V);
        initGgO = 0;
        // 7. ################ Update Ping Ends #######################
        // 6. ################ Bmm2 Pong Starts #######################
        if (n1Value != -1) {
            if (m0Value == 1) {
                copy_ubuf_to_cbuf((__cbuf__ half *)l1pPongBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lpUbuf + Pongflag * L0AB_HALF_BUF_SIZE, 0, n1 / BLOCK_SIZE,
                    1, fm - 1, 0);  // Gap
            } else {
                copy_ubuf_to_cbuf((__cbuf__ half *)l1pPongBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lpUbuf + Pongflag * L0AB_HALF_BUF_SIZE, 0,
                    1, pSize / BLOCK_SIZE, 0, 0);
            }
            set_flag(PIPE_MTE3, PIPE_V, Pongflag);
            set_flag(PIPE_MTE3, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_MTE3, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag);

            // 16 is blocksize in format zN
            if (m0Value == 1) {
                load_cbuf_to_ca((__ca__ half *)l0aBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1pPongBuf + Pongflag * L0AB_HALF_BUF_SIZE, 0, 1, 1, 0, 0);
            } else if (n1 == BLOCK_SIZE) {
                load_cbuf_to_ca((__ca__ half *)l0aBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1pPongBuf + Pongflag * L0AB_HALF_BUF_SIZE, 0, fm / BLOCK_SIZE, 1, 0, 0);
            } else {
                for (int32_t l0aLoadIdx = 0; l0aLoadIdx < (fm / BLOCK_SIZE); ++l0aLoadIdx) { // (fm, n0)
                    load_cbuf_to_ca((__ca__ half *)l0aBuf + Pongflag * L0AB_HALF_BUF_SIZE +
                        l0aLoadIdx * n1 * BLOCK_SIZE,
                        (__cbuf__ half *)l1pPongBuf + Pongflag * L0AB_HALF_BUF_SIZE + l0aLoadIdx * CUBE_MATRIX_SIZE,
                        0, n1 / BLOCK_SIZE, fm / BLOCK_SIZE, 0, 0);
                }
            }
            set_flag(PIPE_MTE1, PIPE_M, Pongflag);
            wait_flag(PIPE_MTE1, PIPE_M, Pongflag);
            // 4. bmm2 part
            if (kvSeqLenAlign <= STRIDE_UPPER_BOUND + n1) {
                copy_gm_to_cbuf((__cbuf__ half *)l1vPongBuf, // load V Nz
                    (__gm__ half *)gmSrcv + (int64_t)srcvOffset +
                    Pongflag * pp_n_scalar * BLOCK_SIZE, 0, fd / BLOCK_SIZE, n1,
                    kvSeqLenAlign - n1, 0, PAD_NONE);
            } else {
                for (int32_t l1vBurstIdx = 0; l1vBurstIdx < (fd / BLOCK_SIZE); ++l1vBurstIdx) {
                    copy_gm_to_cbuf((__cbuf__ half *)l1vPongBuf +
                        l1vBurstIdx * n1 * BLOCK_SIZE,
                        (__gm__ half *)gmSrcv + (int64_t)srcvOffset + Pongflag * pp_n_scalar * BLOCK_SIZE +
                        l1vBurstIdx * kvSeqLenAlign * BLOCK_SIZE,
                        0, 1, n1, 0, 0, PAD_NONE);
                }
            }

            set_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, Pongflag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);

            // 16 is blocksize in format zN
            if (fd == 16) {
                load_cbuf_to_cb((__cb__ half *)l0bBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1vPongBuf, 0, n1 / BLOCK_SIZE, 1, 0, 1);
            } else {
                for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (n1 / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                    load_cbuf_to_cb((__cb__ half *)l0bBuf + Pongflag * L0AB_HALF_BUF_SIZE +
                        l0bLoadIdx * fd * BLOCK_SIZE,
                        (__cbuf__ half *)l1vPongBuf + l0bLoadIdx * CUBE_MATRIX_SIZE, 0,
                        fd / BLOCK_SIZE, n1 / BLOCK_SIZE, 0, 1); // transpose
                }
            }

            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, Pongflag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, Pongflag + 2);
            wait_flag(PIPE_V, PIPE_M, Pongflag);

            mad((__cc__ float *)l0cBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                (__ca__ __fp16 *)l0aBuf + Pongflag * L0AB_HALF_BUF_SIZE,
                (__cb__ __fp16 *)l0bBuf + Pongflag * L0AB_HALF_BUF_SIZE, m1Value, n1Value, fd, 1);

            set_flag(PIPE_M, PIPE_MTE1, Pongflag);
            // 2 for double buffer
            set_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);   // 2 is offset
            wait_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);  // 2 is offset
            set_flag(PIPE_M, PIPE_MTE1, Pongflag + 2);   // 2 is offset
            set_flag(PIPE_M, PIPE_V, Pongflag);
            wait_flag(PIPE_M, PIPE_V, Pongflag);
        }
        // 6. ################ Bmm2 Pong Ends #######################
        // 8. ################ Update Pong Starts #######################
        if (n1Value != -1) {
            copy_matrix_cc_to_ubuf((__ubuf__ float *)loUbuf,
                (__cc__ float *)l0cBuf + Pongflag * L0AB_HALF_BUF_SIZE, 0, 1, oSize / CUBE_MATRIX_SIZE, 0, 0,
                CRMODE_NONE);
            set_flag(PIPE_V, PIPE_M, Pongflag);
            pipe_barrier(PIPE_V);

            // 5. update for outer loop
            if (initGgO == 0) { // 需要更新O
                vconv_f162f32((__ubuf__ float *)tvUbuf, (__ubuf__ half *)dmUbuf + Pongflag * UB_HALF_LINE_SIZE,
                    mD64, 1, 1, uint16_t(8), uint16_t(4));

                pipe_barrier(PIPE_V);

                vexp((__ubuf__ float *)tvUbuf, // e^(m_j-1 - m_j)
                    (__ubuf__ float *)tvUbuf, mD64, 1, 1, uint16_t(8), uint16_t(8));
                pipe_barrier(PIPE_V);
                VecMulFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1
                    (__ubuf__ float *)tvUbuf, (__ubuf__ float *)glUbuf, mD64, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);

                VecAddFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1 + row_sum(Pj)
                    (__ubuf__ float *)glUbuf, (__ubuf__ float *)llUbuf + Pongflag * UB_FLOAT_LINE_SIZE, mD64, 1,
                    1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);

                ExpandToBlockHalf((__ubuf__ half *)tvUbuf, // broadcast(m_j-1 - m_j)
                    (__ubuf__ half *)dmUbuf + Pongflag * UB_HALF_LINE_SIZE, fm);

                vconv_f162f32((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, (__ubuf__ half *)tvUbuf,
                    fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, uint16_t(8), uint16_t(4));
                pipe_barrier(PIPE_V);

                vexp((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, // e^broadcast(m_j-1 - m_j)
                    (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                    fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, uint16_t(8),
                    uint16_t(8));
                pipe_barrier(PIPE_V);

                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }

                for (int32_t vmulIdx = 0; vmulIdx < (fd / BLOCK_SIZE); ++vmulIdx) { // e^broadcast(m_j-1 - m_j) * Oj_1
                    VecMulFp32((__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                        (__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                        (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2,
                        fm * BLOCK_SIZE / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
                }
                pipe_barrier(PIPE_V);

                // 2 for double buffer
                for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) { // update Oj
                    VecAddFp32((__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                        (__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                        (__ubuf__ float *)loUbuf + vaddIdx * oSize / 2,
                        oSize / 2 / FLOAT_VECTOR_SIZE, 1, 1, 1, 8, 8, 8);
                }
                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32((__ubuf__ float *)glUbuf,
                    (__ubuf__ float *)llUbuf + Pongflag * UB_FLOAT_LINE_SIZE, 0, 1, fm / 8, 0, 0);
                pipe_barrier(PIPE_V);
                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }

                CopyUbufToUbufFp32((__ubuf__ float *)goUbuf, (__ubuf__ float *)loUbuf, 0, 1, oSize / 8, 0, 0);
                pipe_barrier(PIPE_V);
            }
            set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
            pipe_barrier(PIPE_V);
            initGgO = 0;
        }
        // 8. ################ Update Pong Ends #######################
        // 9. ################ Line Output Starts #####################
        if (wrapO == 1) {
            vconv_f322f16((__ubuf__ half *)glUbuf, // lj fp32->fp16
                (__ubuf__ float *)glUbuf, mD64, 1, 1, uint16_t(4), uint16_t(8));

            pipe_barrier(PIPE_V);
            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f322f16((__ubuf__ half *)goUbuf + vconvIdx * oSize / 2, // Oi fp32->f16
                    (__ubuf__ float *)goUbuf + vconvIdx * oSize / 2, oSize / 2 / FLOAT_VECTOR_SIZE,
                    1, 1, uint16_t(4), uint16_t(8));
                pipe_barrier(PIPE_V);
            }

            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)glUbuf, fm); // 广播

            for (int32_t vdivIdx = 0; vdivIdx < (fd / BLOCK_SIZE); ++vdivIdx) { // Oi / li
                VecDivFp32((__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                     (__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                     (__ubuf__ half *)tvUbuf, m0Value * BLOCK_SIZE / VECTOR_SIZE,
                     1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            int32_t blockV = VECTOR_SIZE / BLOCK_SIZE;
            if (m0Value % blockV != 0) {
                SetMaskFa(m0Value * BLOCK_SIZE % 128);    // 128 is 128 byte
                VecDivFp32((__ubuf__ half *)goUbuf + m0Value * BLOCK_SIZE / 128 * 128,
                    (__ubuf__ half *)goUbuf + m0Value * BLOCK_SIZE / 128 * 128,
                    (__ubuf__ half *)tvUbuf + m0Value / blockV * blockV * 16, fd / BLOCK_SIZE,
                    1, 1, 1, fm, fm, 0);
                set_vector_mask(-1, -1);
            }
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);

            // move O to gm
            if (qSeqLenAlign <= STRIDE_UPPER_BOUND + fm) {
                copy_ubuf_to_gm((__gm__ half *)gmDsto + (int64_t)dstoOffset, // (fm, fd)
                                (__ubuf__ half *)goUbuf, 0, fd / BLOCK_SIZE,  // nburst
                                fm, 0, qSeqLenAlign - fm);   // dstStride
            } else {
                for (int32_t gmBurstIdx = 0; gmBurstIdx < (fd / BLOCK_SIZE); ++gmBurstIdx) {
                    copy_ubuf_to_gm((__gm__ half *)gmDsto +
                    (int64_t)dstoOffset + gmBurstIdx * qSeqLenAlign * BLOCK_SIZE,
                    (__ubuf__ half *)goUbuf + gmBurstIdx * fm * BLOCK_SIZE, 0, 1, fm, 0, 0);
                }
            }

            if (vmPingpongFlag == 0) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                vmPingpongFlag = 1;
            }
        }
        // 9. ################ Line Output Ends #####################

        set_flag(PIPE_MTE1, PIPE_MTE2, Pingflag);
        set_flag(PIPE_MTE1, PIPE_MTE2, Pongflag);
    }

     __aicore__ inline void FlashAttentionNzDecoderCompute(const int32_t fm, const int32_t fn, const int32_t fk,
        const int32_t m0Value, const int32_t n0Value, const int32_t fnLoop = 1)
    {
        if (fnLoop == 0) {
            return;
        }
        int32_t fn0 = fn;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1qBuf = l1qBufAddr;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1kBuf = l1kBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1vBuf = l1vBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        // 4 for ping-pong memory offset in L1
        __cbuf__ uint8_t *l1pBuf = l1pBufAddr + l1PingpongFlag * 4 * L1_UINT8_BLOCK_SIZE;
        int32_t oSize = fm * fk;
        int32_t mD64 = (fm + 63) / 64;       // fm round up to 64
        int32_t mD128 = (fm + 127) / 128;    // fm round up to 128
        int32_t initGg = (initG == 1) ? 1 : 0;
        // inner loop
        for (int32_t fnIdx = 0; fnIdx < 1; ++fnIdx) {
            // 1. bmm1 part
            int32_t n0 = fn0;
            int32_t pSize = fm * n0;
            if (initGg == 1) {
                // (fm, fk)
                // Load Q Nz Gm to ND L1
                copy_gm_to_cbuf((__cbuf__ half *)l1qBuf, (__gm__ half *)gmSrcq + (int64_t)srcqOffset, 0,
                    fk / BLOCK_SIZE, 1, qSeqLenAlign - 1, 0, PAD_NONE);
                set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
                wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            }
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);
            // Load ND Q to L0A
            load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1qBuf, 0,
                    1, 1, 0, 0);
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            copy_gm_to_cbuf((__cbuf__ half *)l1kBuf + fnIdx * L0AB_HALF_BUF_SIZE,
                            (__gm__ half *)gmSrck + (int64_t)fnIdx * fn0 * BLOCK_SIZE + (int64_t)srckOffset,
                            0, fk / BLOCK_SIZE, n0, kvSeqLenAlign - n0, 0, PAD_NONE);
            set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, // Nz ->(transpose) nZ -> nZ
                (__cbuf__ half *)l1kBuf + fnIdx * L0AB_HALF_BUF_SIZE, 0, fk * n0 / CUBE_MATRIX_SIZE, 1, 0, 0);
            // load V to L1
            copy_gm_to_cbuf((__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE, // load V Nz
                (__gm__ half *)gmSrcv + fnIdx * fn0 * BLOCK_SIZE + (int64_t)srcvOffset, 0, fk / BLOCK_SIZE, n0,
                kvSeqLenAlign - n0, 0, PAD_NONE);
            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            wait_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            mad((__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, m0Value, fk, n0Value, 1);
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);
            // 2 for double buffer
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            set_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_MTE3, PIPE_V, ibPingpongFlag);

            // [#####] Vector calc starts here. We got fm * fn from L0c.
            copy_matrix_cc_to_ubuf((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, 0, 1, pSize / CUBE_MATRIX_SIZE, 0, 0,
                CRMODE_F32toF16_NONE); // fp32 -> fp16
            set_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            pipe_barrier(PIPE_V);
            // 2. mask(attention score * tor)
            vmuls((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                tor, (n0 + 127) / 128,          // 127、128 ues to calculate repeat, n0 round up to 128
                1, fm,                     // dstBlockStride, srcBlockStride
                8, fm * 8);                // dstRepeatStride, srcRepeatStride
            pipe_barrier(PIPE_V);

            //  2.1 decoder mask
            set_flag(PIPE_V, PIPE_MTE2, ibPingpongFlag);
            wait_flag(PIPE_V, PIPE_MTE2, ibPingpongFlag);
            if (gmSrcm != nullptr) {
                copy_gm_to_ubuf((__ubuf__ half *)loUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,  // Nz load
                                (__gm__ half *)gmSrcm + srcmOffset,
                                0, n0 / BLOCK_SIZE, 1,                       // lenBurst
                                maskStride - 1,          // srcStride，尾-头,32byte
                                0);
            }
            set_flag(PIPE_MTE2, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_MTE2, PIPE_V, ibPingpongFlag);

            int32_t repeatMaskAdd = n0Value / 128;     // 128: repeat
            if (gmSrcm != nullptr) {
                VecAddFp32((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)loUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, repeatMaskAdd, 1, 1, 1, 8, 8, 8);
            }
            pipe_barrier(PIPE_V);

            if (gmSrcm != nullptr && n0Value % 128 != 0) {    // 128: n0Value
                SetMaskFa(n0Value % 128);  // 128: mask
                VecAddFp32((__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + repeatMaskAdd * 128,
                    (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + repeatMaskAdd * 128,
                    (__ubuf__ half *)loUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + repeatMaskAdd * 128,
                    1, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
            }
            // 3. softmax part
            if (n0Value <= 128) {                 // 128: mask
                if (n0Value != 128) {            // 128: mask
                    SetMaskFa(n0Value % 128);    // 128: mask
                }
                vcmax((__ubuf__ half *)lmUbuf,
                      (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                      1,     // repeat, fm is always 16
                      1, 1, 8);
                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32(
                    (__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, 0,
                    1,          // nBurst
                    8, 8, 8           // dstStride
                );
                pipe_barrier(PIPE_V);
                if (n0Value % 128 != 0) {        // 128: mask
                    SetMaskFa(n0Value % 128);    // 128: mask
                }
                VecMaxFp32((__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)tvUbuf,
                    (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE + 128,
                    1, 1, 1, 1,
                    8, 8, 8);

                pipe_barrier(PIPE_V);
                set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
                vcmax((__ubuf__ half *)lmUbuf,
                (__ubuf__ half *)tvUbuf,
                1,     // repeat, fm is always 16
                1, 1, 8);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask(-1, -1);
            if (initGg == 0) { // need update m_j, (fm, )
                VecMaxFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf, (__ubuf__ half *)gmUbuf,
                    1, 1, 1, 1,
                    8, 8, 8);
                pipe_barrier(PIPE_V);
                VecSubFp32((__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE,
                    (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, 1, 1, 1, 1,
                    8, 8, 8);
                pipe_barrier(PIPE_V);
            } else {
                CopyUbufToUbufFp32((__ubuf__ half *)hmUbuf, (__ubuf__ half *)lmUbuf, 0, 1, 1, 0, 0);
                pipe_barrier(PIPE_V);
            }
            // update m_j
            CopyUbufToUbufFp32(
                (__ubuf__ half *)gmUbuf, (__ubuf__ half *)hmUbuf, 0,
                1,
                1,
                0, 0);
            pipe_barrier(PIPE_V);
            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)hmUbuf, fm); // (fm,) -> (fm, 16)
            VecSubFp32(
                (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ubuf__ half *)tvUbuf,
                (n0 + 127) / 128,
                1, 1, 0,
                8, 8, 0);

            pipe_barrier(PIPE_V);
            vconv_f162f32((__ubuf__ float *)ls32Ubuf, (__ubuf__ half *)lsUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (n0 + 63) / 64, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            vexp((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                (n0 + 63) / 64, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vconv_f322f16((__ubuf__ half *)lpUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, (__ubuf__ float *)ls32Ubuf,
                (n0 + 63) / 64, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
            if (n0Value < 64) {         // 64: noValue
                if (n0Value != 64) {    // 64: noValue
                    set_vector_mask(0x0, ((long)1 << n0Value) - 1);
                }
                vcadd((__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE, (__ubuf__ float *)ls32Ubuf,
                    1, 1, 1,  // srcBlockStride
                    2);
                set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
            } else {
                for (int64_t vcalcIdx = 1; vcalcIdx < n0Value / 64; vcalcIdx++) {    // 64: noValue
                    VecAddFp32((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                        (__ubuf__ float *)ls32Ubuf + vcalcIdx * 64,    // 64: noValue
                        1, 1, 1, 1,
                        8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                if (n0Value % 64 != 0) {        // 64: noValue
                    SetMaskFa(n0Value % 64);    // 64: noValue
                    VecAddFp32((__ubuf__ float *)ls32Ubuf, (__ubuf__ float *)ls32Ubuf,
                        (__ubuf__ float *)ls32Ubuf + n0Value / 64 * 64,    // 64: noValue
                        1, 1, 1, 1,
                        8, 8, 8);
                    pipe_barrier(PIPE_V);
                    set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
                }

                vcadd((__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE, (__ubuf__ float *)ls32Ubuf,
                      1, 1, 1,  // srcBlockStride
                      2);
            }
            pipe_barrier(PIPE_V);
            set_ctrl(sbitset0(get_ctrl(), 56)); // 56为寄存器参数
            set_vector_mask(0xffffffffffffffff, 0xffffffffffffffff);
            set_flag(PIPE_V, PIPE_MTE3, ibPingpongFlag);
            wait_flag(PIPE_V, PIPE_MTE3, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_MTE3, l1PingpongFlag);
            // Load Nz UB to L1 ND
            copy_ubuf_to_cbuf((__cbuf__ half *)l1pBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__ubuf__ half *)lpUbuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, 0,
                    1,                      // nBurst
                    n0 / 16,                // lenBurst
                    0, 0);             // srcGap, dstGap
            set_flag(PIPE_MTE3, PIPE_V, ibPingpongFlag);
            set_flag(PIPE_MTE3, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE3, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);

            load_cbuf_to_ca((__ca__ half *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1pBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, 0, 1, 1, 0, 0);

            set_flag(PIPE_MTE1, PIPE_MTE3, l1PingpongFlag);
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag);

            // 4. bmm2 part
            set_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            wait_flag(PIPE_MTE2, PIPE_MTE1, l1PingpongFlag);
            // 2 for double buffer
            wait_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            // 16 is blocksize in format zN
            if (fk == 16) {
                load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                    (__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE, 0, n0 / BLOCK_SIZE, 1, 0, 1);
            } else {
                for (int32_t l0bLoadIdx = 0; l0bLoadIdx < (n0 / BLOCK_SIZE); ++l0bLoadIdx) { // Nz -> nZ
                    load_cbuf_to_cb((__cb__ half *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE +
                        l0bLoadIdx * fk * BLOCK_SIZE,
                        (__cbuf__ half *)l1vBuf + fnIdx * L0AB_HALF_BUF_SIZE + l0bLoadIdx * CUBE_MATRIX_SIZE, 0,
                        fk / BLOCK_SIZE, n0 / BLOCK_SIZE, 0,
                        1); // transpose
                }
            }
            // 2 for double buffer
            set_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            // 2 for double buffer
            wait_flag(PIPE_MTE1, PIPE_M, ibPingpongFlag + 2);
            wait_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            mad((__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__ca__ __fp16 *)l0aBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE,
                (__cb__ __fp16 *)l0bBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, m0Value, n0Value, fk, 1);
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag);
            // 2 for double buffer
            set_flag(PIPE_M, PIPE_MTE1, ibPingpongFlag + 2);
            set_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            wait_flag(PIPE_M, PIPE_V, ibPingpongFlag);
            copy_matrix_cc_to_ubuf((__ubuf__ float *)loUbuf,
                (__cc__ float *)l0cBuf + ibPingpongFlag * L0AB_HALF_BUF_SIZE, 0, 1, oSize / CUBE_MATRIX_SIZE, 0, 0,
                CRMODE_NONE);
            set_flag(PIPE_V, PIPE_M, ibPingpongFlag);
            pipe_barrier(PIPE_V);
            // 5. update for outer loop
            if (initGg == 0) { // 需要更新O
                vconv_f162f32((__ubuf__ float *)tvUbuf, (__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE,
                    mD64, 1, 1, uint16_t(8), uint16_t(4));
                pipe_barrier(PIPE_V);
                vexp((__ubuf__ float *)tvUbuf, // e^(m_j-1 - m_j)
                    (__ubuf__ float *)tvUbuf, mD64, 1, 1, uint16_t(8), uint16_t(8));
                pipe_barrier(PIPE_V);
                VecMulFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1
                    (__ubuf__ float *)tvUbuf, (__ubuf__ float *)glUbuf, mD64, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                VecAddFp32((__ubuf__ float *)glUbuf, // e^(m_j-1 - m_j) * l_j-1 + row_sum(Pj)
                    (__ubuf__ float *)glUbuf, (__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE, mD64, 1,
                    1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                ExpandToBlockHalf((__ubuf__ half *)tvUbuf, // broadcast(m_j-1 - m_j)
                    (__ubuf__ half *)dmUbuf + ibPingpongFlag * UB_HALF_LINE_SIZE, fm);
                vconv_f162f32((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, (__ubuf__ half *)tvUbuf,
                    fm * BLOCK_SIZE / 64, 1, 1, uint16_t(8), uint16_t(4));
                pipe_barrier(PIPE_V);
                vexp((__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, // e^broadcast(m_j-1 - m_j)
                    (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, fm * BLOCK_SIZE / 64, 1, 1, uint16_t(8),
                    uint16_t(8));
                pipe_barrier(PIPE_V);
                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }
                for (int32_t vmulIdx = 0; vmulIdx < (fk / BLOCK_SIZE); ++vmulIdx) { // e^broadcast(m_j-1 - m_j) * Oj_1
                    VecMulFp32((__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                        (__ubuf__ float *)goUbuf + vmulIdx * fm * BLOCK_SIZE,
                        (__ubuf__ float *)tvUbuf + fm * BLOCK_SIZE / 2, fm * BLOCK_SIZE / 64, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
                // 2 for double buffer
                for (int32_t vaddIdx = 0; vaddIdx < 2; ++vaddIdx) { // update Oj
                    VecAddFp32((__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                        (__ubuf__ float *)goUbuf + vaddIdx * oSize / 2,
                        (__ubuf__ float *)loUbuf + vaddIdx * oSize / 2, oSize / 2 / 64, 1, 1, 1, 8, 8, 8);
                    pipe_barrier(PIPE_V);
                }
            } else {
                CopyUbufToUbufFp32((__ubuf__ float *)glUbuf,
                    (__ubuf__ float *)llUbuf + ibPingpongFlag * UB_FLOAT_LINE_SIZE, 0, 1, fm / 8, 0, 0);
                pipe_barrier(PIPE_V);
                if (vmPingpongFlag == 1) {
                    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                    vmPingpongFlag = 0;
                }
                CopyUbufToUbufFp32((__ubuf__ float *)goUbuf, (__ubuf__ float *)loUbuf, 0, 1, oSize / 8, 0, 0);
                pipe_barrier(PIPE_V);
            }
            initGg = 0;
            ibPingpongFlag = 1 - ibPingpongFlag;
        }
        if (wrapO == 1) {
            vconv_f322f16((__ubuf__ half *)glUbuf, // lj fp32->fp16
                (__ubuf__ float *)glUbuf, mD64, 1, 1, uint16_t(4), uint16_t(8));
            pipe_barrier(PIPE_V);
            // 2 for double buffer
            for (int32_t vconvIdx = 0; vconvIdx < 2; ++vconvIdx) {
                vconv_f322f16((__ubuf__ half *)goUbuf + vconvIdx * oSize / 2, // Oi fp32->f16
                    (__ubuf__ float *)goUbuf + vconvIdx * oSize / 2, oSize / 2 / 64, 1, 1, uint16_t(4), uint16_t(8));
                pipe_barrier(PIPE_V);
            }
            ExpandToBlockHalf((__ubuf__ half *)tvUbuf, (__ubuf__ half *)glUbuf, fm); // BroadCast(fm, ) -> (fm, 16)

            for (int32_t vdivIdx = 0; vdivIdx < (fk / BLOCK_SIZE); ++vdivIdx) { // Oi / li
                VecDivFp32((__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                        (__ubuf__ half *)goUbuf + vdivIdx * fm * BLOCK_SIZE,
                        (__ubuf__ half *)tvUbuf, m0Value * BLOCK_SIZE / VECTOR_SIZE,
                        1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
            }
            int32_t blockV = VECTOR_SIZE / BLOCK_SIZE;
            if (m0Value % blockV != 0) {
                SetMaskFa(m0Value * BLOCK_SIZE % 128);    // 128: repeat
                VecDivFp32((__ubuf__ half *)goUbuf + m0Value * BLOCK_SIZE / 128 * 128,    // 128: repeat
                        (__ubuf__ half *)goUbuf + m0Value * BLOCK_SIZE / 128 * 128,    // 128: repeat
                        (__ubuf__ half *)tvUbuf + m0Value / blockV * blockV * 16, fk / BLOCK_SIZE,
                        1, 1, 1, fm, fm, 0);
                set_vector_mask(-1, -1);
            }

            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
            // move O to gm
            copy_ubuf_to_gm((__gm__ half *)gmDsto + (int64_t)dstoOffset, (__ubuf__ half *)goUbuf,
                            0, fk / BLOCK_SIZE, fm, 0,
                            qSeqLenAlign - fm);
            if (vmPingpongFlag == 0) {
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
                vmPingpongFlag = 1;
            }
        }
        l1PingpongFlag = 1 - l1PingpongFlag;
    }
private:
    __aicore__ void SetMaskFa(int32_t len)
    {
        int32_t highMask = len - 64 > 0 ? len - 64 : 0;
        int32_t lowMask = len - 64 >= 0 ? 64 : len;
        if (len < 64) {    // 64 is len
            set_vector_mask(0x0, ((uint64_t) 1 << lowMask) - 1);
        } else {
            set_vector_mask(((uint64_t) 1 << highMask) - 1, 0xffffffffffffffff);
        }
    }

    __aicore__ void SetVcgMaskFa(int32_t len)
    {
        uint64_t subMask = ((uint64_t) 1 << len) - 1;
        uint64_t maskValue = (subMask << 48) + (subMask << 32) + (subMask << 16) + subMask;
        set_vector_mask(maskValue, maskValue);
    }

    __aicore__ void ExpandToBlockHalf(__ubuf__ half *dst, __ubuf__ half *src, int32_t len)
    {
        for (int32_t vaddsIdx = 0; vaddsIdx < 2; ++vaddsIdx) { // (len,) -> len / 16 个 (16, 16)
            vadds((__ubuf__ half *)dst + vaddsIdx * 8 * BLOCK_SIZE, (__ubuf__ half *)src, (half)(0.0),
                len / BLOCK_SIZE,                 // repeat
                1, 0, uint16_t(16), uint16_t(1)); // dstBlockStride srcBlockStride dstRepeatStride srcRepeatStride
        }
        pipe_barrier(PIPE_V);
        for (int32_t vtransIdx = 0; vtransIdx < (len / BLOCK_SIZE); ++vtransIdx) { // (16, len) -> (len, 16)
            vtranspose((__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE,
                (__ubuf__ uint16_t *)dst + vtransIdx * CUBE_MATRIX_SIZE);
        }
        pipe_barrier(PIPE_V);
    }

    template <typename T>
    __aicore__ inline void VecMaxFp32(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmax(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecAddFp32(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vadd(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecSubFp32(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vsub(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecMulFp32(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vmul(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void VecDivFp32(T *dst, T *src0, T *src1, uint16_t repeat, uint16_t dstBlockStride,
        uint16_t src0BlockStride, uint16_t src1BlockStride, uint16_t dstRepeatStride, uint16_t src0RepeatStride,
        uint16_t src1RepeatStride)
    {
        vdiv(dst, src0, src1, repeat, dstBlockStride, src0BlockStride, src1BlockStride, dstRepeatStride,
            src0RepeatStride, src1RepeatStride);
    }

    template <typename T>
    __aicore__ inline void CopyUbufToUbufFp32(T *dst, T *src, uint16_t sid, uint16_t nBurst,
        uint16_t lenBurst, uint16_t srcGap, uint16_t dstGap)
    {
        copy_ubuf_to_ubuf(dst, src, sid, nBurst, lenBurst, srcGap, dstGap);
    }

    template <typename T>
    __aicore__ inline void VecExpFp32(T *dst, T *src, uint8_t expRepeat)
    {
        vexp(dst, src, expRepeat, 1, 1, DEFAULT_REPEAT_STRIDE, DEFAULT_REPEAT_STRIDE);
    }

    __aicore__ inline void CopyGmToL1Fp32(__cbuf__ half *dst, __gm__ half *src, uint16_t num, uint16_t len,
                                      uint16_t srcStride, uint16_t dstStride)
    {
        copy_gm_to_cbuf(dst, src, 0, num, len, srcStride, dstStride, PAD_NONE);
    }

    __aicore__ inline void CopyUbToGmFp32(__gm__ half *dst, __ubuf__ half *src, uint16_t num, uint16_t len,
                                      uint16_t srcStride, uint16_t dstStride)
    {
        copy_ubuf_to_gm(dst, src, 0, num, len, srcStride, dstStride);
    }

private:
    int32_t l1PingpongFlag = 0;
    int32_t ibPingpongFlag = 0;
    int32_t vmPingpongFlag = 1;

    __gm__ uint8_t *__restrict__ gmSrcq;
    __gm__ uint8_t *__restrict__ gmSrck;
    __gm__ uint8_t *__restrict__ gmSrcv;
    __gm__ uint8_t *__restrict__ gmSrcm;
    __gm__ uint8_t *__restrict__ gmDsto;

    __cbuf__ uint8_t *l1qBufAddr;
    __cbuf__ uint8_t *l1kBufAddr;
    __cbuf__ uint8_t *l1pBufAddr;
    __cbuf__ uint8_t *l1vBufAddr;
    __cbuf__ uint8_t *l1maxkBufAddr;

    __ca__ uint8_t *l0aBuf;
    __cb__ uint8_t *l0bBuf;
    __cc__ uint8_t *l0cBuf;

    __ubuf__ uint8_t *lsUbuf;
    __ubuf__ uint8_t *lpUbuf;
    __ubuf__ uint8_t *ls32Ubuf;
    __ubuf__ uint8_t *loUbuf;
    __ubuf__ uint8_t *lmUbuf;
    __ubuf__ uint8_t *hmUbuf;
    __ubuf__ uint8_t *gmUbuf;
    __ubuf__ uint8_t *dmUbuf;
    __ubuf__ uint8_t *llUbuf;
    __ubuf__ uint8_t *glUbuf;
    __ubuf__ uint8_t *tvUbuf;
    __ubuf__ uint8_t *goUbuf;

    half tor = 0;
    int32_t m = 0;
    int32_t n = 0;
    int32_t k = 0;
    int32_t d = 0;
    int32_t ntokensQ = 0;
    int32_t qSeqLenAlign = 0;
    int32_t kvSeqLenAlign = 0;

    int64_t srcqOffset = 0;
    int64_t srckOffset = 0;
    int32_t srcvOffset = 0;
    int64_t dstoOffset = 0;
    int32_t srcmOffset = 0;
    int32_t maskStride = 0;

    int32_t initG = 0;
    int32_t wrapO = 0;
};
} // namespace AscendC

namespace {
extern "C" __global__ __aicore__ void flash_attention_softmax_fp32(GM_ADDR query, GM_ADDR key, GM_ADDR value,
    GM_ADDR qSeqLen, GM_ADDR kvSeqLen, GM_ADDR mask, GM_ADDR attnOut, GM_ADDR workspace, GM_ADDR tiling)
{
    set_ctrl(sbitset0(get_ctrl(), 56));
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
    set_padding(uint16_t(0));
    set_atomic_none();

    uint32_t *tiling_para_ub_static;
    __ubuf__ int32_t *tiling_para_ub;

    if (tiling == nullptr) {
        GET_TILING_DATA(tilingData, tiling);
        tiling_para_ub_static = const_cast<uint32_t *>(tilingData.tilingParam);
    } else {
        tiling_para_ub = (__ubuf__ int32_t *)get_imm(
            4 * UB_UINT8_BLOCK_SIZE + 9 * UB_UINT8_LINE_SIZE); // use left ub space to store tilingData
        // 6 = 48 *sizeof(uint32_t) / 32 = 41 * 4 / 32 = 6
        copy_gm_to_ubuf((__ubuf__ int32_t *)tiling_para_ub, (__gm__ int32_t *)tiling, 0, 1, 6, 0, 0);
    }

    __ubuf__ int32_t *qSeqlenUb = (__ubuf__ int32_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE +
        9 * UB_UINT8_LINE_SIZE + MAX_TILING);

    __ubuf__ int32_t *kvSeqlenUb = (__ubuf__ int32_t *)get_imm(4 * UB_UINT8_BLOCK_SIZE +
        9 * UB_UINT8_LINE_SIZE + MAX_TILING + 8 * 4); // 偏移8*4=32位

    // qSeqlenUb实际长度为1，为确保对齐，这里偏移32位
    copy_gm_to_ubuf((__ubuf__ int32_t *)qSeqlenUb, (__gm__ int32_t *)qSeqLen, 0, 1, 1, 0, 0);
    // kvSeqlenUb实际长度为1，为确保对齐，这里偏移32位
    copy_gm_to_ubuf((__ubuf__ int32_t *)kvSeqlenUb, (__gm__ int32_t *)kvSeqLen, 0, 1, 1, 0, 0);

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    uint32_t batchSize;
    uint32_t qtokens;
    uint32_t kvHead;
    int64_t heads;
    int32_t embd;
    half tor;
    uint32_t maskBatchStride;
    uint32_t maskHeadStride;
    uint32_t startBatch = 0;
    uint32_t endBatch = 0;
    uint32_t startBlk = 0;
    uint32_t endBlk = 0;

    if (tiling == nullptr) {
        batchSize = (uint32_t)(*((int32_t *)tiling_para_ub_static));
        qtokens = (uint32_t)(*((int32_t *)tiling_para_ub_static + 1)); // 1 is index
        kvHead = (uint32_t)(*((int32_t *)tiling_para_ub_static + 2));  // 2 is index
        heads = (int32_t)(*((int32_t *)tiling_para_ub_static + 3));    // 3 is index
        embd = (int32_t)(*((int32_t *)tiling_para_ub_static + 4));     // 4 is index
        tor = (half)(*((float *)tiling_para_ub_static + 5));           // 5 is index

        maskBatchStride = (uint32_t)(*((int32_t *)tiling_para_ub_static + 6));  // 6 is index
        maskHeadStride = (uint32_t)(*((int32_t *)tiling_para_ub_static + 7));   // 7 is index

        TILING_CORE(0);    // 0 is the value of block_idx
        TILING_CORE(1);    // 1 is the value of block_idx
        TILING_CORE(2);    // 2 is the value of block_idx
        TILING_CORE(3);    // 3 is the value of block_idx
        TILING_CORE(4);    // 4 is the value of block_idx
        TILING_CORE(5);    // 5 is the value of block_idx
        TILING_CORE(6);    // 6 is the value of block_idx
        TILING_CORE(7);    // 7 is the value of block_idx
    } else {
        batchSize = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub));
        qtokens = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 1));  // 1 is index
        kvHead = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 2));   // 2 is index
        heads = (int32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 3));     // 3 is index
        embd = (int32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 4));      // 4 is index
        tor = (half)(*((__ubuf__ float *)tiling_para_ub + 5));            // 5 is index

        maskBatchStride = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 6));   // 6 is index
        maskHeadStride = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 7));    // 7 is index

        startBatch = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 11 + block_idx * 4 + 0));  // 11、4、0 is offset
        endBatch = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 11 + block_idx * 4 + 1));    // 11、4、1 is offset
        startBlk = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 11 + block_idx * 4 + 2));     // 11、4、2 is offset
        endBlk = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 11 + block_idx * 4 + 3));       // 11、4、3 is offset
    }

    int32_t kvRealHeads = kvHead > 0 ? kvHead : heads;
    int32_t groupNum = heads / kvRealHeads;

    AscendC::FlashAttentionSoftmaxFp32 foo(query, key, value, mask, attnOut, tor);

    set_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_MTE2, EVENT_ID0);

    int32_t curBatch = 0;

    int32_t qSeq = *((__ubuf__ int32_t *)qSeqlenUb);
    int32_t kvSeq = *((__ubuf__ int32_t *)kvSeqlenUb);

    set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);

    if (qSeq != 1) {
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID2);
        set_flag(PIPE_M, PIPE_MTE1, EVENT_ID3);
        set_flag(PIPE_V, PIPE_M, EVENT_ID0);
        set_flag(PIPE_V, PIPE_M, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE1, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID3);
        set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
    }

    for (uint32_t currQblk = startBlk; currQblk < endBlk; currQblk++) {
        // get tiling args
        int32_t offsetTiling = TILING_PARA_SIZE * curBatch;

        uint32_t ppMScalar = 0;
        uint32_t ppNScalar = 0;
        uint32_t curProcNum = 0;
        uint32_t curTotalQblk = 0;

        if (tiling == nullptr) {
            ppMScalar = (uint32_t)(*((uint32_t *)tiling_para_ub_static + 8));
            ppNScalar = (uint32_t)(*((uint32_t *)tiling_para_ub_static + 9));
            curProcNum = heads * (uint32_t)(*((uint32_t *)tiling_para_ub_static + 10));
            curTotalQblk = (startBatch + curBatch + 1) * curProcNum;
        } else {
            ppMScalar = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 8));
            ppNScalar = (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 9));
            curProcNum = heads * (uint32_t)(*((__ubuf__ int32_t *)tiling_para_ub + 10));
            curTotalQblk =  (startBatch + curBatch + 1) * curProcNum;
        }

        int32_t qAlign = (qSeq + BLOCK_SIZE -1) / BLOCK_SIZE * BLOCK_SIZE;
        int32_t kvAlign = (kvSeq + BLOCK_SIZE -1) / BLOCK_SIZE * BLOCK_SIZE;

        uint32_t maskStride = qAlign;
        uint32_t maskBatchOffset = 0;
        maskBatchOffset = ((startBatch + curBatch) % maskBatchStride) * maskHeadStride * qAlign * kvAlign;

        uint64_t strideQo = qAlign * embd;
        uint64_t strideKv = kvAlign * embd;

        uint32_t curQblkId = currQblk - (curTotalQblk - curProcNum);
        int32_t mLoop = (qAlign + ppMScalar - 1) / ppMScalar;
        int32_t nLoop = (kvAlign + ppNScalar - 1) / ppNScalar;

        int32_t start = curQblkId * nLoop;
        int32_t end = start + nLoop;

        if (qSeq == 1) {
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
        }

        int32_t stride = 1;
        if (qSeq != 1) {
            stride = 2;
        }
        for (int32_t loop_idx = start; loop_idx < end; loop_idx += stride) {
            int32_t head_idx0 = loop_idx / (mLoop * nLoop);
            int32_t m_idx0 = loop_idx % (mLoop * nLoop) / nLoop;
            int32_t n_idx0 = loop_idx % (mLoop * nLoop) % nLoop;

            int64_t qOffset = (startBatch + curBatch) * qAlign * heads* embd +
                               head_idx0 * strideQo + m_idx0 * ppMScalar * BLOCK_SIZE;
            int64_t kOffset = (startBatch + curBatch) * kvAlign * kvHead * embd +
                               head_idx0 / groupNum * strideKv + n_idx0 * ppNScalar * BLOCK_SIZE;
            int64_t vOffset = (startBatch + curBatch) * kvAlign * kvHead * embd +
                               head_idx0 / groupNum * strideKv + n_idx0 * ppNScalar * BLOCK_SIZE;
            int64_t oOffset = (startBatch + curBatch) * qAlign * heads* embd +
                               head_idx0 * strideQo + m_idx0 * ppMScalar * BLOCK_SIZE;

            uint32_t maskHeadOffset = 0;
            maskHeadOffset = (head_idx0 % maskHeadStride) * qAlign * kvAlign;
            int64_t maskOffset = maskBatchOffset +
                                 maskHeadOffset +
                                 m_idx0 * ppMScalar * BLOCK_SIZE + n_idx0 * maskStride * ppNScalar;
            int32_t last_n_loop = 0;
            if (qSeq != 1) {
                last_n_loop = (n_idx0 == (nLoop - 1) || (n_idx0 + 1) == (nLoop - 1)) ? 1 : 0;
            } else {
                last_n_loop = (n_idx0 == (nLoop - 1)) ? 1 : 0;
            }
            int32_t warpO = last_n_loop;
            int32_t initG = (n_idx0 == 0) ? 1 : 0;

            int32_t m0Value = (m_idx0 == (mLoop - 1)) ? (qSeq - m_idx0 * ppMScalar) : ppMScalar;
            int32_t n0Value = (n_idx0 == (nLoop - 1)) ? (kvSeq - n_idx0 * ppNScalar) : ppNScalar;
            int32_t m1Value = 0;
            int32_t n1Value = 0;
            if (qSeq != 1) {
                m1Value = (m_idx0 == (mLoop - 1)) ? (qSeq - m_idx0 * ppMScalar) : ppMScalar;
                n1Value = ((n_idx0 + 1) == (nLoop - 1)) ? (kvSeq - (n_idx0 + 1) * ppNScalar) : ppNScalar;
            }

            int32_t k0Value = embd;
            int32_t roundM0 = (m0Value + 15) / 16 * 16;
            int32_t roundN0 = (n0Value + 15) / 16 * 16;
            int32_t roundK0 = (k0Value + 15) / 16 * 16;

            int32_t roundM1 = 0;
            int32_t roundN1 = 0;
            int32_t roundK1 = 0;

            if (qSeq != 1) {
                roundM1 = (m1Value + 15) / 16 * 16;
                roundN1 = (n1Value + 15) / 16 * 16;
                roundK1 = (k0Value + 15) / 16 * 16;

                if ((n_idx0 + 1) == (nLoop)) {
                    n1Value = -1;
                }
            }
            foo.Init(roundM0, roundN0, roundK0, qOffset, kOffset, vOffset, maskOffset, oOffset,
                     initG, warpO, qtokens, maskStride, qAlign, kvAlign);
            if (qSeq != 1) {
                foo.FlashAttentionNzPrefillCompute(roundM0, roundN0, roundK0, roundM1, roundN1, roundK1,
                                                m0Value, n0Value, m1Value, n1Value, ppNScalar);
            } else {
                foo.FlashAttentionNzDecoderCompute(roundM0, roundN0, roundK0, m0Value, n0Value);
            }
        }
        if (curQblkId == curProcNum - 1) {
            curBatch++;
        }
        if (qSeq == 1) {
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
    }
    if (qSeq != 1) {
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE1, EVENT_ID1);
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
    }
    pipe_barrier(PIPE_ALL);
}
}