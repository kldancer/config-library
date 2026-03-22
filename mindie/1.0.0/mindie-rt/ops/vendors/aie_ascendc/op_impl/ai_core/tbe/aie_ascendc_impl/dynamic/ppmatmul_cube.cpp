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
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN_FP16 = 16384;
constexpr int32_t L0AB_PINGPONG_BUFFER_LEN_INT8 = 32768;
constexpr int32_t BLOCK_SIZE_16 = 16;
constexpr int32_t BLOCK_SIZE_32 = 32;
constexpr int32_t CUBE_MATRIX_SIZE_256 = 256;
constexpr int32_t CUBE_MATRIX_SIZE_512 = 512;
constexpr int64_t L1_PINGPONG_BUFFER_LEN_FP16 = 131072;
constexpr int64_t L1_PINGPONG_BUFFER_LEN_INT8 = 262144;
constexpr int64_t ND2NZ_STRIDE_LIMIT = 65536;
}

namespace AscendC {
template<bool TA, bool TB, bool SPLIT_K = false, bool HAVE_BIAS = false, bool IS_INT8 = false,
    typename IN_DTYPE = half, typename OUT_DTYPE = half, typename TBIAS = float>
class PpmatmulCube {
public:
    __aicore__ inline PpmatmulCube() {};
    __aicore__ inline void SetArgs(__gm__ uint8_t *__restrict__ a,
                                   __gm__ uint8_t *__restrict__ b,
                                   __gm__ uint8_t *__restrict__ c,
                                   int32_t batchA,
                                   int32_t batchB,
                                   int32_t mReal,
                                   int32_t nReal,
                                   int32_t kReal,
                                   int32_t m0Real,
                                   int32_t n0Real,
                                   int32_t k0Real,
                                   int32_t headLoop,
                                   int32_t tailNum)
    {
        gmA = reinterpret_cast<__gm__ IN_DTYPE *>(a);
        gmB = reinterpret_cast<__gm__ IN_DTYPE *>(b);
        gmC = reinterpret_cast<__gm__ OUT_DTYPE *>(c);
        batchSize = batchA;
        batchSizeB = batchB;
        m = mReal;
        n = nReal;
        k = kReal;
        m0 = m0Real;
        n0 = n0Real;
        k0 = k0Real;

        blockSize = BLOCK_SIZE_16;
        cubeMatrixSize = CUBE_MATRIX_SIZE_256;
        l1PingPongBufferLen = L1_PINGPONG_BUFFER_LEN_FP16;
        l0abPingPongBufferLen = L0AB_PINGPONG_BUFFER_LEN_FP16;

        int32_t aL1Size = m0 * k0 * sizeof(IN_DTYPE);
        l1BaseB = reinterpret_cast<__cbuf__ IN_DTYPE *>(
                        (uintptr_t) ((aL1Size + cubeMatrixSize - 1) / cubeMatrixSize * cubeMatrixSize));
        coreNum = get_block_num();
        coreIdx = get_block_idx();
        mLoop = (m + m0 - 1) / m0;
        nLoop = (n + n0 - 1) / n0;
        kLoop = (k + k0 - 1) / k0;
        coreLoop = coreIdx < tailNum ? headLoop + 1 : headLoop;
        pingFlag = 1;
    }

    __aicore__ inline void SetArgs(__gm__ uint8_t *__restrict__ a,
                                   __gm__ uint8_t *__restrict__ b,
                                   __gm__ uint8_t *__restrict__ c,
                                   __gm__ uint8_t *__restrict__ bias,
                                   int32_t batchA,
                                   int32_t batchB,
                                   int32_t mReal,
                                   int32_t nReal,
                                   int32_t kReal,
                                   int32_t m0Real,
                                   int32_t n0Real,
                                   int32_t k0Real,
                                   int32_t headLoop,
                                   int32_t tailNum)
    {
        gmBias = reinterpret_cast<__gm__ TBIAS *>(bias);
        SetArgs(a, b, c, batchA, batchB, mReal, nReal, kReal, m0Real, n0Real, k0Real, headLoop, tailNum);
    }

    __aicore__ inline void CopyGmToCbufMultiND2NZKernel(__cbuf__ IN_DTYPE *l1_buf,
                                                        __gm__ IN_DTYPE *gm,
                                                        int64_t offset,
                                                        int32_t nValue,
                                                        int32_t dValue,
                                                        int32_t srcDValue,
                                                        int32_t dstNzC0Stride)
    {
        if (srcDValue < ND2NZ_STRIDE_LIMIT) {
            copy_gm_to_cbuf_multi_nd2nz_b16(
                l1_buf,
                gm + offset,
                0,
                1,
                nValue,
                dValue,
                0,
                srcDValue,
                dstNzC0Stride,
                1,
                0
            );
        } else {
            auto gm_src = gm + offset;
            for (int i = 0; i < nValue; i++) {
                copy_gm_to_cbuf_multi_nd2nz_b16(
                    // 16 for 32Byte / sizeof(half)
                    l1_buf + i * 16,
                    gm_src + i * srcDValue,
                    0,
                    1,
                    1,
                    dValue,
                    0,
                    0,
                    dstNzC0Stride,
                    0,
                    0
                );
            }
        }
    }

    __aicore__ inline void CopyGmToCbufMultiND2NZ(bool T,
                                                  __cbuf__ IN_DTYPE *l1_buf,
                                                  __gm__ IN_DTYPE *gm,
                                                  int64_t offset,
                                                  int32_t a,
                                                  int32_t b,
                                                  int32_t aActual,
                                                  int32_t bActual,
                                                  int32_t aRound,
                                                  int32_t bRound)
    {
        auto nValue = 0;
        auto dValue = 0;
        auto srcDValue = 0;
        auto dstNzC0Stride = 0;
        if (T) {
            nValue = bActual;
            dValue = aActual;
            srcDValue = a;
            dstNzC0Stride = bRound;
        } else {
            nValue = aActual;
            dValue = bActual;
            srcDValue = b;
            dstNzC0Stride = aRound;
        }
        CopyGmToCbufMultiND2NZKernel(l1_buf, gm, offset, nValue, dValue, srcDValue, dstNzC0Stride);
    }

    __aicore__ inline void Run()
    {
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);

        for (int32_t loopIdx = 0; loopIdx < coreLoop; ++loopIdx) {
            int32_t index = coreIdx + coreNum * loopIdx;
            int64_t mIdx = index / nLoop % mLoop;
            int64_t nIdx = index % nLoop;
            int64_t batchIdx = index / nLoop / mLoop;
            int64_t batchBIdx = batchIdx % batchSizeB;

            int64_t offsetA;
            int64_t offsetB;
            int64_t offsetBais;
            int64_t offsetANext;
            int64_t offsetBNext;

            int64_t offsetC = batchIdx * m * n + mIdx * m0 * n + nIdx * n0;
            int32_t mActual = (mIdx == (mLoop - 1)) ? (m - mIdx * m0) : m0;
            int32_t nActual = (nIdx == (nLoop - 1)) ? (n - nIdx * n0) : n0;
            int32_t mRound = 0;
            int32_t nRound = 0;

            mRound = (mActual + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16 * BLOCK_SIZE_16;
            nRound = (nActual + BLOCK_SIZE_16 - 1) / BLOCK_SIZE_16 * BLOCK_SIZE_16;

            int32_t mn_max = mRound > nRound ? mRound : nRound;
            int32_t kPartLen = 0;
            kPartLen = l0abPingPongBufferLen / mn_max / BLOCK_SIZE_16 * BLOCK_SIZE_16;

            if (TA) {
                offsetA = batchIdx * m * k + mIdx * m0;
            } else {
                offsetA = batchIdx * m * k + mIdx * m0 * k;
            }

            if (TB) {
                offsetB = batchBIdx * k * n + nIdx * n0 * k;
            } else {
                offsetB = batchBIdx * k * n + nIdx * n0;
            }

            offsetBais = nIdx * n0;

            int32_t kActual = (kLoop == 1) ? k : k0;
            int32_t kRound = (kActual + blockSize - 1) / blockSize * blockSize;

            auto l1BufA = pingFlag ? l1BaseA : l1BaseA + l1PingPongBufferLen;
            auto l1BufB = pingFlag ? l1BaseB : l1BaseB + l1PingPongBufferLen;
            auto l0ABuf = pingFlag ? l0aBase : l0aBase + l0abPingPongBufferLen;
            auto l0BBuf = pingFlag ? l0BBase : l0BBase + l0abPingPongBufferLen;
            auto eventId = pingFlag ? EVENT_ID0 : EVENT_ID1;

            if (HAVE_BIAS) {
                pipe_barrier(PIPE_MTE2);
                copy_gm_to_cbuf_multi_nd2nz_b32s(
                    ((__cbuf__ TBIAS *)l1Bias),
                    ((__gm__ TBIAS *)gmBias) + offsetBais,
                    0,
                    1,
                    1,
                    nActual,
                    0,
                    n,
                    1,
                    1,
                    0
                );
                set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
                copy_cbuf_to_bt(((uint64_t)btBias), ((__cbuf__ TBIAS *)l1Bias),
                                // 4 for sizeof(float), 64 for 256 Byte / sizeof(float)
                                (uint16_t)0ULL, 1, (nActual * 4 + 64 - 1) / 64, 0, 0);
                set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
                set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
                wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID1);
            }

            wait_flag(PIPE_MTE1, PIPE_MTE2, eventId);

            if ((m == 1) || (mActual == 1 && !TA)) {
                copy_gm_to_cbuf(
                    l1BufA,
                    gmA + offsetA,
                    0,
                    1,
                    kRound / blockSize,
                    0,
                    0,
                    PAD_NONE);
            } else {
                CopyGmToCbufMultiND2NZ(TA, l1BufA, gmA, offsetA, m, k, mActual, kActual, mRound, kRound);
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, eventId);

            // 2 for event id index
            wait_flag(PIPE_MTE1, PIPE_MTE2, eventId + 2);
            CopyGmToCbufMultiND2NZ(TB, l1BufB, gmB, offsetB, k, n, kActual, nActual, kRound, nRound);
            // 2 for event id index
            set_flag(PIPE_MTE2, PIPE_MTE1, eventId + 2);

            if (HAVE_BIAS) {
                wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
            }

            for (int64_t kIdx = 0; kIdx < kLoop; kIdx++) {
                if (TA) {
                    offsetA = batchIdx * m * k + kIdx * k0 * m + mIdx * m0;
                } else {
                    offsetA = batchIdx * m * k + mIdx * m0 * k + kIdx * k0;
                }

                if (TB) {
                    offsetB = batchBIdx * k * n + nIdx * n0 * k + kIdx * k0;
                } else {
                    offsetB = batchBIdx * k * n + kIdx * k0 * n + nIdx * n0;
                }

                int32_t kActual = (kIdx == (kLoop - 1)) ? (k - kIdx * k0) : k0;
                int32_t kRound = (kActual + blockSize - 1) / blockSize * blockSize;
                int32_t kPartLoop = (kActual + kPartLen - 1) / kPartLen;

                __cbuf__ IN_DTYPE *l1BufA = pingFlag ? l1BaseA : l1BaseA + l1PingPongBufferLen;
                __cbuf__ IN_DTYPE *l1BufB = pingFlag ? l1BaseB : l1BaseB + l1PingPongBufferLen;
                auto eventId = pingFlag ? EVENT_ID0 : EVENT_ID1;

                if (kIdx < kLoop - 1) {
                    if (TA) {
                        offsetANext = batchIdx * m * k + (kIdx + 1) * k0 * m + mIdx * m0;
                    } else {
                        offsetANext = batchIdx * m * k + mIdx * m0 * k + (kIdx + 1) * k0;
                    }

                    if (TB) {
                        offsetBNext = batchBIdx * k * n + nIdx * n0 * k + (kIdx + 1) * k0;
                    } else {
                        offsetBNext = batchBIdx * k * n + (kIdx + 1) * k0 * n + nIdx * n0;
                    }

                    int32_t kActualNext = ((kIdx + 1) == (kLoop - 1)) ? (k - (kIdx + 1) * k0) : k0;
                    int32_t kRoundNext = (kActualNext + blockSize - 1) / blockSize * blockSize;

                    __cbuf__ IN_DTYPE *l1BufANext = (1 - pingFlag) ? l1BaseA : l1BaseA + l1PingPongBufferLen;
                    __cbuf__ IN_DTYPE *l1BufBNext = (1 - pingFlag) ? l1BaseB : l1BaseB + l1PingPongBufferLen;
                    auto eventIdNext = (1 - pingFlag) ? EVENT_ID0 : EVENT_ID1;

                    wait_flag(PIPE_MTE1, PIPE_MTE2, eventIdNext);

                    if ((m == 1) || (mActual == 1 && !TA)) {
                        copy_gm_to_cbuf(
                            l1BufANext,
                            gmA + offsetANext,
                            0,
                            1,
                            kRoundNext / blockSize,
                            0,
                            0,
                            PAD_NONE);
                    } else {
                        CopyGmToCbufMultiND2NZ(TA, l1BufANext, gmA, offsetANext, m, k, \
                            mActual, kActualNext, mRound, kRoundNext);
                    }
                    set_flag(PIPE_MTE2, PIPE_MTE1, eventIdNext);

                    // 2 for event id index
                    wait_flag(PIPE_MTE1, PIPE_MTE2, eventIdNext + 2);
                    CopyGmToCbufMultiND2NZ(TB, l1BufBNext, gmB, offsetBNext, k, n, kActualNext, \
                        nActual, kRoundNext, nRound);
                    // 2 for event id index
                    set_flag(PIPE_MTE2, PIPE_MTE1, eventIdNext + 2);
                }
                
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

                for (int32_t kPartIdx = 0; kPartIdx < kPartLoop; kPartIdx++) {
                    int32_t k0Round = (kPartIdx < kPartLoop - 1) ?
                        kPartLen : kRound - kPartIdx * kPartLen;
                    int32_t k0Actual = (kPartIdx < kPartLoop - 1) ?
                        kPartLen : kActual - kPartIdx * kPartLen;
                    
                    auto mte1MadPingFlag = 1 - kPartIdx % 2;
                    auto mte1MadEventId = mte1MadPingFlag ? EVENT_ID0 : EVENT_ID1;
                    // 2 for double buffer
                    auto l0ABuf = l0aBase + (kPartIdx % 2) * l0abPingPongBufferLen;
                    // 2 for double buffer
                    auto l0BBuf = l0BBase + (kPartIdx % 2) * l0abPingPongBufferLen;

                    if (kPartIdx == 0) {
                        wait_flag(PIPE_MTE2, PIPE_MTE1, eventId);
                    }
                    wait_flag(PIPE_M, PIPE_MTE1, mte1MadEventId);
                    if ((m == 1) || (mActual == 1 && !TA)) {
                        load_cbuf_to_ca(
                            l0ABuf,
                            l1BufA + kPartIdx * kPartLen,
                            0,
                            (k0Round + cubeMatrixSize - 1) / cubeMatrixSize,
                            1,
                            0,
                            0,
                            false,
                            inc);
                    } else {
                        if (TA) {
                            for (int64_t i = 0; i < mRound / BLOCK_SIZE_16; i++) {
                                load_cbuf_to_ca(
                                    l0ABuf + i * k0Round * BLOCK_SIZE_16,
                                    l1BufA + kPartIdx * kPartLen * BLOCK_SIZE_16 +
                                        i * kRound * BLOCK_SIZE_16,
                                    0,
                                    k0Round / BLOCK_SIZE_16,
                                    1,
                                    0,
                                    0,
                                    true,
                                    inc);
                            }
                        } else {
                            for (int64_t i = 0; i < mRound / BLOCK_SIZE_16; i++) {
                                load_cbuf_to_ca(
                                    l0ABuf + i * k0Round * BLOCK_SIZE_16,
                                    l1BufA + kPartIdx * kPartLen * mRound +
                                        i * cubeMatrixSize,
                                    0,
                                    k0Round / blockSize,
                                    mRound / BLOCK_SIZE_16,
                                    0,
                                    0,
                                    false,
                                    inc);
                            }
                        }
                    }
                    if (kPartIdx == kPartLoop - 1) {
                        set_flag(PIPE_MTE1, PIPE_MTE2, eventId);
                    }

                    if (kPartIdx == 0) {
                        // 2 for double buffer
                        wait_flag(PIPE_MTE2, PIPE_MTE1, eventId + 2);
                    }
                    if (TB) {
                        load_cbuf_to_cb(
                            l0BBuf,
                            l1BufB + kPartIdx * kPartLen * nRound,
                            0,
                            k0Round * nRound / cubeMatrixSize,
                            1,
                            0,
                            0,
                            false,
                            inc);
                    } else {
                        for (int64_t i = 0; i < k0Round / BLOCK_SIZE_16; i++) {
                            load_cbuf_to_cb(
                                l0BBuf + i * nRound * BLOCK_SIZE_16,
                                l1BufB + (kPartIdx * kPartLen + i * BLOCK_SIZE_16) * BLOCK_SIZE_16,
                                0,
                                nRound / BLOCK_SIZE_16,
                                kRound / BLOCK_SIZE_16,
                                0,
                                0,
                                true,
                                inc);
                        }
                    }
                    if (kPartIdx == kPartLoop - 1) {
                        // 2 for double buffer
                        set_flag(PIPE_MTE1, PIPE_MTE2, eventId + 2);
                    }

                    set_flag(PIPE_MTE1, PIPE_M, mte1MadEventId);
                    wait_flag(PIPE_MTE1, PIPE_M, mte1MadEventId);

                    bool init_c = (kIdx == 0 && kPartIdx == 0);
                    if (init_c) {
                        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
                    }
                    if (HAVE_BIAS) {
                        mad((__cc__ float *)l0cBuf,
                            (__ca__ half *)l0ABuf,
                            (__cb__ half *)l0BBuf,
                            ((uint64_t)btBias),
                            mActual,
                            k0Actual,
                            nActual,
                            0,
                            0,
                            init_c,
                            0);
                    } else {
                        if (m != 1 && mActual == 1 && TA) {
                            mad(l0cBuf,
                                l0ABuf,
                                l0BBuf,
                                BLOCK_SIZE_16,
                                k0Actual,
                                nActual,
                                0,
                                0,
                                0,
                                init_c);
                        } else {
                            mad(l0cBuf,
                                l0ABuf,
                                l0BBuf,
                                mActual,
                                k0Actual,
                                nActual,
                                0,
                                0,
                                0,
                                init_c);
                        }
                    }
                    pipe_barrier(PIPE_M);
                    set_flag(PIPE_M, PIPE_MTE1, mte1MadEventId);
                }

                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
                wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
                pingFlag = 1 - pingFlag;
            }

            set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
            wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

            copy_matrix_cc_to_gm(
                gmC + offsetC,
                l0cBuf,
                0,
                nActual,
                mActual,
                n,
                mRound,
                0,
                F322F16,
                0,
                false,
                true);
            set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        }

        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
        pipe_barrier(PIPE_ALL);
    }

private:
    __gm__ IN_DTYPE *__restrict__ gmA{nullptr};
    __gm__ IN_DTYPE *__restrict__ gmB{nullptr};
    __gm__ TBIAS *__restrict__ gmBias{nullptr};
    __gm__ OUT_DTYPE *__restrict__ gmC{nullptr};

    __cbuf__ IN_DTYPE *l1BaseA = reinterpret_cast<__cbuf__ IN_DTYPE *>((uintptr_t)0);
    // 128 * 1024 is 128 KB
    __cbuf__ IN_DTYPE *l1BaseB = reinterpret_cast<__cbuf__ IN_DTYPE *>((uintptr_t)(128 * 1024));

    __ca__ IN_DTYPE *l0aBase = reinterpret_cast<__ca__ IN_DTYPE *>((uintptr_t)0);
    __cb__ IN_DTYPE *l0BBase = reinterpret_cast<__cb__ IN_DTYPE *>((uintptr_t)0);

    __cc__ TBIAS *l0cBuf = reinterpret_cast<__cc__ TBIAS *>((uintptr_t)0);

    __cbuf__ TBIAS *l1Bias = reinterpret_cast<__cbuf__ TBIAS *>((uintptr_t)0);
    uint16_t btBias{0};

    int32_t coreNum{0};

    int32_t batchSize{0};
    int32_t batchSizeB{0};
    int32_t m{0};
    int32_t k{0};
    int32_t n{0};

    int32_t m0{0};
    int32_t k0{0};
    int32_t n0{0};

    int32_t mLoop{0};
    int32_t nLoop{0};
    int32_t kLoop{0};
    int32_t coreLoop{0};
    int32_t coreIdx{0};
    int32_t pingFlag{0};
    int32_t blockSize{0};
    int32_t cubeMatrixSize{0};

    int64_t l1PingPongBufferLen{0};
    int32_t l0abPingPongBufferLen{0};
};
}

namespace {
extern "C" __global__ __aicore__ void ppmatmul_cube(GM_ADDR gmA, GM_ADDR gmB, GM_ADDR gmBias, GM_ADDR gmC,
                                                    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    AscendC::PpmatmulCube<false, false, false> matmul;
    AscendC::PpmatmulCube<true, false, false> matmulTransA;
    AscendC::PpmatmulCube<false, true, false> matmulTransB;
    AscendC::PpmatmulCube<true, true, false> matmulTransAB;

    AscendC::PpmatmulCube<false, false, false, true> matmulBias;
    AscendC::PpmatmulCube<true, false, false, true> matmulBiasTransA;
    AscendC::PpmatmulCube<false, true, false, true> matmulBiasTransB;
    AscendC::PpmatmulCube<true, true, false, true> matmulBiasTransAB;

#ifdef __DAV_C220_CUBE__
    set_padding(uint16_t(0));
    set_atomic_none();
    uint64_t config = 0x1;
    set_nd_para(config);

    int32_t batchSize = tiling_data.batchSize;
    int32_t batchSizeB = tiling_data.batchSizeB;
    int32_t m = tiling_data.m;
    int32_t k = tiling_data.k;
    int32_t n = tiling_data.n;
    int32_t m0 = tiling_data.m0;
    int32_t k0 = tiling_data.k0;
    int32_t n0 = tiling_data.n0;
    int32_t mLoop = tiling_data.mLoop;
    int32_t kLoop = tiling_data.kLoop;
    int32_t nLoop = tiling_data.nLoop;
    int32_t coreHeadLoop = tiling_data.coreHeadLoop;
    int32_t coreTailNum = tiling_data.coreTailNum;
    int32_t tilingKey = tiling_data.tilingKey;

    switch (tilingKey) {
        // 0 is 0b00000: transA = 0, transB = 0, splitk = 0, bias = 0, int8 = 0
        case 0 :
            matmul.SetArgs(gmA, gmB, gmC, batchSize, batchSizeB, m, n, k, m0, n0, k0, coreHeadLoop, coreTailNum);
            matmul.Run();
            break;
        // 2 is 0b00010: transA = 0, transB = 0, splitk = 0, bias = 1, int8 = 0
        case 2 :
            matmulBias.SetArgs(gmA, gmB, gmC, gmBias, batchSize, batchSizeB, m, n, k, m0, n0, k0,
                               coreHeadLoop, coreTailNum);
            matmulBias.Run();
            break;
        // 16 is 0b10000: trans1 = 0, transB = 0, splitk = 0, bias = 0, int8 = 0
        case 16 :
            matmulTransA.SetArgs(gmA, gmB, gmC, batchSize, batchSizeB, m, n, k, m0, n0, k0,
                                 coreHeadLoop, coreTailNum);
            matmulTransA.Run();
            break;
        // 8 is 0b01000: transA = 0, transB = 1, splitk = 0, bias = 0, int8 = 0
        case 8 :
            matmulTransB.SetArgs(gmA, gmB, gmC, batchSize, batchSizeB, m, n, k, m0, n0, k0,
                                 coreHeadLoop, coreTailNum);
            matmulTransB.Run();
            break;
        // 10 is 0b01010: transA = 0, transB = 1, splitk = 0, bias = 1, int8 = 0
        case 10 :
            matmulBiasTransB.SetArgs(gmA, gmB, gmC, gmBias, batchSize, batchSizeB, m, n, k, m0, n0, k0,
                                     coreHeadLoop, coreTailNum);
            matmulBiasTransB.Run();
            break;
        // 10 is 0b01010: transA = 1, transB = 1, splitk = 0, bias = 0, int8 = 0
        case 24 :
            matmulTransAB.SetArgs(gmA, gmB, gmC, batchSize, batchSizeB, m, n, k, m0, n0, k0,
                                  coreHeadLoop, coreTailNum);
            matmulTransAB.Run();
            break;
        default :
            break;
    }
#endif
}
}