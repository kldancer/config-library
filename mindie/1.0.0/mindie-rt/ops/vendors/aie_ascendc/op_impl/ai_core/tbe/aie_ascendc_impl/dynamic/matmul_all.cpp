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
namespace {
constexpr size_t BLOCK_M = 64;
constexpr size_t BLOCK_N = 64;

// 规避k不能被64整除时的bug
constexpr uint32_t BLOCK_K_1 = 16;
constexpr uint32_t BLOCK_K_2 = 32;
constexpr uint32_t BLOCK_K_3 = 64;

constexpr uint32_t ADD = 1; // +
constexpr uint32_t SUB = 2; // -
constexpr uint32_t MUL = 3; // x
constexpr uint32_t DIV = 4; // /

constexpr size_t BLOCKSIZE = 32; // 搬运的最小字节数
constexpr size_t BURSTLEN = 16;  // 搬运的最少个数

constexpr uint32_t MAX_M = 1024;
constexpr uint32_t MAX_N = 1024;
constexpr uint32_t MAX_K = 1024;
constexpr uint32_t ALIGN = 128;

class MatmulAll {
public:
    __aicore__ inline MatmulAll(uint32_t isTrans, uint32_t isBias, uint32_t operate,
        uint32_t m, uint32_t n, uint32_t k, __gm__ uint8_t *inputBias, __gm__ uint8_t *inputExtra)
    {
        this->isTrans = isTrans;
        this->isBias = isBias;
        this->operate = operate;
        this->m = m;
        this->n = n;
        this->k = k;

        groupIdx = get_block_idx(); // 获取当前核ID
        coreNums = get_block_num(); // 获取当前核个数，

        if (k % BLOCK_K_3 == 0) {
            blockK = BLOCK_K_3;
        } else if (k % BLOCK_K_2 == 0) {
            blockK = BLOCK_K_2;
        } else {
            blockK = BLOCK_K_1;
        }

        if (coreNums == 0) {
            coreNums = 1;
        }
        rowsPerBlock = m / coreNums;

        int blocksReamin = m % coreNums;

        if (groupIdx < blocksReamin) {
            blocksOffset = (rowsPerBlock + 1) * groupIdx;
            rowsPerBlock += 1;
        } else {
            blocksOffset = blocksReamin + groupIdx * rowsPerBlock;
        }
    };

    __aicore__ inline void Init(__gm__ uint8_t *inputX, __gm__ uint8_t *inputWeight, __gm__ uint8_t *inputBias,
        __gm__ uint8_t *inputExtra, __gm__ uint8_t *outMatrix)
    {
        gmX.SetGlobalBuffer((__gm__ half *)inputX + blocksOffset * k);
        gmW.SetGlobalBuffer((__gm__ half *)inputWeight);
        gmB.SetGlobalBuffer((__gm__ half *)inputBias);
        gmE.SetGlobalBuffer((__gm__ half *)inputExtra);

        gmO.SetGlobalBuffer((__gm__ half *)outMatrix + blocksOffset * n);

        pipe.InitBuffer(a1Buf, blockM * blockK * sizeof(half));
        pipe.InitBuffer(a2Buf, blockM * blockK * sizeof(half));

        pipe.InitBuffer(b1Buf, blockK * blockN * sizeof(half));
        pipe.InitBuffer(b2Buf, blockK * blockN * sizeof(half));

        pipe.InitBuffer(co1fp32Buf, blockM * blockN * sizeof(float));
        pipe.InitBuffer(co2fp32Buf, blockM * blockN * sizeof(float));
        pipe.InitBuffer(co2fp16Buf, blockM * blockN * sizeof(half));

        pipe.InitBuffer(biasBuf, blockN * sizeof(half));
        pipe.InitBuffer(biasfillBuf, blockM * blockN * sizeof(half));
        pipe.InitBuffer(biasfp32Buf, blockM * blockN * sizeof(float));

        pipe.InitBuffer(extraBuf, blockM * blockN * sizeof(half));
        pipe.InitBuffer(outputBuf, blockM * blockN * sizeof(half));
    };

    __aicore__ inline void Process()
    {
        int mLoop = (rowsPerBlock + blockM - 1) / blockM;
        int kLoop = (k + blockK - 1) / blockK;
        int nLoop = (n + blockN - 1) / blockN;

        int mTail = rowsPerBlock % blockM;
        int kTail = k % blockK;
        int nTail = n % blockN;

        for (int mm = 0; mm < mLoop; ++mm) {
            int mReal = blockM;
            if (mTail > 0 && mm == (mLoop - 1)) {
                mReal = mTail;
            }
            for (int nn = 0; nn < nLoop; ++nn) {
                int nReal = blockN;
                if (nTail > 0 && nn == (nLoop - 1)) {
                    nReal = nTail;
                }
                LocalTensor<float> c1fp32Local = co1fp32Buf.Get<float>(blockM * blockN);
                Duplicate<float>(c1fp32Local, (half)0.0, blockM * blockN);

                for (int kk = 0; kk < kLoop; ++kk) {
                    int kReal = blockK;
                    if (kTail > 0 && kk == (kLoop - 1)) {
                        kReal = kTail;
                    }

                    c1fp32Local = Mad(c1fp32Local, mm, kk, nn, mReal, kReal, nReal);
                }
                pipe_barrier(PIPE_ALL);

                // stage 3.0: y from CO1 to CO2, NZ -> NZ, use DataCopy
                LocalTensor<half> output = outputBuf.Get<half>(blockM * blockN);
                Duplicate<half>(output, (half)0.0, blockM * blockN);
                output = ProcessY(c1fp32Local, output);
                // stage4.0: bias from GM to CO2, format is ND
                output = ProcessBias(output, nn, nReal);
                // stage 5.0: extra from GM to CO2, format is ND
                ProcessOperate(output, mReal, nReal, mm, nn);
            }
        }
    }

private:
    __aicore__ inline LocalTensor<half> ProcessInputX(LocalTensor<half> &a1X, LocalTensor<half> &a2X, uint32_t kReal,
        uint32_t mReal, uint32_t kk, uint32_t mm)
    {
        Nd2NzParams xGm2A1Params;
        xGm2A1Params.ndNum = 1;
        xGm2A1Params.nValue = mReal;
        xGm2A1Params.dValue = kReal;
        xGm2A1Params.srcNdMatrixStride = mReal * kReal;
        xGm2A1Params.srcDValue = k;
        xGm2A1Params.dstNzC0Stride = BURSTLEN * blockM * sizeof(half) / BLOCKSIZE;
        xGm2A1Params.dstNzNStride = BURSTLEN * sizeof(half) / BLOCKSIZE;
        xGm2A1Params.dstNzMatrixStride = blockM * blockK;
        DataCopy(a1X, gmX[mm * blockM * k + kk * blockK], xGm2A1Params);

        // stage 0.1: x from A1 to A2, NZ -> zZ, use LoadData

        LoadData2dParams xA12A2Params;
        xA12A2Params.repeatTimes = blockK / BURSTLEN;
        xA12A2Params.srcStride = blockM * BURSTLEN * sizeof(half) / 512; // 512字节，16*16*2
        xA12A2Params.ifTranspose = false;
        pipe_barrier(PIPE_ALL); // 不能删

        for (int i = 0; i < (blockM / BURSTLEN); ++i) {
            int srcOffset = i * BURSTLEN * BURSTLEN;
            int dstOffset = i * BURSTLEN * blockK;
            LoadData(a2X[dstOffset], a1X[srcOffset], xA12A2Params);
        }
        return a2X;
    }

    __aicore__ inline LocalTensor<half> ProcessTrans(LocalTensor<half> &b1W, LocalTensor<half> &b2W, uint32_t kReal,
        uint32_t nReal, uint32_t kk, uint32_t nn)
    {
        if (isTrans == 0) {
            // stage 1.0: w from gm（k, n） to B1, ND -> NZ, use DataCopy
            Nd2NzParams wGm2B1Params;
            wGm2B1Params.ndNum = 1;
            wGm2B1Params.nValue = kReal;
            wGm2B1Params.dValue = nReal;
            wGm2B1Params.srcNdMatrixStride = kReal * nReal;
            wGm2B1Params.srcDValue = n;
            wGm2B1Params.dstNzC0Stride = BURSTLEN * blockK * sizeof(half) / BLOCKSIZE;
            wGm2B1Params.dstNzNStride = BURSTLEN * sizeof(half) / BLOCKSIZE;
            wGm2B1Params.dstNzMatrixStride = blockK * blockN;

            DataCopy(b1W, gmW[kk * blockK * n + nn * blockN], wGm2B1Params);
            pipe_barrier(PIPE_ALL);

            // stage 1.1: w from B1 to B2, NZ -> nZ, use LoadData
            LoadData2dParams wB12B2Params;
            wB12B2Params.repeatTimes = blockN / BURSTLEN;
            wB12B2Params.srcStride = blockK * BURSTLEN * sizeof(half) / 512; // 512字节，16*16*2
            wB12B2Params.ifTranspose = true;

            for (int i = 0; i < (blockK / BURSTLEN); ++i) {
                int srcOffset = i * BURSTLEN * BURSTLEN;
                int dstOffset = i * BURSTLEN * blockN;
                LoadData(b2W[dstOffset], b1W[srcOffset], wB12B2Params);
                pipe_barrier(PIPE_ALL);
            }
        } else {
            for (int i = 0; i < kReal / BURSTLEN; ++i) {
                int srcOffset = i * BURSTLEN;
                int dstOffset = i * BURSTLEN * blockN;
                DataCopyParams wGm2B1Params;
                wGm2B1Params.blockCount = nReal;
                wGm2B1Params.blockLen = BURSTLEN * sizeof(half) / BLOCKSIZE;
                wGm2B1Params.srcStride = (k - BURSTLEN) * sizeof(half) / BLOCKSIZE;
                wGm2B1Params.dstStride = 0;
                DataCopyEnhancedParams enhancel12l2Params;
                enhancel12l2Params.blockMode = BlockMode::BLOCK_MODE_NORMAL;

                DataCopy(b1W[dstOffset], gmW[nn * blockN * k + kk * blockK + srcOffset], wGm2B1Params,
                    enhancel12l2Params);
                pipe_barrier(PIPE_ALL);
            }

            int srcOffset2 = 0;
            int dstOffset2 = 0;
            // L0B
            LoadData2dParams loadDataParams;
            loadDataParams.repeatTimes = blockN * blockK / (BURSTLEN * BURSTLEN);
            loadDataParams.srcStride = 1;
            loadDataParams.ifTranspose = false;
            LoadData(b2W[dstOffset2], b1W[srcOffset2], loadDataParams);
            pipe_barrier(PIPE_ALL);
        }
        return b2W;
    };

    __aicore__ inline LocalTensor<half> ProcessY(LocalTensor<float> &c1fp32Local, LocalTensor<half> &output)
    {
        LocalTensor<float> c2fp32Local = co2fp32Buf.Get<float>(blockM * blockN);
        DataCopyParams yC12C2Params;
        yC12C2Params.blockCount = blockN / BURSTLEN;
        yC12C2Params.blockLen = BURSTLEN * blockM * sizeof(float) / 1024; // 1024字节
        yC12C2Params.srcStride = 0;
        yC12C2Params.dstStride = 0;

        DataCopyEnhancedParams yC12C2EnhanceParams;
        yC12C2EnhanceParams.blockMode = BlockMode::BLOCK_MODE_MATRIX;

        DataCopy(c2fp32Local, c1fp32Local, yC12C2Params, yC12C2EnhanceParams);

        // stage 3.1: fp32 to fp16
        LocalTensor<half> c2fp16Local = co2fp16Buf.Get<half>(blockM * blockN);
        Duplicate<half>(c2fp16Local, (half)0.0, blockM * blockN);
        UnaryRepeatParams castfp32Tofp16Params;
        castfp32Tofp16Params.dstBlkStride = 1; // 1 单词迭代内不同block间地址步长
        castfp32Tofp16Params.srcBlkStride = 1; // 1 单词迭代内不同block间地址步长
        castfp32Tofp16Params.dstRepStride = 4; // 4 迭代间
        castfp32Tofp16Params.srcRepStride = 8; // 8 迭代间
        // 64表示64byte
        Cast(c2fp16Local, c2fp32Local, RoundMode::CAST_NONE, 64, (blockM * blockN / 64), castfp32Tofp16Params);

        // stage 3.3: y from CO2 to CO2, NZ -> ND
        DataCopyParams yNZ2NDParams;
        yNZ2NDParams.blockCount = blockM;
        yNZ2NDParams.blockLen = BURSTLEN * sizeof(half) / BLOCKSIZE;
        yNZ2NDParams.srcStride = 0;
        yNZ2NDParams.dstStride = (blockN - BURSTLEN) * sizeof(half) / BLOCKSIZE;

        for (int j = 0; j < blockN / BURSTLEN; ++j) {
            int srcOffset = j * BURSTLEN * blockM;
            int dstOffset = j * BURSTLEN;
            DataCopy(output[dstOffset], c2fp16Local[srcOffset], yNZ2NDParams);
            pipe_barrier(PIPE_ALL);
        }
        return output;
    }

    __aicore__ inline LocalTensor<half> ProcessBias(LocalTensor<half> &output, uint32_t nn, uint32_t nReal)
    {
        LocalTensor<half> bias = biasBuf.Get<half>(blockN);
        Duplicate<half>(bias, half(0.0), blockN);

        if (isBias == 1) {
            for (int i = 0; i < blockM; i++) {
                DataCopy(bias, gmB[nn * blockN], nReal);
                pipe_barrier(PIPE_ALL);
                Add(output[i * blockN], output[i * blockN], bias, blockN);
            }
        }
        return output;
    }

    __aicore__ inline void ProcessOperate(LocalTensor<half> &output, uint32_t mReal, uint32_t nReal, uint32_t mm,
        uint32_t nn)
    {
        LocalTensor<half> biasfill = biasfillBuf.Get<half>(blockM * blockN);
        Duplicate<half>(biasfill, half(0.0), blockM * blockN);

        DataCopyParams yC22GMParams;
        yC22GMParams.blockCount = mReal;
        yC22GMParams.blockLen = nReal * sizeof(half) / BLOCKSIZE;
        yC22GMParams.srcStride = (blockN - nReal) * sizeof(half) / BLOCKSIZE;
        ;
        yC22GMParams.dstStride = (n - nReal) * sizeof(half) / BLOCKSIZE;

        if (operate != 0) {
            LocalTensor<half> extra = extraBuf.Get<half>(blockM * blockN);

            Duplicate<half>(extra, (half)0.0, blockM * blockN);
            DataCopyParams extraCopyeParams;
            extraCopyeParams.blockCount = mReal;
            extraCopyeParams.blockLen = nReal * sizeof(half) / BLOCKSIZE;
            extraCopyeParams.srcStride = (n - nReal) * sizeof(half) / BLOCKSIZE;
            extraCopyeParams.dstStride = (blockN - nReal) * sizeof(half) / BLOCKSIZE;

            DataCopyEnhancedParams dataCopyEnhancedParams;
            dataCopyEnhancedParams.blockMode = BlockMode::BLOCK_MODE_NORMAL;

            DataCopy(biasfill, gmE[blocksOffset * n + mm * blockM * n + nn * blockN], extraCopyeParams);
            pipe_barrier(PIPE_ALL);

            if (operate == ADD) { // 1 means “+”
                Add(extra, output, biasfill, blockM * blockN);
            } else if (operate == SUB) { // 1 means “-”
                Sub(extra, output, biasfill, blockM * blockN);
            } else if (operate == MUL) { // 3 means “x”
                Mul(extra, output, biasfill, blockM * blockN);
            } else { // means “/”
                Div(extra, output, biasfill, blockM * blockN);
            }

            pipe_barrier(PIPE_ALL);
            DataCopy(gmO[mm * blockM * n + nn * blockN], extra, yC22GMParams);
        } else {
            DataCopy(gmO[mm * blockM * n + nn * blockN], output, yC22GMParams);
        }
    };

    __aicore__ inline LocalTensor<float> Mad(LocalTensor<float> &c1fp32Local, uint32_t mm, uint32_t kk, uint32_t nn,
        uint32_t mReal, uint32_t kReal, uint32_t nReal)
    {
        // stage 0.0: x from gm to A1, ND -> NZ, use DataCopy
        LocalTensor<half> a1X = a1Buf.Get<half>(blockM * blockK);
        Duplicate<half>(a1X, (half)0.0, blockM * blockK);
        LocalTensor<half> a2X = a2Buf.Get<half>(blockM * blockK);
        Duplicate<half>(a2X, (half)0.0, blockM * blockK);
        a2X = ProcessInputX(a1X, a2X, kReal, mReal, kk, mm);

        LocalTensor<half> b1W = b1Buf.Get<half>(blockK * blockN);
        Duplicate<half>(b1W, (half)0.0, blockK * blockN);
        LocalTensor<half> b2W = b2Buf.Get<half>(blockK * blockN);
        Duplicate<half>(b2W, (half)0.0, blockK * blockN);

        b2W = ProcessTrans(b1W, b2W, kReal, nReal, kk, nn);

        // stage 2.0: matmul, format is NZ
        if (kk == 0) {
            MmadParams yMmadInfo;
            yMmadInfo.m = blockM;
            yMmadInfo.n = blockN;
            yMmadInfo.k = blockK;
            yMmadInfo.isBias = false;
            Mmad(c1fp32Local, a2X, b2W, yMmadInfo);
        } else {
            MmadParams yMmadInfo;
            yMmadInfo.m = blockM;
            yMmadInfo.n = blockN;
            yMmadInfo.k = blockK;
            yMmadInfo.isBias = true;
            Mmad(c1fp32Local, a2X, b2W, yMmadInfo);
        }
        return c1fp32Local;
    }

private:
    TPipe pipe;

    GlobalTensor<half> gmX, gmW, gmB, gmE, gmO;

    TBuf<TPosition::A1> a1Buf;
    TBuf<TPosition::A2> a2Buf;
    TBuf<TPosition::B1> b1Buf;
    TBuf<TPosition::B2> b2Buf;

    TBuf<TPosition::CO1> co1fp32Buf;
    TBuf<TPosition::CO2> co2fp32Buf;
    TBuf<TPosition::CO2> co2fp16Buf;

    TBuf<TPosition::CO2> biasBuf;
    TBuf<TPosition::CO2> biasfillBuf;
    TBuf<TPosition::CO2> biasfp32Buf;

    TBuf<TPosition::CO2> extraBuf;
    TBuf<TPosition::CO2> outputBuf;

    uint32_t groupIdx, coreNums, rowsPerBlock, blocksOffset;

    size_t blockM = BLOCK_M;
    size_t blockN = BLOCK_N;
    size_t blockK = BLOCK_K_3;

    uint32_t isTrans;
    uint32_t isBias;
    uint32_t operate;

    uint32_t m;
    uint32_t n;
    uint32_t k;
};
}

namespace {

extern "C" __global__ __aicore__ void matmul_all(GM_ADDR inputX, GM_ADDR inputWeight, GM_ADDR inputBias,
    GM_ADDR inputExtra, GM_ADDR outMatrix, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);

    MatmulAll op(tiling_data.needTrans, tiling_data.withBias, tiling_data.operateType, tiling_data.m,
        tiling_data.n, tiling_data.k, inputBias, inputExtra);
    op.Init(inputX, inputWeight, inputBias, inputExtra, outMatrix);
    op.Process();
}
}