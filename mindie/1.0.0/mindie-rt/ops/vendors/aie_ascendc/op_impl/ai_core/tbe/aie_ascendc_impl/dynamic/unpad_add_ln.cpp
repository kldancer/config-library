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
constexpr size_t MAX_H = 32;
constexpr size_t FP32_MASK = 64;
constexpr size_t FP16_MASK = 128;
constexpr uint64_t HIDDEN_SIZE_MAX = 1024;
constexpr uint64_t NTOKENS_MAX = 16384;
constexpr uint32_t BASE_DIM = 64;
}

namespace AscendC {
class UnpadAddLN {
public:
    __aicore__ inline UnpadAddLN(uint32_t wTotal, uint32_t hTotal, uint32_t groupNums, half pScale, half nScale)
        : wTotal(wTotal), hTotal(hTotal), groupNums(groupNums), pScale(pScale), nScale(nScale)
    {
        castfp16Tofp32Params.dstBlkStride = 1;
        castfp16Tofp32Params.srcBlkStride = 1;
        // compute 64 nums in each repeat, cost 8 * 32Bytes for fp32
        castfp16Tofp32Params.dstRepStride = 8;
        // compute 64 nums in each repeat, cost 4 * 32Bytes for fp16
        castfp16Tofp32Params.srcRepStride = 4;

        castfp32Tofp16Params.dstBlkStride = 1;
        castfp32Tofp16Params.srcBlkStride = 1;
        // compute 64 nums in each repeat, cost 4 * 32Bytes for fp16
        castfp32Tofp16Params.dstRepStride = 4;
        // compute 64 nums in each repeat, cost 8 * 32Bytes for fp32
        castfp32Tofp16Params.srcRepStride = 8;
    }

    __aicore__ inline void Process(GM_ADDR xAddr, GM_ADDR residualAddr, GM_ADDR epsilonAddr,
                                   GM_ADDR weightAddr, GM_ADDR biasAddr, GM_ADDR oAddr)
    {
        size_t hRealNums = ProcessInit(xAddr, residualAddr, epsilonAddr, weightAddr, biasAddr, oAddr);
        size_t hLoops = (hRealNums + MAX_H - 1) / MAX_H;

        // load epsilon value
        // alloc 32Bytes at least, 8 for 32 / sizeof(float)
        LocalTensor<float> epTensor = epsilonBuf.Get<float>(8);
        // alloc 32Bytes at least, 8 for 32 / sizeof(float)
        DataCopy(epTensor, gmEpsilon, 8);
        pipe_barrier(PIPE_ALL);
        epValue = epTensor.GetValue(0);
        pipe_barrier(PIPE_ALL);

        // load weight and bias tensor
        LocalTensor<half> weightTensor = weightBuf.Get<half>(wTotal);
        LocalTensor<half> biasTensor = biasBuf.Get<half>(wTotal);
        DataCopy(weightTensor, gmWeight, wTotal);
        DataCopy(biasTensor, gmBias, wTotal);
        pipe_barrier(PIPE_ALL);

        for (size_t i = 0; i < hLoops; ++i) {
            size_t hReal = (i == hLoops - 1) ? (hRealNums - i * MAX_H) : MAX_H;
            // load x and residual slice
            LocalTensor<half> xTensor = xBuf.Get<half>(MAX_H * wTotal);
            LocalTensor<half> subTensor = subLayerBuf.Get<half>(MAX_H * wTotal);
            LocalTensor<half> outTensor = oBuf.Get<half>(MAX_H * wTotal);

            DataCopy(xTensor, gmX[i * MAX_H * wTotal], hReal * wTotal);
            DataCopy(subTensor, gmSubLayer[i * MAX_H * wTotal], hReal * wTotal);
            pipe_barrier(PIPE_ALL);

            // 1. compute x add residual
            Add(xTensor, xTensor, subTensor, hReal * wTotal);
            pipe_barrier(PIPE_ALL);

            // 2. compute and sub mean
            LocalTensor<half> sumTempTensor = sumTempBuf.Get<half>(MAX_H * FP32_MASK);
            LocalTensor<half> meanTensor = meanBuf.Get<half>(MAX_H);

            LocalTensor<float> sumfp32Tensor = sumfp32Buf.Get<float>(MAX_H * FP32_MASK);
            LocalTensor<float> sumTempfp32Tensor = sumTempfp32Buf.Get<float>(MAX_H * FP32_MASK);
            LocalTensor<float> meanfp32Tensor = meanfp32Buf.Get<float>(MAX_H);

            Duplicate<float>(sumfp32Tensor, (float)0.0, MAX_H * FP32_MASK);
            pipe_barrier(PIPE_ALL);

            xUB2UBParams.blockCount = hReal;
            // uint is 32 Bytes
            xUB2UBParams.blockLen = FP32_MASK * sizeof(half) / 32;
            // uint is 32 Bytes
            xUB2UBParams.srcStride = (wTotal - FP32_MASK) * sizeof(half) / 32;
            xUB2UBParams.dstStride = 0;

            uint64_t castRepeat = hReal / FP32_MASK;
            uint64_t castRemain = hReal % FP32_MASK;
            uint64_t castOffset = castRepeat * FP32_MASK;
            ProcessComputeAndSubMean (sumTempTensor, xTensor, sumTempfp32Tensor, sumfp32Tensor, hReal,
                meanfp32Tensor, meanTensor, castRepeat, castRemain, castOffset);

            // 3. compute variance
            ProcessVariance (sumTempTensor, xTensor, sumTempfp32Tensor, sumfp32Tensor, hReal, meanfp32Tensor,
                meanTensor, castRepeat, castRemain, castOffset);

            // 4. norm
            ProcessNorm(hReal, meanTensor, outTensor, xTensor, weightTensor, biasTensor);
            DataCopy(gmO[i * MAX_H * wTotal], outTensor, hReal * wTotal);
            pipe_barrier(PIPE_ALL);
        }
    }

private:
    __aicore__ inline size_t ProcessInit(GM_ADDR xAddr, GM_ADDR residualAddr, GM_ADDR epsilonAddr,
        GM_ADDR weightAddr, GM_ADDR biasAddr, GM_ADDR oAddr)
    {
        if (groupNums == 0) {
            return -1;
        }
        size_t hRepeat = hTotal / groupNums;
        size_t hRealNums = (block_idx == (groupNums - 1)) ? (hTotal - block_idx * hRepeat) : hRepeat;

        gmX.SetGlobalBuffer((__gm__ half *)xAddr + block_idx * hRepeat * wTotal, hRealNums * wTotal);
        gmSubLayer.SetGlobalBuffer((__gm__ half *)residualAddr + block_idx * hRepeat * wTotal, hRealNums * wTotal);
        gmWeight.SetGlobalBuffer((__gm__ half *)weightAddr);
        gmBias.SetGlobalBuffer((__gm__ half *)biasAddr);
        gmEpsilon.SetGlobalBuffer((__gm__ float *)epsilonAddr);
        gmO.SetGlobalBuffer((__gm__ half *)oAddr + block_idx * hRepeat * wTotal, hRealNums * wTotal);

        pipe.InitBuffer(xBuf, MAX_H * wTotal * sizeof(half));
        pipe.InitBuffer(subLayerBuf, MAX_H * wTotal * sizeof(half));
        pipe.InitBuffer(oBuf, MAX_H * wTotal * sizeof(half));

        pipe.InitBuffer(weightBuf, wTotal * sizeof(half));
        pipe.InitBuffer(biasBuf, wTotal * sizeof(half));
        pipe.InitBuffer(sumTempBuf, MAX_H * FP32_MASK * sizeof(half));
        pipe.InitBuffer(meanBuf, MAX_H * sizeof(half));

        pipe.InitBuffer(sumfp32Buf, MAX_H * FP32_MASK * sizeof(float));
        pipe.InitBuffer(sumTempfp32Buf, MAX_H * FP32_MASK * sizeof(float));
        pipe.InitBuffer(meanfp32Buf, MAX_H * sizeof(float));
        // alloc 32Bytes at least, 8 for 32 / sizeof(float)
        pipe.InitBuffer(epsilonBuf, 8 * sizeof(float));

        return hRealNums;
    }

    __aicore__ inline void ProcessComputeAndSubMean(LocalTensor<half> &sumTempTensor, LocalTensor<half> &xTensor,
        LocalTensor<float> &sumTempfp32Tensor, LocalTensor<float> &sumfp32Tensor, size_t hReal,
        LocalTensor<float> &meanfp32Tensor, LocalTensor<half> &meanTensor, uint64_t castRepeat,
        uint64_t castRemain, uint64_t castOffset)
    {
        for (size_t j = 0; j < (wTotal / FP32_MASK); ++j) {
            size_t offset = j * FP32_MASK;
            DataCopy(sumTempTensor, xTensor[offset], xUB2UBParams);
            pipe_barrier(PIPE_ALL);
            Cast(sumTempfp32Tensor, sumTempTensor, RoundMode::CAST_NONE, FP32_MASK, hReal, castfp16Tofp32Params);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Add(sumfp32Tensor, sumfp32Tensor, sumTempfp32Tensor, FP32_MASK, hReal, { 1, 1, 1, 8, 8, 8 });
            pipe_barrier(PIPE_ALL);
        }

        // 8 is for src repeat stride
        WholeReduceSum(meanfp32Tensor, sumfp32Tensor, FP32_MASK, hReal, 1, 1, 8);
        pipe_barrier(PIPE_ALL);
        Muls(meanfp32Tensor, meanfp32Tensor, (float)nScale, hReal);
        pipe_barrier(PIPE_ALL);

        if (castRepeat > 0) {
            Cast(meanTensor, meanfp32Tensor, RoundMode::CAST_NONE, FP32_MASK, castRepeat, castfp32Tofp16Params);
            pipe_barrier(PIPE_ALL);
        }
        if (castRemain > 0) {
            Cast(meanTensor[castOffset], meanfp32Tensor[castOffset], RoundMode::CAST_NONE,
                castRemain, 1, castfp32Tofp16Params);
            pipe_barrier(PIPE_ALL);
        }

        for (size_t j = 0; j < hReal; ++j) {
            half curMean = meanTensor.GetValue(j);
            size_t offset = j * wTotal;
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src repeat stride
            Adds(xTensor[offset], xTensor[offset], curMean, wTotal);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessVariance(LocalTensor<half> &sumTempTensor, LocalTensor<half> &xTensor,
        LocalTensor<float> &sumTempfp32Tensor, LocalTensor<float> &sumfp32Tensor, size_t hReal,
        LocalTensor<float> &meanfp32Tensor, LocalTensor<half> &meanTensor, uint64_t castRepeat,
        uint64_t castRemain, uint64_t castOffset)
    {
        Duplicate<float>(sumfp32Tensor, (float)0.0, MAX_H * FP32_MASK);
        pipe_barrier(PIPE_ALL);

        for (size_t j = 0; j < (wTotal / FP32_MASK); ++j) {
            size_t offset = j * FP32_MASK;
            DataCopy(sumTempTensor, xTensor[offset], xUB2UBParams);
            pipe_barrier(PIPE_ALL);
            Cast(sumTempfp32Tensor, sumTempTensor, RoundMode::CAST_NONE, FP32_MASK, hReal, castfp16Tofp32Params);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Mul(sumTempfp32Tensor, sumTempfp32Tensor, sumTempfp32Tensor, FP32_MASK, hReal, { 1, 1, 1, 8, 8, 8 });
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Add(sumfp32Tensor, sumfp32Tensor, sumTempfp32Tensor, FP32_MASK, hReal, { 1, 1, 1, 8, 8, 8 });
            pipe_barrier(PIPE_ALL);
        }

        // 8 is for src repeat stride
        WholeReduceSum(meanfp32Tensor, sumfp32Tensor, FP32_MASK, hReal, 1, 1, 8);
        pipe_barrier(PIPE_ALL);
        Muls(meanfp32Tensor, meanfp32Tensor, (float)pScale, hReal);
        pipe_barrier(PIPE_ALL);
        Adds(meanfp32Tensor, meanfp32Tensor, epValue, hReal);
        pipe_barrier(PIPE_ALL);
        Sqrt(meanfp32Tensor, meanfp32Tensor, hReal);
        pipe_barrier(PIPE_ALL);

        if (castRepeat > 0) {
            Cast(meanTensor, meanfp32Tensor, RoundMode::CAST_NONE, FP32_MASK, castRepeat, castfp32Tofp16Params);
            pipe_barrier(PIPE_ALL);
        }
        if (castRemain > 0) {
            Cast(meanTensor[castOffset], meanfp32Tensor[castOffset], RoundMode::CAST_NONE,
                castRemain, 1, castfp32Tofp16Params);
            pipe_barrier(PIPE_ALL);
        }
    }

    __aicore__ inline void ProcessNorm(size_t hReal, LocalTensor<half> &meanTensor, LocalTensor<half> &outTensor,
        LocalTensor<half> &xTensor, LocalTensor<half> &weightTensor, LocalTensor<half> &biasTensor)
    {
        for (size_t h = 0; h < hReal; ++h) {
            size_t offset = h * wTotal;
            half curValue = meanTensor.GetValue(h);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst repeat stride
            Duplicate<half>(outTensor[h * wTotal], curValue, wTotal);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Div(xTensor[h * wTotal], xTensor[h * wTotal], outTensor[h * wTotal], wTotal);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Mul(outTensor[offset], xTensor[offset], weightTensor, wTotal);
            pipe_barrier(PIPE_ALL);
            // 8 is for dst/src0/src1 repeat stride
            Add(outTensor[offset], outTensor[offset], biasTensor, wTotal);
            pipe_barrier(PIPE_ALL);
        }
    }
private:
    TPipe pipe;
    TBuf<TPosition::VECIN> xBuf, subLayerBuf, tmpBuf, weightBuf, biasBuf, meanBuf, epsilonBuf;
    TBuf<TPosition::VECIN> sumfp32Buf, sumTempfp32Buf, meanfp32Buf;
    TBuf<TPosition::VECOUT> oBuf, sumTempBuf;

    GlobalTensor<half> gmX, gmSubLayer, gmWeight, gmBias, gmO;
    GlobalTensor<float> gmEpsilon;

    uint32_t wTotal = 0;
    uint32_t hTotal = 0;
    uint32_t groupNums = 0;

    half pScale = 0.0;
    half nScale = 0.0;

    float epValue = 0.0;

    UnaryRepeatParams castfp16Tofp32Params;
    UnaryRepeatParams castfp32Tofp16Params;

    DataCopyParams xUB2UBParams;
};
}

namespace {
    extern "C" __global__ __aicore__ void unpad_add_ln(GM_ADDR hiddenStates, GM_ADDR residual, GM_ADDR epsilon,
        GM_ADDR weight, GM_ADDR bias, GM_ADDR outStates, GM_ADDR workspace, GM_ADDR tiling)
    {
        GET_TILING_DATA(tilingData, tiling);
        const half *pScale = static_cast<const half *>(static_cast<void *>(&tilingData.scale));
        const half *nScale = static_cast<const half *>(static_cast<void *>(&tilingData.nScale));

        AscendC::UnpadAddLN op(tilingData.wTotal, tilingData.hTotal, tilingData.coreNums, *pScale, *nScale);
        op.Process(hiddenStates, residual, epsilon, weight, bias, outStates);
    }
}