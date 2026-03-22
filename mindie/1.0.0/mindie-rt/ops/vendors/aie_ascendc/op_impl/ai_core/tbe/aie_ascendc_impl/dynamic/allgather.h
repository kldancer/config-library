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

#ifndef SRC_OP_KERNEL_LCCL_ALLGATHER_H
#define SRC_OP_KERNEL_LCCL_ALLGATHER_H

#include "collectives.h"

using namespace AscendC;

constexpr int64_t MEM_DMA_UNIT_SIZE = MEM_DMA_UNIT_INT_NUM * sizeof(int64_t);
constexpr int64_t STEP1 = 1; // 1: 算法步骤1

template <typename T> class AllGather : public Collectives {
public:
    __aicore__ inline AllGather(int64_t rank, int64_t rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());

        // 共享数据区起点的偏移
        baseOffsetSize = IPC_DATA_OFFSET;

        // 计算step1数据分片，input-->share，所有core参与搬运
        GetBlockDataCount(len, blockNum, offsetFromInput, countToShare);
        offsetToShare = offsetFromInput;

        // 当前block的input分片
        inputGm.SetGlobalBuffer((__gm__ T *)input + offsetFromInput, countToShare);

        // 计算step2数据分片，share-->output，当前core负责一个rank的部分或全部数据搬运
        // rankSize > 0, tiling阶段保证；blockNum > 0
        blockNumPerRank = blockNum / rankSize; // 均分core至每个rank，多余的core不使用
        GetBlockDataCount(len, blockNumPerRank, offsetFromShare, countToOutput);
        blockRank = blockIdx / blockNumPerRank;
        offsetToOutput = blockRank * len + offsetFromShare;

        // 当前block的output分片
        outputGm.SetGlobalBuffer((__gm__ T *)output + offsetToOutput, countToOutput);
    }

    __aicore__ inline void Process()
    {
        // step1: 拷贝input至共享内存
        shareGm.SetGlobalBuffer((__gm__ T *)(shareAddrs[rank] + baseOffsetSize) + offsetToShare, countToShare);
        if (countToShare > 0) {
            CpGM2GM<T>(shareGm, inputGm, countToShare, COPYONLY);
        }
        // 卡内同步，确保数据已拷贝至共享内存
        sync.SetInnerFlag(magic, STEP1);                 // 当前rank当前核的数据搬运已完成
        sync.WaitRankInnerFlag(magic, STEP1, blockRank); // 等待目标rank的数据全部搬运完成
        // step2：拷贝共享内存至output
        shareGm.SetGlobalBuffer((__gm__ T *)(shareAddrs[blockRank] + baseOffsetSize) + offsetFromShare, countToOutput);
        if (countToOutput > 0) {
            CpGM2GM<T>(outputGm, shareGm, countToOutput, COPYONLY);
        }
    }

private:
    __aicore__ inline void GetBlockDataCount(const int64_t dataLen, const int64_t useBlockNum, int64_t &blockDataOffset,
        int64_t &blockDataCount)
    {
        // 向上整除获取每个core切分的数据个数
        blockDataCount = CeilDiv(dataLen, useBlockNum);
        // 设置每个core数据下限
        blockDataCount =
            blockDataCount > MEM_DMA_UNIT_SIZE / sizeof(T) ? blockDataCount : MEM_DMA_UNIT_SIZE / sizeof(T);
        // 极小数据量情况，core分配到数据下限，后面若干个core数据量为0
        blockDataOffset = blockIdx % useBlockNum * blockDataCount; // 使用当前block在useBlock里的相对index计算偏移
        if (blockDataOffset >= dataLen) {
            blockDataOffset = dataLen;
            blockDataCount = 0;
            return;
        }
        // 非整除情况，最后一个core数据量为剩余数据量
        if (blockDataOffset + blockDataCount > dataLen) {
            blockDataCount = dataLen - blockDataOffset;
        }
    }

private:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> shareGm;

    int64_t baseOffsetSize; // 共享数据区起点的偏移（Bytes）
    // step1数据切片
    int64_t offsetFromInput; // 从input拷贝数据的地址偏移
    int64_t offsetToShare;   // 拷贝至share[rank]数据的地址偏远
    int64_t countToShare;    // 拷贝至share[rank]数据的个数
    // step2数据切片
    int64_t blockNumPerRank; // 单个rank负责搬运数据的core数量
    int64_t blockRank;       // 当前core负责搬运数据的rank
    int64_t offsetFromShare; // 从share[blockRank]拷贝数据的地址偏移
    int64_t offsetToOutput;  // 拷贝至output数据的地址偏远
    int64_t countToOutput;   // 拷贝至output数据的个数
};

#endif // SRC_OP_KERNEL_LCCL_ALLGATHER_H