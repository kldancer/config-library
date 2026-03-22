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

#ifndef SRC_OP_KERNEL_LCCL_ALLGATHER_BIGDATA_H
#define SRC_OP_KERNEL_LCCL_ALLGATHER_BIGDATA_H

#include "collectives.h"
#include "ipc_queue.h"

using namespace AscendC;

constexpr int64_t INPUT_TO_SHARE_CORE_NUM = 8;  // input->share拷贝使用核数
constexpr int64_t SHARE_TO_OUTPUT_CORE_NUM = 8; // share->output拷贝使用核数
constexpr int64_t SHARE_QUE_DEPTH = 32;         // 共享队列深度

enum TurnType {
    NORMAL = 0, // 常规轮次
    FINAL = 1,  // 最后一轮
    TURNS = 2,  // 不同轮次使用不同大小的数据
};

enum CoreGroup {
    INPUT_TO_SHARE,  // 负责从input拷贝数据至share的核
    SHARE_TO_OUTPUT, // 负责从share拷贝数据至output的核
    RESERVE,
};

template <typename T> class AllGatherBigData : public Collectives {
public:
    __aicore__ inline AllGatherBigData(int64_t rank, int64_t rankSize, uint32_t extraFlag)
        : Collectives(rank, rankSize, extraFlag)
    {}

    __aicore__ inline void Init(KERNELS_ARGS_FUN())
    {
        Collectives::Init(KERNELS_ARGS_CALL());

        // 初始化输入输出
        inputGm.SetGlobalBuffer((__gm__ T *)input, len);
        outputGm.SetGlobalBuffer((__gm__ T *)output, len * rankSize);
        // 初始化核分组
        InitCoreGroup();
        // 初始化共享内存队列
        InitShareQue();
        // 初始化数据切片
        InitDataSlice();
    }

    __aicore__ inline void InitCoreGroup()
    {
        // 计算单个rank的多核并行度，用于切分共享内存和分配任务
        if (rankSize < SHARE_TO_OUTPUT_CORE_NUM) {
            coreNumPerRank = SHARE_TO_OUTPUT_CORE_NUM / rankSize;
        } else {
            coreNumPerRank = 1;
        }
        if (coreNumPerRank > INPUT_TO_SHARE_CORE_NUM) {
            coreNumPerRank = INPUT_TO_SHARE_CORE_NUM;
        }

        // 预计算每个que的大小，如果输入数据量太小，仅用一个core和一个que即可
        int64_t preQueElemLen = IPC_BUFF_MAX_SIZE / sizeof(T) / coreNumPerRank / SHARE_QUE_DEPTH;
        if (len < preQueElemLen) {
            coreNumPerRank = 1;
        }

        // 多核按功能分组
        if (blockIdx < coreNumPerRank) {
            coreGroup = INPUT_TO_SHARE;
            groupCoreIdx = blockIdx;
        } else if (blockIdx >= INPUT_TO_SHARE_CORE_NUM &&
            blockIdx < INPUT_TO_SHARE_CORE_NUM + coreNumPerRank * rankSize) {
            coreGroup = SHARE_TO_OUTPUT;
            groupCoreIdx = blockIdx - INPUT_TO_SHARE_CORE_NUM;
        } else {
            coreGroup = RESERVE;
        }
    }

    __aicore__ inline void InitShareQue()
    {
        // 初始化共享queue
        queLenPerCore = IPC_BUFF_MAX_SIZE / sizeof(T) / coreNumPerRank;
        queElemLen = queLenPerCore / SHARE_QUE_DEPTH;
        if (coreGroup == INPUT_TO_SHARE) {
            int64_t shareOffsetSize = groupCoreIdx * queLenPerCore * sizeof(T);
            que.Init(&sync, magic, shareAddrs[rank] + IPC_DATA_OFFSET + shareOffsetSize, queLenPerCore, queElemLen);
        } else if (coreGroup == SHARE_TO_OUTPUT) {
            int64_t shareOffsetSize = groupCoreIdx % coreNumPerRank * queLenPerCore * sizeof(T);
            targetRank = groupCoreIdx / coreNumPerRank;
            que.Init(&sync, magic, shareAddrs[targetRank] + IPC_DATA_OFFSET + shareOffsetSize, queLenPerCore,
                queElemLen);
        }

        // 初始化rankList
        for (auto r = 0; r < rankSize; ++r) {
            rankList[r] = r;
        }
    }

    __aicore__ inline void InitDataSlice()
    {
        // que的数量与step2中负责单个rank核数相同
        int64_t queNum = coreNumPerRank;
        if (coreGroup == INPUT_TO_SHARE) {
            SplitData(len, queNum, groupCoreIdx, inputOffset, inputLen);
            sliceNum = CeilDiv(inputLen, queElemLen);
        } else if (coreGroup == SHARE_TO_OUTPUT) {
            int64_t rankOutputOffset;
            SplitData(len, queNum, groupCoreIdx % coreNumPerRank, rankOutputOffset, outputLen);
            outputOffset = groupCoreIdx / coreNumPerRank * len + rankOutputOffset;
            sliceNum = CeilDiv(outputLen, queElemLen);
        }
    }

    __aicore__ inline void InputToSharePipeline()
    {
        int64_t sliceIdx = 0;
        int64_t inputPos = inputOffset;
        int64_t waitIdx = rank * coreNumPerRank % SHARE_TO_OUTPUT_CORE_NUM + groupCoreIdx + INPUT_TO_SHARE_CORE_NUM;

        int64_t dataLen = queElemLen;
        while (sliceIdx < sliceNum) {
            if (sliceIdx == sliceNum - 1) {
                dataLen = inputLen - queElemLen * sliceIdx;
            }

            que.DeQue(rankList, rankSize, waitIdx); // 所有rank的与组内位置相同的标志位
            shareGm = que.EnQue();
            CpGM2GMPingPong<T>(dataLen * sizeof(T), inputGm[inputPos], shareGm, -1);
            sync.SetInnerFlag(magic, sliceIdx); // set input->share的同步位

            sliceIdx++;
            inputPos += dataLen;
        }
    }

    __aicore__ inline void ShareToOutputPipeline()
    {
        int64_t sliceIdx = 0;
        int64_t outputPos = outputOffset;
        int64_t waitIdx = groupCoreIdx % coreNumPerRank; // 使用等待que的编号来区分标志位位置
        targetRank = groupCoreIdx / coreNumPerRank;

        // 等待input->output完成
        sync.WaitInnerFlag(magic, sliceIdx, targetRank, waitIdx);
        int64_t eventIdx = sync.GetInnerFlag(targetRank, waitIdx) & EVENT_ID_MASK;

        int64_t dataLen = queElemLen;
        while (sliceIdx < sliceNum) {
            if (sliceIdx == sliceNum - 1) {
                dataLen = outputLen - queElemLen * sliceIdx;
            }

            if (eventIdx < sliceIdx) { // 标志位滞后时，才会更新标志位，否则复用标志位
                eventIdx = sync.GetInnerFlag(targetRank, waitIdx) & EVENT_ID_MASK;
                continue;
            }

            shareGm = que.ReadFront();
            CpGM2GMPingPong<T>(dataLen * sizeof(T), shareGm, outputGm[outputPos], -1);
            sync.SetInnerFlag(magic, sliceIdx); // set share->output的同步位

            sliceIdx++;
            outputPos += dataLen;
        }
    }

    __aicore__ inline void Process()
    {
        if (coreGroup == INPUT_TO_SHARE) {
            InputToSharePipeline();
        } else if (coreGroup == SHARE_TO_OUTPUT) {
            ShareToOutputPipeline();
        }
    }

private:
    __aicore__ inline void SplitData(const int64_t totalLen, const int64_t useCoreNum, const int64_t useCoreIdx,
        int64_t &dataOffset, int64_t &dataLen)
    {
        // 向上整除获取每个core切分的数据个数
        dataLen = CeilDiv(totalLen, useCoreNum);
        // 数据量极小或略微超过核数的情况，后面若干个core数据量为0
        dataOffset = useCoreIdx * dataLen; // 使用当前block在useBlock里的相对索引来计算偏移
        if (dataOffset >= totalLen) {
            dataOffset = totalLen;
            dataLen = 0;
            return;
        }
        // 非整除情况，最后一个core数据量为剩余数据量
        if (dataOffset + dataLen > totalLen) {
            dataLen = totalLen - dataOffset;
        }
    }

private:
    GlobalTensor<T> inputGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<T> shareGm;

    int rankList[LCAL_MAX_RANK_SIZE];
    IpcQueue<T> que;       // 共享内存队列
    int64_t queLenPerCore; // 共享队列长度（以T计）
    int64_t queElemLen;    // 共享队列里每个元素大小（以T计）

    CoreGroup coreGroup;    // 当前核的功能分组
    int64_t groupCoreIdx;   // 当前核在组内的索引
    int64_t coreNumPerRank; // step2时，每个rank分配core数
    int64_t targetRank;     // step2时，当前core负责的rank

    int64_t sliceNum;     // 当前core负责的切片总数
    int64_t inputOffset;  // 当前core负责的input偏移（以T计）
    int64_t inputLen;     // 当前core负责的input长度（以T计）
    int64_t outputOffset; // 当前core负责的output偏移（以T计）
    int64_t outputLen;    // 当前core负责的output长度（以T计）
};

#endif // SRC_OP_KERNEL_LCCL_ALLGATHER_BIGDATA_H
