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

#ifndef SRC_OP_KERNEL_LCCL_COMM_ARGS_H
#define SRC_OP_KERNEL_LCCL_COMM_ARGS_H

#include "kernel_operator.h"

constexpr int64_t LCAL_MAX_RANK_SIZE = 16;
constexpr int64_t IPC_BUFF_MAX_SIZE = 100 * 1024 * 1024;
constexpr int64_t FLAG_NUM = 256 * 1024;             // 512K个int64_t
constexpr int64_t IPC_DATA_OFFSET = 2 * 1024 * 1024; // 前2MB作为flag标志位，之后100MB作为数据存储
constexpr int64_t SYNC_FLAG_BIT_NUM = 10;
constexpr int64_t MEM_DMA_UNIT_INT_NUM = 4;
constexpr int64_t EVENT_ID_MASK = 0xFFFFFFFF;
constexpr int64_t PING_PONG_SIZE = 2;
constexpr int64_t UB_SINGLE_DMA_SIZE_MAX = 190 * 1024;
constexpr int64_t UB_SINGLE_PING_PONG_ADD_SIZE_MAX = UB_SINGLE_DMA_SIZE_MAX / 2;
constexpr int64_t RANK_SIZE_TWO = 2; // 可用SIO的规模，以及是否需要跨卡搬运数据核的分界规模

enum Op {
    COPYONLY = -1,
    ADD = 0,
    MUL = 1,
    MAX = 2,
    MIN = 3
};

struct ExtraFlag {
    static constexpr uint32_t RDMA = 1;
    static constexpr uint32_t DETERMINISTIC = 1 << 3;
    static constexpr uint32_t QUANT_FP16 = 1 << 4;
    static constexpr uint32_t QUANT_FP32 = 1 << 5;
};

struct CommArgs {
    int rank;
    int rankSize;
    uint32_t extraFlag;
    GM_ADDR peerMems[LCAL_MAX_RANK_SIZE];
    int64_t sendCountMatrix[LCAL_MAX_RANK_SIZE * LCAL_MAX_RANK_SIZE];
};

#endif // SRC_OP_KERNEL_LCCL_COMM_ARGS_H