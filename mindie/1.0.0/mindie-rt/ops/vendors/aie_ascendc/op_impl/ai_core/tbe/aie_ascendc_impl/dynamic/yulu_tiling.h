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
#ifndef OP_YULU_TILING_KERNEL_H
#define OP_YULU_TILING_KERNEL_H


const uint32_t NUMBER_OF_DOUBLE = 2;
constexpr uint32_t DEFAULT_MIN_BLOCK_SIZE = 32; // 最小的数据块长度，32Bytes
const uint32_t PACK_SIZE = 512; // pack unit in cache 512B
const uint32_t HEAD_AND_TAIL = 2; // pack unit in cache 512B

const uint32_t DEFAULT_COMBINE_FACTOR = 2;

// tiling for Yulu Vector on one VectorCore
struct YuluTilingKernel {
    uint64_t blockLength = 0; // number of calculations on this core
    uint32_t tileLength = 0;  // number of calculations in one tile
    uint64_t tileLoopNum = 0; // number of tile(tileLength per tile) on this core, don't include tailTile
    uint32_t tailTileLen = 0; // number of calculations in tail tile

    uint64_t gmOffset = 0;

    bool warmup;
    uint64_t headElemNum;
    uint64_t tailElemNum;

    // calc tiling data
    __aicore__ void GetTilingAndOffset(GM_ADDR tiling_gm, uint32_t inputDTypeLen)
    {
        GET_TILING_DATA(tilingDataIn, tiling_gm);
        const YuluTilingData* tempTilingGm = &tilingDataIn;
        blockLength = tempTilingGm->singleCoreBlockLens[get_block_idx()];
        tileLength = tempTilingGm->maxTileLen;
        tileLoopNum = blockLength / tileLength;
        tailTileLen = blockLength - tileLength * tileLoopNum;

        gmOffset = 0;
        for (uint32_t i = 0; i < get_block_idx(); i++) {
            gmOffset += tempTilingGm->singleCoreBlockLens[i];
        }

        uint64_t packElemNum = (inputDTypeLen == 0) ? PACK_SIZE : (PACK_SIZE / inputDTypeLen);
        warmup = tileLoopNum > 1 && tileLength >= packElemNum * HEAD_AND_TAIL;
        if (warmup) {
            headElemNum = ((tileLength / HEAD_AND_TAIL) + packElemNum - 1) / packElemNum * packElemNum;
            tailElemNum = tileLength - headElemNum;
        }
    }
};

#endif  // OP_YULU_TILING_KERNEL_H
