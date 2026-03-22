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

#ifndef ASCENDIE_SHUFFLE_LAYER_H
#define ASCENDIE_SHUFFLE_LAYER_H

#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ShuffleLayer definition
 */
class ShuffleLayerImpl;
class ShuffleLayer : public BaseLayer {
public:
    /**
     * @brief 设置第一次转置参数，ShuffleLayer会增加一个转置操作。
     * @note rank必须等于input输入张量的维数，order[MAX_DIM]中轴序号不可以有重复。
     *       不能在SetReshapeDimensions之后调用。
     * @param [in] conf: PermutationConf枚举值包含rank转置后的维数，转置后轴序号的映射关系order[MAX_DIM]。
     * (例如 input dims [2, 3, 4], conf1 { 3, { 2, 0, 1 } }, 转置后为 [4, 2, 3])
     */
    void SetFirstTranspose(PermutationConf conf) noexcept;

    /**
     * @brief 获取第一次转置参数
     * @note 默认值为{ -1, {} }
     * @return PermutationConf: 枚举值包含rank转置后的维数，转置后轴序号的映射关系order[MAX_DIM]。
     */
    PermutationConf GetFirstTranspose() const noexcept;

    /**
     * @brief 设置一个变换张量形状操作
     * @note 每一位必须大于0，且乘积等于输入张量的维度乘积。
     *       不可以在SetSecondTranspose之后调用。
     * @param [in] dimensions: 变换后的张量形状。
     */
    void SetReshapeDimensions(Dims dimensions) noexcept;

    /**
     * @brief 获取变换张量形状的目标形状
     * @note 默认值为Dims({})
     * @return Dims: 目标形状
     */
    Dims GetReshapeDimensions() const noexcept;

    /**
     * @brief 设置第二次转置参数，ShuffleLayer会增加一个转置操作。
     * @note rank必须等于input输入张量的维数，order[MAX_DIM]中轴序号不可以有重复。
     * @param [in] conf: PermutationConf枚举值包含rank转置后的维数，转置后轴序号的映射关系order[MAX_DIM]。
     * (例如 input dims [2, 3, 4], conf1 { 3, { 2, 0, 1 } }, 转置后为 [4, 2, 3])
     */
    void SetSecondTranspose(PermutationConf conf) noexcept;

    /**
     * @brief 获取第二次转置参数
     * @note 默认值为{ -1, {} }
     * @return PermutationConf: 枚举值包含rank转置后的维数，转置后轴序号的映射关系order[MAX_DIM]。
     */
    PermutationConf GetSecondTranspose() const noexcept;

    /* *
     * \brief Add or replace input tensor to current layer.
     * \param index the index of input to modify
     * \param tensor the new input.
     *
     * The ShuffleLayer layer currently allows to set 2 index, currently only support to set once.
     * - 0: To replace the tensor to be reshape or Transpose.
     * - 1: The shape tensor. 1D int64_t shape tensor.
     */
    using BaseLayer::SetInput;

protected:
    /**
     * @brief 管理ShuffleLayer具体实现的指针。
     */
    std::shared_ptr<ShuffleLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif // ASCENDIE_SHUFFLE_LAYER_H