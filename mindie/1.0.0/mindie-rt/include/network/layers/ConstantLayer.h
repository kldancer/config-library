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

#ifndef ASCENDIE_CONSTANT_LAYER_H
#define ASCENDIE_CONSTANT_LAYER_H

#include "common/Dims.h"
#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ConstantLayer definition
 */
class ConstantLayerImpl;
class ConstantLayer : public BaseLayer {
public:
    /**
     * @brief 设置ConstantLayer的权重信息。
     * @note weights可以为空，但需要保证dimensions所有维度相乘等于weights.count。
     * @param [in] weights: ConstantLayer的权重信息。
     */
    void SetWeights(WeightsBuf weights) noexcept;

    /**
     * @brief 获取ConstantLayer的权重信息。
     * @note 若还未调用SetWeights接口，返回默认值{0,0,0,0}。
     * @return 返回ConstantLayer的权重信息。
     */
    WeightsBuf GetWeights() const;

    /**
     * @brief 设置ConstantLayer的维度信息。
     * @note dimensions中最多8个元素，每个元素需要≥0，所有维度乘积小于UINT64上限。
     * @param [in] dinmensions：ConstantLayer的维度信息。
     */
    void SetDimensions(Dims dimensions) noexcept;

    /**
     * @brief 获取ConstantLayer的维度信息。
     * @note 若还未调用SetDimensions接口，返回默认值 {0,0,0,0}。
     * @return 返回ConstantLayer的维度信息。
     */
    Dims GetDimensions() const;

    /**
     * @brief 标记权重节点是否需要运行时更新
     * @note 默认权重节点不需要更新
     * @param [in] enableRuntimeMutation: 更新标志, true: 需要更新, false: 不需要更新
     */
    void EnableRuntimeMutation(bool enableRuntimeMutation) noexcept;

protected:
    /**
     * @brief 管理ConstantLayer具体实现的指针。
     */
    std::shared_ptr<ConstantLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif // ASCENDIE_CONSTANT_LAY