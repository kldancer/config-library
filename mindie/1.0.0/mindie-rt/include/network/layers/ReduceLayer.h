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

#ifndef ASCENDIE_REDUCE_LAYER_H
#define ASCENDIE_REDUCE_LAYER_H

#include "network/layers/BaseLayer.h"
#include "common/Common.h"
#include "common/Dims.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ReduceLayer definition
 */
class ReduceLayerImpl;
class ReduceLayer : public BaseLayer {
public:
    /**
     * @brief 获取ReduceLayer的操作类型
     * @note 当前支持的操作类型包括AVG、MAX、MIN、SUM、STD
     * @return 返回ReduceLayer的操作类型
     */
    ReduceOperation GetOperation() const noexcept;

    /**
     * @brief 设置并更新RedeceLayer中需要进行reduce操作的维度
     * @note 进行reduce操作的维度会被消除或设置为1，通过keepDimensions参数设置。
     * @param [in] reduceAxes:需要进行reduce操作的维度下标
     */
    void SetReduceAxes(std::vector<int64_t> reduceAxes) noexcept;

    /**
     * @brief 获取需要进行reduce操作的维度
     * @note 传入int64_t类型vector，需要保证下标值在[0, inputDims.size())
     * @return 返回包含维度下标信息的int64_t数组
     */
    std::vector<int64_t> GetReduceAxes() const noexcept;

    /**
     * @brief 设置是否保留输出tensor的reduce维度，并设置为1。
     * @note 无
     * @param [in] keepDimensions:布尔值
     */
    void SetKeepDimensions(bool keepDimensions) noexcept;

    /**
     * @brief 获取KeepDimensions参数。
     * @note 无
     * @return 返回RedeceLayer的KeepDimensions参数，布尔值。
     */
    bool GetKeepDimensions() const noexcept;

protected:
    /**
     * @brief 管理ReduceLayer具体实现的指针。
     */
    std::shared_ptr<ReduceLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif // ASCENDIE_REDUCE_LAYER_H