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

#ifndef ASCENDIE_CLIP_LAYER_H
#define ASCENDIE_CLIP_LAYER_H

#include "common/Common.h"
#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ClipLayer definition
 */
class ClipLayerImpl;
class ClipLayer : public BaseLayer {
public:
    /**
     * @brief 设置最小值。
     * @param [in] min: 最小值。
     */
    void SetMin(float min) noexcept;

    /**
     * @brief 获取最小值。
     * @return 返回最小值。
     */
    float GetMin() const noexcept;

    /**
     * @brief 设置最大值。
     * @param [in] max: 最大值。
     */
    void SetMax(float max) noexcept;

    /**
     * @brief 获取最大值。
     * @return 返回最大值。
     */
    float GetMax() const noexcept;

    /**
     * @brief ClipLayer的默认析构函数。
     */
    ~ClipLayer() override = default;

protected:
    /**
     * @brief 管理ClipLayer具体实现的指针。
     */
    std::shared_ptr<ClipLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif // ASCENDIE_CLIP_LAYER_H