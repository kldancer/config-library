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

#ifndef ASCENDIE_SHAPELAYER_H
#define ASCENDIE_SHAPELAYER_H

#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ShapeLayer definition
 */
class ShapeLayerImpl;
class ShapeLayer : public BaseLayer {
public:
    /**
     * @brief 设置ShapeLayer的输出数据类型
     * @note 默认输出类型为int32
     */
    void SetOutputType(DataType type) noexcept;

    /**
     * @brief 获取ShapeLayer的输出数据类型
     * @note 默认输出类型为int32
     */
    DataType GetOutputType() const noexcept;

    /**
     * @brief ShapeLayer的析构函数。
     */
    ~ShapeLayer() noexcept override = default;

protected:
    /**
     * @brief 管理ShapeLayer具体实现的指针。
     */
    std::shared_ptr<ShapeLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_SHAPELAYER_H */