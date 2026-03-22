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
#ifndef ASCENDIE_ACTIVATIONLAYER_H
#define ASCENDIE_ACTIVATIONLAYER_H
#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ActivationLayer definition
 */
class ActivationLayerImpl;
class ActivationLayer : public BaseLayer {
public:
    /**
     * @brief 获取要执行的激活函数类型。
     * @note 支持的类型为RELU、SIGMOID、LEAKY_RELU、HARD_SIGMOID、TANH、SWISH、
     * GELU、ELU、SELU、SOFTPLUS、MISH 和 RELU6。
     * @return 返回要执行的激活函数类型。
     */
    ActivationKind GetActivationKind() const noexcept;

    /**
     * @brief 设置激活函数的Alpha参数。
     * @note alpha参数由以下激活函数使用：LeakyRelu、Clip、HardSigmoid、Elu。被其他激活函数忽略。
     * @param [in] a：Alpha参数的值。
     */
    void SetAlpha(float a) noexcept;

    /**
     * @brief 获取激活函数的Alpha参数。
     * @note 若还未调用SetAlpha接口，返回默认值0.0。
     * @return 返回激活函数的Alpha参数。
     */
    float GetAlpha() const noexcept;

    /**
     * @brief 设置激活函数的beta参数。
     * @note beta参数由以下激活函数使用：Clip、HardSigmoid、Softplus。被其他激活函数忽略。
     * @param [in] b：Beta参数的值。
     */
    void SetBeta(float b) noexcept;

    /**
     * @brief 获取激活函数的Beta参数。
     * @note 若还未调用SetBeta接口，返回默认值0.0。
     * @return 返回激活函数的Beta参数。
     */
    float GetBeta() const noexcept;

    /**
     * @brief 设置激活函数的scale参数。
     * @note scale参数由以下激活函数使用：SWISH、ELU。被其他激活函数忽略。
     * @param [in] scale：scale参数的值。
     */
    void SetScale(float scale) noexcept;

    /**
     * @brief 获取激活函数的scale参数。
     * @note 若还未调用SetScale接口，返回默认值1.0。
     * @return 返回激活函数的scale参数。
     */
    float GetScale() const noexcept;

    /**
     * @brief 设置激活函数的inputScale参数。
     * @note inputScale参数由以下激活函数使用：ELU。被其他激活函数忽略。
     * @param [in] inputScale：inputScale参数的值。
     */
    void SetInputScale(float inputScale) noexcept;

    /**
     * @brief 获取激活函数的inputScale参数。
     * @note 若还未调用SetInputScale接口，返回默认值1.0。
     * @return 返回激活函数的inputScale参数。
     */
    float GetInputScale() const noexcept;
    
    /**
     * @brief 传入阈值。
     * @note 当前仅softplus函数使用，当输入数据大于等于此值时恢复为线性函数。
     * @param [in] threshold: 输入的阈值，类型为float。
     */
    void SetThreshold(float threshold) noexcept;

    /**
     * @brief 获取阈值。
     * @note 当前仅softplus函数使用，当输入数据大于等于此值时恢复为线性函数。
     * @return 返回传入的阈值。
     */
    float GetThreshold() const noexcept;

    /**
     * @brief ActivationLayer的析构函数。
     */
    ~ActivationLayer() noexcept override = default;

protected:
    /**
     * @brief 管理ActivationLayer具体实现的指针。
     */
    std::shared_ptr<ActivationLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_ACTIVATIONLAYER_H */