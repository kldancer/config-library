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
#ifndef ASCENDIE_CONVOLUTION_LAYER_H
#define ASCENDIE_CONVOLUTION_LAYER_H
#include <memory>
#include "common/Dims.h"
#include "network/layers/BaseLayer.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief ConvolutionLayer definition
 */
class ConvolutionLayerImpl;
class ConvolutionLayer : public BaseLayer {
public:
    /**
     * @brief 设置卷积的步长参数。
     * @note strides 不能为空，必须为4维或2维。每一维取值范围要求[1,16]。如果维度是4维，N维和C维必须设置为1。Default: (1, 1, 1, 1)。
     * @param [in] strides: 步长参数的维度。
     */
    void SetStrides(Dims strides) noexcept;

    /**
     * @brief 获取卷积的步长参数的维度。
     * @note 若还未调用SetStrides接口，返回默认值 (1, 1, 1, 1)。
     * @return 返回卷积的步长参数的维度。
     */
    Dims GetStrides() const noexcept;

    /**
     * @brief 设置卷积的填充参数。
     * @note paddings 不能为空，对于Conv2D必须为4维或2维，对于Conv3D必须为6维或3维。每一维取值范围要求[0,255]。
     * @param [in] paddings: 填充参数的维度。
     */
    void SetPaddings(Dims paddings) noexcept;

    /**
     * @brief 获取卷积的填充参数的维度。
     * @note 若还未调用SetStrides接口，返回默认值 (0, 0, 0, 0)。
     * @return 返回卷积的填充参数的维度。
     */
    Dims GetPaddings() const noexcept;

    /**
     * @brief 设置卷积的扩张参数。
     * @note dilations 不能为空，必须为4维或2维。每一维取值范围[1,255]。
     * @param [in] dilations: 扩张参数的维度。
     */
    void SetDilations(Dims dilations) noexcept;

    /**
     * @brief 获取卷积的扩张参数的维度。
     * @note 若还未调用SetDilations接口，返回默认值 (1, 1, 1, 1)。
     * @return 返回卷积的扩张参数的维度。
     */
    Dims GetDilations() const noexcept;

    /**
     * @brief 设置卷积的分组数量。
     * @note groupNum 不能为0。要保证输入通道数和输出通道数都能被 "groupNum" 整除。
     * @param [in] groupNum: 分组数量。
     */
    void SetGroupNum(int32_t groupNum) noexcept;

    /**
     * @brief 获取卷积的分组数量。
     * @note 若还未调用SetGroupNum接口，返回默认值 1。
     * @return 返回卷积的分组数量。
     */
    int32_t GetGroupNum() const noexcept;

    /**
     * @brief ConvolutionLayer的析构函数。
     */
    ~ConvolutionLayer() noexcept override = default;

protected:
    /**
     * @brief 管理ConvolutionLayer具体实现的指针。
     */
    std::shared_ptr<ConvolutionLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_CONVOLUTION_LAYER_H */