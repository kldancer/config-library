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

#ifndef ASCENDIE_POOLINGLAYER_H
#define ASCENDIE_POOLINGLAYER_H

#include "network/layers/BaseLayer.h"
#include "common/Dims.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief PoolingLayer definition
 */
class PoolingLayerImpl;
class PoolingLayer : public BaseLayer {
public:
    /**
     * @brief 获取池化类型。
     * @return 返回该PoolingLayer的池化类型。
     */
    virtual PoolingKind GetPoolingKind() const noexcept;

    /**
     * @brief 设置池化窗口大小。
     * @note 不支持自适应场景。
     * @param [in] windowSize: 输入的windowSize，支持2D和3D。ADAPTIVE_AVERAGE类型输入{}，
     * MAX和AVERAGE类型设置维度为2/3维，windowSize[i]取值范围[1~255]，
     * windowSize[0] * windowSize[1]取值范围[1, 8192]。
     */
    virtual void SetWindowSize(Dims windowSize) noexcept;

    /**
     * @brief 获取池化窗口大小。
     * @note 不支持自适应场景。
     * @return 返回该PoolingLayer的池化窗口大小。
     */
    virtual Dims GetWindowSize() const noexcept;

    /**
     * @brief 设置池化的步长。
     * @note 不支持自适应场景。
     * @param [in] stride: 具体的步长。stride维度取值范围[2, 5]。
     * 每一维取值范围要求[1, 255]
     */
    virtual void SetStride(Dims stride) noexcept;

    /**
     * @brief 获取池化的步长。
     * @note 不支持自适应场景。
     * @return 返回该PoolingLayer的池化的步长。
     */
    virtual Dims GetStride() const noexcept;

    /**
     * @brief 设置池化的填充shape。
     * @note 不支持自适应场景。
     * @param [in] padding: 具体的填充shape。2维Pooling要求padding为4维，3维Poolinng要求padding为6维。
     * 每一维取值范围要求[0,255]。
     */
    virtual void SetPadding(Dims padding) noexcept;

    /**
     * @brief 获取池化的填充shape。
     * @return 返回池化的填充shape。
     */
    virtual Dims GetPadding() const noexcept;

    /**
     * @brief 设置池化的填充类型。
     * @note 不支持自适应场景。
     * @param [in] paddingMode: 具体的填充类型。包括：EXPLICIT_ROUND_DOWN,
     * EXPLICIT_ROUND_UP, SAME_UPPER, SAME_LOWER。
     */
    virtual void SetPaddingMode(PaddingMode paddingMode) noexcept;

    /**
     * @brief 获取池化的填充类型。
     * @note 不支持自适应场景。
     * @return 池化的填充类型。
     */
    virtual PaddingMode GetPaddingMode() const noexcept;

    /**
     * @brief 设置计算均值时是否排除填充。
     * @param [in] exclusive: 是否排除填充。默认为true，排除填充。
     */
    virtual void SetAverageCountExcludesPadding(bool exclusive) noexcept;

    /**
     * @brief 获取计算均值时是否排除填充参数exclusive_。
     * @note 不支持自适应场景。
     * @return 计算均值时是否排除填充参数exclusive_的值。
     */
    virtual bool GetAverageCountExcludesPadding() const noexcept;

    /**
     * @brief PoolingLayer的析构函数。
     */
    ~PoolingLayer() noexcept override = default;

protected:
    /**
     * @brief 管理PoolingLayer具体实现的指针。
     */
    std::shared_ptr<PoolingLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_POOLINGLAYER_H */