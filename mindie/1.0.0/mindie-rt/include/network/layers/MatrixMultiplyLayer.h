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
#ifndef ASCENDIE_MATRIX_MULTIPLY_LAYER_H
#define ASCENDIE_MATRIX_MULTIPLY_LAYER_H

#include "network/layers/BaseLayer.h"
#include "common/Common.h"
#include "common/Dims.h"

#include <memory>

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * @brief MatrixMultiplyLayer definition
 */
class MatrixMultiplyLayerImpl;
class MatrixMultiplyLayer : public BaseLayer {
public:
    /**
     * @brief 获取对应索引输入的矩阵乘运算类型。
     * @note NONE表示不转置，TRANSPOSE表示转置。
     * @return 返回对应索引输入的矩阵乘运算类型。
     */
    MatrixOperation GetMatrixOperation(int32_t index) const noexcept;
    
protected:
    /**
     * @brief 管理MatrixMultiplyLayer具体实现的指针。
     */
    std::shared_ptr<MatrixMultiplyLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif // ASCENDIE_MATRIX_MULTIPLY_LAYER_H