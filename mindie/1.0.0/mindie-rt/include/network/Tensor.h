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
#ifndef ASCENDIE_TENSOR_H
#define ASCENDIE_TENSOR_H

#include <memory>
#include "common/Common.h"
#include "common/Dims.h"
#include "common/NoCopy.h"

#pragma GCC visibility push(default)
namespace AscendIE {
class TensorImpl;
class Tensor : public NoCopy {
public:
    /**
     * @brief 设置tensor的name信息。
     * @note name不能为空，不能出现无效字符：'\n', '\f', '\r', '\b', '\u007f'，
     * 不能超出长度上限4096。
     * @param [in] name: tensor的name信息。
    */
    void SetName(const char* name) noexcept;

    /**
     * @brief 获取tensor的name信息。
     * @note 若还未调用SetName接口，返回默认值""。
     * @return 返回tensor的name信息。
    */
    const char* GetName() const noexcept;

    /**
     * @brief 获取tensor的维度信息。
     * @note 若还未传入维度信息，则返回默认值{}。
     * @return 返回tensor的维度信息。
    */
    Dims GetDimensions() const noexcept;

    /**
     * @brief 获取tensor的DataLayout信息。
     * @note 若还未传入DataLayout信息，则返回默认值NCHW。
     * @return 返回tensor的DataLayout信息。
    */
    DataLayout GetLayout() const noexcept;

    /**
     * @brief 获取tensor的dType信息。
     * @note 若还未传入dType信息，则返回默认值float。
     * @return 返回tensor的DataType信息。
    */
    DataType GetType() const noexcept;

    /**
     * @brief 判断tensor是否是输入。
     * @note 默认返回false。
     * @return 返回tensor是否是输入。
    */
    bool IsInput() noexcept;

    /**
     * @brief 判断tensor是否是输出。
     * @note 默认返回false。
     * @return 返回tensor是否是输出。
    */
    bool IsOutput() noexcept;

protected:
    std::shared_ptr<TensorImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_TENSOR_H */