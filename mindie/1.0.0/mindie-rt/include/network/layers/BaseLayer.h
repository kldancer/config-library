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
#ifndef ASCENDIE_BASE_LAYER_H
#define ASCENDIE_BASE_LAYER_H

#include "common/Common.h"
#include "common/NoCopy.h"
#include "network/Tensor.h"

#pragma GCC visibility push(default)
namespace AscendIE {
/**
 * Layer structure in neural network
 */
class BaseLayerImpl;
class BaseLayer : public NoCopy {
public:

    void SetName(const char* name) noexcept;

    const char* GetName() const noexcept;

    void SetInput(uint32_t index, Tensor *tensor) noexcept;

    Tensor *GetInput(int32_t index) const noexcept;

    Tensor *GetOutput(int32_t index) noexcept;

    int32_t GetInputCount() const noexcept;

    int32_t GetOutputCount() noexcept;

    ~BaseLayer() override = default;

protected:
    std::shared_ptr<BaseLayerImpl> impl_;
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_BASE_LAYER_H */