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

#ifndef ASCENDIE_BUILDER_H
#define ASCENDIE_BUILDER_H

#include <memory>
#include "common/Common.h"
#include "network/Network.h"
#include "builder/BuilderConfig.h"

#pragma GCC visibility push(default)
namespace AscendIE {
class BuilderImpl;
class Builder : public NoCopy {
public:
    static Builder* CreateInferBuilder(const char* socName) noexcept;
    
    ~Builder() noexcept override;

    Network* CreateNetwork() noexcept;

    ModelData BuildModel(Network* network, const BuilderConfig& config) noexcept;

protected:
    std::unique_ptr<BuilderImpl> builderImpl_;
};
} // end of namespace

#pragma GCC visibility pop
#endif /* ASCENDIE_BUILDER_H */