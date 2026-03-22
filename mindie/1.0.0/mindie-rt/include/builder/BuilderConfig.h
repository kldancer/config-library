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

#ifndef ASCENDIE_BUILDER_CONFIG_H
#define ASCENDIE_BUILDER_CONFIG_H

#include "common/Common.h"

namespace AscendIE {
class BuilderConfigImpl;
class BuilderConfig {
public:
    BuilderConfig();

    ~BuilderConfig() noexcept;

    void SetFlag(BuilderFlag builderFlag) noexcept;

    bool GetFlag(BuilderFlag builderFlag) const noexcept;

    BuilderConfig(BuilderConfig const& other);

    BuilderConfig& operator=(BuilderConfig const& other);
private:
    std::unique_ptr<BuilderConfigImpl> builderConfigImpl_;
};
}

#endif // ASCENDIE_BUILDER_CONFIG_H
