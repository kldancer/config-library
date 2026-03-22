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

#ifndef ASCENDIE_MODEL_PARSER_H
#define ASCENDIE_MODEL_PARSER_H

#include "network/Network.h"

#pragma GCC visibility push(default)
namespace AscendIE {
class ModelParser {
public:
    ModelParser() {}

    virtual bool Parse(AscendIE::Network *network, const char* file) = 0;

    virtual ~ModelParser() {}
};
}
#pragma GCC visibility pop
#endif /* ASCENDIE_MODEL_PARSER_H */