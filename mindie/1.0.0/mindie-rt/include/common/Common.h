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

#ifndef ASCENDIE_COMMON_H
#define ASCENDIE_COMMON_H

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace AscendIE {
struct ModelData {
    ModelData(std::shared_ptr<uint8_t> dataPtr, size_t dataSize) : data(dataPtr), size(dataSize) {}
    ModelData() {};
    std::shared_ptr<uint8_t> data = nullptr;
    size_t size = 0;
};

// Strongly recommend to invoke Finalize() where application ends
void Finalize() noexcept;

enum class BuilderFlag : size_t {
    FP16               = 0,
    FP32               = 1,
    MIX_FP32_WITH_FP16 = 2,
    MIX_FP16           = 3,
    QINT8              = 4
};

enum class RangeSelector : size_t {
    RANGE_MIN = 0,
    RANGE_MAX = 1
};

struct Status {
    Status();
    Status(size_t errorCode, const char* errorDesc);
    Status& operator=(const Status& other);
    ~Status();
    
    size_t code = 0;
    const char* desc = nullptr;
};

enum class DataType {
    UNKNOWN = -1,
    FLOAT = 0,
    FLOAT16 = 1,
    INT8 = 2,
    INT32 = 3,
    UINT8 = 4,
    INT16 = 6,
    UINT16 = 7,
    UINT32 = 8,
    INT64 = 9,
    UINT64 = 10,
    DOUBLE = 11,
    BOOL = 12,
    STRING = 13
};

enum class DataLayout {
    UNKNOWN = -1,
    NCHW = 0,
    NHWC = 1,
    ND = 2,
    NC1HWC0 = 3,
    FRACTAL_Z = 4,
    NC1HWC0_C04 = 12,
    HWCN = 16,
    NDHWC = 27,
    FRACTAL_NZ = 29,
    NCDHW = 30,
    NDC1HWC0 = 32,
    Z_3D = 33
};

enum class ActivationKind : int32_t {
    RELU = 0,
    SIGMOID,
    LEAKY_RELU,
    HARD_SIGMOID,
    TANH,
    SWISH,
    GELU,
    ELU,
    SELU,
    SOFTPLUS,
    MISH,
    RELU6
};

enum class PoolingKind : int32_t {
    MAX = 0,
    AVERAGE = 1,
    ADAPTIVE_AVERAGE // support 2d,as the last enum element. add new element before `ADAPTIVE_AVERAGE`
};

enum class PaddingMode : int32_t {
    EXPLICIT_ROUND_DOWN = 0,   // ceil_mode is false
    EXPLICIT_ROUND_UP = 1,    // ceil_mode is true
    SAME_UPPER = 2,          // auto_pad is SAME_UPPER
    // as the last enum element. add new element before `SAME_LOWER`
    SAME_LOWER              // auto_pad is SAME_LOWER
};

enum class ElementWiseOperation : int32_t {
    MAX = 0,
    MIN,
    ADD,
    SUB,
    MUL,
    DIV,
    LESS,
    POW,
    FLOOR_DIV,
    MOD,
    EQUAL,
    GREATER,
    GREATER_EQUAL,
    LESS_EQUAL,
    NOT_EQUAL,
    OR,
    AND,
    BITWISE_AND,
    BITWISE_OR,
    BITWISE_XOR,
    RIGHTSHIFT
};

enum class MatrixOperation : int32_t {
    NONE = 0,
    TRANSPOSE = 1,
};

enum class ReduceOperation : int32_t {
    AVG = 0,
    MAX = 1,
    MIN = 2,
    SUM = 3,
    STD = 4,
    PROD // as the last enum element. add new element before `PROD`
};

struct PermutationConf {
    static constexpr uint64_t MAX_DIM = 8;
    // number of dimentions
    int64_t rank;
    int64_t order[MAX_DIM];
};

struct WeightsBuf {
    DataType type;
    // pointer to memory location
    const void *value;
    // element count of weights
    uint64_t count;
};
}

#endif // ASCENDIE_COMMON_H

