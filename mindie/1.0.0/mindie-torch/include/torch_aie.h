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

#ifndef TORCH_AIE_CPP_TORCHAIE_INCLUDE
#define TORCH_AIE_CPP_TORCHAIE_INCLUDE

#include <set>
#include <string>
#include <vector>
#include <iostream>

#include <torch/custom_class.h>

namespace torch_aie {
enum class PrecisionPolicy {
    PREF_FP32 = 0,
    FP16 = 1,
    FP32 = 2,
    PREF_FP16 = 3,
};

enum class DataType {
    UNKNOWN = -1,
    FLOAT = 0,
    FLOAT16 = 1,
    INT8 = 2,
    INT32 = 3,
    UINT8 = 4,
    INT16 = 5,
    UINT16 = 6,
    UINT32 = 7,
    INT64 = 8,
    UINT64 = 9,
    DOUBLE = 10,
    BOOL = 11,
    STRING = 12
};

enum class TensorFormat {
    UNKNOWN = -1,
    NCHW = 0,
    NHWC = 1,
    ND = 2,
    NC1HWC0 = 3
};

struct Input : torch::CustomClassHolder {
public:
    explicit Input(bool isShapeRange = false) : isShapeRange(isShapeRange) {};

    explicit Input(std::vector<int64_t> shape, TensorFormat format = TensorFormat(TensorFormat::ND));

    Input(std::vector<int64_t> shape, DataType dtype, TensorFormat format = TensorFormat(TensorFormat::ND));

    Input(std::vector<int64_t> minShape, std::vector<int64_t> maxShape,
        TensorFormat format = TensorFormat(TensorFormat::ND));
    
    Input(std::vector<int64_t> minShape, std::vector<int64_t> maxShape, DataType dtype,
        TensorFormat format = TensorFormat(TensorFormat::ND));
    
    inline bool IsShapeRange() const
    {
        return isShapeRange;
    }

    std::vector<int64_t> minShape;
    std::vector<int64_t> maxShape;
    std::vector<int64_t> shape;
    DataType dtype = DataType(DataType::UNKNOWN);
    TensorFormat format = TensorFormat(TensorFormat::UNKNOWN);

private:
    bool isShapeRange = false;
};

using InputProfile  = std::vector<Input>;
struct GraphInputs {
    std::vector<torch::jit::IValue> inputSignatures;
    std::vector<InputProfile> inputs;
};

void set_device(const int npu_id);

// Strongly recommend to invoke finalize() where application ends
void finalize() noexcept;

namespace torchscript {
struct CompileSpec {
    explicit CompileSpec(std::vector<std::vector<int64_t>> staticShapes);
    explicit CompileSpec(std::vector<Input> inputs);
    explicit CompileSpec(std::vector<InputProfile> inputs);

    GraphInputs graphInputs;
    PrecisionPolicy precision_policy = PrecisionPolicy::FP16;
    std::string soc_version = "Ascend310P3";
    size_t minBlockSize = 1;
    size_t optimizationLevel = 0;
    std::vector<size_t> defaultBufferSizeVec = { 500, };
    bool requireFullCompilation = false;
    bool truncateLongAndDouble = true;
    bool allowTensorReplaceInt = false;
    std::vector<std::string> torchExecutedOps;
    std::vector<std::string> torchExecutedModules;
};

bool check_method_operator_support(const torch::jit::Module& module, const std::string& methodName);

torch::jit::Module compile(const torch::jit::Module& module, const CompileSpec& info);

std::string export_engine(const torch::jit::Module& module, const std::string& methodName, const CompileSpec& info);
} // namespace torchscript
} // namespace torch_aie
#endif // TORCH_AIE_CPP_TORCHAIE_INCLUDE
