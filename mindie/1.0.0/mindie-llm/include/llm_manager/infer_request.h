/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
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

#ifndef MINDIE_LLM_INFERENCE_REQUEST_H
#define MINDIE_LLM_INFERENCE_REQUEST_H

#include <memory>
#include <string>
#include "infer_tensor.h"
#include "infer_request_id.h"
#include "utils/status.h"
namespace mindie_llm {
class InferRequestImpl;
/// The InferRequest class is used to manage the input tensors of the inference process.
///
/// The InferRequest class provides methods to add and set input tensors, get request id,
/// and get the status of the request, and it also provides methods required by the prefill and decode separation.
class InferRequest {
public:
    explicit InferRequest(InferRequestId requestId);

    /// Add a tensor to the inference request.
    /// This method adds a tensor to the inference request with the specified name. The tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to add.
    /// \param tensor The tensor to add.
    /// \return The status of the operation AddTensor.
    Status AddTensor(const std::string& tensorName, TensorPtr &tensor);

    /// Set a tensor to the inference request.
    ///
    /// This method sets a tensor to the inference request with the specified name. The tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to be set.
    /// \param tensor The tensor to be set.
    void SetTensor(const std::string& tensorName, TensorPtr &tensor);

    /// Get a tensor from the inference request with the specified name.
    ///
    /// This method gets a tensor from the inference request with the specified name,
    /// the tensor is stored in the request
    ///
    /// \param tensorName The name of the tensor to be acquired.
    /// \param tensor The tensor to be acquired.
    /// \return The status of the operation GetTensorByName.
    Status GetTensorByName(const std::string& tensorName, TensorPtr &tensor);

    /// Delete a tensor from the inference request with the specified name.
    ///
    /// This method deletes a tensor from the inference request with the specified name,
    /// the tensor is stored in the request.
    ///
    /// \param name The name of the tensor to be deleted.
    /// \return The status of the operation DelTensorByName.
    Status DelTensorByName(const std::string &name);

    /// Get the request id of the inference request.
    ///
    /// This method gets the request id of the inference request.
    ///
    /// \return The request id of the inference request.
    InferRequestId GetRequestId() const;

    /// Set the MaxOutputLen of the inference request.
    Status SetMaxOutputLen(uint32_t maxOutputLen);

    /// Get the MaxOutputLen of the inference request.
    uint32_t GetMaxOutputLen() const;

    std::shared_ptr<InferRequestImpl> GetRequestInner() const;

    /// Get the immutable inputs of the inference request.
    ///
    /// This method retrieves all tensors from the request and returns them as an tensor map.
    /// \return The collection of tensors in TensorMap format.
    const TensorMap &ImmutableInputs() const;

    /// Set the request type of the inference request.
    void SetReqType(mindie_llm::InferReqType reqType);

    /// Get the request type of the inference request.
    mindie_llm::InferReqType GetReqType() const;

    bool IsPrefillReq() const;

    bool IsDecodeReq() const;

    void SetDTarget(std::string &dTarget);

    std::string GetDTarget() const;

    void SetPrefillAddr(std::string &prefillAddr);

    std::string GetPrefillAddr() const;

    void SetSrcBlockTable(const std::vector<int64_t> &srcBlockTable);

    std::vector<int64_t> GetSrcBlockTable() const;

    ~InferRequest();
private:
    std::shared_ptr<InferRequestImpl> impl_;
};
}

#endif // MINDIE_LLM_INFERENCE_REQUEST_H
