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
#ifndef MINDIE_LLM_CALLBACK_H
#define MINDIE_LLM_CALLBACK_H
#include <string>
#include <vector>
#include <unordered_set>
#include <memory>
#include <functional>

#include "infer_request.h"
#include "infer_tensor.h"
namespace mindie_llm {
/// The eum class of callback type, which includes control signal status and request enqueue status.
/// The CONTROL_SIGNAL_STATUS is used to identify that the Staus Response is for the request
/// with control operation enque status. The REQUEST_ENQUEUE_STATUS is used to identify that
/// the Staus Response is for the request enqueue status.
enum class StatusResponseType {
    CONTROL_SIGNAL_STATUS = 0,
    REQUEST_ENQUEUE_STATUS = 1,
};

/// The eum class of operation type, which includes STOP and RELEASE_KV.
/// The STOP is used to identify that the operation is to stop the request inference.
/// The RELEASE_KV is used to identify that the operation is to release the kv cache.
enum class Operation {
    STOP = 1,
    RELEASE_KV = 2,
};

/// Use the std::function to define a callback function for retriving requests.
/// The callback function should return a vector of shared_ptr of InferRequest.
///
/// \return std::vector<std::shared_ptr<InferRequest>>
using GetRequestsCallback = std::function<std::vector<std::shared_ptr<InferRequest>>()>;

/// Use the std::function to define a callback function for sending responses.
/// The callback function has 4 parameters:
///
/// \param InferRequestId request_id: The id of the request.
/// \param TensorMap TensorMap& results: The results of the request, which contains the response tensormap
/// \param bool Whether the request's inference is final.
/// \param string error_msg: The error message if the request's inference is not correctly inferenced.
using SendResponsesCallback = std::function<void(InferRequestId, const TensorMap&, bool, const std::string&)>;

/// Use the std::function to define a callback function for sending the status of requests being queued.
/// The callback function has 2 parameters:
///
/// \param InferRequestId request_id: The id of the request, whose type is InferRequestId class.
/// \param Status status: The status of requests being queued.
/// \param StatusResponseType status_type: The type of the status.
using SendStatusResponseCallback = std::function<void(InferRequestId, Status, StatusResponseType)>;

/// Use the std::function to define a callback function for retrieving the rqeuest with the given operation,
/// the operation is defined in the 'Operation' .
///
/// \returns std::vector<std::pair<InferRequestId, Operation>>: The vector of the request id and operation pair.
using ControlSignalCallback = std::function<std::vector<std::pair<InferRequestId, Operation>>()>;

/// Use the std::function to define a callback function for Getting the LLM Manager statistics.
/// The callback function has 1 parameter:
///
/// \param std::string stats: The statistics of the LLM Manager.
using LlmManagerStatsCallback = std::function<void(const std::string&)>;
}
#endif // MINDIE_LLM_CALLBACK_H
