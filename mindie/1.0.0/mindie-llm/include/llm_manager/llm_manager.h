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
#ifndef MINDIE_LLM_MANAGER_H
#define MINDIE_LLM_MANAGER_H

#include <memory>
#include <string>
#include <map>
#include <set>
#include "callback.h"
namespace mindie_llm {
/// A component called the llmmanager, to support continous batching of requesets
///
/// This class is a manager to to support continous batching of requesets, provides basic functions such as
/// initialize the LlmManager，get running params，shutdown the LlmManager and functions of PD Separation.
class LlmManager {
public:
    /// This Constructor initializes a LlmManager object with the following parameters
    ///
    /// \param llmConfigPath The path of the LLM configuration file
    /// \param getRequest The callback function for getting requests
    /// \param sendResponse The callback function for retrieving response tensor from the llmmanger
    /// \param controlCallback The callback function for acquiring requests with control operations
    /// \param statusCallback The callback function for retrieving status information from the llmmanger
    /// \param statusResponseCallback The callback function for obtaining the status of requests being queued and the
    /// \param ipInfo The map saved params need to be set in modelConfig and the
    /// execution status of requests with control operations
    LlmManager(const std::string &llmConfigPath, mindie_llm::GetRequestsCallback getRequest,
        mindie_llm::SendResponsesCallback sendResponse, mindie_llm::ControlSignalCallback controlCallback,
        mindie_llm::LlmManagerStatsCallback statusCallback,
        mindie_llm::SendStatusResponseCallback statusResponseCallback,
        std::map<std::string, std::string> ipInfo = std::map<std::string, std::string>());
    
    uint32_t GetMaxPositionEmbeddings() const;

    /// Get model's parameters,
    /// this function is used to get the parameters of the model, such as the maximum number of tokens. etc
    ///
    /// \return map<std::string, std::string> format, which stores configuration information
    std::map<std::string, std::string> GetModelParams() const;

    void Shutdown();

    /// Assign the DMI role to the specified request,
    /// this function is used to assign the DMI role to the specified request
    ///
    /// \param runtimeRequest The request to assign the DMI role to
    /// \param isForceRelease Whether to force release the DMI role
    ///
    /// \return true if the DMI role was successfully assigned, false otherwise
    bool UpdateEngineInfo(std::shared_ptr<mindie_llm::InferRequest> &runtimeRequest, bool isForceRelease);

    /// This function initializes the LLM manager with the specified model instance ID and NPU device IDs.
    ///
    /// \param modelInstanceId The model instance ID used to initialize the LLM manager
    /// \param npuDeviceIds The NPU device IDs used to initialize the LLM manager
    ///
    /// \return Status indicating whether the initialization is successful
    Status Init(uint32_t modelInstanceId, std::set<size_t> npuDeviceIds);
    ~LlmManager();

private:
    class LlmManagerImpl;
    std::shared_ptr<LlmManagerImpl> impl_;
};
}
#endif // MINDIE_LLM_MANAGER_H
