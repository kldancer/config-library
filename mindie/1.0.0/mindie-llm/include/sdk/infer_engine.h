/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2024. All rights reserved.
 */
#ifndef SDK_INFER_ENGINE_H
#define SDK_INFER_ENGINE_H
#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>

#include "sdk/callback.h"
#include "sdk/infer_request.h"
#include "sdk/infer_response.h"
#include "sdk/status.h"

namespace SimpleLLMInference {
enum class Operation {
    STOP = 1,
    RELEASE_KV = 2,
};

class InferenceEngine {
public:
    InferenceEngine();

    ~InferenceEngine();

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status Init(const SendResponseCallback &callback = nullptr,
        const std::string &configPath = "");

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status Forward(std::shared_ptr<InferenceRequest> &request, bool validRequest = false);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status GetProcessingRequest(uint64_t *num);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status GetRequestBlockQuotas(uint64_t *remainBlocks, uint64_t *remainPrefillSlots,
        uint64_t *remainPrefillTokens);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status ControlRequest(const RequestId &requestId, Operation operation);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status Finalize();

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status GetMaxBatchSize(uint64_t *batchSize);

private:
    class InferenceEngineInner;

    std::shared_ptr<InferenceEngineInner> engineInner_{ nullptr };
};
}

#endif