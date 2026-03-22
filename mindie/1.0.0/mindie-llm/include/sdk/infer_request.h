/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SDK_INFER_REQUEST_H
#define SDK_INFER_REQUEST_H
#pragma once

#include <deque>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "sdk/sampling_params.h"
#include "sdk/callback.h"
#include "sdk/common.h"
#include "sdk/status.h"
#include "infer_request_id.h"
#include "sdk/error.h"

namespace SimpleLLMInference {
class InferenceRequest {
public:
    class [[deprecated]] Input {
    public:
        Input(const std::string &name, LLM_ENGINE_DataType datatype, const int64_t *shape, uint64_t dimCount);

        Input() = delete;

        ~Input() = default;

        infrastructure::Status Allocate(size_t size);

        const std::string &Name() const;

        LLM_ENGINE_DataType DType() const;

        const std::vector<int64_t> &Shape() const;

        const uint64_t &DimCount() const;

        uint64_t ByteSize() const;

        void *Buffer();

        infrastructure::Status SetRelease(bool status);

        class InputInner;
        std::shared_ptr<InputInner> GetInputInner() const;

    private:
        std::shared_ptr<InputInner> inputInner_;
    };

    explicit InferenceRequest(const RequestId &reqId);

    InferenceRequest() = delete;

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    bool SetMaxOutputLen(uint32_t maxOutputLen);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    uint32_t MaxOutputLen() const;

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    const RequestId &GetRequestId();

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status AddOriginalInput(const std::string &name, LLM_ENGINE_DataType datatype, const int64_t *shape,
        uint64_t dimCount, Input **input);
    // add input named "INPUT_IDS"
    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status AddOriginalInput(LLM_ENGINE_DataType datatype, const int64_t *shape, uint64_t dimCount,
        Input **input);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    infrastructure::Status RemoveOriginalInput(const std::string &name);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    void SetSamplingParams(SamplingParams samplingParams);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    const SamplingParams &GetSamplingParams() const;

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    bool HasSampling();

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    void SetSendResponseCallback(const SendResponseCallback &callback);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    void SetInputText(std::string &text);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    std::string &GetInputText();

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    bool IsInputText();

    class InferenceRequestInner;
    [[deprecated("This function is deprecated on December 30th, 2024")]]
    std::shared_ptr<InferenceRequestInner> GetRequestInner() const;

private:
    std::shared_ptr<InferenceRequestInner> requestInner_;
};

}

#endif
