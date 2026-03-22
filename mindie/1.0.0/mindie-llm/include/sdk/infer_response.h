/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */

#ifndef SDK_INFER_RESPONSE_H
#define SDK_INFER_RESPONSE_H
#pragma once

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <shared_mutex>
#include <mutex>
#include "sdk/common.h"
#include "sdk/status.h"
#include "sdk/infer_request_id.h"

namespace SimpleLLMInference {
// InferenceResponse GetFlags接口获取到的标志定义
enum class [[deprecated]] InferResponseEndFlag {
    // 请求继续迭代执行
    INFER_RESPONSE_CONTINUE = 0,
    // 请求正常结束
    INFER_RESPONSE_EOS = 1,
    // 请求被主动CANCEL或STOP，用户不感知，丢弃响应
    INFER_RESPONSE_CANCEL = 2,
    // 请求执行中出错，响应输出为空，err_msg非空
    INFER_RESPONSE_EXEC_ERROR = 3,
    // 请求输入校验异常，响应输出为空，err_msg非空
    INFER_RESPONSE_ILLEGAL_INPUT = 4,
    // 请求因达到最大序列长度而结束，响应为最后一轮迭代输出
    INFER_RESPONSE_REACH_MAX_SEQ_LEN = 5,
    // 请求因达到最大输出长度（包括请求和模型粒度）而结束，响应为最后一轮迭代输出
    INFER_RESPONSE_REACH_MAX_OUTPUT_LEN = 6,
};

class InferenceResponse {
public:
    /* *
     * @brief output tensor
     */
    class [[deprecated("This class is deprecated on December 30th, 2024")]] Output {
    public:
        virtual ~Output() = default;
        virtual const char *Name() const noexcept = 0;
        virtual LLM_ENGINE_DataType DType() const noexcept = 0;
        virtual const std::vector<int64_t> &Shape() const noexcept = 0;
        virtual uint64_t DimCount() const noexcept = 0;
        virtual void *Buffer() noexcept = 0;
        virtual const void *Buffer() const noexcept = 0;
        virtual uint64_t ByteSize() const noexcept = 0;
    };

    virtual ~InferenceResponse() = default;
    [[deprecated("This function is deprecated on December 30th, 2024")]]
    virtual const RequestId &GetRequestId() const noexcept = 0;

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    virtual bool IsEOS() const noexcept = 0;

    /* *
     * @brief Obtains the finish type.
     * @return value of the finish type.
     * @details flags has the following values to indicate the finish type:
     * 1: The request is finished normally.
     * 2: The request is actively canceled or stopped. The user is unaware of the operation and discards the response.
     * 3: An error occurs during request execution. The response output is empty, and err_msg is not empty.
     * 4: The request input verification is abnormal. The response output is empty, and err_msg is not empty.
     * 5: The request finished because the maximum sequence length is reached. The response is the output of the last
     * iteration.
     * 6: The request ends because the maximum output length (including the request and model granularity) is reached.
     * The response is the output of the last iteration.
     */
     [[deprecated("This function is deprecated on December 30th, 2024")]]
    virtual uint32_t GetFlags() const noexcept = 0;
    
    [[deprecated("This function is deprecated on December 30th, 2024")]]
    virtual infrastructure::Status ImmutableOutput(const std::string &name, Output **output) const noexcept = 0;
};

}

#endif
