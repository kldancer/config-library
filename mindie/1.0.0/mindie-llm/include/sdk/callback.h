/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SDK_CALLBACK_H
#define SDK_CALLBACK_H
#pragma once

#include <functional>
#include <memory>
#include "sdk/infer_response.h"
#include "sdk/infer_request_id.h"

namespace SimpleLLMInference {

using SendResponseCallback [[deprecated]] = std::function<void(std::shared_ptr<InferenceResponse> &)>;

using ReleaseCallback [[deprecated]] = std::function<void(const RequestId &)>;

}
#endif

