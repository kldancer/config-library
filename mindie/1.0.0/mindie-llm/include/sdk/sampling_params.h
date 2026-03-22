/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SDK_SAMPLING_PARAMS_H
#define SDK_SAMPLING_PARAMS_H
#pragma once

#include <cstdint>

namespace SimpleLLMInference {
struct [[deprecated]] SamplingParams {
    float temperature = 1.0;
    uint32_t topK = 0;
    float topP = 1.0;
    float typicalP = 1.0;
    bool doSample = false;
    uint32_t seed = 1;
    float repetitionPenalty = 1.0;
    bool watermark = false;
    float frequencyPenalty = 0.0f;
    float presencePenalty = 0.0f;
    };
}
#endif
