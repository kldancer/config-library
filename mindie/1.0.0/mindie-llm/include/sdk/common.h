/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef SDK_COMMON_H
#define SDK_COMMON_H
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

[[deprecated]] typedef enum LLM_ENGINE_parametertype_enum {
    LLM_ENGINE_PARAMETER_STRING,
    LLM_ENGINE_PARAMETER_INT,
    LLM_ENGINE_PARAMETER_BOOL,
    LLM_ENGINE_PARAMETER_BYTES
} LLM_ENGINE_ParameterType;

[[deprecated]] typedef enum LLM_ENGINE_DataType_enum {
    TYPE_INVALID = 0,

    TYPE_BOOL = 1,
    TYPE_UINT8 = 2,
    TYPE_UINT16 = 3,
    TYPE_UINT32 = 4,
    TYPE_UINT64 = 5,

    TYPE_INT8 = 6,
    TYPE_INT16 = 7,
    TYPE_INT32 = 8,
    TYPE_INT64 = 9,

    TYPE_FP16 = 10,
    TYPE_FP32 = 11,
    TYPE_FP64 = 12,

    TYPE_STRING = 13,

    TYPE_BF16 = 14,
    TYPE_BUTT,
} LLM_ENGINE_DataType;

[[deprecated]] typedef enum LLMENGINE_memorytype_enum {
    LLMENGINE_MEMORY_CPU,
    LLMENGINE_MEMORY_CPU_PINNED,
    LLMENGINE_MEMORY_GPU
} LLM_ENGINE_MemoryType;

#ifdef __cplusplus
}
#endif

#endif