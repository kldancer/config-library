/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 */
#ifndef SDK_INFER_REQUEST_ID_H
#define SDK_INFER_REQUEST_ID_H

#pragma once

#include <functional>
#include <string>
#include <map>
#include <vector>

namespace SimpleLLMInference {
class RequestId {
public:
    enum class [[deprecated]] DataType {
        UINT64,
        STRING
    };

    explicit RequestId(const std::string &requestLabel);

    explicit RequestId(uint64_t requestIndex);

    RequestId &operator = (const uint64_t rhs);

    RequestId &operator = (const std::string &rhs);

    RequestId &operator = (const RequestId &rhs);

    RequestId(const SimpleLLMInference::RequestId &other);

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    DataType Type() const
    {
        return idType_;
    }

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    const std::string &StringValue() const
    {
        return requestLabel_;
    }

    [[deprecated("This function is deprecated on December 30th, 2024")]]
    uint64_t UnsignedIntValue() const
    {
        return requestIndex_;
    }

    struct Compare {
        bool operator () (const RequestId &lhs, const RequestId &rhs) const
        {
            if (lhs.Type() == RequestId::DataType::STRING) {
                return std::hash<std::string>()(lhs.StringValue()) < std::hash<std::string>()(rhs.StringValue());
            } else {
                return lhs.UnsignedIntValue() < rhs.UnsignedIntValue();
            }
        }
    };

private:
    friend bool operator == (const RequestId inputLhs, const RequestId inputRhs);

    friend bool operator != (const RequestId lhs, const RequestId rhs);

    std::string requestLabel_;

    uint64_t requestIndex_;

    DataType idType_;
};

}

#endif