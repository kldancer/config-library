/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
 */
#ifndef MIES_INFRA_STATUS_H
#define MIES_INFRA_STATUS_H

#include <string>

#include "sdk/error.h"

namespace infrastructure {

// Slave node Health Status
enum class [[deprecated]] NodeHealthStatus {
    READY,
    ABNORMAL
};

class [[deprecated]] Status {
public:
    // Construct a status from a code with no message.
    explicit Status(Error::Code code = Error::Code::OK) noexcept
    {
        error_ = Error(code);
    }

    // Construct a status from a code and message.
    explicit Status(Error::Code code, const std::string &msg)
    {
        error_ = Error(code, msg);
    }

    // Construct a status from a code and message.
    explicit Status(const Error &error)
    {
        error_ = error;
    }

    bool IsOk() const
    {
        return error_.IsOk();
    }

    Error::Code StatusCode() const
    {
        return error_.ErrorCode();
    }
    static const Status success;

    std::string StatusMsg() const
    {
        return error_.Message();
    }

    bool operator==(const Status& other) const
    {
        return error_.ErrorCode() == other.error_.ErrorCode();
    }

private:
    Error error_;
};

} // namespace infrastructure

// using infrastructure::Status;

#endif