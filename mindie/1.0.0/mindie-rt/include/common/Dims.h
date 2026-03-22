/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#ifndef ASCENDIE_DIMS_H
#define ASCENDIE_DIMS_H

#include <vector>
#include <array>
#include <initializer_list>
#include <cstdint>
#include <cstddef>

#pragma GCC visibility push(default)
namespace AscendIE {
    class Dims {
    public:
        static constexpr int64_t MAX_DIMS = 8;

        Dims();

        ~Dims();

        Dims(int32_t ndims, const int64_t* shape);

        Dims(std::initializer_list<int64_t> shape);

        explicit Dims(const std::vector<int64_t>& shape);

        Dims& operator=(const Dims& dims);

        bool operator==(const Dims& dims) const;

        bool operator!=(const Dims& dims) const;

        Dims(const Dims& dims);

        int64_t& operator[](size_t index);

        const int64_t& operator[](size_t index) const;

        bool Empty() const;

        size_t Size() const;

        const int64_t* Data() const;
    private:
        int64_t ndims_ = 0;

        std::array<int64_t, MAX_DIMS> shape_ {};
    };
}
#pragma GCC visibility pop
#endif // ASCENDIE_DIMS_H
