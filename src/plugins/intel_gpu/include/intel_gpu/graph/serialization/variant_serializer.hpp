// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>
#include <variant>

#include "buffer.hpp"

namespace cldnn {

template <typename BufferType, typename... Variants>
class Serializer<BufferType, std::variant<Variants...>, std::enable_if_t<std::is_base_of_v<OutputBuffer<BufferType>, BufferType>>> {
public:
    static void save(BufferType& buffer, const std::variant<Variants...>& scalar) {
        buffer << scalar.index();
        std::visit(
            [&](auto&& val) {
                buffer << val;
            },
            scalar);
    }
};

template <typename BufferType, typename... Variants>
class Serializer<BufferType, std::variant<Variants...>, std::enable_if_t<std::is_base_of_v<InputBuffer<BufferType>, BufferType>>> {
public:
    using VariantType = std::variant<Variants...>;
    static void load(BufferType& buffer, VariantType& scalar) {
        size_t idx = 0;
        buffer >> idx;
        scalar = read_variant(buffer, idx);
    }

private:
    template<size_t I = 0>
    static VariantType read_variant(BufferType& buffer, size_t idx) {
        if constexpr (I < std::variant_size_v<VariantType>) {
            if (I == idx) {
                std::variant_alternative_t<I, VariantType> val;
                buffer >> val;
                return val;
            }
            return read_variant<I + 1>(buffer, idx);
        }

        OPENVINO_THROW("[GPU] Invalid variant index: ", idx);
    }
};

}  // namespace cldnn
