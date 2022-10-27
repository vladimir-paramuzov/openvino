// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <ostream>

#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/runtime/tensor.hpp"

namespace cldnn {

/// @brief Primitives implementation type.
enum class impl_types : uint8_t {
    cpu = 1 << 0,
    common = 1 << 1,
    ocl = 1 << 2,
    onednn = 1 << 3,
    any = 0xFF,
};

/// @brief Primitives implementation type.
enum class shape_types : uint8_t {
    static_shape = 1 << 0,
    dynamic_shape = 1 << 1,
    any = 0xFF,
};

inline impl_types operator&(impl_types a, impl_types b) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(static_cast<underlying_type>(a) & static_cast<underlying_type>(b));
}

inline impl_types operator|(impl_types a, impl_types b) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(static_cast<underlying_type>(a) | static_cast<underlying_type>(b));
}

inline impl_types operator~(impl_types a) {
    typedef std::underlying_type<impl_types>::type underlying_type;
    return static_cast<impl_types>(~static_cast<underlying_type>(a));
}

inline std::ostream& operator<<(std::ostream& out, const impl_types& impl_type) {
    switch (impl_type) {
        case impl_types::cpu: out << "cpu"; break;
        case impl_types::common: out << "common"; break;
        case impl_types::ocl: out << "ocl"; break;
        case impl_types::onednn: out << "onednn"; break;
        case impl_types::any: out << "any"; break;
        default: out << "unknown"; break;
    }

    return out;
}

inline std::ostream& operator<<(std::ostream& out, const shape_types& shape_type) {
    switch (shape_type) {
        case shape_types::static_shape: out << "static_shape"; break;
        case shape_types::dynamic_shape: out << "dynamic_shape"; break;
        default: out << "unknown"; break;
    }

    return out;
}

/// @brief Description of primitives implementation.
struct implementation_desc {
    format::type output_format;  ///< Output format.
    std::string kernel_name;     ///< GPU kernel name.
    impl_types impl_type;        ///< GPU implementation type.

    implementation_desc() :
        output_format(format::any),
        kernel_name(""),
        impl_type(impl_types::any) {}

    implementation_desc(format::type output_format,
                        std::string kernel_name,
                        impl_types impl_type = impl_types::any) :
        output_format(output_format),
        kernel_name(kernel_name),
        impl_type(impl_type) {}
};

using implementation_forcing_map = std::map<primitive_id, implementation_desc>;

}  // namespace cldnn
