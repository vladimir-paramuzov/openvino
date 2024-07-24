// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_map.hpp"
#include "program_node.h"

namespace cldnn {


shape_types ImplementationManagerBase::get_shape_type(const kernel_impl_params& impl_params) {
    for (auto& in_shape : impl_params.input_layouts) {
        if (in_shape.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    if (impl_params.get_output_layout().is_dynamic())
        return shape_types::dynamic_shape;

    return shape_types::static_shape;
}

shape_types ImplementationManagerBase::get_shape_type(const program_node& node) {
    for (auto& in_layout : node.get_input_layouts()) {
        if (in_layout.is_dynamic()) {
            return shape_types::dynamic_shape;
        }
    }
    if (node.get_output_layout().is_dynamic())
        return shape_types::dynamic_shape;

    return shape_types::static_shape;
}

bool ImplementationManagerBase::is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types supported_shape_type) {
    auto target_shape_type = get_shape_type(node);

    if ((target_shape_type & supported_shape_type) != target_shape_type)
        return false;

    auto key = implementation_key()(node.get_input_layout(0));
    if (!supported_keys.empty() && supported_keys.find(key) == supported_keys.end())
        return false;

    return true;
}

} // namespace cldnn
