// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "adaptive_pooling_inst.h"
#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id adaptive_pooling::type_id() {
    static primitive_type_base<adaptive_pooling> instance;
    return &instance;
}

layout adaptive_pooling_inst::calc_output_layout(const adaptive_pooling_node& node, kernel_impl_params const& impl_param) {
    const auto data_layout = impl_param.input_layouts[0];
    const auto prim = impl_param.typed_desc<adaptive_pooling>();
    return {data_layout.data_type, data_layout.format, prim->output_size};
}

std::string adaptive_pooling_inst::to_string(const adaptive_pooling_node& node) {
    const auto prim = node.get_primitive();

    std::stringstream primitive_description;

    json_composite info;
    const auto mode = prim->mode == adaptive_pooling_mode::max ? "max" : "average";
    info.add("mode", mode);
    info.add("output_size", prim->output_size);

    auto node_info = node.desc_to_json();
    node_info->add("adaptive_pooling_info", info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}
}  // namespace cldnn
