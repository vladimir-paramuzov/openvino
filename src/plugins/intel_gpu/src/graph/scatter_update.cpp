// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_update_inst.h"

#include "primitive_type_base.h"
#include "intel_gpu/runtime/error_handler.hpp"
#include "json_object.h"
#include <string>

namespace cldnn {
primitive_type_id scatter_update::type_id() {
    static primitive_type_base<scatter_update> instance;
    return &instance;
}

layout scatter_update_inst::calc_output_layout(scatter_update_node const& node, kernel_impl_params const& impl_param) {
    auto desc = impl_param.typed_desc<scatter_update>();

    auto input_layout = impl_param.input_layouts[0];

    auto output_shape = input_layout.get_tensor();
    auto input_format = input_layout.format;
    auto output_type = input_layout.data_type;

    if (impl_param.has_fused_primitives()) {
        output_type = impl_param.get_fused_output_layout().data_type;
    }

    return layout{output_type, input_format, output_shape};
}

std::string scatter_update_inst::to_string(scatter_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_update_info;
    scatter_update_info.add("input id", input.id());
    scatter_update_info.add("axis", desc->axis);

    node_info->add("scatter_update info", scatter_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_update_inst::typed_primitive_inst(network& network, scatter_update_node const& node) : parent(network, node) {}

}  // namespace cldnn
