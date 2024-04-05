// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "softmax_inst.h"
#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(softmax)

layout softmax_inst::calc_output_layout(softmax_node const& node, kernel_impl_params const& impl_param) {
    assert(static_cast<bool>(impl_param.desc->output_data_types[0]) == false &&
           "Output data type forcing is not supported for softmax_node!");

    auto output_layout = impl_param.get_input_layout();

    if (impl_param.has_fused_primitives())
        output_layout.data_type = impl_param.get_fused_output_layout().data_type;

    return output_layout;
}

std::string softmax_inst::to_string(softmax_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();

    std::stringstream primitive_description;

    json_composite softmax_info;
    softmax_info.add("dimension", desc->dimension);

    node_info->add("softmax_info", softmax_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

softmax_inst::typed_primitive_inst(network& network, softmax_node const& node) : parent(network, node) { }
}  // namespace cldnn
