// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "scatter_nd_update_inst.h"

#include "primitive_type_base.h"
#include "json_object.h"
#include <string>

#include "scatter_nd_base_shape_inference.hpp"

namespace cldnn {
GPU_DEFINE_PRIMITIVE_TYPE_ID(scatter_nd_update)

template<typename ShapeType>
std::vector<layout> scatter_nd_update_inst::calc_output_layouts(scatter_nd_update_node const& /*node*/, const kernel_impl_params& impl_param) {
    const auto& input0_layout = impl_param.get_input_layout(0);
    const auto& input1_layout = impl_param.get_input_layout(1);
    const auto& input2_layout = impl_param.get_input_layout(2);

    std::vector<ShapeType> input_shapes = {
        input0_layout.get<ShapeType>(),     // inputs_shape
        input1_layout.get<ShapeType>(),     // indices_shape,
        input2_layout.get<ShapeType>(),     // updates_shape,
    };

    ov::op::v3::ScatterNDUpdate op;
    std::vector<ShapeType> output_shapes = shape_infer(&op, input_shapes);

    return { layout{output_shapes[0], input0_layout.data_type, input0_layout.format} };
}

template std::vector<layout>
scatter_nd_update_inst::calc_output_layouts<ov::PartialShape>(scatter_nd_update_node const& node, const kernel_impl_params& impl_param);

std::string scatter_nd_update_inst::to_string(scatter_nd_update_node const& node) {
    auto desc = node.get_primitive();
    auto node_info = node.desc_to_json();
    auto& input = node.input();

    std::stringstream primitive_description;

    json_composite scatter_nd_update_info;
    scatter_nd_update_info.add("input id", input.id());

    node_info->add("scatter_nd_update info", scatter_nd_update_info);
    node_info->dump(primitive_description);

    return primitive_description.str();
}

scatter_nd_update_inst::typed_primitive_inst(network& network, scatter_nd_update_node const& node) : parent(network, node) {}

void scatter_nd_update_inst::on_execute() {
    auto input1_shape = _impl_params->input_layouts[1].get_partial_shape();
    auto input2_shape = _impl_params->input_layouts[2].get_partial_shape();
    auto same_layouts = _impl_params->input_layouts[0] == _impl_params->output_layouts[0];

    if (same_layouts && ((ov::shape_size(input1_shape.to_shape()) == 0) || (ov::shape_size(input2_shape.to_shape()) == 0)))
        update_output_memory();
}

void scatter_nd_update_inst::update_output_memory() {
    if (_outputs.size() > 0 && static_cast<bool>(_outputs[0])
        && _network.get_engine().is_the_same_buffer(output_memory(), input_memory()))
        return;

    if (_node != nullptr)
        build_deps();

    _outputs = {_network.get_engine().reinterpret_buffer(input_memory(), _impl_params->get_output_layout())};
    _mem_allocated = false;
}
}  // namespace cldnn
