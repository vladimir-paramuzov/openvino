// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/roll.hpp"
#include "openvino/op/constant.hpp"

#include "intel_gpu/primitives/roll.hpp"
#include "intel_gpu/plugin/common_utils.hpp"
#include "intel_gpu/plugin/program.hpp"

namespace ov {
namespace intel_gpu {

namespace {

void CreateRollOp(Program& p, const std::shared_ptr<ov::op::v7::Roll>& op) {
    validate_inputs_count(op, {3});

    const auto inputs = p.GetInputInfo(op);
    const auto layer_name = layer_type_name_ID(op);
    const auto& op_friendly_name = op->get_friendly_name();
    const auto& input_pshape = op->get_input_partial_shape(0);
    OPENVINO_ASSERT(input_pshape.is_static(), "Dynamic shapes are not supported for Roll operation yet");
    const auto& input_shape = input_pshape.to_shape();
    const auto rank = static_cast<int>(input_shape.size());
    const auto format = cldnn::format::get_default_format(rank);
    const auto default_rank = format.dimension();

    auto shift_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(1));
    OPENVINO_ASSERT(shift_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op_friendly_name, " (", op->get_type_name(), ")");
    const auto shift_raw = shift_constant->cast_vector<int32_t>();

    auto axes_constant = std::dynamic_pointer_cast<ov::op::v0::Constant>(op->get_input_node_shared_ptr(2));
    OPENVINO_ASSERT(axes_constant != nullptr, "[GPU] Unsupported parameter nodes type in ", op_friendly_name, " (", op->get_type_name(), ")");
    auto axes_raw = axes_constant->cast_vector<int32_t>();

    // Normalize axes and sum shift
    std::vector<int32_t> shift(default_rank);
    for (size_t a = 0; a < axes_raw.size(); ++a) {
        auto& axis = axes_raw[a];
        if (axis < 0) {
            axis += rank;
        }
        if (axis < 0 || axis >= rank) {
            OPENVINO_THROW(op_friendly_name, " Incorrect axis value: ", axis);
        }
        shift[axis] += shift_raw[a];
    }

    // Normalize shift
    for (int s = 0; s < rank; ++s) {
        auto& sh = shift[s];
        const auto dim = static_cast<int32_t>(input_shape[s]);
        sh %= dim;
        if (sh < 0) {
            sh += dim;
        }
    }

    const cldnn::roll roll_prim(layer_name, inputs.front(), {format, shift});
    p.add_primitive(*op, roll_prim);
}

}  // namespace

REGISTER_FACTORY_IMPL(v7, Roll);

}  // namespace intel_gpu
}  // namespace ov
