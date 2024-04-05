// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/op/internal_primitive.hpp"
#include "openvino/core/node_vector.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

InternalPrimitive::InternalPrimitive(const OutputVector& inputs, std::shared_ptr<Node> original_node, std::shared_ptr<cldnn::primitive> prim)
    : Op(inputs)
    , m_original_node(original_node)
    , m_primitive(prim) {
    validate_and_infer_types();
}

InternalPrimitive::InternalPrimitive(const NodeVector& inputs, std::shared_ptr<Node> original_node, std::shared_ptr<cldnn::primitive> prim)
    : InternalPrimitive(as_output_vector(inputs), original_node, prim) {}

bool InternalPrimitive::visit_attributes(ov::AttributeVisitor& visitor) {
    return true;
}

void InternalPrimitive::validate_and_infer_types() {
    m_original_node->validate_and_infer_types();
    for (size_t i = 0; i < m_original_node->get_output_size(); i++) {
        set_output_type(i, m_original_node->get_output_element_type(i), m_original_node->get_output_partial_shape(i));
    }
}

std::shared_ptr<Node> InternalPrimitive::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    check_new_args_count(this, new_args);
    return std::make_shared<InternalPrimitive>(new_args, m_original_node, m_primitive);
}

}  // namespace op
}  // namespace intel_gpu
}  // namespace ov
