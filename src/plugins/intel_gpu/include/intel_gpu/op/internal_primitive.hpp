// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/node_vector.hpp"
#include "openvino/op/op.hpp"
#include "intel_gpu/primitives/primitive.hpp"


namespace ov {
namespace intel_gpu {
namespace op {

// Common node for v1::Convolution and v1::GroupConvolution with few extensions
//  - Relaxed type requirements
//  - Bias support
//  - Asymmetric quantization support
class InternalPrimitive : public ov::op::Op {
public:
    OPENVINO_OP("InternalPrimitive", "gpu_opset");

    InternalPrimitive() = default;

    InternalPrimitive(const OutputVector& inputs, std::shared_ptr<Node> original_node, std::shared_ptr<cldnn::primitive> prim);
    InternalPrimitive(const NodeVector& inputs, std::shared_ptr<Node> original_node, std::shared_ptr<cldnn::primitive> prim);

    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

protected:
    std::shared_ptr<Node> m_original_node;
    std::shared_ptr<cldnn::primitive> m_primitive;
};

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
