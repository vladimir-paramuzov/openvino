// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/convolution.hpp"

namespace ov {
namespace intel_gpu {
namespace op {

class WithOptionalInputsInterface {
public:
    virtual bool has_input_at_port(size_t i) = 0;
};

// Common node for v1::Convolution and v1::GroupConvolution with few extensions
//  - Relaxed type requirements
//  - Bias support
//  - Asymmetric quantization support
class Convolution : public ov::op::util::ConvolutionFwdPropBase, public WithOptionalInputsInterface {
public:
    OPENVINO_OP("Convolution", "gpu_opset", ov::op::util::ConvolutionFwdPropBase);

    Convolution() = default;

    Convolution(const ov::Output<Node>& data_batch,
                const ov::Output<Node>& filters,
                const ov::Output<Node>& bias,
                const ov::Strides& strides,
                const ov::CoordinateDiff& pads_begin,
                const ov::CoordinateDiff& pads_end,
                const ov::Strides& dilations,
                const int64_t& groups,
                const ov::op::PadType& auto_pad,
                const ov::element::Type& output_type);

    Convolution(const ov::Output<Node>& data_batch,
                const ov::Output<Node>& filters,
                const ov::Output<Node>& bias,
                const ov::Output<Node>& activations_zero_point,
                const ov::Output<Node>& weights_zero_point,
                const ov::Output<Node>& compensations,
                const ov::Strides& strides,
                const ov::CoordinateDiff& pads_begin,
                const ov::CoordinateDiff& pads_end,
                const ov::Strides& dilations,
                const int64_t& groups,
                const ov::op::PadType& auto_pad,
                const ov::element::Type& output_type);


    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor& visitor) override;

    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector& new_args) const override;

    bool has_groups() const;
    int64_t get_groups() const;

    bool is_asymmetric() const;

    bool has_input_at_port(size_t i) override;

protected:
    int64_t m_groups = -1; // negative value means no groups
    bool m_asymmetric = false;
    ov::element::Type m_output_type = ov::element::undefined;
};

std::vector<ov::PartialShape> shape_infer(const Convolution* op,
                                          const std::vector<ov::PartialShape>& input_shapes,
                                          CoordinateDiff& pads_begin,
                                          CoordinateDiff& pads_end);

}   // namespace op
}   // namespace intel_gpu
}   // namespace ov
