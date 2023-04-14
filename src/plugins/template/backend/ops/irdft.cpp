// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph/runtime/reference/irdft.hpp"

#include "evaluate_node.hpp"
#include "evaluates_map.hpp"
#include "ngraph/runtime/reference/fft.hpp"

namespace irfft_v9 {
struct InfoForIRFFT9 {
    std::vector<float> input_data;
    std::vector<int64_t> axes_data;
    ngraph::Shape input_data_shape;
    ngraph::Shape axes_data_shape;
    ngraph::Shape fft_output_shape;
    ngraph::Shape output_shape;
    int64_t last_signal_size;
};

InfoForIRFFT9 get_info_for_irfft9_eval(const std::vector<std::shared_ptr<ngraph::HostTensor>>& inputs) {
    InfoForIRFFT9 result;

    result.input_data_shape = inputs[0]->get_shape();
    result.axes_data_shape = inputs[1]->get_shape();
    result.input_data = get_floats(inputs[0], result.input_data_shape);
    result.axes_data = get_integers(inputs[1], result.axes_data_shape);

    auto fft_output_shape = result.input_data_shape;
    auto output_shape = result.input_data_shape;

    int64_t input_rank = static_cast<int64_t>(result.input_data_shape.size());
    int64_t complex_data_rank = input_rank - 1;
    auto canonicalized_axes = ngraph::runtime::reference::canonicalize_axes(result.axes_data.data(),
                                                                            result.axes_data_shape,
                                                                            complex_data_rank);

    size_t num_of_axes = result.axes_data.size();
    auto signal_size = get_signal_size(inputs, num_of_axes);

    const auto last_axis = canonicalized_axes.back();
    for (size_t i = 0; i < num_of_axes; ++i) {
        int64_t current_axis = canonicalized_axes[i];
        int64_t current_signal_size = signal_size[i];
        if (current_signal_size != -1) {
            fft_output_shape[current_axis] = static_cast<size_t>(current_signal_size);
            output_shape[current_axis] = static_cast<size_t>(current_signal_size);
        }
    }
    result.last_signal_size = signal_size.back();
    if (signal_size.back() == -1) {
        output_shape[last_axis] = 2 * (result.input_data_shape[last_axis] - 1);
        fft_output_shape[last_axis] = 2 * (result.input_data_shape[last_axis] - 1);
        result.last_signal_size = 2 * (result.input_data_shape[last_axis] - 1);
    }

    output_shape.pop_back();

    result.fft_output_shape = fft_output_shape;
    result.output_shape = output_shape;
    result.axes_data = canonicalized_axes;

    return result;
}
}  // namespace irfft_v9

template <ngraph::element::Type_t ET>
bool evaluate(const std::shared_ptr<ngraph::op::v9::IRDFT>& op,
              const ngraph::HostTensorVector& outputs,
              const ngraph::HostTensorVector& inputs) {
    auto info = irfft_v9::get_info_for_irfft9_eval(inputs);
    outputs[0]->set_shape(info.output_shape);

    std::vector<float> irfft_result(ngraph::shape_size(info.output_shape), 0.0f);
    ngraph::runtime::reference::irdft(info.input_data,
                                      info.input_data_shape,
                                      info.axes_data,
                                      irfft_result.data(),
                                      info.fft_output_shape,
                                      info.output_shape,
                                      info.last_signal_size);

    const auto output_type = op->get_input_element_type(0);
    ngraph::runtime::reference::fft_postprocessing(outputs, output_type, irfft_result);
    return true;
}

template <>
bool evaluate_node<ngraph::op::v9::IRDFT>(std::shared_ptr<ngraph::Node> node,
                                          const ngraph::HostTensorVector& outputs,
                                          const ngraph::HostTensorVector& inputs) {
    auto element_type = node->get_output_element_type(0);
    if (ov::is_type<ngraph::op::v1::Select>(node) || ov::is_type<ngraph::op::util::BinaryElementwiseComparison>(node))
        element_type = node->get_input_element_type(1);

    switch (element_type) {
    case ngraph::element::Type_t::boolean:
        return evaluate<ngraph::element::Type_t::boolean>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node),
                                                          outputs,
                                                          inputs);
    case ngraph::element::Type_t::bf16:
        return evaluate<ngraph::element::Type_t::bf16>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::f16:
        return evaluate<ngraph::element::Type_t::f16>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::f64:
        return evaluate<ngraph::element::Type_t::f64>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::f32:
        return evaluate<ngraph::element::Type_t::f32>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::i4:
        return evaluate<ngraph::element::Type_t::i4>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::i8:
        return evaluate<ngraph::element::Type_t::i8>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::i16:
        return evaluate<ngraph::element::Type_t::i16>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::i32:
        return evaluate<ngraph::element::Type_t::i32>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::i64:
        return evaluate<ngraph::element::Type_t::i64>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u1:
        return evaluate<ngraph::element::Type_t::u1>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u4:
        return evaluate<ngraph::element::Type_t::u4>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u8:
        return evaluate<ngraph::element::Type_t::u8>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u16:
        return evaluate<ngraph::element::Type_t::u16>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u32:
        return evaluate<ngraph::element::Type_t::u32>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    case ngraph::element::Type_t::u64:
        return evaluate<ngraph::element::Type_t::u64>(ov::as_type_ptr<ngraph::op::v9::IRDFT>(node), outputs, inputs);
    default:
        throw ngraph::ngraph_error(std::string("Unhandled data type ") + node->get_element_type().get_type_name() +
                                   std::string("in evaluate_node()"));
    }
}