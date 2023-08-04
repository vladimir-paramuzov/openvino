// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/compress_float_constants.hpp"

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/decompression.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"
#include "transformations/rt_info/old_api_map_element_type_attribute.hpp"
#include "ngraph/runtime/reference/convert.hpp"

#define POSTPONED_FP16_COMPRESSION 1

namespace {
// TODO: Make this definitiion unique, now it is duplicated between this file and serialize.cpp
const std::string& postponed_fp16_compression_tag = "postponed_fp16_compression";
}  // namespace


namespace {
template <ov::element::Type_t PREC_FROM>
std::shared_ptr<ov::Node> change_constant_precision_to_fp16(std::shared_ptr<ov::op::v0::Constant>& constant,
                                                            bool postponed = false) {
    using src_type = typename ov::element_type_traits<PREC_FROM>::value_type;

    const auto* src_data = constant->get_data_ptr<src_type>();
    const auto size = ov::shape_size(constant->get_shape());

    auto new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::f16, constant->get_shape());
    auto* dst_data = const_cast<ov::float16*>(reinterpret_cast<const ov::float16*>(new_constant->get_data_ptr()));
    if (dst_data == nullptr)
        return nullptr;

    // TODO: Speed this part up by applying code similar to Convert::evaluate
    int num_out_of_range = 0;
    //#pragma parallel for reduction(+ : num_out_of_range)
    for (size_t i = 0; i < size; ++i) {
        // if abs value is smaller than the smallest positive fp16, but not zero
        if (std::abs(src_data[i]) < ov::float16::from_bits(0x0001) && src_data[i] != 0.0f) {
            num_out_of_range++;
        } else if (src_data[i] > std::numeric_limits<ov::float16>::max()) {
            dst_data[i] = std::numeric_limits<ov::float16>::max();
            num_out_of_range++;
        } else if (src_data[i] < std::numeric_limits<ov::float16>::lowest()) {
            dst_data[i] = std::numeric_limits<ov::float16>::lowest();
            num_out_of_range++;
        } else {
            dst_data[i] = static_cast<ov::float16>(src_data[i]);
        }
    }

    // if more than 75% of a FP32 constant do not fit into FP16 keep in FP32
    float keep_threshold = 0.75f;
    float out_of_range_proportion = static_cast<float>(num_out_of_range) / static_cast<float>(size);

    if (out_of_range_proportion >= keep_threshold) {
        return nullptr;
    }

    if (postponed) {
        // dispose just converted constant to avoid allocation too much memory
        // it will be converted again while serialization
        return constant;
    } else {
        return new_constant;
    }
}
}  // namespace

ov::pass::CompressFloatConstantsImpl::CompressFloatConstantsImpl(bool postponed) {
    MATCHER_SCOPE(CompressFloatConstantsImpl);
    auto const_pattern = pattern::wrap_type<ov::op::v0::Constant>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& const_node_pattern = pattern_map.at(const_pattern);

        auto const_node = std::dynamic_pointer_cast<ov::op::v0::Constant>(const_node_pattern.get_node_shared_ptr());
        if (!const_node)
            return false;

        if (ov::fp16_compression_is_disabled(const_node))
            return false;

        auto c_type = const_node->get_element_type();
        std::shared_ptr<ov::Node> new_const;
        if (c_type == ov::element::f32) {
            if(postponed) { // New optimized implemenation based on count_out_of_f16_range, replace postposed by false to disable
                auto size = shape_size(const_node->get_output_shape(0));
                auto num_out_of_range = ngraph::runtime::reference::count_out_of_f16_range(const_node->get_data_ptr<ov::element::f32>(), size);

                // if more than 75% of a FP32 constant do not fit into FP16 keep in FP32
                float keep_threshold = 0.75f;
                float out_of_range_proportion = static_cast<float>(num_out_of_range) / static_cast<float>(size);

                if (out_of_range_proportion >= keep_threshold) {
                    return false;
                }
                new_const = const_node;
            } else {
                new_const = change_constant_precision_to_fp16<ov::element::Type_t::f32>(const_node, postponed);
            }
        } else if (c_type == ov::element::f64) {
            new_const = change_constant_precision_to_fp16<ov::element::Type_t::f64>(const_node, postponed);
        } else {
            return false;
        }

        if (!new_const) {
            return false;
        }
        auto constant_target_inputs = const_node->get_output_target_inputs(0);
        auto convert = std::make_shared<ov::op::v0::Convert>(new_const, const_node->get_element_type());

        convert->set_friendly_name(const_node->get_friendly_name());
        new_const->set_friendly_name(const_node->get_friendly_name() + "_compressed");
        ov::copy_runtime_info(const_node, convert);
        ov::mark_as_decompression(convert);
        if (postponed) {
            // Value of postponed_fp16_compression_tag rt_info entry always should be true, value itself is ignored
            // TODO: choose another type for that
            new_const->get_rt_info()[postponed_fp16_compression_tag] = true;
            new_const->get_output_tensor(0).get_rt_info()[postponed_fp16_compression_tag] = true;
            for (const auto& target_input : constant_target_inputs) {
                target_input.replace_source_output(convert);
            }
        } else {
            ov::replace_node(const_node, convert);
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(const_pattern, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::AddOldApiMapToParameters::AddOldApiMapToParameters() {
    MATCHER_SCOPE(AddOldApiMapToParameters);
    auto param_pattern = pattern::wrap_type<ov::op::v0::Parameter>();

    ov::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto node = pattern_map.at(param_pattern).get_node_shared_ptr();

        auto param_node = std::dynamic_pointer_cast<ov::op::v0::Parameter>(node);
        if (!param_node)
            return false;
        auto p_type = param_node->get_element_type();
        if (p_type == ov::element::f32 || p_type == ov::element::f64) {
            ov::set_old_api_map_element_type(node, ov::OldApiMapElementType(ov::element::Type_t::f16));
        } else {
            return false;
        }
        return true;
    };

    auto m = std::make_shared<pattern::Matcher>(param_pattern, matcher_name);
    this->register_matcher(m, callback);
}
