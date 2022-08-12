// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/resample.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "resample_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;

namespace shape_infer_tests {

using InterpolateOp = ov::op::v4::Interpolate;
struct InterpolateAttrs {
        InterpolateAttrs(InterpolateOp::ShapeCalcMode shape_calc_mode) : shape_calc_mode(shape_calc_mode) {}
        InterpolateOp::InterpolateMode mode = InterpolateOp::InterpolateMode::LINEAR;
        InterpolateOp::ShapeCalcMode shape_calc_mode = InterpolateOp::ShapeCalcMode::SIZES;
        std::vector<size_t> pads_begin;
        std::vector<size_t> pads_end;
        InterpolateOp::CoordinateTransformMode coordinate_transformation_mode = InterpolateOp::CoordinateTransformMode::HALF_PIXEL;
        InterpolateOp::NearestMode nearest_mode = InterpolateOp::NearestMode::ROUND_PREFER_FLOOR;
        bool antialias = false;
        double cube_coeff = -0.75f;
};

struct interpolate_test_params {
    layout in_layout;
    layout pattern_layout;
    std::vector<int64_t> pattern_data;
    ov::PartialShape output_partial_shape;
    std::vector<float> scales;
    std::vector<int64_t> axes;
    InterpolateAttrs attrs;
    layout expected_layout;
};

class interpolate_test_two_inputs : public testing::TestWithParam<interpolate_test_params> { };
TEST_P(interpolate_test_two_inputs, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto pattern_prim = std::make_shared<input_layout>("pattern", p.pattern_layout);
    auto resample_prim = std::make_shared<resample>("output", "input", "pattern", p.scales, p.axes, p.output_partial_shape,
                                                    p.attrs.pads_begin, p.attrs.pads_end, p.attrs.antialias, p.attrs.cube_coeff,
                                                    p.attrs.mode, p.attrs.shape_calc_mode,
                                                    p.attrs.coordinate_transformation_mode, p.attrs.nearest_mode);
    cldnn::program prog(engine);

    auto pattern_mem = engine.allocate_memory(p.pattern_layout);
    set_values(pattern_mem, p.pattern_data);

    auto& input_node = prog.get_or_create(input_prim);
    auto& pattern_node = prog.get_or_create(pattern_prim);
    auto& resample_node = prog.get_or_create(resample_prim);
    program_wrapper::add_connection(prog, input_node, resample_node);
    program_wrapper::add_connection(prog, pattern_node, resample_node);
    auto params = resample_node.get_kernel_impl_params();

    auto res_wo_data = resample_inst::calc_output_layouts<ov::PartialShape>(resample_node, *params);

    params->memory_deps = {{1, pattern_mem}};
    auto res_w_data = resample_inst::calc_output_layouts<ov::PartialShape>(resample_node, *params);

    layout expected_layout_wo_data{p.output_partial_shape, p.expected_layout.data_type, p.expected_layout.format};
    ASSERT_EQ(res_wo_data.size(), 1);
    ASSERT_EQ(res_wo_data[0], expected_layout_wo_data);

    ASSERT_EQ(res_w_data.size(), 1);
    ASSERT_EQ(res_w_data[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, interpolate_test_two_inputs,
    testing::ValuesIn(std::vector<interpolate_test_params>{
        {
            layout{ov::PartialShape{1, 2, 48, 80}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {-1, -1, -1, -1}, ov::PartialShape::dynamic(4),
            {0.5, 2.0}, {2, 3}, InterpolateAttrs{InterpolateOp::ShapeCalcMode::SCALES},
            layout{ov::PartialShape{1, 2, 24, 160}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 2, 3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {2, 2, 2, 3}, ov::PartialShape::dynamic(4),
            {}, {}, InterpolateAttrs(InterpolateOp::ShapeCalcMode::SIZES),
            layout{ov::PartialShape{2, 2, 2, 3}, data_types::f32, format::bfyx}
        }
    }));

class interpolate_test_single_input : public testing::TestWithParam<interpolate_test_params> { };
TEST_P(interpolate_test_single_input, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input_prim = std::make_shared<input_layout>("input", p.in_layout);
    auto resample_prim = std::make_shared<resample>("output", "input", p.pattern_data, p.scales, p.axes, p.output_partial_shape,
                                                    p.attrs.pads_begin, p.attrs.pads_end, p.attrs.antialias, p.attrs.cube_coeff,
                                                    p.attrs.mode, p.attrs.shape_calc_mode,
                                                    p.attrs.coordinate_transformation_mode, p.attrs.nearest_mode);
    cldnn::program prog(engine);

    auto& input_node = prog.get_or_create(input_prim);
    auto& resample_node = prog.get_or_create(resample_prim);
    program_wrapper::add_connection(prog, input_node, resample_node);

    auto params = resample_node.get_kernel_impl_params();
    auto res = resample_inst::calc_output_layouts<ov::PartialShape>(resample_node, *params);

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, interpolate_test_single_input,
    testing::ValuesIn(std::vector<interpolate_test_params>{
        {
            layout{ov::PartialShape{1, 2, 48, 80}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {-1, -1, -1, -1}, ov::PartialShape::dynamic(),
            {0.5, 2.0}, {2, 3}, InterpolateAttrs{InterpolateOp::ShapeCalcMode::SCALES},
            layout{ov::PartialShape{1, 2, 24, 160}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 2, 3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {2, 2, 2, 3}, ov::PartialShape::dynamic(),
            {}, {}, InterpolateAttrs(InterpolateOp::ShapeCalcMode::SIZES),
            layout{ov::PartialShape{2, 2, 2, 3}, data_types::f32, format::bfyx}
        },
        {
            layout{ov::PartialShape{2, 2, 3, 2}, data_types::f32, format::bfyx},
            layout{ov::PartialShape{4}, data_types::i64, format::bfyx}, {}, ov::PartialShape::dynamic(2),
            {}, {}, InterpolateAttrs(InterpolateOp::ShapeCalcMode::SIZES),
            layout{ov::PartialShape::dynamic(2), data_types::f32, format::bfyx}
        }
    }));

}  // shape_infer_tests
