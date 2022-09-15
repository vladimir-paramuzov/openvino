// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>

#include "eltwise_inst.h"

#include "program_wrapper.h"

#include <cmath>
#include <algorithm>

using namespace cldnn;
using namespace ::tests;
using namespace ov::op;
using namespace ov;

namespace shape_infer_tests {

struct eltwise_test_params {
    layout input1_layout;
    layout input2_layout;
    eltwise_mode mode;
    AutoBroadcastSpec auto_broadcast_spec;
    layout expected_layout;
    std::vector<tensor> stride;
};

std::string to_string(const eltwise_mode mode) {
    std::string str_mode;
    switch (mode) {
        case eltwise_mode::sum:
            str_mode = "sum";
            break;
        case eltwise_mode::sub:
            str_mode = "subtract";
            break;
        case eltwise_mode::max:
            str_mode = "max";
            break;
        case eltwise_mode::prod:
            str_mode = "product";
            break;
        case eltwise_mode::div:
            str_mode = "div";
            break;
        case eltwise_mode::min:
            str_mode = "min";
            break;
        case eltwise_mode::pow:
            str_mode = "pow";
            break;
        case eltwise_mode::squared_diff:
            str_mode = "squared_diff";
            break;
        case eltwise_mode::mod:
            str_mode = "mod";
            break;
        case eltwise_mode::eq:
            str_mode = "equal";
            break;
        case eltwise_mode::ne:
            str_mode = "not equal";
            break;
        case eltwise_mode::lt:
            str_mode = "less";
            break;
        case eltwise_mode::le:
            str_mode = "less-or-equal";
            break;
        case eltwise_mode::gt:
            str_mode = "greater";
            break;
        case eltwise_mode::ge:
            str_mode = "greater-or-equal";
            break;
        case eltwise_mode::logic_and:
            str_mode = "and";
            break;
        case eltwise_mode::logic_or:
            str_mode = "or";
            break;
        case eltwise_mode::logic_xor:
            str_mode = "xor";
            break;
        case eltwise_mode::floor_mod:
            str_mode = "floor_mod";
            break;
        default:
            str_mode = "not supported mode";
            break;
    }
    return str_mode;
}

std::string to_string(const cldnn::layout& l) {
    std::stringstream s;
    s << "{" << data_type_traits::name(l.data_type) << ","
      << l.format.to_string() << ","
      << l.get_partial_shape() << "}";
    return s.str();
}

std::ostream& operator<<(std::ostream& ost, const eltwise_test_params& params) {
    ost << "{ IN1:" << to_string(params.input1_layout) << ","
        << "IN2:" << to_string(params.input2_layout) << ","
        << to_string(params.mode) << ","
        << "{" << params.auto_broadcast_spec.m_type << ", " << std::to_string(params.auto_broadcast_spec.m_axis) << "},"
        << "EXPECTED:" << to_string(params.expected_layout) << ","
        << "STRIDE:{";
    for (auto& s : params.stride) {
        ost << s << ",";
    }
    ost << "}\n";
    return ost;
}

class eltwise_si_test : public testing::TestWithParam<eltwise_test_params> { };

TEST_P(eltwise_si_test, shape_infer) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto input1_prim = std::make_shared<input_layout>("input1", p.input1_layout);
    auto input2_prim = std::make_shared<input_layout>("input2", p.input2_layout);
    auto eltwise_prim = std::make_shared<eltwise>("output", "input1", "input2", p.stride, p.mode, p.auto_broadcast_spec);

    cldnn::program prog(engine);

    auto& input1_node = prog.get_or_create(input1_prim);
    auto& input2_node = prog.get_or_create(input2_prim);
    auto& eltwise_node = prog.get_or_create(eltwise_prim);
    program_wrapper::add_connection(prog, input1_node, eltwise_node);
    program_wrapper::add_connection(prog, input2_node, eltwise_node);
    auto res = eltwise_inst::calc_output_layouts<ov::PartialShape>(eltwise_node, *eltwise_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

TEST_P(eltwise_si_test, shape_infer_const_data) {
    auto p = GetParam();

    auto& engine = get_test_engine();

    auto const_data = engine.allocate_memory(p.input2_layout);

    auto input1_prim = std::make_shared<input_layout>("input1", p.input1_layout);
    auto const_data_prim = std::make_shared<data>("const_data", const_data);
    auto eltwise_prim = std::make_shared<eltwise>("output", "input1", "const_data", p.stride, p.mode, p.auto_broadcast_spec);

    cldnn::program prog(engine);

    auto& input1_node = prog.get_or_create(input1_prim);
    auto& const_data_node = prog.get_or_create(const_data_prim);
    auto& eltwise_node = prog.get_or_create(eltwise_prim);
    program_wrapper::add_connection(prog, input1_node, eltwise_node);
    program_wrapper::add_connection(prog, const_data_node, eltwise_node);
    auto res = eltwise_inst::calc_output_layouts<ov::PartialShape>(eltwise_node, *eltwise_node.get_kernel_impl_params());

    ASSERT_EQ(res.size(), 1);
    ASSERT_EQ(res[0], p.expected_layout);
}

INSTANTIATE_TEST_SUITE_P(smoke, eltwise_si_test,
    testing::ValuesIn(std::vector<eltwise_test_params>{
    {{{2, 1, 5}, data_types::f32, format::bfyx},                {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NONE},      {{2, 1, 5}, data_types::f32, format::bfyx},                     {}},
    {{{2, 1, 5}, data_types::f32, format::bfyx},                {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, 4, 5}, data_types::f32, format::bfyx},                     {}},
    {{{1, 1, 5}, data_types::f32, format::bfyx},                {{5, 2, 1, 3}, data_types::f32, format::bfyx},              eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 5, 3}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{4, 5}, data_types::f32, format::bfyx},                    eltwise_mode::sum,      {AutoBroadcastType::PDPD, -1},  {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{1, 3}, data_types::f32, format::bfyx},                    eltwise_mode::sum,      {AutoBroadcastType::PDPD},      {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::sum,      {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::f32, format::bfyx},                  {}},
    // test for dynamic shape
    {{{1, 1, 5}, data_types::f32, format::bfyx},                {{5, 2, 1, 3}, data_types::f32, format::bfyx},              eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 5, 3}, data_types::f32, format::bfyx},                  {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{2, 3, 4, 5}, data_types::f32, format::bfyx},              eltwise_mode::sum,      {AutoBroadcastType::PDPD},      {PartialShape::dynamic(4), data_types::f32, format::bfyx},      {}},
    {{{2, -1, 5}, data_types::f32, format::bfyx},               {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, 4, 5}, data_types::f32, format::bfyx},                     {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{-1, 4, -1}, data_types::f32, format::bfyx},                   {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{2, -1, 5}, data_types::f32, format::bfyx},                    {}},
    {{PartialShape::dynamic(3), data_types::f32, format::bfyx}, {{1, 4, 1}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::PDPD},      {PartialShape::dynamic(3), data_types::f32, format::bfyx},      {}},
    {{{-1, -1, 1024, 512}, data_types::f32, format::bfyx},      {{1,   1, 512,  1}, data_types::f32, format::bfyx},         eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {ov::PartialShape::dynamic(4), data_types::f32, format::bfyx},  {}},
    // test for output data type of logic and comparison operations
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::eq,       {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f16, format::bfyx},             {{3}, data_types::f16, format::bfyx},                       eltwise_mode::ne,       {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f16, format::bfyx},             {{3}, data_types::f16, format::bfyx},                       eltwise_mode::lt,       {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i32, format::bfyx},             {{3}, data_types::i32, format::bfyx},                       eltwise_mode::le,       {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i64, format::bfyx},             {{3}, data_types::i64, format::bfyx},                       eltwise_mode::gt,       {AutoBroadcastType::NUMPY},     {{3, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::u8,  format::bfyx},             {{3}, data_types::u8,  format::bfyx},                       eltwise_mode::ge,       {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::i8,  format::bfyx},             {{3}, data_types::i8,  format::bfyx},                       eltwise_mode::logic_and,{AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::logic_or, {AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    {{{2, 3, 4, 5}, data_types::f32, format::bfyx},             {{3}, data_types::f32, format::bfyx},                       eltwise_mode::logic_xor,{AutoBroadcastType::PDPD, 1},   {{2, 3, 4, 5}, data_types::i8, format::bfyx},                   {}},
    // test stride
    {{{5, 2, 1, 20}, data_types::f32, format::bfyx},            {{1, 1, 40}, data_types::f32, format::bfyx},                eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {{5, 2, 1, 5}, data_types::f32, format::bfyx},                  {{1,3,4,2}}},
    {{{2, 3, 40,50}, data_types::f32, format::bfyx},            {{40, 50}, data_types::f32, format::bfyx},                  eltwise_mode::sum,      {AutoBroadcastType::PDPD, -1},  {{2, 3, 5, 10}, data_types::f32, format::bfyx},                 {{1,1,5,8}}},
    {{PartialShape::dynamic(4), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::NUMPY},     {PartialShape::dynamic(4), data_types::f32, format::bfyx},      {{1,1,5,8}}},
    {{PartialShape::dynamic(4), data_types::f32, format::bfyx}, {{2, 1, 5}, data_types::f32, format::bfyx},                 eltwise_mode::sum,      {AutoBroadcastType::PDPD, 1},   {PartialShape::dynamic(4), data_types::f32, format::bfyx},      {{1,1,3,8}}},
}));

}  // shape_infer_tests
