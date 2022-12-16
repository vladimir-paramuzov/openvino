// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "test_utils.h"
#include "fusion_test_common.hpp"

#include <intel_gpu/primitives/input_layout.hpp>
#include <intel_gpu/primitives/quantize.hpp>
#include <intel_gpu/primitives/eltwise.hpp>
#include <intel_gpu/primitives/data.hpp>
#include <intel_gpu/primitives/gather.hpp>

#include <cmath>

using namespace cldnn;
using namespace ::tests;

namespace {
struct gather_test_params {
    tensor dictionary_shape;
    tensor indices_shape;
    ov::Shape out_shape;
    int64_t axis;
    data_types data_type;
    format input_format;
    data_types default_type;
    format default_format;
    size_t expected_fused_primitives;
    size_t expected_not_fused_primitives;
};

class GatherPrimitiveFusingTest : public ::BaseFusingTest<gather_test_params> {
public:
    void execute(gather_test_params& p) {
        auto input_prim = get_mem(get_input_layout(p));
        network network_not_fused(this->engine, this->topology_non_fused, cfg_not_fused);
        network network_fused(this->engine, this->topology_fused, cfg_fused);
        network_fused.set_input_data("input", input_prim);
        network_not_fused.set_input_data("input", input_prim);

        compare(network_not_fused, network_fused, p);
    }

    layout get_input_layout(gather_test_params& p) {
        return layout{ p.data_type, p.input_format, p.dictionary_shape };
    }

    layout get_indices_layout(gather_test_params& p) {
        return layout{ p.data_type, format::bfyx, p.indices_shape };
    }

    size_t get_axis_dim(gather_test_params& p) {
        auto in_layout = get_input_layout(p);
        return in_layout.get_dims()[p.axis];
    }

    layout get_per_channel_layout(gather_test_params& p) {
        return layout{ p.default_type, p.default_format, tensor{ 1, static_cast<tensor::value_type>(p.out_shape[1]), 1, 1 } };
    }
};
}  // namespace

/* ----------------------------------------------------------------------------------------------------- */
/* ------------------------------------------ Gather cases --------------------------------------------- */
/* ----------------------------------------------------------------------------------------------------- */

#define CASE_GATHER_FP32_1 { 2, 3, 1, 4 }, { 4, 1, 1, 1 }, { 4, 3, 4, 1 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_2 { 3, 2, 1, 2 }, { 2, 3, 1, 1 }, { 2, 3, 2, 2 }, 0, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_3 { 3, 1, 1, 2 }, { 2, 1, 1, 1 }, { 3, 2, 2, 1 }, 1, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_4 { 5, 3, 2, 2 }, { 3, 1, 1, 1 }, { 5, 2, 3, 2 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx
#define CASE_GATHER_FP32_5 { 2, 3, 1, 2 }, { 1, 3, 1, 1 }, { 2, 3, 1, 3 }, 2, data_types::f32, format::bfyx, data_types::f32, format::bfyx

#define CASE_GATHER_FP16_1 { 2, 3, 1, 4 }, { 4, 1, 1, 1 }, { 4, 3, 4, 1 }, 0, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_2 { 3, 2, 1, 2 }, { 2, 3, 1, 1 }, { 2, 3, 2, 2 }, 0, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_3 { 3, 1, 1, 2 }, { 2, 1, 1, 1 }, { 3, 2, 2, 1 }, 1, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_4 { 5, 3, 2, 2 }, { 3, 1, 1, 1 }, { 5, 2, 3, 2 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx
#define CASE_GATHER_FP16_5 { 2, 3, 1, 2 }, { 1, 3, 1, 1 }, { 2, 3, 1, 3 }, 2, data_types::f16, format::bfyx, data_types::f16, format::bfyx

#define CASE_GATHER_5D_FP32_1 { 2, 3, 1, 4, 1 }, { 4, 1, 1, 1 }, { 4, 3, 1, 4, 1 }, 0, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_2 { 2, 3, 2, 2, 2 }, { 2, 1, 1, 1 }, { 2, 2, 2, 2, 2 }, 1, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_3 { 5, 3, 2, 2, 2 }, { 3, 1, 1, 1 }, { 5, 3, 2, 3, 2 }, 3, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_4 { 2, 3, 1, 4, 4 }, { 2, 1, 1, 1 }, { 2, 3, 2, 4, 1 }, 2, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx
#define CASE_GATHER_5D_FP32_5 { 3, 1, 5, 2, 1 }, { 2, 1, 1, 1 }, { 3, 1, 1, 2, 2 }, 4, data_types::f32, format::bfzyx, data_types::f32, format::bfzyx

#define CASE_GATHER_5D_FP16_1 { 3, 2, 1, 2, 1 }, { 2, 1, 1, 1 }, { 2, 2, 1, 2, 2 }, 0, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_2 { 1, 3, 1, 2, 1 }, { 2, 1, 1, 1 }, { 1, 2, 1, 2, 1 }, 1, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_3 { 2, 3, 1, 3, 3 }, { 1, 2, 1, 1 }, { 2, 3, 3, 2, 1 }, 3, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_4 { 3, 2, 2, 2, 2 }, { 2, 1, 1, 1 }, { 3, 2, 2, 2, 2 }, 2, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx
#define CASE_GATHER_5D_FP16_5 { 1, 1, 2, 1, 1 }, { 3, 1, 1, 1 }, { 1, 1, 1, 1, 3 }, 4, data_types::f16, format::bfzyx, data_types::f16, format::bfzyx

class gather_quantize : public GatherPrimitiveFusingTest {};
TEST_P(gather_quantize, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p) - 1))),
        data("in_lo", get_mem(get_per_channel_layout(p), min_random, 0)),
        data("in_hi", get_mem(get_per_channel_layout(p), 1, max_random)),
        data("out_lo", get_mem(get_single_element_layout(p), -127)),
        data("out_hi", get_mem(get_single_element_layout(p), 127)),
        gather("gather_prim", input_info("input"), input_info("gather_indices"), p.axis, p.out_shape),
        quantize("quantize", input_info("gather_prim"), input_info("in_lo"), input_info("in_hi"),
                 input_info("out_lo"), input_info("out_hi"), 255, data_types::i8),
        reorder("reorder_bfyx", input_info("quantize"), p.default_format, data_types::f32)
    );

    tolerance = 1.f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_quantize, ::testing::ValuesIn(std::vector<gather_test_params>{
    gather_test_params{ CASE_GATHER_FP32_1, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_2, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_3, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_4, 2, 3 },
    gather_test_params{ CASE_GATHER_FP32_5, 2, 3 },

    gather_test_params{ CASE_GATHER_FP16_1, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_2, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_3, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_4, 2, 3 },
    gather_test_params{ CASE_GATHER_FP16_5, 2, 3 },

    gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 3 },

    gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 3 },
    gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 3 },
}));

class gather_eltwise_activation : public GatherPrimitiveFusingTest {};
TEST_P(gather_eltwise_activation, basic) {
    auto p = GetParam();
    create_topologies(
        input_layout("input", get_input_layout(p)),
        data("gather_indices", get_mem(get_indices_layout(p), 0, static_cast<int>(get_axis_dim(p) - 1))),
        data("eltwise_data", get_mem(get_per_channel_layout(p), -10, 10)),
        gather("gather_prim", input_info("input"), input_info("gather_indices"), p.axis, p.out_shape),
        activation("activation", input_info("gather_prim"), activation_func::abs),
        eltwise("eltwise", { input_info("activation"), input_info("eltwise_data") }, eltwise_mode::prod),
        reorder("reorder_bfyx", input_info("eltwise"), p.default_format, data_types::f32)
    );

    tolerance = 1e-5f;
    execute(p);
}

INSTANTIATE_TEST_SUITE_P(fusings_gpu, gather_eltwise_activation, ::testing::ValuesIn(std::vector<gather_test_params>{
    gather_test_params{ CASE_GATHER_FP32_1, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_2, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_3, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_4, 2, 4 },
    gather_test_params{ CASE_GATHER_FP32_5, 2, 4 },

    gather_test_params{ CASE_GATHER_FP16_1, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_2, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_3, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_4, 2, 4 },
    gather_test_params{ CASE_GATHER_FP16_5, 2, 4 },

    gather_test_params{ CASE_GATHER_5D_FP32_1, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_2, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_3, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_4, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP32_5, 2, 4 },

    gather_test_params{ CASE_GATHER_5D_FP16_1, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_2, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_3, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_4, 2, 4 },
    gather_test_params{ CASE_GATHER_5D_FP16_5, 2, 4 },
}));
