// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/deconvolution.hpp"
#include "openvino/core/except.hpp"
#include "pass_manager.h"
#include "gemm_inst.h"
#include "program_node.h"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "to_string_utils.h"
#include <iostream>
#include <sstream>


using namespace cldnn;

namespace {

#ifdef ENABLE_ONEDNN_FOR_GPU
void print_selected_formats(const program_node& n) {
    std::stringstream ss;
    ov::write_all_to_stream(ss, "select_preferred_formats:", n.id(), ":\n");

    const auto& in_fmts = n.get_preferred_input_fmts();
    const auto& out_fmts = n.get_preferred_output_fmts();

    for (size_t i = 0; i < in_fmts.size(); i++) {
        ss << "\tIn " << i << ": " << fmt_to_str(in_fmts[i]) << std::endl;
    }

    for (size_t i = 0; i < out_fmts.size(); i++) {
        ss << "\tOut " << i << ": " << fmt_to_str(out_fmts[i]) << std::endl;
    }
    GPU_DEBUG_LOG << ss.str() << std::endl;
}

static void optimize_conv_permute(program_node& node) {
    // In conv-permute pattern, sets the output format of conv to byxf so that permute can be optimized.
    // ex) oneDNN convolution -> (byxf) -> permute -> (bfyx) -> output
    //     output layout of convolution: byxf [b:1, f:128, y:2, x:2]
    //     output layout of permute:     bfyx [b:1, f:2, y:2, x:128]
    // In this case, it can be handled by changing only the shape of permute without the kernel execution.
    if (node.get_output_layout().get_rank() == 4
        && node.get_users().size() == 1 && node.get_users().front()->is_type<permute>()) {
        auto& pnode = node.get_users().front()->as<permute>();
        auto can_optimize_permute = pnode.get_output_layout().data_type == node.get_output_layout().data_type
            && !pnode.has_fused_primitives()
            && !pnode.is_output() && pnode.get_input_layout(0).is_static()
            && pnode.is_rotating_except_batch();
        if (can_optimize_permute) {
            node.set_preferred_output_fmt(0, format::byxf);
            pnode.init_preferred_fmt(1, 1);
            pnode.set_preferred_input_fmt(0, cldnn::format::byxf);
            pnode.set_preferred_output_fmt(0, cldnn::format::bfyx);
            pnode.can_be_optimized(true);
        }
    }
}
#endif  // ENABLE_ONEDNN_FOR_GPU
} // namespace

void select_preferred_formats::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::select_preferred_formats");

    auto& engine = p.get_engine();
    const auto& device_info = engine.get_device_info();

    if (!device_info.supports_immad)
        return;

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Fallback to ocl when asymmetric weights convolution is existed.
    if (_lo.get_optimization_attributes().use_onednn_impls) {
        for (auto n : p.get_processing_order()) {
            if (n->is_type<convolution>() && n->as<convolution>().weights_zero_points_term())
                return;
        }
    }

    auto forcing_map = _lo.get_implementation_forcing();

    engine.create_onednn_engine(p.get_config());
    for (auto n : p.get_processing_order()) {
        n->recalc_output_layout();
        if (n->is_input() || !n->can_use(impl_types::onednn)) {
            continue;
        }

        // skip to set preferred_formats if forcing_impl is not onednn.
        if (std::find_if(forcing_map.begin(), forcing_map.end(),
                [&n](std::map<primitive_id, std::pair<format::type, impl_types>>::value_type const& it) {
                    return (it.first == n->id() && it.second.second != impl_types::onednn);
                }) != forcing_map.end())
            continue;

        // Onednn primitive descriptor creation may fail, for example, due to asymmetric weight.
        try {
            n->select_preferred_formats(impl_types::onednn);

            if (n->is_type<convolution>() || n->is_type<deconvolution>()) {
                optimize_conv_permute(*n);
            }

            print_selected_formats(*n);
        } catch(std::exception &exception) {
            GPU_DEBUG_INFO << "WARNING(select_preferred_formats): " << exception.what() << std::endl;
        }
    }
#else
    (void)_lo;
#endif  // ENABLE_ONEDNN_FOR_GPU
}
