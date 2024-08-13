// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "program_node.h"
#include "openvino/core/except.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "to_string_utils.h"
#include <iostream>
#include <sstream>


using namespace cldnn;

namespace {

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

} // namespace

void select_preferred_formats::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::select_preferred_formats");

    auto& engine = p.get_engine();

#ifdef ENABLE_ONEDNN_FOR_GPU
    // Fallback to ocl when asymmetric weights convolution is existed.
    if (_lo.get_optimization_attributes().use_onednn_impls) {
        for (auto n : p.get_processing_order()) {
            if (n->is_type<convolution>() && n->as<convolution>().weights_zero_points_term())
                return;
        }
        engine.create_onednn_engine(p.get_config());
    }
#endif  // ENABLE_ONEDNN_FOR_GPU

    const auto& forcing_map = _lo.get_implementation_forcing();

    for (auto n : p.get_processing_order()) {
        n->recalc_output_layout();
        if (n->is_input() || !n->is_in_data_flow()) {
            continue;
        }

        auto forced_type_it = std::find_if(forcing_map.begin(), forcing_map.end(),
                [&n](std::map<primitive_id, std::pair<format::type, impl_types>>::value_type const& it) {
                    return it.first == n->id();
                });

        auto impl_type = impl_types::any;

        if (forced_type_it != forcing_map.end()) {
            impl_type = forced_type_it->second.second;
        }

        const auto& params = n->get_kernel_impl_params();
        auto shape_type = ImplementationManager::get_shape_type(*params);
        if (auto factory = n->type()->choose_impl(*n, *n->get_kernel_impl_params(), impl_type, shape_type)) {
            try {
                auto fmts = factory->query_formats(*n);
                for (size_t i = 0; i < fmts.first.size(); i++) {
                    n->set_preferred_input_fmt(i, fmts.first[i]);
                }
                for (size_t i = 0; i < fmts.second.size(); i++) {
                    n->set_preferred_output_fmt(i, fmts.second[i]);
                }

                if (factory->get_impl_type() == impl_types::onednn && (n->is_type<convolution>() || n->is_type<deconvolution>())) {
                    optimize_conv_permute(*n);
                }
            } catch (std::exception& exception) {
                GPU_DEBUG_INFO << "WARNING(select_preferred_formats): " << exception.what() << std::endl;
            }
            print_selected_formats(*n);
        }
    }
}
