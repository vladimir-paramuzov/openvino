// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////

#include "pass_manager.h"
#include "cldnn/graph/program_node.hpp"
#include "layout_optimizer.h"
#include "cldnn/graph/program.hpp"
#include "program_helpers.h"
#include "fully_connected_inst.h"

using namespace cldnn;

pre_optimize_bias::pre_optimize_bias(reorder_factory& rf_ref) : base_pass("pre_optimize_bias"), _rf(rf_ref) {}

void pre_optimize_bias::run(program& p) { run(p, _rf); }

// function which prepares given primitive for weights optimization
template <typename T>
void pre_optimize_bias::optimize_bias(T& node, reorder_factory& rf, program& p) {
    size_t weights_offset = node.get_primitive()->input_ids.size();
    size_t bias_offset = weights_offset + program_helpers::wrap_if_single(node.get_primitive()->weights).size();
    for (size_t i = bias_offset; i < node.get_dependencies().size() - node.get_fused_inputs_count(); ++i) {
        // find weights primitive with given pimitive_id and add it to weights_optimizer
        const program_node& bias = node.get_dependency(i);
        auto new_layout = layout(bias.get_output_layout().data_type,
                                 format::bfyx,
                                 { 1, static_cast<tensor::value_type>(bias.get_output_layout().count()), 1, 1 });
        auto reorder = rf.get_reorder(bias.id(),
                                      bias.get_output_layout(),
                                      new_layout);

        if (reorder.first)
            p.add_intermediate(reorder.first, node, i, !reorder.second);
    }
}
template void pre_optimize_bias::optimize_bias<convolution_node>(convolution_node& node,
                                                                 reorder_factory& rf,
                                                                 program& p);
template void pre_optimize_bias::optimize_bias<deconvolution_node>(deconvolution_node& node,
                                                                   reorder_factory& rf,
                                                                   program& p);
template void pre_optimize_bias::optimize_bias<fully_connected_node>(fully_connected_node& node,
                                                                     reorder_factory& rf,
                                                                     program& p);

void pre_optimize_bias::run(program& p, reorder_factory& rf) {
    for (auto& prim : p.get_processing_order()) {
        if (prim->type() == convolution::type_id()) {
            optimize_bias(prim->as<convolution>(), rf, p);
        } else if (prim->type() == deconvolution::type_id()) {
            optimize_bias(prim->as<deconvolution>(), rf, p);
        } else if (prim->type() == fully_connected::type_id()) {
            optimize_bias(prim->as<fully_connected>(), rf, p);
        }
    }
}
