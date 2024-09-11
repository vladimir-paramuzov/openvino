// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//


#include "pass_manager.h"
#include "program_node.h"
#include "convert_color_inst.h"
#include "fully_connected_inst.h"
#include "assign_inst.h"
#include "mvn_inst.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <stdexcept>

using namespace cldnn;

/*
This pass checks if data formats (layouts) of output/input in hidden layers match.
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr
*/
void add_required_reorders::add_reorder(program& p, program_node* node, program_node* usr, bool keep_original_dt) {
    layout reorder_layout = node->get_output_layout();
    reorder_layout.format = usr->get_output_layout().format;
    reorder_layout.data_type = usr->get_output_layout().data_type;

    if (keep_original_dt)
        reorder_layout.data_type = node->get_output_layout().data_type;

    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);
    new_reorder_node.set_output_layout(reorder_layout, false);

    // ToDo: add a method to program class which adds an intermediate node given a node and its user
    auto it = std::find_if(usr->get_dependencies().begin(), usr->get_dependencies().end(),
    [&](const std::pair<program_node*, int32_t>& dep) {
        return node == dep.first;
    });
    if (it == usr->get_dependencies().end()) {
        throw std::runtime_error("Inconcistency in topology description: user of a node is not present among its dependecies.");
    }
    auto idx = it - usr->get_dependencies().begin();
    if (idx < 0 || (size_t)idx >= usr->get_dependencies().size()) {
        throw std::runtime_error("Internal Error: container index out of range exception.");
    }
    p.add_intermediate(new_reorder_node, *usr, idx);
}

void add_required_reorders::run(program& p) {
    bool optimize_data = p.get_config().get_property(ov::intel_gpu::optimize_data);
    auto usr_itr = p.get_processing_order().begin();
    while (usr_itr != p.get_processing_order().end()) {
        auto& usr = *usr_itr++;
        if (usr->get_dependencies().size() == 0)
            continue;  // only nodes with dependencies
        if (usr->is_type<data>())
            continue;

        if (!usr->is_all_valid_output_layouts()) {
            usr->recalc_output_layouts(false);
        }

        // If usr is assign and input and output data types are different
        // add reorder with usr's output data type between dep and usr
        if (usr->is_type<assign>()) {
            auto& dep = usr->get_dependency(0);
            auto dep_layout = dep.get_output_layout();
            auto out_layout = usr->get_output_layout();
            bool required_reorder = out_layout.data_type != dep_layout.data_type;
            if (required_reorder) {
                auto new_reorder = std::make_shared<reorder>(dep.id() + "_reorder_" + usr->id(), dep.id(), out_layout.format, out_layout.data_type);
                auto& new_reorder_node = p.get_or_create(new_reorder);
                p.add_intermediate(new_reorder_node, *usr, dep);
                new_reorder_node.recalc_output_layouts(false);
            }
        }

        if (optimize_data) {
            auto fused_ops = usr->get_fused_primitives();
            auto out_layout = usr->get_output_layout();
            // If there is a fused reorder at the end, then we use input layout of reorder
            // as target one for fused ops, as code generator in many kernels is expecting that, not final output layout
            // However, the condition below may need some adjustment in the future, if codegen of some primitives behave differently
            if (!fused_ops.empty() && fused_ops.back().is_type<reorder>()) {
                out_layout = fused_ops.back().input_layout;
            }
            for (auto& fused_op : fused_ops) {
                // Some kernels use blocked aligned subgroup reads for a vector of elements from dependency tensor
                // In that case jitter checks that layout of input tensor from fused op is same as output layout or broadcast is possible
                // The code below is intended to insert additional reorder node for const eltwise dependency to ensure jitter can process such fusion
                if (!fused_op.is_type<eltwise>() && !(fused_op.is_type<activation>() && fused_op.total_num_deps == 2))
                    continue;

                if (!fused_op.has_outer_dep())
                    continue;
                auto dep_id = fused_op.outer_dep_start_idx;
                auto& dep = usr->get_dependency(dep_id);
                if (!dep.is_type<data>())
                    continue;

                auto dep_layout = dep.get_output_layout();

                bool valid_broadcast_case = out_layout.is_static() && dep_layout.is_static() &&
                                            (static_cast<size_t>(out_layout.feature()) == dep_layout.count() || dep_layout.count() == 1);

                bool requires_reorder = out_layout.format != dep_layout.format && !valid_broadcast_case;
                if (requires_reorder) {
                    auto new_reorder = std::make_shared<reorder>(dep.id() + "_reorder_" + usr->id(), dep.id(), out_layout.format, dep_layout.data_type);
                    auto& new_reorder_node = p.get_or_create(new_reorder);
                    p.add_intermediate(new_reorder_node, *usr, dep);
                    new_reorder_node.recalc_output_layout(false);
                }
            }
        }

        // Remove padded-inputs in spatial axes not to use ref kernel which causes huge perf drop
        if (usr->is_type<mvn>() && usr->as<mvn>().input().is_padded_spatial()) {
            auto out_layout = usr->get_output_layout();
            // Check formats of implemented opt kernels without a spatial padding support
            if (out_layout.format == format::b_fs_yx_fsv16 || out_layout.format == format::b_fs_zyx_fsv16 ||
                out_layout.format == format::bs_fs_yx_bsv32_fsv16 || out_layout.format == format::bs_fs_yx_bsv32_fsv32) {
                auto& dep = usr->as<mvn>().input();
                cldnn::layout layout_wo_padding = dep.get_output_layout();
                layout_wo_padding.data_padding = cldnn::padding{};
                auto new_reorder = std::make_shared<reorder>(dep.id() + "_no_pad_reorder", dep.id(), layout_wo_padding);
                auto& new_reorder_node = p.get_or_create(new_reorder);
                p.add_intermediate(new_reorder_node, *usr, dep);
                new_reorder_node.recalc_output_layout(false);
            }
        }

        if (usr->type()->does_an_implementation_exist(*usr)) {
            if (usr->get_preferred_impl_type() != impl_types::onednn) {
                continue;
            } else {
                // oneDNN doesn't support padded memory, so add reorder directly if needed
                for (size_t idx = 0; idx < usr->get_dependencies().size(); idx++) {
                    auto& input = usr->get_dependency(idx);
                    if (!input.is_in_data_flow() || input.is_constant())
                        continue;

                    auto& in_layout = input.get_output_layout();
                    auto& in_padding = in_layout.data_padding;
                    if (static_cast<bool>(in_padding)) {
                        bool spatial_padding = false;
                        for (size_t i = 0; i < in_layout.get_spatial_rank(); ++i) {
                            spatial_padding |= (in_padding._lower_size[2 + i] != 0);
                        }
                        for (size_t i = 0; i < in_layout.get_spatial_rank(); ++i) {
                            spatial_padding |= (in_padding._upper_size[2 + i] != 0);
                        }

                        bool feature_padding = false;
                        feature_padding |= (in_padding._lower_size[1] != 0);
                        feature_padding |= (in_padding._upper_size[1] != 0);

                        bool batch_padding = false;
                        batch_padding |= (in_padding._lower_size[0] != 0);
                        batch_padding |= (in_padding._upper_size[0] != 0);

                        if (batch_padding && !feature_padding && !spatial_padding) {
                            batch_padding = false;
                        }

                        if (spatial_padding || batch_padding) {
                            cldnn::layout layout_padding = input.get_output_layout();
                            cldnn::layout layout_wo_padding = input.get_output_layout();
                            layout_wo_padding.data_padding = cldnn::padding{};
                            layout_wo_padding.data_padding._lower_size[1] = layout_padding.data_padding._lower_size[1];
                            layout_wo_padding.data_padding._upper_size[1] = layout_padding.data_padding._upper_size[1];
                            auto new_reorder = std::make_shared<reorder>(input.id() + "_padding_reorder_" + usr->id(), input.id(), layout_wo_padding);
                            auto& new_reorder_node = p.get_or_create(new_reorder);
                            p.add_intermediate(new_reorder_node, *usr, idx);
                            new_reorder_node.recalc_output_layouts(false);
                        } else {
                            continue;
                        }
                    }
                }
                continue;
            }
        }

        bool correct_layout_selected = false;
        bool weights_data = (usr->is_type<convolution>() || usr->is_type<deconvolution>() || usr->is_type<fully_connected>());

        layout original_layout = usr->get_output_layout();

        for (auto& node : usr->get_dependencies()) {
            if (!node.first->is_in_data_flow() && !weights_data) {
                if (cldnn::format::dimension(original_layout.format) == cldnn::format::dimension(node.first->get_output_layout().format)) {
                    /*
                        ToDo: Here we should handle also the situation where primitive usr has data inputs in different
                       formats
                    */
                    layout current_layout(original_layout.get_partial_shape(),
                                          original_layout.data_type,
                                          node.first->get_output_layout().format);
                    usr->set_output_layout(current_layout, false);
                    if (usr->type()->does_possible_implementation_exist(*usr)) {
                        correct_layout_selected = true;
                        break;
                    }
                }

                OPENVINO_ASSERT(correct_layout_selected,
                                "[GPU] No layout format available for ", usr->id(),  ", impl_type: ", usr->get_preferred_impl_type(),
                                " (format: ", original_layout.format.to_string(),
                                ", data_type: ", ov::element::Type(original_layout.data_type), ") ",
                                "compatible with ", node.first->id(),
                                " (format: ", node.first->get_output_layout().format.to_string(),
                                ", data_type: ", ov::element::Type(node.first->get_output_layout().data_type), ")");
            }
        }

        if (!correct_layout_selected) {
            std::vector<cldnn::format> preferred_layout_formats;
            size_t max_in_dims = std::max(cldnn::format::dimension(original_layout.format), static_cast<size_t>(4));
            for (auto& node : usr->get_dependencies()) {
                if (format::is_weights_format(node.first->get_output_layout().format))
                    continue;
                max_in_dims = std::max(cldnn::format::dimension(node.first->get_output_layout().format), max_in_dims);
            }
            // This list of preferred layouts has been selected arbitrary due to developers' experience
            preferred_layout_formats = { cldnn::format::get_default_format(max_in_dims) };
            if (max_in_dims == 5) {
                preferred_layout_formats.push_back(cldnn::format::bzyxf);
            } else if (max_in_dims == 4) {
                preferred_layout_formats.push_back(cldnn::format::yxfb);
                preferred_layout_formats.push_back(cldnn::format::byxf);
            }

            if (original_layout.is_dynamic() && usr->type()->does_dynamic_implementation_exist(*usr)) {
                correct_layout_selected = true;
            }

            if (usr->get_preferred_impl_type() == impl_types::onednn) {
                usr->set_preferred_impl_type(impl_types::ocl);
                usr->set_output_layout(original_layout, false);
                if (usr->type()->does_possible_implementation_exist(*usr)) {
                    correct_layout_selected = true;
                }
            }

            if (!correct_layout_selected) {
                for (auto new_layout_format : preferred_layout_formats) {
                    layout current_layout(original_layout.get_partial_shape(), original_layout.data_type, new_layout_format);
                    usr->set_output_layout(current_layout, false);
                    if (usr->type()->does_possible_implementation_exist(*usr)) {
                        correct_layout_selected = true;
                        break;
                    }
                }
            }
        }

        // layout is selected now add required reorders
        auto dep_itr = usr->get_dependencies().begin();
        while (dep_itr != usr->get_dependencies().end()) {
            auto node = *dep_itr++;
            // do not add a reorder if usr or node are reorders or does not belong to data_flow
            if (!usr->is_type<reorder>() && node.first->is_in_data_flow()) {
                if (usr->is_type<convert_color>()) {
                    auto reorder_prim = node.first->as<reorder>().get_primitive();
                    if (reorder_prim->has_surface_input())
                        continue;
                }

                if (usr->get_output_layout() != node.first->get_output_layout()) {
                    // Preserve original data type to prevent Convolution input data type from changing
                    // in the following sequence: Node(U8, unsupported format) -> Conv(FP16, bfyx).
                    // Without this condition, inserted reorder will change Conv's input to FP16, instead of
                    // expected U8 format.
                    bool keep_original_dt = false;
                    if (usr->is_type<convolution>())
                        keep_original_dt = true;
                    add_reorder(p, node.first, usr, keep_original_dt);
                }
            }
        }
    }
}
