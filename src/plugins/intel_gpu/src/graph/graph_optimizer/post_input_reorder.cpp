// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pass_manager.h"
#include "fully_connected_inst.h"
#include <memory>
#include <stdexcept>

using namespace cldnn;

/*
This pass checks if if primitive's input format matches implementation's input format
If not than required reorder is added to the network.
*/

/*
Add a reorder in between node and usr with reorder_layout as layout
*/
program_node& post_input_reorder::add_reorder(program& p,
                                              program_node* node,
                                              program_node* usr,
                                              const layout& reorder_layout) {
    auto new_reorder = std::make_shared<reorder>(node->id() + "_reorder_" + usr->id(), node->id(), reorder_layout);
    auto& new_reorder_node = p.get_or_create(new_reorder);

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
    return new_reorder_node;
}

void post_input_reorder::run(program& p) {
    auto node_itr = p.get_processing_order().begin();
    while (node_itr != p.get_processing_order().end()) {
        auto& node = *node_itr++;
        const auto impl = node->get_selected_impl();
        if (!impl)
            continue;
        // add a reorder if primitive's input format doesn't match implementation's input format
        if (node->is_type<fully_connected>()) {
            auto layout_format = impl->get_preferred_input_fmt(0);
            if (layout_format == format::any)
                continue;

            auto& input = node->get_dependencies()[0].first;
            auto input_layout = input->get_output_layout();

            if (input_layout.format != layout_format) {
                auto previous_layout = node->get_output_layout();
                layout current_layout(input_layout.get_partial_shape(),
                                      input_layout.data_type,
                                      layout_format,
                                      input_layout.data_padding);
                auto& reorder = add_reorder(p, input, node, current_layout);
                reorder.set_unique_id();
                reorder.get_output_layout(false);
                node->set_output_layout(previous_layout, false);
                reorder.set_selected_impl(reorder.type()->choose_impl(reorder));
                if (auto impl = reorder.get_selected_impl()) {
                    impl->add_to_cache(p.get_kernels_cache());
                }
            }
        }
    }
}
