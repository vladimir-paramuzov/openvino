// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "impls/registry/registry.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/itt.hpp"

#include "pass_manager.h"
#include "program_node.h"

#include "intel_gpu/primitives/data.hpp"
#include "intel_gpu/primitives/mutable_data.hpp"

#include <iostream>
#include <cmath>

#include "openvino/runtime/threading/cpu_streams_executor.hpp"

using namespace cldnn;

void compile_graph::run(program& p) {
    OV_ITT_SCOPED_TASK(ov::intel_gpu::itt::domains::intel_gpu_plugin, "pass::CompileGraph");
    for (auto& node : p.get_processing_order()) {
        node->set_unique_id();
        if (!node->is_type<data>()) {
            node->get_output_layout();
        }
    }

    auto task_executor = p.get_task_executor();
    auto& proc_order = p.get_processing_order();
    std::vector<ov::threading::Task> tasks;
    std::exception_ptr exception;

    for (size_t idx = 0; idx < proc_order.size(); idx++) {
        auto& node = *(std::next(proc_order.begin(), idx));

        bool can_select_impl = !node->is_type<data>() &&
                               !(node->is_type<mutable_data>() && node->get_dependencies().empty());

        if (can_select_impl) {
    //     tasks.push_back([node, &exception, change_initial_impl, original_impl_type] {
                try {
                    node->selected_impl = node->type()->choose_impl(*node);
                    std::cerr << "-----------------------------\n";
                    auto all_impls = node->type()->get_available_impl_types(*node);
                    std::cerr << node->id() << " " << node->get_primitive()->type_string()
                            << " is_dynamic = " << node->is_dynamic() << " preferred initial: " << node->get_preferred_impl_type() << std::endl;
                    std::cerr << "available:\n";
                    for (auto& impl : all_impls) {
                        std::cerr << "\t" << impl << " ";
                    }
                    std::cerr << std::endl;


                    std::cerr << " SELECTED: " << static_cast<void*>(node->selected_impl.get()) << std::endl;
                    if (node->selected_impl) {
                        auto factory = node->type()->get_best_impl(node->get_preferred_impl_type(), ImplementationManager::get_shape_type(*node));
                        std::cerr <<"\t" << node->selected_impl->get_kernel_name() << std::endl;
                        std::cerr <<"\t" << factory->get_impl_type() << std::endl;
                        std::cerr <<"\t" << factory->get_shape_type() << std::endl;

                    }
                    std::cerr << "preferred: " << node->get_preferred_impl_type() << std::endl;;

                    std::cerr << "query via new API: \n";
                    const auto& all_list = node->type()->get_all_implementations();
                    std::cerr << "ALL: \n";
                    for (auto& impl : all_list) {
                        std::cerr << "Impl! " << static_cast<void*>(impl.get()) << " " << impl->get_impl_type() << " " << impl->get_shape_type() << std::endl;
                    }

                    if (node->is_dynamic()) {
                        node->available_impls = node->type()->get_supported_implementations(*node);
                    }
                    std::cerr << "Supported: \n";
                    for (auto& impl : node->available_impls) {
                        std::cerr << "Impl! " << static_cast<void*>(impl.get()) << " " << impl->get_impl_type() << " " << impl->get_shape_type() << std::endl;
                    }
                    std::cerr << "-----------------------------\n";
                } catch(...) {
                    exception = std::current_exception();
                }
        // } );
        }
    }

    // task_executor->run_and_wait(tasks);
    // tasks.clear();

    if (exception) {
        std::rethrow_exception(exception);
    }
}
