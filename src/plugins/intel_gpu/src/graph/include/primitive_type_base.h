// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/layout.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include "intel_gpu/runtime/utils.hpp"
#include "primitive_type.h"
#include "program_node.h"
#include "primitive_inst.h"
#include "intel_gpu/graph/network.hpp"
#include "implementation_map.hpp"

#include <memory>
#include <string>

namespace cldnn {
template <class PType>
struct primitive_type_base : primitive_type {
    std::shared_ptr<cldnn::program_node> create_node(program& program,
                                                     const std::shared_ptr<primitive> prim) const override {
        OPENVINO_ASSERT(prim->type == this, "[GPU] primitive_type_base::create_node: primitive type mismatch");
        return std::make_shared<typed_program_node<PType>>(std::static_pointer_cast<PType>(prim), program);
    }

    std::shared_ptr<cldnn::primitive_inst> create_instance(network& network, const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::create_instance: primitive type mismatch");
        return std::make_shared<typed_primitive_inst<PType>>(network, node);
    }

    // TODO: Should we get rid of engine type in impl map? Or we must pass internal build engine to get real ocl type?
    std::unique_ptr<primitive_impl> choose_impl(const cldnn::program_node& node) const override {
        return choose_impl(node, *node.get_kernel_impl_params());
    }

    in_out_fmts_t query_preferred_formats(const cldnn::program_node& node, impl_types impl_type) const  override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::choose_impl: primitive type mismatch");
        auto shape_type = ImplementationManager::get_shape_type(node);
        if (auto factory = implementation_map<PType>::get(node.get_preferred_impl_type(), shape_type))
            return factory->query_formats(node);
        return {};
    }

    std::unique_ptr<primitive_impl> choose_impl(const cldnn::program_node& node, const kernel_impl_params& runtime_params) const override {
        try {
            OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::choose_impl: primitive type mismatch");
            auto impl_type = node.get_preferred_impl_type();
            auto shape_type = ImplementationManager::get_shape_type(runtime_params);
            if (auto factory = implementation_map<PType>::get(impl_type, shape_type))
                return factory->create(node, runtime_params);
            OPENVINO_THROW("[GPU] Could not find any implementatio with",
                           " impl_type: ", impl_type,
                           " shape_type: ", shape_type);
        } catch (std::exception& e) {
            std::stringstream ss;
            const auto& p = node.get_primitive();
            ov::write_all_to_stream(ss, "[GPU] Can't choose implementation for ", node.id(), " node (type=", p->type_string(), ")\n",
                                        "[GPU] Original name: ", p->origin_op_name, "\n"
                                        "[GPU] Original type: ", p->origin_op_type_name, "\n"
                                        "[GPU] Reason: ", e.what());
            OPENVINO_THROW(ss.str());
        }
    }

    std::map<impl_types, const ImplementationManager*> get_available_impls(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::get_available_impls: primitive type mismatch");
        auto shape_type = ImplementationManager::get_shape_type(node);
        auto all_impls = implementation_map<PType>::get_available_impls(shape_type);
        std::map<impl_types, const ImplementationManager*> supported_impls;
        for (const auto& impl : all_impls) {
            auto factory = implementation_map<PType>::get(impl, shape_type);
            if (factory->validate(node))
                supported_impls.insert({impl, factory});
        }

        return supported_impls;
    }


    bool is_node_supported(const cldnn::program_node& node) const override {
        auto shape_type = ImplementationManager::get_shape_type(node);
        return is_node_supported(node, node.get_preferred_impl_type(), shape_type);
    }

    bool is_node_supported(const cldnn::program_node& node, impl_types impl_type) const override {
        auto shape_type = ImplementationManager::get_shape_type(node);
        return is_node_supported(node, impl_type, shape_type);
    }

    bool is_node_supported(const cldnn::program_node& node, shape_types shape_type) const override {
        return is_node_supported(node, node.get_preferred_impl_type(), shape_type);
    }

    bool is_node_supported(const cldnn::program_node& node, impl_types impl_type, shape_types shape_type) const override {
        if (auto factory = implementation_map<PType>::get(impl_type, shape_type))
            return factory->validate(node);
        return false;
    }

    cldnn::layout calc_output_layout(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::calc_output_layout: primitive type mismatch");
        for (auto& t : impl_param.input_layouts) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " input tensor: " << t.to_short_string() << std::endl;
        }
        auto res = typed_primitive_inst<PType>::calc_output_layout(node, impl_param);

        GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " output tensor: " << res.to_short_string() << std::endl;
        return res;
    }

    std::vector<cldnn::layout> calc_output_layouts(const cldnn::program_node& node, const kernel_impl_params& impl_param) const override {
        OPENVINO_ASSERT(node.type() == this, "primitive_type_base::calc_output_layouts: primitive type mismatch");

        for (auto& t : impl_param.input_layouts) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " input tensor: " << t.to_short_string() << std::endl;
        }

        auto res = typed_primitive_inst<PType>::template calc_output_layouts<ov::PartialShape>(node, impl_param);

        for (auto& t : res) {
            GPU_DEBUG_TRACE_DETAIL << impl_param.desc->id << " output tensor: " << t.to_short_string() << std::endl;
        }

        return res;
    }

    kernel_impl_params get_fake_aligned_params(kernel_impl_params const& orig_impl_param) const override {
        return typed_primitive_inst<PType>::get_fake_aligned_params(orig_impl_param);
    }

    std::string to_string(const cldnn::program_node& node) const override {
        OPENVINO_ASSERT(node.type() == this, "[GPU] primitive_type_base::to_string: primitive type mismatch");
        return typed_primitive_inst<PType>::to_string(node);
    }
};

}  // namespace cldnn
