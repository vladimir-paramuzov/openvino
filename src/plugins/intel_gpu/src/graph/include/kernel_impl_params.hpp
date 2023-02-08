// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/serialization/binary_buffer.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "intel_gpu/runtime/tensor.hpp"
#include "intel_gpu/primitives/primitive.hpp"

#include "fused_primitive_desc.h"

#include <cstdint>
#include <string>
#include <vector>
#include <memory>

namespace cldnn {

struct program;


struct kernel_impl_params {
    const program *prog;
    std::shared_ptr<const primitive> desc;
    size_t unique_id;
    std::vector<layout> input_layouts;
    std::vector<layout> output_layouts;
    std::vector<tensor> input_offsets;
    std::vector<cldnn::fused_primitive_desc> fused_desc;
#ifdef ENABLE_ONEDNN_FOR_GPU
    std::vector<cldnn::fused_primitive_desc_onednn> fused_desc_onednn;
#endif // ENABLE_ONEDNN_FOR_GPU

    optional_layout weights_layout = optional_layout();

    optional_layout bias_layout = optional_layout();
    optional_layout weights_zero_points_layout = optional_layout();
    optional_layout activations_zero_points_layout = optional_layout();
    optional_layout compensation_layout = optional_layout();

    std::map<size_t, memory::ptr> memory_deps = {};
    size_t primary_input_idx = 0;

    memory::ptr reordered_weights = nullptr;

    kernel_impl_params() : prog(nullptr) {}

    kernel_impl_params(program& _prog,
                       std::shared_ptr<const primitive> _desc,
                       size_t _uid,
                       const std::vector<layout>& _in_layouts,
                       const std::vector<layout>& _out_layouts,
                       const std::vector<cldnn::fused_primitive_desc>& _fused_descs)
                       : prog(&_prog)
                       , desc(_desc)
                       , unique_id(_uid)
                       , input_layouts(_in_layouts)
                       , output_layouts(_out_layouts)
                       , fused_desc(_fused_descs)
                       , primary_input_idx(0) {
    }

    layout get_input_layout(size_t idx = 0) const {
        OPENVINO_ASSERT(input_layouts.size() > idx,
                        "The size of input layouts must be greater than the requested index: ",
                        "Requested index is ", idx, ", ",
                        "but the size of input layouts is ", input_layouts.size());
        return input_layouts[idx];
    }

    layout get_non_padded_input_layout(size_t idx = 0) const {
        auto input_layout = get_input_layout(idx);
        auto result = layout({input_layout.get_partial_shape(), input_layout.data_type, input_layout.format});
        return result;
    }

    layout get_output_layout(size_t idx = 0) const {
        OPENVINO_ASSERT(output_layouts.size() > idx,
                        "The size of output layouts must be greater than the requested index: ",
                        "Requested index is ", idx, ",",
                        "but the size of output layouts is ", output_layouts.size());
        return output_layouts[idx];
    }

    bool has_fused_primitives() const { return !fused_desc.empty(); }

    layout get_fused_output_layout() const {
        if (fused_desc.empty())
            return layout(data_types::f32, format::bfyx, tensor());
        return fused_desc.back().output_layout;
    }

    template <class PType>
    std::shared_ptr<const PType> typed_desc() const { return std::static_pointer_cast<const PType>(desc); }

    void save(BinaryOutputBuffer& ob) const;
    void load(BinaryInputBuffer& ib);
    const program& get_program() const {
        OPENVINO_ASSERT(prog != nullptr, "[GPU] Program pointer in kernel_impl_params in not initialized");
        return *prog;
    }
};

}  // namespace cldnn
