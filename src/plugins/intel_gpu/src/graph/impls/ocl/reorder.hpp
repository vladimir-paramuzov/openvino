// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/implementation_manager.hpp"
#include "intel_gpu/primitives/reorder.hpp"
#include "program_node.h"

#include <memory>
namespace cldnn {
namespace ocl {

struct ReorderImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("ReorderImplementationOCL")
    ReorderImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::ocl, shape_type) {}

    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;
    std::unique_ptr<primitive_impl> create_impl(const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reorder>());

        const auto& output_layout = node.get_output_layout(0);
        auto output_fmt = output_layout.format;
        if (output_fmt == format::custom)
            return false;

        return true;
    }

    in_out_fmts_t query_formats(const program_node&) const override { OPENVINO_NOT_IMPLEMENTED; }
    bool support_shapes(const kernel_impl_params&) const override { return true; }
};

}  // namespace ocl
}  // namespace cldnn
