// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/reorder_onednn.hpp"
#endif
#if OV_GPU_WITH_OCL
    #include "impls/ocl/reorder.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

static std::vector<format> supported_dyn_formats = {
    format::bfyx,
    format::bfzyx,
    format::bfwzyx,
    format::b_fs_yx_fsv16
};

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reorder>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ReorderImplementationManager, shape_types::static_shape),
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::static_shape),
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::dynamic_shape,
            [](const program_node& node) {
                const auto& in_layout = node.get_input_layout(0);
                const auto& out_layout = node.get_output_layout(0);
                if (!one_of(in_layout.format, supported_dyn_formats) || !one_of(out_layout.format, supported_dyn_formats))
                    return false;
                return true;
            }),
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::static_shape),
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::dynamic_shape),
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
