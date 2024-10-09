// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/group_normalization.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/group_normalization_ref.hpp"
    #include "impls/ocl/group_normalization_bfyx_opt.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<group_normalization>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GroupNormalizationBfyxOpt, shape_types::any)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GroupNormalizationRef, shape_types::static_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
