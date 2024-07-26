// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "implementation_registry.hpp"
#include "register.hpp"
#include "intel_gpu/primitives/deconvolution.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/deconvolution_onednn.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<deconvolution>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_INSTANCE_ONEDNN(onednn::DeconvolutionImplementationManager),
        OV_GPU_INSTANCE_OCL(deconvolution, shape_types::static_shape),
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
