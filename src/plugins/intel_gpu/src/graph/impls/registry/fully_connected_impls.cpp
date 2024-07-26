// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/fully_connected.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/fully_connected_onednn.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fully_connected>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::FullyConnectedImplementationManager),
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::static_shape),
        OV_GPU_GET_INSTANCE_OCL(fully_connected, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
