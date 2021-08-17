// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"

#ifdef GPU_ENABLE_ZE_BACKEND
#include "ze/ze_device_detector.hpp"
#endif  // GPU_ENABLE_ZE_BACKEND

#include <map>
#include <string>

namespace cldnn {

device_query::device_query(engine_types engine_type, runtime_types runtime_type, void* user_context, void* user_device) {
    switch (engine_type) {
    case engine_types::ocl: {
        if (runtime_type != runtime_types::ocl)
            throw std::runtime_error("Unsupported runtime type for ocl engine");

        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device);
        break;
    }
#ifdef GPU_ENABLE_ZE_BACKEND
    case engine_types::ze: {
        if (runtime_type != runtime_types::ze)
            throw std::runtime_error("Unsupported runtime type for ze engine");

        ze::ze_device_detector ze_detector;
        _available_devices = ze_detector.get_available_devices(user_context, user_device);
        break;
    }
#endif  // GPU_ENABLE_ZE_BACKEND
    default: throw std::runtime_error("Unsupported engine type in device_query");
    }

    if (_available_devices.empty()) {
        throw std::runtime_error("No suitable devices found for requested engine and runtime types");
    }
}
}  // namespace cldnn
