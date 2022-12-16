// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "utils.hpp"

#include <string>
#include <stdexcept>
#include <thread>
#include <threading/ie_cpu_streams_executor.hpp>

namespace cldnn {

/// @brief Defines available engine types
enum class engine_types : int32_t {
    ocl,
};

/// @brief Defines available runtime types
enum class runtime_types : int32_t {
    ocl,
};

}  // namespace cldnn
