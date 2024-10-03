// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "event.hpp"
#include "kernel.hpp"
#include "kernel_args.hpp"
#include "execution_config.hpp"

#include <memory>
#include <vector>


namespace cldnn {

class command_list {
public:
    using ptr = std::shared_ptr<command_list>;

    command_list() = default;
    virtual ~command_list() = default;

    virtual void start();
    virtual void close();
};

}  // namespace cldnn
