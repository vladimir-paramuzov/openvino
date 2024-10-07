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
class kernel;

class command_list {
public:
    using ptr = std::shared_ptr<command_list>;

    command_list() = default;
    virtual ~command_list() = default;

    virtual void start() = 0;
    virtual void close() = 0;

    virtual void add(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) = 0;

    bool is_mutable() { return true; }
};

}  // namespace cldnn
