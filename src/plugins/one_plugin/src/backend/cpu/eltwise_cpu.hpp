// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/implementation_params.hpp"
#include "extension/op_implementation.hpp"

namespace ov {
namespace cpu {

class SomeEltwiseCPUImpl : public OpImplementation {
public:
    SomeEltwiseCPUImpl() : OpImplementation("SomeEltwiseCPUImpl") {}

    OpExecutor::Ptr get_executor(const ImplementationParameters* params) const override;
    bool supports(const ImplementationParameters* params) const override;
};

}  // namespace cpu
}  // namespace ov
