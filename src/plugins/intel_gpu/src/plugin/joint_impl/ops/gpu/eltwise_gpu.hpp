// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/implementation_params.hpp"
#include "joint_impl/op_implementation.hpp"

namespace ov {
namespace gpu {

class SomeEltwiseGPUImpl : public OpImplementation {
public:
    SomeEltwiseGPUImpl() : OpImplementation("SomeEltwiseGPUImpl", Type::OCL) {}

    OpExecutor::Ptr get_executor(const ImplementationParameters* params) const override;
    bool supports(const ImplementationParameters* params) const override;
};

}  // namespace gcpu
}  // namespace ov
