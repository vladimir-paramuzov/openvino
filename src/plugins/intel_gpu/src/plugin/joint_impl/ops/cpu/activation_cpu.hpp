// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "joint_impl/implementation_params.hpp"
#include "joint_impl/op_implementation.hpp"

namespace ov {
namespace cpu {

class SomeActivationCPUImpl : public OpImplementation {
public:
    SomeActivationCPUImpl() : OpImplementation("SomeActivationCPUImpl") {}

    OpExecutor::Ptr get_executor(const ImplementationParameters* params) const override;
    bool supports(const ImplementationParameters* params) const override;
};

}  // namespace cpu
}  // namespace ov
