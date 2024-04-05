// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace intel_gpu {

class ConvertToInternalOpset : public ov::pass::ModelPass {
public:
    OPENVINO_RTTI("ConvertToInternalOpset", "0");
    ConvertToInternalOpset(cldnn::engine& engine, ExecutionConfig& config);
    bool run_on_model(const std::shared_ptr<ov::Model>& m) override;
private:
    cldnn::engine& m_engine;
    ExecutionConfig& m_config;
};

}  // namespace intel_gpu
}  // namespace ov
