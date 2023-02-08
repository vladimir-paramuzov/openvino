// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/graph/kernels_cache.hpp"
#include <functional>
#include <memory>

namespace cldnn {

class ICompilationContext {
public:
    using Task = std::function<void(KernelsCache&)>;
    virtual void push_task(size_t key, Task&& task) = 0;
    virtual void cancel() noexcept = 0;
    virtual ~ICompilationContext() = default;

    static std::unique_ptr<ICompilationContext> create(cldnn::engine& engine, const ExecutionConfig& config, size_t program_id);
};

}  // namespace cldnn
