// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <cpp_interfaces/impl/ie_infer_async_request_thread_safe_default.hpp>
#include "infer_request.hpp"

namespace gpu {

class AsyncInferRequest : public InferenceEngine::AsyncInferRequestThreadSafeDefault {
public:
    using Parent = InferenceEngine::AsyncInferRequestThreadSafeDefault;
    AsyncInferRequest(const ::gpu::InferRequest::Ptr &inferRequest,
                      const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                      const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor);

    ~AsyncInferRequest();

    void Infer_ThreadUnsafe() override;
    void StartAsync_ThreadUnsafe() override;

private:
    ::gpu::InferRequest::Ptr _inferRequest;
    InferenceEngine::ITaskExecutor::Ptr _waitExecutor;
};

}  // namespace CLDNNPlugin
