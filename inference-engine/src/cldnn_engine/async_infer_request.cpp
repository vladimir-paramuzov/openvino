// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "async_infer_request.hpp"
#include "cldnn_itt.h"
#include <memory>

gpu::AsyncInferRequest::AsyncInferRequest(const gpu::InferRequest::Ptr &inferRequest,
                                          const InferenceEngine::ITaskExecutor::Ptr& taskExecutor,
                                          const InferenceEngine::ITaskExecutor::Ptr& waitExecutor,
                                          const InferenceEngine::ITaskExecutor::Ptr& callbackExecutor)
    : AsyncInferRequestThreadSafeDefault(inferRequest, taskExecutor, callbackExecutor), _inferRequest(inferRequest), _waitExecutor(waitExecutor) {
    _pipeline = {};

    if (!_inferRequest->use_external_queue()) {
        _pipeline.push_back({taskExecutor,
                    [this] {
                        OV_ITT_SCOPED_TASK(itt::domains::GPUPlugin, "AsyncInferRequest::PreprocessingAndStartPipeline");
                        _inferRequest->enqueue();
        } });
    }
    _pipeline.push_back({_waitExecutor,
                    [this] {
                        OV_ITT_SCOPED_TASK(itt::domains::GPUPlugin, "AsyncInferRequest::WaitPipeline");
                        _inferRequest->wait();
                    }});
}

void gpu::AsyncInferRequest::Infer_ThreadUnsafe() {
    if (_inferRequest->use_external_queue()) {
        _inferRequest->enqueue();
    }
    Parent::Infer_ThreadUnsafe();
}

void gpu::AsyncInferRequest::StartAsync_ThreadUnsafe() {
    if (_inferRequest->use_external_queue()) {
        _inferRequest->enqueue();
    }
    Parent::StartAsync_ThreadUnsafe();
}

gpu::AsyncInferRequest::~AsyncInferRequest() {
    StopAndWait();
}
