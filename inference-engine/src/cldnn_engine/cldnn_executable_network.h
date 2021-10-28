// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <string>
#include <utility>
#include "ie_blob.h"
#include "cpp/ie_cnn_network.h"
#include <cpp_interfaces/impl/ie_executable_network_thread_safe_default.hpp>
#include "cldnn_graph.h"
#include "cldnn_config.h"
#include "cldnn_remote_context.h"

namespace CLDNNPlugin {

class CLDNNExecNetwork : public InferenceEngine::ExecutableNetworkThreadSafeDefault {
public:
    typedef std::shared_ptr<CLDNNExecNetwork> Ptr;

    CLDNNExecNetwork(InferenceEngine::CNNNetwork &network, std::shared_ptr<InferenceEngine::RemoteContext> context, Config config);

    std::shared_ptr<ngraph::Function> GetExecGraphInfo() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequest() override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(InferenceEngine::InputsDataMap networkInputs,
                                                                       InferenceEngine::OutputsDataMap networkOutputs) override;
    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImpl(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                                                                       const std::vector<std::shared_ptr<const ov::Node>>& outputs) override;

    InferenceEngine::IInferRequestInternal::Ptr CreateInferRequestImplLegacy(InferenceEngine::InputsDataMap networkInputs,
                                                                             InferenceEngine::OutputsDataMap networkOutputs);

    InferenceEngine::Parameter GetMetric(const std::string &name) const override;
    InferenceEngine::Parameter GetConfig(const std::string &name) const override;
    std::shared_ptr<InferenceEngine::RemoteContext> GetContext() const override;

    std::vector<std::shared_ptr<CLDNNGraph>> m_graphs;
    InferenceEngine::gpu::ClContext::Ptr m_context;
    Config m_config;
    InferenceEngine::ITaskExecutor::Ptr m_taskExecutor;
    InferenceEngine::ITaskExecutor::Ptr m_waitExecutor;

private:
    bool isNewAPI() const;
};

};  // namespace CLDNNPlugin
