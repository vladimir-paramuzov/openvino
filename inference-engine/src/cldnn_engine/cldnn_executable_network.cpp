// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_metric_helpers.hpp"
#include "ie_icore.hpp"
#include <chrono>
#include <cmath>
#include <algorithm>
#include "cldnn_graph.h"
#include "cldnn_itt.h"

#include <description_buffer.hpp>
#include "cldnn_infer_request.h"
#include <threading/ie_executor_manager.hpp>
#include "cldnn_async_infer_request.h"
#include "async_infer_request.hpp"
#include <fstream>
#include <utility>
#include <sys/types.h>

#include "cldnn_executable_network.h"
#include "threading/ie_cpu_streams_executor.hpp"
#include "cpp_interfaces/interface/ie_iinfer_request_internal.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::details;

namespace CLDNNPlugin {

CLDNNExecNetwork::CLDNNExecNetwork(InferenceEngine::CNNNetwork &network, std::shared_ptr<RemoteContext> context, Config config) :
    InferenceEngine::ExecutableNetworkThreadSafeDefault{[&]()->InferenceEngine::ITaskExecutor::Ptr {
        if (config.exclusiveAsyncRequests) {
            //exclusiveAsyncRequests essentially disables the streams (and hence should be checked first) => aligned with the CPU behavior
            return ExecutorManager::getInstance()->getExecutor("GPU");
        }  else if (config.throughput_streams > 1) {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"CLDNNPlugin executor", config.throughput_streams});
        } else {
            return std::make_shared<InferenceEngine::CPUStreamsExecutor>(
                IStreamsExecutor::Config{"CLDNNPlugin executor", 1});
        }
    }()},
    m_config(config),
    m_taskExecutor{ _taskExecutor },
    m_waitExecutor(InferenceEngine::ExecutorManager::getInstance()->getIdleCPUStreamsExecutor({ "GPUWaitExecutor" })) {
    auto casted_context = std::dynamic_pointer_cast<InferenceEngine::gpu::ClContext>(context);

    if (nullptr == casted_context) {
        IE_THROW() << "Invalid remote context";
    }

    m_context = casted_context;

    auto graph_base = std::make_shared<CLDNNGraph>(network, m_context, m_config, 0);
    for (uint16_t n = 0; n < m_config.throughput_streams; n++) {
        auto graph = n == 0 ? graph_base : std::make_shared<CLDNNGraph>(graph_base, n);
        m_graphs.push_back(graph);
    }
}

IInferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequestImpl(InputsDataMap networkInputs,
                                                                    OutputsDataMap networkOutputs) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecNetwork::CreateInferRequestImplLegacy");
    if (m_graphs.empty()) {
        IE_THROW(NetworkNotLoaded);
    }

    for (auto& graph : m_graphs) {
        if (graph == nullptr) {
            IE_THROW(NetworkNotLoaded);
        }

        if (!graph->IsLoaded()) {
            IE_THROW(NetworkNotLoaded) << ": no networks created";
        }
    }

    if (!isNewAPI()) {
        return CreateInferRequestImplLegacy(networkInputs, networkOutputs);
    }

    auto ptr = std::make_shared<::gpu::InferRequest>(networkInputs, networkOutputs,
                                                     std::static_pointer_cast<CLDNNExecNetwork>(shared_from_this()),
                                                     m_graphs.front());
    if (m_config.throughput_streams > 1) {
        ptr->enable_streams();
    }
    if (m_config.useProfiling)
        ptr->enable_profiling();

    if (m_graphs.front()->use_external_queue()) {
        ptr->enable_external_queue();
    }

    return ptr;
}

IInferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequestImplLegacy(InputsDataMap networkInputs,
                                                                          OutputsDataMap networkOutputs) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecNetwork::CreateInferRequestImplLegacy");
    auto ptr = std::make_shared<CLDNNInferRequest>(networkInputs, networkOutputs,
                                                   std::static_pointer_cast<CLDNNExecNetwork>(shared_from_this()));
    if (m_config.throughput_streams > 1) {
        ptr->EnableStreams();
    }
    if (m_config.useProfiling)
        ptr->EnableProfiling();

    if (m_graphs.front()->use_external_queue()) {
        ptr->enable_external_queue();
    }
    ptr->SetGraph(m_graphs.front());

    return ptr;
}

IInferRequestInternal::Ptr CLDNNExecNetwork::CreateInferRequest() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "CLDNNExecNetwork::CreateInferRequest");
    if (isNewAPI()) {
        auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
        internalRequest->setPointerToExecutableNetworkInternal(shared_from_this());
        return std::make_shared<::gpu::AsyncInferRequest>(std::static_pointer_cast<::gpu::InferRequest>(internalRequest),
                                                        m_taskExecutor,
                                                        m_waitExecutor,
                                                        _callbackExecutor);
    } else {
        auto internalRequest = CreateInferRequestImpl(_networkInputs, _networkOutputs);
        internalRequest->setPointerToExecutableNetworkInternal(shared_from_this());
        return std::make_shared<CLDNNAsyncInferRequest>(std::static_pointer_cast<CLDNNInferRequest>(internalRequest),
                                                        m_taskExecutor,
                                                        m_waitExecutor,
                                                        _callbackExecutor);
    }
}

std::shared_ptr<ngraph::Function> CLDNNExecNetwork::GetExecGraphInfo() {
    if (m_graphs.empty())
        IE_THROW(NetworkNotLoaded);

    return m_graphs.front()->GetExecGraphInfo();
}

InferenceEngine::Parameter CLDNNExecNetwork::GetConfig(const std::string &name) const {
    auto it = m_config.key_config_map.find(name);
    if (it != m_config.key_config_map.end()) {
        return it->second;
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork config key: " << name;
    }
}

InferenceEngine::Parameter CLDNNExecNetwork::GetMetric(const std::string &name) const {
    if (name == METRIC_KEY(NETWORK_NAME)) {
        IE_ASSERT(!m_graphs.empty());
        IE_SET_METRIC_RETURN(NETWORK_NAME, m_graphs[0]->getName());
    } else if (name == METRIC_KEY(SUPPORTED_METRICS)) {
        std::vector<std::string> metrics;
        metrics.push_back(METRIC_KEY(NETWORK_NAME));
        metrics.push_back(METRIC_KEY(SUPPORTED_METRICS));
        metrics.push_back(METRIC_KEY(SUPPORTED_CONFIG_KEYS));
        metrics.push_back(METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS));
        metrics.push_back(GPU_METRIC_KEY(MEMORY_STATISTICS));
        IE_SET_METRIC_RETURN(SUPPORTED_METRICS, metrics);
    } else if (name == METRIC_KEY(SUPPORTED_CONFIG_KEYS)) {
        std::vector<std::string> configKeys;
        for (auto && value : m_config.key_config_map)
            configKeys.push_back(value.first);
        IE_SET_METRIC_RETURN(SUPPORTED_CONFIG_KEYS, configKeys);
    } else if (name == METRIC_KEY(OPTIMAL_NUMBER_OF_INFER_REQUESTS)) {
        unsigned int nr = m_config.throughput_streams;
        if (m_config.perfHintsConfig.ovPerfHint != CONFIG_VALUE(LATENCY))
            nr *= 2;
        IE_SET_METRIC_RETURN(OPTIMAL_NUMBER_OF_INFER_REQUESTS, nr);
    } else if (name == GPU_METRIC_KEY(MEMORY_STATISTICS)) {
        std::map<std::string, uint64_t> statistics;
        if (m_context != nullptr) {
            auto impl = getContextImpl(m_context);
            impl->acquire_lock();
            std::shared_ptr<cldnn::engine> eng = impl->GetEngine();
            eng->get_memory_statistics(&statistics);
            impl->release_lock();
        }
        IE_SET_METRIC_RETURN(GPU_MEMORY_STATISTICS, statistics);
    } else {
        IE_THROW() << "Unsupported ExecutableNetwork metric: " << name;
    }
}

std::shared_ptr<RemoteContext> CLDNNExecNetwork::GetContext() const {
    return m_context;
}

bool CLDNNExecNetwork::isNewAPI() const {
    return _plugin->GetCore()->isNewAPI();
}

};  // namespace CLDNNPlugin
