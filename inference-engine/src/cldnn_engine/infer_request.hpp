// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <map>
#include <vector>
#include <memory>
#include <atomic>
#include "cldnn_graph.h"
#include <threading/ie_istreams_executor.hpp>

namespace CLDNNPlugin {
class CLDNNExecNetwork;
}  // namespace CLDNNPlugin

namespace gpu {

class InferRequest : public InferenceEngine::IInferRequestInternal {
public:
    using Ptr = std::shared_ptr<InferRequest>;
    void checkBlobs() override {}
    void InferImpl() override;

    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> GetPerformanceCounts() const override;

    InferRequest(InferenceEngine::InputsDataMap networkInputs, InferenceEngine::OutputsDataMap networkOutputs,
                 const std::shared_ptr<CLDNNPlugin::CLDNNExecNetwork>& execNetwork,
                 std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph);

    InferRequest(const InferRequest &) = delete;

    virtual ~InferRequest() = default;

    InferenceEngine::Blob::Ptr GetBlob(const std::string& name) override;
    void SetBlob(const std::string& name, const InferenceEngine::Blob::Ptr &data) override;

    void enable_profiling() { m_useProfiling = true; }
    void enable_streams() { m_useStreams = true; }
    void enable_external_queue() { m_useExternalQueue = true; }

    bool use_external_queue() const { return m_useExternalQueue; }

    void enqueue();
    void wait();

private:
    std::map<std::string, cldnn::memory::ptr> _device_inputs;
    std::map<std::string, cldnn::memory::ptr> _device_outputs;
    std::map<std::string, cldnn::primitive_id> inputs_map;
    std::map<std::string, cldnn::primitive_id> outputs_map;

    bool m_useProfiling;
    bool m_useStreams;
    bool m_useExternalQueue;
    std::shared_ptr<CLDNNPlugin::CLDNNGraph> m_graph;
    cldnn::network::ptr m_network;

    InferenceEngine::IStreamsExecutor* streamExecutor = nullptr;

    void prepare_input(const cldnn::primitive_id &inputName, InferenceEngine::Blob::Ptr &inputBlob,
                       std::vector<cldnn::event::ptr>& dependencies);
    void prepare_output(const cldnn::primitive_id& outputName, InferenceEngine::Blob::Ptr& outputBlob);

    InferenceEngine::Blob::Ptr create_host_blob(const InferenceEngine::TensorDesc& desc);
    InferenceEngine::Blob::Ptr create_host_blob(const InferenceEngine::TensorDesc& desc, void* mem_ptr);
    InferenceEngine::Blob::Ptr create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout);

    cldnn::memory::ptr get_device_memory_for_blob(const InferenceEngine::Blob::Ptr& blob);

    void copy_output_data(cldnn::memory::ptr outputMemory, InferenceEngine::Blob::Ptr bptr);
    void copy_input_data(std::shared_ptr<cldnn::network> network, const cldnn::primitive_id &inputName,
                         const cldnn::layout& inputLayout, const InferenceEngine::Blob &inputBlob);

    void check_blob(std::string name, const InferenceEngine::Blob::Ptr& blob) const;

    bool is_input(std::string name) const;

    std::map<cldnn::primitive_id, cldnn::network_output> internal_outputs;
};

};  // namespace gpu
