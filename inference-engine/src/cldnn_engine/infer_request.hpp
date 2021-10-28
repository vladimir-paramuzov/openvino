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

    void setGraph(std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph);
    void prepare_input(const cldnn::primitive_id &input_name);
    void prepare_output(const cldnn::primitive_id& output_name);
    void get_or_alloc_input(const cldnn::primitive_id& input_name, const InferenceEngine::InputInfo::Ptr info);
    void get_or_alloc_output(const cldnn::primitive_id& output_name, const InferenceEngine::DataPtr info);

    InferenceEngine::Blob::Ptr create_host_blob(const InferenceEngine::TensorDesc& desc);
    InferenceEngine::Blob::Ptr create_host_blob(const InferenceEngine::TensorDesc& desc, void* mem_ptr);
    InferenceEngine::Blob::Ptr create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout);

    void create_device_input(const std::string& name, const InferenceEngine::Precision& prec, cldnn::memory::ptr mem = nullptr);
    void create_device_output(const std::string& name, cldnn::memory::ptr mem = nullptr);

    cldnn::event::ptr copy_output_data(cldnn::memory::ptr output_memory, InferenceEngine::Blob::Ptr blob);
    cldnn::event::ptr copy_input_data(InferenceEngine::Blob::Ptr blob, cldnn::memory::ptr input_memory);

    void check_blob(std::string name, const InferenceEngine::Blob::Ptr& blob) const;

    bool is_input(const std::string& name) const;

    std::vector<cldnn::event::ptr> m_result_events;
};

};  // namespace gpu
