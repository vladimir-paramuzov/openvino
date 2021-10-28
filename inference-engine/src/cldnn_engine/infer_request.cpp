// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <description_buffer.hpp>
#include "infer_request.hpp"
#include "cldnn_remote_context.h"
#include "cldnn_executable_network.h"
#include "cldnn_itt.h"
#include "cldnn/runtime/debug_configuration.hpp"
#include <ie_algorithm.hpp>
#include <debug.h>

using namespace InferenceEngine;

namespace gpu {

// ----------------------------------------------------------------------------------------- //
// ---------------------------- IE API impl ------------------------------------------------ //
// ----------------------------------------------------------------------------------------- //
Blob::Ptr InferRequest::GetBlob(const std::string& name) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::GetBlob");
    Blob::Ptr blob;

    // Maybe SetBlob can be reused here? Like
    // if (not allocated)
    //    blob = allocate();
    // SetBlob(name, blob);
    // return blob

    if (is_input(name)) {
        if (_inputs.find(name) == _inputs.end()) {
            // create blob for input
            // _inputs[name] = something
        }
        blob = _inputs[name];
    } else {
        if (_outputs.find(name) == _outputs.end()) {
            // create blob for output
            // _output[name] = something
        }
        blob = _outputs[name];
    }
    return blob;
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::SetBlob");

    check_blob(name, data);

    auto remote_ptr = data->as<InferenceEngine::gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;
    if (is_remote) {
        auto impl = CLDNNPlugin::getBlobImpl(remote_ptr);
        if (!impl->is_allocated()) {
            impl->allocate();
        }
    }

    if (is_input(name)) {
        _inputs[name] = data;
        _device_inputs[name] = get_device_memory_for_blob(data);
    } else {
        _outputs[name] = data;
        _device_outputs[name] = get_device_memory_for_blob(data);
    }
}

void InferRequest::InferImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::InferImpl");
    enqueue();
    wait();
}

InferRequest::InferRequest(const std::vector<std::shared_ptr<const ov::Node>>& inputs,
                           const std::vector<std::shared_ptr<const ov::Node>>& outputs,
                           const CLDNNPlugin::CLDNNExecNetwork::Ptr& execNetwork,
                           std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph)
        : IInferRequestInternal(inputs, outputs)
        , m_useProfiling(false)
        , m_useStreams(false)
        , m_useExternalQueue(false)
        , m_graph(graph) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());

    if (m_graph == nullptr) {
        IE_THROW(NetworkNotLoaded) << "null graph pointer is assigned to InferRequest instance";
    }

    m_network = m_graph->GetNetwork();
}

// ----------------------------------------------------------------------------------------- //
// ------------------------------------- pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //

void InferRequest::enqueue() {
    m_graph->wait(CLDNNPlugin::CLDNNGraph::Stage::EXECUTE);
    int streamID = 0;
    auto& streamGraphs = static_cast<CLDNNPlugin::CLDNNExecNetwork*>(_exeNetwork.get())->m_graphs;
    if (nullptr != streamExecutor) {
        streamID = streamExecutor->GetStreamId();
        int numGraphs = streamGraphs.size();
        streamID = streamID % numGraphs;
    }
    m_graph = streamGraphs[streamID];
    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;
    for (auto& item : _inputs) {
        std::string inputName = item.first;
        Blob::Ptr& inputBlob = item.second;
        prepare_input(inputName, inputBlob, dependencies);
        // if (copy_needed)
        // copy_input();
    }

    for (auto& item : _outputs) {
        std::string outputName = item.first;
        Blob::Ptr& outputBlob = item.second;
        prepare_output(outputName, outputBlob);
    }

    internal_outputs.clear();
    internal_outputs = m_network->execute(dependencies);

    // for (auto& item : _outputs) {
    //     std::string outputName = item.first;
    //     Blob::Ptr& outputBlob = item.second;
    //     if (copy_needed)
    //          copy_output();
    // }
}

void InferRequest::wait() {
    if (internal_outputs.empty()) {
        IE_THROW() << "Inference was not started!\n";
    }

    for (auto& no : _networkOutputs) {
        Blob::Ptr bptr = _outputs[no.first];
        std::string outputID = outputs_map.at(no.first);
        internal_outputs.at(outputID).get_event()->wait();
        // TODO: wait for copy_output events as well
    }

    // finally collect profiling info
    if (m_useProfiling) {
        m_graph->UpdatePerfStatistics();
    }
    m_graph->notify(CLDNNPlugin::CLDNNGraph::Stage::EXECUTE);
}

// ----------------------------------------------------------------------------------------- //
// ---------------------------- internal utils --------------------------------------------- //
// ----------------------------------------------------------------------------------------- //

bool InferRequest::is_input(std::string name) const {
    InputInfo::Ptr foundInput;
    DataPtr foundOutput;

    return findInputAndOutputBlobByName(name, foundInput, foundOutput);
}

void InferRequest::check_blob(std::string name, const InferenceEngine::Blob::Ptr &blob) const {
    if (name.empty()) {
        IE_THROW(NotFound) << "Blob name can't be empty";
    }

    if (!blob)
        IE_THROW(NotAllocated) << "Blob with name: \'" << name << "\' is nullptr";

    if (blob->size() == 0) {
        IE_THROW() << "Blob with name: \'" << name << "\' is empty";
    }

    if (std::dynamic_pointer_cast<RemoteBlob>(blob) == nullptr) {
        if (blob->buffer() == nullptr)
            IE_THROW(NotAllocated) << "Blob with name: \'" << name << "\' has invalid buffer";
    }

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;

    bool is_input = findInputAndOutputBlobByName(name, foundInput, foundOutput);

    const TensorDesc& desc = is_input
        ? foundInput->getTensorDesc()
        : foundOutput->getTensorDesc();

    if (desc.getPrecision() != blob->getTensorDesc().getPrecision()) {
        IE_THROW(ParameterMismatch) << (is_input ? "Input" : "Output") << " blob precision doesn't match corresponding "
                                    << (is_input ? "Parameter" : "Result") << " precision";
    }
}

cldnn::memory::ptr InferRequest::get_device_memory_for_blob(const InferenceEngine::Blob::Ptr& blob) {
    auto remote_ptr = blob->as<InferenceEngine::gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;
    cldnn::memory::ptr res = nullptr;
    if (is_remote) {
        // TODO: extract handle from remote blob and create cldnn::memory on top of it
        // res = engine.shared_buffer/usm(...);
    } else {
        // TODO: allocate device mem
        // res = engine.allocate_memory(...);
    }

    return res;
}

Blob::Ptr InferRequest::create_host_blob(const TensorDesc& desc) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::create_host_blob");
    const Precision& p = desc.getPrecision();
    switch (p) {
    case Precision::FP32: return make_shared_blob<float>(desc);
    case Precision::FP16: return make_shared_blob<uint16_t>(desc);
    case Precision::I16: return make_shared_blob<int16_t>(desc);
    case Precision::U16: return make_shared_blob<uint16_t>(desc);
    case Precision::I32: return make_shared_blob<int32_t>(desc);
    case Precision::I64: return make_shared_blob<int64_t>(desc);
    case Precision::I8: return make_shared_blob<int8_t>(desc);
    case Precision::U8: return make_shared_blob<uint8_t>(desc);
    case Precision::BOOL: return make_shared_blob<uint8_t>(desc);
    default: IE_THROW(NotImplemented) << "The plugin does not support " << p.name() << " blob precision";
    }
}

Blob::Ptr InferRequest::create_host_blob(const TensorDesc& desc, void* mem_ptr) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::create_host_blob");
    const Precision& p = desc.getPrecision();

    if (!mem_ptr)
        return create_host_blob(desc);

    switch (p) {
    case Precision::FP32: return make_shared_blob<float>(desc, reinterpret_cast<float*>(mem_ptr));
    case Precision::FP16: return make_shared_blob<uint16_t>(desc, reinterpret_cast<uint16_t*>(mem_ptr));
    case Precision::I16: return make_shared_blob<int16_t>(desc, reinterpret_cast<int16_t*>(mem_ptr));
    case Precision::U16: return make_shared_blob<uint16_t>(desc, reinterpret_cast<uint16_t*>(mem_ptr));
    case Precision::I32: return make_shared_blob<int32_t>(desc, reinterpret_cast<int32_t*>(mem_ptr));
    case Precision::I64: return make_shared_blob<int64_t>(desc, reinterpret_cast<int64_t*>(mem_ptr));
    case Precision::I8: return make_shared_blob<int8_t>(desc, reinterpret_cast<int8_t*>(mem_ptr));
    case Precision::U8: return make_shared_blob<uint8_t>(desc, reinterpret_cast<uint8_t*>(mem_ptr));
    case Precision::BOOL: return make_shared_blob<uint8_t>(desc, reinterpret_cast<uint8_t*>(mem_ptr));
    default: IE_THROW(NotImplemented) << "The plugin does not support " << p.name() << " blob precision";
    }
}

void InferRequest::copy_input_data(std::shared_ptr<cldnn::network> network,
                                        const cldnn::primitive_id &inputName,
                                        const cldnn::layout& inputLayout,
                                        const Blob &inputBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::copy_input_data");

    cldnn::primitive_id internalName = "parameter:" + inputName;
    auto locked = inputBlob.cbuffer();
    auto ptr = locked.as<void*>();
    network->set_input_data(internalName, network->get_engine().attach_memory(inputLayout, ptr));
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::GetPerformanceCounts");
    if (!m_useProfiling) {
        IE_THROW() << "Performance counters were not enabled";
    } else {
        return m_graph->GetPerformanceCounts();
    }
}

void InferRequest::prepare_input(const cldnn::primitive_id& inputName, Blob::Ptr& inputBlob,
                                 std::vector<cldnn::event::ptr>& dependencies) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::prepare_input");
    auto input_mem = _device_inputs.at(inputName);
    cldnn::primitive_id internalName = "parameter:" + inputName;
    m_network->set_input_data(internalName, input_mem);
}

void InferRequest::prepare_output(const cldnn::primitive_id& outputName, Blob::Ptr& outputBlob) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::prepare_output");
    auto output_mem = _device_outputs.at(outputName);
    cldnn::primitive_id internalName = outputs_map[outputName];
    m_network->set_output_memory(internalName, output_mem);
}

InferenceEngine::Blob::Ptr InferRequest::create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout) {
    auto blobPtr = std::make_shared<CLDNNPlugin::CLDNNRemoteCLbuffer>(m_graph->GetContext(), m_network->get_stream(), desc, layout);
    getBlobImpl(blobPtr.get())->allocate();
    return blobPtr;
}

}  // namespace gpu
