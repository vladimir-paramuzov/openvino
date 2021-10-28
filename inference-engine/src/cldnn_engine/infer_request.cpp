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

    InputInfo::Ptr foundInput;
    DataPtr foundOutput;

    bool is_input = findInputAndOutputBlobByName(name, foundInput, foundOutput);

    if (is_input) {
        get_or_alloc_input(name, foundInput);
        blob = _inputs[name];
    } else {
        get_or_alloc_output(name, foundOutput);
        blob = _outputs[name];
    }
    return blob;
}

void InferRequest::SetBlob(const std::string& name, const Blob::Ptr& data) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::SetBlob");

    check_blob(name, data);

    auto remote_ptr = data->as<InferenceEngine::gpu::ClBlob>();
    bool is_remote = remote_ptr != nullptr;
    cldnn::memory::ptr dev_ptr = nullptr;
    if (is_remote) {
        auto impl = CLDNNPlugin::getBlobImpl(remote_ptr);
        if (!impl->is_allocated()) {
            impl->allocate();
        }
        dev_ptr = impl->getMemory();
    }

    if (is_input(name)) {
        create_device_input(name, data->getTensorDesc().getPrecision(), dev_ptr);
        _inputs[name] = data;
    } else {
        create_device_output(name, dev_ptr);
        _outputs[name] = data;
    }
}

void InferRequest::InferImpl() {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::InferImpl");
    enqueue();
    wait();
}

InferRequest::InferRequest(InputsDataMap networkInputs, OutputsDataMap networkOutputs,
                           const CLDNNPlugin::CLDNNExecNetwork::Ptr& execNetwork,
                           std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph)
        : IInferRequestInternal(networkInputs, networkOutputs)
        , m_useProfiling(false)
        , m_useStreams(false)
        , m_useExternalQueue(false) {
    IE_ASSERT(nullptr != execNetwork);
    streamExecutor = dynamic_cast<InferenceEngine::IStreamsExecutor*>(execNetwork->m_taskExecutor.get());
    setGraph(graph);
}

// ----------------------------------------------------------------------------------------- //
// ------------------------------------- pipeline stages ----------------------------------- //
// ----------------------------------------------------------------------------------------- //

void InferRequest::enqueue() {
    int streamID = 0;
    auto& streamGraphs = static_cast<CLDNNPlugin::CLDNNExecNetwork*>(_exeNetwork.get())->m_graphs;
    if (nullptr != streamExecutor) {
        streamID = streamExecutor->GetStreamId();
        int numGraphs = streamGraphs.size();
        streamID = streamID % numGraphs;
    }
    setGraph(streamGraphs[streamID]);
    m_graph->wait(CLDNNPlugin::CLDNNGraph::Stage::EXECUTE);

    // here we need some check that everythin is allocated properly.
    for (auto& item : _networkInputs) {
        get_or_alloc_input(item.first, item.second);
    }
    for (auto& item : _networkOutputs) {
        get_or_alloc_output(item.first, item.second);
    }

    // set input and output memory from request blob maps
    // into the network object primitives
    std::vector<cldnn::event::ptr> dependencies;
    for (auto& item : _inputs) {
        std::string inputName = item.first;
        Blob::Ptr& inputBlob = item.second;
        prepare_input(inputName);
        if (true /*copy_needed*/)
            dependencies.push_back(copy_input_data(inputBlob, _device_inputs[inputName]));
    }

    for (auto& item : _outputs) {
        prepare_output(item.first);
    }

    m_result_events.clear();

    m_network->execute(dependencies);

    for (auto& item : _outputs) {
        std::string outputName = item.first;
        Blob::Ptr& outputBlob = item.second;
        m_result_events.push_back(m_network->get_primitive_event(outputs_map.at(outputName)));
        if (1 /*copy_needed*/)
            m_result_events.push_back(copy_output_data(_device_outputs[outputName], outputBlob));
    }
}

void InferRequest::wait() {
    for (auto& e : m_result_events) {
        e->wait();
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
void InferRequest::setGraph(std::shared_ptr<CLDNNPlugin::CLDNNGraph> graph) {
    m_graph = graph;
    if (m_graph == nullptr) {
        IE_THROW(NetworkNotLoaded) << "null graph pointer is assigned to InferRequest instance";
    }
    m_network = m_graph->GetNetwork();
}

bool InferRequest::is_input(const std::string& name) const {
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

void InferRequest::create_device_input(const std::string& name, const Precision& prec, cldnn::memory::ptr mem) {
    auto& inputLayouts = m_graph->GetInputLayouts();
    auto litr = inputLayouts.find(name);
    if (litr == inputLayouts.end()) {
        IE_THROW() << "Input layout for " << name << " is not found";
    }
    auto input_layout = litr->second;
    if (prec == Precision::I16 || prec == Precision::U16) {
        input_layout.data_type = cldnn::data_types::f32;
    }
    if (mem)
        _device_inputs[name] = mem;
    else
        _device_inputs[name] = m_graph->GetEngine()->allocate_memory(input_layout);
}

void InferRequest::create_device_output(const std::string& name, cldnn::memory::ptr mem) {
    std::string outputID = m_graph->MapOutputName(name);
    const cldnn::layout output_layout = m_graph->GetNetwork()->get_output_memory(outputID)->get_layout();
    if (mem)
        _device_inputs[name] = mem;
    else
        _device_outputs[name] = m_graph->GetEngine()->allocate_memory(output_layout);
    outputs_map[name] = outputID;
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

void InferRequest::get_or_alloc_input(const cldnn::primitive_id& input_name, const InputInfo::Ptr info) {
    if (_inputs.find(input_name) == _inputs.end()) {
        const auto& desc = info->getTensorDesc();
        auto blob = create_host_blob(desc);
        blob->allocate();
        _inputs[input_name] = blob;
        create_device_input(input_name, desc.getPrecision());
    }
}

void InferRequest::get_or_alloc_output(const cldnn::primitive_id& output_name, const DataPtr info) {
    if (_outputs.find(output_name) == _outputs.end()) {
        const auto& desc = info->getTensorDesc();
        auto blob = create_host_blob(desc);
        blob->allocate();
        _outputs[output_name] = blob;
        create_device_output(output_name);
    }
}

cldnn::event::ptr InferRequest::copy_input_data(Blob::Ptr blob, cldnn::memory::ptr input_memory) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::copy_input_data");

    auto locked = blob->cbuffer();
    auto ptr = locked.as<void*>();
    auto event = input_memory->copy_from(m_network->get_stream(), ptr);
    return event;
}

cldnn::event::ptr InferRequest::copy_output_data(cldnn::memory::ptr output_memory, InferenceEngine::Blob::Ptr blob) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::copy_output_data");

    // Technically the code below is not correct, as blob gets unmapped before copy operation is finished
    // thus the pointer may become invalid
    // but this copy must happen only for host blobs, so map/unmap doesn't have any effect there and pointer remains valid.
    // TODO: consider using some better approach
    auto locked = blob->cbuffer();
    auto ptr = locked.as<void*>();
    auto event = output_memory->copy_to(m_network->get_stream(), ptr);
    return event;
}

std::map<std::string, InferenceEngineProfileInfo> InferRequest::GetPerformanceCounts() const {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::GetPerformanceCounts");
    if (!m_useProfiling) {
        IE_THROW() << "Performance counters were not enabled";
    } else {
        return m_graph->GetPerformanceCounts();
    }
}

void InferRequest::prepare_input(const cldnn::primitive_id& input_name) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::prepare_input");
    auto input_mem = _device_inputs.at(input_name);
    cldnn::primitive_id internalName = "parameter:" + input_name;
    m_network->set_input_data(internalName, input_mem);
}

void InferRequest::prepare_output(const cldnn::primitive_id& output_name) {
    OV_ITT_SCOPED_TASK(itt::domains::CLDNNPlugin, "InferRequest::prepare_output");
    auto output_mem = _device_outputs.at(output_name);
    cldnn::primitive_id internalName = outputs_map[output_name];
    m_network->set_output_memory(internalName, output_mem);
}

InferenceEngine::Blob::Ptr InferRequest::create_device_blob(const InferenceEngine::TensorDesc& desc, const cldnn::layout& layout) {
    auto blobPtr = std::make_shared<CLDNNPlugin::CLDNNRemoteCLbuffer>(m_graph->GetContext(), m_network->get_stream(), desc, layout);
    getBlobImpl(blobPtr.get())->allocate();
    return blobPtr;
}

}  // namespace gpu
