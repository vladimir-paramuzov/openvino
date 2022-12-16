// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "intel_gpu/plugin/custom_layer.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"
#include "intel_gpu/graph/network.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include <ie_performance_hints.hpp>
#include <threading/ie_cpu_streams_executor.hpp>

namespace ov {
namespace intel_gpu {


struct Config {
    Config(std::string device_id = "0") : device_id(device_id),
                                          throughput_streams(1),
                                          useProfiling(false),
                                          dumpCustomKernels(false),
                                          exclusiveAsyncRequests(false),
                                          enableDynamicBatch(false),
                                          enableInt8(true),
                                          nv12_two_inputs(false),
                                          queuePriority(cldnn::priority_mode_types::med),
                                          queueThrottle(cldnn::throttle_mode_types::med),
                                          max_dynamic_batch(1),
                                          customLayers({}),
                                          kernels_cache_dir(""),
                                          inference_precision(ov::element::undefined),
                                          task_exec_config({"GPU plugin internal task executor",                        // name
                                                    std::max(1, static_cast<int>(std::thread::hardware_concurrency())), // # of streams
                                                    1,                                                                  // # of threads per streams
                                                    InferenceEngine::IStreamsExecutor::ThreadBindingType::HYBRID_AWARE, // thread binding type
                                                    1,                                                                  // thread binding step
                                                    0,                                                                  // thread binding offset
                                                    1,                                                                  // # of threads
                                                    InferenceEngine::IStreamsExecutor::Config::ANY}),                   // preferred core type
                                          enable_loop_unrolling(true) {
        GPU_DEBUG_GET_INSTANCE(debug_config);
        GPU_DEBUG_IF(debug_config->serialize_compile == 1) {
            task_exec_config._streams = 1;
        }

        adjustKeyMapValues();
    }

    uint32_t GetDefaultNStreamsForThroughputMode() const {
        return 2;
    }
    void UpdateFromMap(const std::map<std::string, std::string>& configMap, const cldnn::device_info& info);
    void adjustKeyMapValues();
    static bool isNewApiProperty(std::string property);
    static std::string ConvertPropertyToLegacy(const std::string& key, const std::string& value);

    bool CanShareContextWith(const Config& other) const;

    std::string device_id;
    uint16_t throughput_streams;
    bool useProfiling;
    bool dumpCustomKernels;
    bool exclusiveAsyncRequests;
    bool enableDynamicBatch;
    bool enableInt8;
    bool nv12_two_inputs;
    cldnn::priority_mode_types queuePriority;
    cldnn::throttle_mode_types queueThrottle;
    int max_dynamic_batch;
    CustomLayerMap customLayers;
    std::string kernels_cache_dir;
    ov::element::Type inference_precision;
    InferenceEngine::IStreamsExecutor::Config task_exec_config;

    bool enable_loop_unrolling;

    std::map<std::string, std::string> key_config_map;
    InferenceEngine::PerfHintsConfig  perfHintsConfig;
};

struct Configs {
    using conf_iter = std::map<std::string, Config>::iterator;
    Configs(Config conf = Config()) : configs({std::make_pair(default_device_id, conf.device_id = default_device_id)}) { }

    void CreateConfig(std::string device_id);
    Config& GetConfig(std::string device_id);
    Config& GetDefaultDeviceConfig();

    void SetDefaultDeviceID(std::string default_device_id) { this->default_device_id = default_device_id; }
    std::string GetDefaultDeviceID() { return default_device_id; }

    conf_iter begin() { return configs.begin(); }
    conf_iter end() { return configs.end(); }

private:
    std::string default_device_id = "0";
    std::map<std::string, Config> configs;
};

}  // namespace intel_gpu
}  // namespace ov
