// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/device_config.hpp"

#include <ie_system_conf.h>
#include <sys/stat.h>

#include <gpu/gpu_config.hpp>
#include <thread>

#include "cpp_interfaces/interface/ie_internal_plugin_config.hpp"
#include "file_utils.h"
#include "ie_api.h"
#include "intel_gpu/plugin/itt.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include <openvino/util/common_util.hpp>

#ifdef _WIN32
#    include <direct.h>
#    ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#        define mkdir(dir, mode) _wmkdir(dir)
#    else
#        define mkdir(dir, mode) _mkdir(dir)
#    endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT
#endif      // _WIN32

using namespace InferenceEngine;

namespace ov {
namespace intel_gpu {

static void createDirectory(std::string _path) {
#if defined(OPENVINO_ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
    std::wstring widepath = ov::util::string_to_wstring(_path.c_str());
    const wchar_t* path = widepath.c_str();
#else
    const char* path = _path.c_str();
#endif

    auto err = mkdir(path, 0755);
    if (err != 0 && errno != EEXIST) {
        IE_THROW() << "Couldn't create directory! (err=" << err << "; errno=" << errno << ")";
    }
}

static int getNumberOfCores(const IStreamsExecutor::Config::PreferredCoreType core_type) {
    const auto total_num_cores = getNumberOfLogicalCPUCores();
    const auto total_num_big_cores = getNumberOfLogicalCPUCores(true);
    const auto total_num_little_cores = total_num_cores - total_num_big_cores;

    int num_cores = total_num_cores;
    if (core_type == IStreamsExecutor::Config::BIG) {
        num_cores = total_num_big_cores;
    } else if (core_type == IStreamsExecutor::Config::LITTLE) {
        num_cores = total_num_little_cores;
    }
    return num_cores;
}

IE_SUPPRESS_DEPRECATED_START
void Config::UpdateFromMap(const std::map<std::string, std::string>& configMap, const cldnn::device_info& info) {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Config::UpdateFromMap");
    for (auto& kvp : configMap) {
        std::string key = kvp.first;
        std::string val = kvp.second;
        const auto hints = perfHintsConfig.SupportedKeys();
        if (hints.end() != std::find(hints.begin(), hints.end(), key)) {
            perfHintsConfig.SetConfig(key, val);
        } else if (key == ov::hint::inference_precision) {
            std::stringstream ss(val);
            ss >> inference_precision;
            OPENVINO_ASSERT(inference_precision == ov::element::f16 ||
                            inference_precision == ov::element::f32 ||
                            inference_precision == ov::element::undefined,
                            "Unexpected inference precision set: ", inference_precision);
        } else if (key.compare(PluginConfigParams::KEY_PERF_COUNT) == 0 || key == ov::enable_profiling) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                useProfiling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                useProfiling = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_DYN_BATCH_ENABLED) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableDynamicBatch = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableDynamicBatch = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
            case 0:
            case 2:
                queuePriority = ov::hint::Priority::MEDIUM;
                break;
            case 1:
                queuePriority = ov::hint::Priority::LOW;
                break;
            case 3:
                queuePriority = ov::hint::Priority::HIGH;
                break;
            default:
                IE_THROW(ParameterMismatch) << "Unsupported queue priority value: " << uVal;
            }
        } else if (key == ov::intel_gpu::hint::queue_priority) {
            std::stringstream ss(val);
            ss >> queuePriority;
        } else if (key.compare(PluginConfigParams::KEY_MODEL_PRIORITY) == 0 || key == ov::hint::model_priority) {
            if (val.compare(PluginConfigParams::MODEL_PRIORITY_HIGH) == 0 ||
                val.compare(ov::util::to_string(ov::hint::Priority::HIGH)) == 0) {
                queuePriority = ov::hint::Priority::HIGH;
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::BIG;
            } else if (val.compare(PluginConfigParams::MODEL_PRIORITY_MED) == 0 ||
                       val.compare(ov::util::to_string(ov::hint::Priority::MEDIUM)) == 0) {
                queuePriority = ov::hint::Priority::MEDIUM;
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
            } else if (val.compare(PluginConfigParams::MODEL_PRIORITY_LOW) == 0 ||
                       val.compare(ov::util::to_string(ov::hint::Priority::LOW)) == 0) {
                queuePriority = ov::hint::Priority::LOW;
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::LITTLE;
            } else {
                IE_THROW() << "Not found appropriate value for config key " << PluginConfigParams::KEY_MODEL_PRIORITY
                           << ".\n";
            }
            if (getAvailableCoresTypes().size() > 1) {
                if (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::BIG ||
                    task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE) {
                    task_exec_config._streams = std::min(task_exec_config._streams,
                                                         getNumberOfCores(task_exec_config._threadPreferredCoreType));
                }
            } else {
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
                task_exec_config._streams =
                    std::min(task_exec_config._streams, static_cast<int>(std::thread::hardware_concurrency()));
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE) == 0) {
            std::stringstream ss(val);
            uint32_t uVal(0);
            ss >> uVal;
            if (ss.fail()) {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
            switch (uVal) {
            case 0:
            case 2:
                queueThrottle = ov::intel_gpu::hint::ThrottleLevel::MEDIUM;
                break;
            case 1:
                queueThrottle = ov::intel_gpu::hint::ThrottleLevel::LOW;
                break;
            case 3:
                queueThrottle = ov::intel_gpu::hint::ThrottleLevel::HIGH;
                break;
            default:
                IE_THROW(ParameterMismatch) << "Unsupported queue throttle value: " << uVal;
            }
        } else if (key == ov::intel_gpu::hint::queue_throttle) {
            std::stringstream ss(val);
            ss >> queueThrottle;
        } else if (key.compare(PluginConfigParams::KEY_CONFIG_FILE) == 0) {
            std::stringstream ss(val);
            std::istream_iterator<std::string> begin(ss);
            std::istream_iterator<std::string> end;
            std::vector<std::string> configFiles(begin, end);
            for (auto& file : configFiles) {
                CustomLayer::LoadFromFile(file, customLayers);
            }
        } else if (key.compare(PluginConfigParams::KEY_CACHE_DIR) == 0 || key == ov::cache_dir) {
            if (!val.empty()) {
                kernels_cache_dir = val;
                createDirectory(kernels_cache_dir);
            }
        } else if (key.compare(PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                exclusiveAsyncRequests = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                exclusiveAsyncRequests = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS) == 0 || key == ov::num_streams) {
            if (val.compare(PluginConfigParams::GPU_THROUGHPUT_AUTO) == 0 ||
                val.compare(ov::util::to_string(ov::streams::AUTO)) == 0) {
                throughput_streams = std::max(GetDefaultNStreamsForThroughputMode(), info.num_ccs);
            } else {
                int val_i;
                try {
                    val_i = std::stoi(val);
                } catch (const std::exception&) {
                    IE_THROW() << "Wrong value for property key " << PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS
                               << ". Expected only positive numbers (#streams) or "
                               << "PluginConfigParams::GPU_THROUGHPUT_AUTO";
                }
                if (val_i > 0)
                    throughput_streams = static_cast<uint16_t>(val_i);
            }
        } else if (key.compare(PluginConfigParams::KEY_DEVICE_ID) == 0 || key == ov::device::id) {
            // Validate if passed value is postivie number.
            try {
                int val_i = std::stoi(val);
                (void)val_i;
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << ov::device::id.name()
                           << ". DeviceIDs are only represented by positive numbers";
            }
            // Set this value.
            device_id = val;
        } else if (key.compare(PluginConfigInternalParams::KEY_LP_TRANSFORMS_MODE) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enableInt8 = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enableInt8 = false;
            } else {
                IE_THROW(NotFound) << "Unsupported property value by plugin: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS) == 0) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                nv12_two_inputs = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                nv12_two_inputs = false;
            } else {
                IE_THROW(NotFound) << "Unsupported NV12 flag value: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_MAX_NUM_THREADS) == 0 || key == ov::compilation_num_threads) {
            int max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
            try {
                int val_i = std::stoi(val);
                if (val_i <= 0 || val_i > max_threads) {
                    val_i = max_threads;
                }
                task_exec_config._streams = std::min(task_exec_config._streams, val_i);
            } catch (const std::exception&) {
                IE_THROW() << "Wrong value for property key " << GPUConfigParams::KEY_GPU_MAX_NUM_THREADS << ": " << val
                           << "\nSpecify the number of threads use for build as an integer."
                           << "\nOut of range value will be set as a default value, maximum concurrent threads.";
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING) == 0 ||
                   key == ov::intel_gpu::enable_loop_unrolling) {
            if (val.compare(PluginConfigParams::YES) == 0) {
                enable_loop_unrolling = true;
            } else if (val.compare(PluginConfigParams::NO) == 0) {
                enable_loop_unrolling = false;
            } else {
                IE_THROW(ParameterMismatch) << "Unsupported KEY_GPU_ENABLE_LOOP_UNROLLING flag value: " << val;
            }
        } else if (key.compare(GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY) == 0 ||
                   key == ov::intel_gpu::hint::host_task_priority) {
            if (val.compare(GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH) == 0 ||
                val.compare(ov::util::to_string(ov::hint::Priority::HIGH)) == 0) {
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::BIG;
            } else if (val.compare(GPUConfigParams::GPU_HOST_TASK_PRIORITY_MEDIUM) == 0 ||
                       val.compare(ov::util::to_string(ov::hint::Priority::MEDIUM)) == 0) {
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::ANY;
            } else if (val.compare(GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW) == 0 ||
                       val.compare(ov::util::to_string(ov::hint::Priority::LOW)) == 0) {
                task_exec_config._threadPreferredCoreType = IStreamsExecutor::Config::LITTLE;
            } else {
                IE_THROW(NotFound) << "Unsupported host task priority by plugin: " << val;
            }
        } else {
            IE_THROW(NotFound) << "Unsupported property key by plugin: " << key;
        }

        adjustKeyMapValues();
    }
}

void Config::adjustKeyMapValues() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "Config::AdjustKeyMapValues");
    if (useProfiling) {
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::YES;
        key_config_map[ov::enable_profiling.name()] = PluginConfigParams::YES;
    } else {
        key_config_map[PluginConfigParams::KEY_PERF_COUNT] = PluginConfigParams::NO;
        key_config_map[ov::enable_profiling.name()] = PluginConfigParams::NO;
    }

    if (exclusiveAsyncRequests)
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_EXCLUSIVE_ASYNC_REQUESTS] = PluginConfigParams::NO;

    if (enableDynamicBatch)
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::YES;
    else
        key_config_map[PluginConfigParams::KEY_DYN_BATCH_ENABLED] = PluginConfigParams::NO;

    if (nv12_two_inputs) {
        key_config_map[GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS] = PluginConfigParams::YES;
    } else {
        key_config_map[GPUConfigParams::KEY_GPU_NV12_TWO_INPUTS] = PluginConfigParams::NO;
    }

    key_config_map[ov::hint::inference_precision.name()] = inference_precision.get_type_name();

    {
        if (queuePriority == ov::hint::Priority::HIGH &&
            (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::BIG ||
             getAvailableCoresTypes().size() == 1)) {
            key_config_map[ov::hint::model_priority.name()] =
                ov::util::to_string(ov::hint::Priority::HIGH);
        } else if (queuePriority == ov::hint::Priority::LOW &&
                   (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE ||
                    getAvailableCoresTypes().size() == 1)) {
            key_config_map[ov::hint::model_priority.name()] =
                ov::util::to_string(ov::hint::Priority::LOW);
        } else if (queuePriority == ov::hint::Priority::MEDIUM &&
                   task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::ANY) {
            key_config_map[ov::hint::model_priority.name()] =
                ov::util::to_string(ov::hint::Priority::MEDIUM);
        }
    }
    {
        std::string qp = "0";
        switch (queuePriority) {
        case ov::hint::Priority::LOW:
            qp = "1";
            break;
        case ov::hint::Priority::MEDIUM:
            qp = "2";
            break;
        case ov::hint::Priority::HIGH:
            qp = "3";
            break;
        default:
            break;
        }
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_PRIORITY] = qp;
    }
    {
        key_config_map[ov::intel_gpu::hint::queue_priority.name()] = ov::util::to_string(queuePriority);
    }
    {
        std::string qt = "0";
        switch (queueThrottle) {
        case ov::intel_gpu::hint::ThrottleLevel::LOW:
            qt = "1";
            break;
        case ov::intel_gpu::hint::ThrottleLevel::MEDIUM:
            qt = "2";
            break;
        case ov::intel_gpu::hint::ThrottleLevel::HIGH:
            qt = "3";
            break;
        default:
            break;
        }
        key_config_map[GPUConfigParams::KEY_GPU_PLUGIN_THROTTLE] = qt;
    }
    {
        key_config_map[ov::intel_gpu::hint::queue_throttle.name()] = ov::util::to_string(queueThrottle);
    }
    {
        std::string hostTaskPriority;
        if (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::LITTLE)
            hostTaskPriority = ov::util::to_string(ov::hint::Priority::LOW);
        else if (task_exec_config._threadPreferredCoreType == IStreamsExecutor::Config::BIG)
            hostTaskPriority = ov::util::to_string(ov::hint::Priority::HIGH);
        else
            hostTaskPriority = ov::util::to_string(ov::hint::Priority::MEDIUM);
        key_config_map[ov::intel_gpu::hint::host_task_priority.name()] = hostTaskPriority;
    }

    key_config_map[PluginConfigParams::KEY_CACHE_DIR] = kernels_cache_dir;
    key_config_map[ov::cache_dir.name()] = kernels_cache_dir;

    key_config_map[PluginConfigParams::KEY_GPU_THROUGHPUT_STREAMS] = std::to_string(throughput_streams);
    key_config_map[ov::num_streams.name()] = std::to_string(throughput_streams);

    key_config_map[PluginConfigParams::KEY_DEVICE_ID] = device_id;
    key_config_map[ov::device::id.name()] = device_id;

    key_config_map[PluginConfigParams::KEY_CONFIG_FILE] = "";

    key_config_map[GPUConfigParams::KEY_GPU_MAX_NUM_THREADS] = std::to_string(task_exec_config._streams);
    key_config_map[ov::compilation_num_threads.name()] = std::to_string(task_exec_config._streams);

    if (enable_loop_unrolling) {
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::YES;
        key_config_map[ov::intel_gpu::enable_loop_unrolling.name()] = PluginConfigParams::YES;
    } else {
        key_config_map[GPUConfigParams::KEY_GPU_ENABLE_LOOP_UNROLLING] = PluginConfigParams::NO;
        key_config_map[ov::intel_gpu::enable_loop_unrolling.name()] = PluginConfigParams::NO;
    }

    key_config_map[PluginConfigParams::KEY_PERFORMANCE_HINT] = perfHintsConfig.ovPerfHint;
    key_config_map[ov::hint::performance_mode.name()] = perfHintsConfig.ovPerfHint;

    key_config_map[PluginConfigParams::KEY_PERFORMANCE_HINT_NUM_REQUESTS] =
        std::to_string(perfHintsConfig.ovPerfHintNumRequests);
}

bool Config::isNewApiProperty(std::string property) {
    static const std::set<std::string> new_api_keys{
        ov::intel_gpu::hint::queue_priority.name(),
        ov::intel_gpu::hint::queue_throttle.name(),
        ov::hint::inference_precision.name(),
        ov::compilation_num_threads.name(),
        ov::num_streams.name(),
    };
    return new_api_keys.find(property) != new_api_keys.end();
}

std::string Config::ConvertPropertyToLegacy(const std::string& key, const std::string& value) {
    if (key == PluginConfigParams::KEY_MODEL_PRIORITY) {
        auto priority = ov::util::from_string(value, ov::hint::model_priority);
        if (priority == ov::hint::Priority::HIGH)
            return PluginConfigParams::MODEL_PRIORITY_HIGH;
        else if (priority == ov::hint::Priority::MEDIUM)
            return PluginConfigParams::MODEL_PRIORITY_MED;
        else if (priority == ov::hint::Priority::LOW)
            return PluginConfigParams::MODEL_PRIORITY_LOW;
    } else if (key == GPUConfigParams::KEY_GPU_HOST_TASK_PRIORITY) {
        auto priority = ov::util::from_string(value, ov::intel_gpu::hint::host_task_priority);
        if (priority == ov::hint::Priority::HIGH)
            return GPUConfigParams::GPU_HOST_TASK_PRIORITY_HIGH;
        else if (priority == ov::hint::Priority::MEDIUM)
            return GPUConfigParams::GPU_HOST_TASK_PRIORITY_MEDIUM;
        else if (priority == ov::hint::Priority::LOW)
            return GPUConfigParams::GPU_HOST_TASK_PRIORITY_LOW;
    }
    IE_THROW() << "Unsupported value for legacy key : " << key;
}

void Configs::CreateConfig(std::string device_id) {
    if (configs.find(device_id) == configs.end()) {
        configs.emplace(device_id, Config(device_id));
    }
}

Config& Configs::GetConfig(std::string device_id) {
    if (device_id.empty()) {
        return GetDefaultDeviceConfig();
    }
    if (configs.find(device_id) == configs.end()) {
        IE_THROW() << "Config for device with " << device_id << " ID is not registered in GPU plugin";
    }
    return configs.find(device_id)->second;
}

Config& Configs::GetDefaultDeviceConfig() {
    return GetConfig(default_device_id);
}

IE_SUPPRESS_DEPRECATED_END

}  // namespace intel_gpu
}  // namespace ov
