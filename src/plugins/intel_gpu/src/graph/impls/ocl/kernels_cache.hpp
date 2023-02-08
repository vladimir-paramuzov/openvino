// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/kernel.hpp"
#include "intel_gpu/runtime/execution_config.hpp"
#include "intel_gpu/graph/kernels_cache.hpp"

#include <map>
#include <mutex>
#include <vector>
#include <memory>
#include <atomic>
#include <string>
#include <set>

#include <threading/ie_cpu_streams_executor.hpp>
#include "kernels_factory.hpp"
#include "ocl/ocl_engine.hpp"

namespace cldnn {
class kernels_cache_ocl : public KernelsCache {
public:
    using source_code = std::vector<std::string>;
    struct batch_program {
        int32_t bucket_id;
        int32_t batch_id;
        size_t hash_value;
        uint32_t kernels_counter;
        source_code source;
        std::string options;
        bool dump_custom_program;
        std::map<std::string, std::string> entry_point_to_id;

        explicit batch_program(int32_t _bucket_id, int32_t _batch_id, std::string _options, const std::vector<std::string>& batch_header_str)
            : bucket_id(_bucket_id),
              batch_id(_batch_id),
              hash_value(0),
              kernels_counter(0),
              source(std::move(batch_header_str)),
              options(_options),
              dump_custom_program(false),
              entry_point_to_id({}) {
        }
    };

    struct kernel_code {
        std::shared_ptr<kernel_string> kernel_strings;
        std::string id;
        bool dump_custom_program;
        size_t hash_value;

        kernel_code(const std::shared_ptr<kernel_string>& _kernel_strings,
                    const std::string& _id,
                    bool _dump_custom_program)
            : kernel_strings(_kernel_strings),
              id(_id),
              dump_custom_program(_dump_custom_program),
              hash_value(_kernel_strings->get_hash()) {}

        bool operator == (const kernel_code& rhs) const {
            return (hash_value == rhs.hash_value);
        }
    };

    struct cmp_kernel_code {
        bool operator()(const kernel_code& x1, const kernel_code& x2) const {
            return (x1.hash_value < x2.hash_value);
        }
    };

    using kernels_code = std::set<kernel_code, cmp_kernel_code>;

private:
    static std::mutex _mutex;
    engine& _engine;
    InferenceEngine::CPUStreamsExecutor::Ptr _task_executor;
    ExecutionConfig _config;
    uint32_t _prog_id = 0;
    kernels_code _kernels_code;
    size_t _kernel_idx = 0;
    std::atomic<bool> _pending_compilation{false};
    std::map<const std::string, kernel::ptr> _kernels;

    void get_program_source(const kernels_code& kernels_source_code, std::vector<batch_program>*) const;
    void build_batch(const engine& build_engine, const batch_program& batch);

    std::string get_cache_path() const;
    bool is_cache_enabled() const;
    size_t get_max_kernels_per_batch() const;

public:
    explicit kernels_cache_ocl(engine& engine, const ExecutionConfig& config, uint32_t prog_id);
    kernel_id set_kernel_source(const std::shared_ptr<kernel_string>& kernel_string, bool dump_custom_program);
    kernel::ptr get_kernel(kernel_id id) const;

    bool validate_simple_kernel_execution(kernel::ptr kernel);

    void remove_kernel(kernel_id id) {
        _kernels.erase(id);
    }
    std::vector<kernel_id> add_kernels_source(std::vector<std::shared_ptr<kernel_string>> kernel_sources, bool dump_custom_program = false);
    void add_kernels(const std::vector<std::string>& kernel_ids, const std::vector<kernel::ptr>& kernels);

    void save(BinaryOutputBuffer& ob) const override;
    void load(BinaryInputBuffer& ib) override;

    void compile_parallel(InferenceEngine::CPUStreamsExecutor::Ptr task_executor) override;
    void compile_sequential() override;
    void reset() override;

    static std::unique_ptr<KernelsCache> create(engine& engine, const ExecutionConfig& config, uint32_t prog_id) {
        return cldnn::make_unique<kernels_cache_ocl>(engine, config, prog_id);
    }
};

}  // namespace cldnn
