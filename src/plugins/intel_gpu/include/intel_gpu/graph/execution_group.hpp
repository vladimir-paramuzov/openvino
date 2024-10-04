// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"

#include <vector>
#include <memory>

namespace cldnn {

class primitive_inst;

using ExecutionOrder = std::vector<std::shared_ptr<primitive_inst>>;
struct ExecutionInterval {
    size_t start = 0;
    size_t end = 0;
};

struct ExecutionGroup {
    ExecutionInterval m_interval;

    const ExecutionOrder& m_exec_order;
    engine& m_engine;
    stream::ptr m_stream;


    ExecutionGroup(const ExecutionOrder& exec_order, engine& engine, stream::ptr stream, ExecutionInterval interval)
        : m_interval(interval)
        , m_exec_order(exec_order)
        , m_engine(engine)
        , m_stream(stream) {}


    event::ptr run(const std::vector<event::ptr>& dep_events);

private:
    std::unique_ptr<command_list> m_list = nullptr;

    void build_list();
    bool requires_update();
    void mutate();
    void execute();
};

}  // namespace cldnn
