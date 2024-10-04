// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/graph/execution_group.hpp"
#include "primitive_inst.h"

namespace cldnn {
event::ptr ExecutionGroup::run(const std::vector<event::ptr>& dep_events) {
    auto rt_type = m_engine.runtime_type();
    if (rt_type == runtime_types::ocl) {
        for (size_t i = m_interval.start; i < m_interval.end; i++) {
            m_exec_order[i]->execute(dep_events);
        }
    } else if (rt_type == runtime_types::ze) {
        if (!m_list || !m_list->is_mutable()) {
            build_list();
            return execute(dep_events);
        } else {
            if (requires_update())
                mutate();
            return execute(dep_events);
        }
    }

    return nullptr;
}

void ExecutionGroup::build_list() {

}
bool ExecutionGroup::requires_update() {
    return false;
}

void ExecutionGroup::mutate() {

}
event::ptr ExecutionGroup::execute(const std::vector<event::ptr>& dep_events) {
    std::vector<event::ptr> ret_events;
    for (size_t i = m_interval.start; i < m_interval.end; i++) {
        ret_events.push_back(m_exec_order[i]->execute(dep_events));
    }

    return m_stream->enqueue_marker(ret_events);
}

}  // namespace cldnn
