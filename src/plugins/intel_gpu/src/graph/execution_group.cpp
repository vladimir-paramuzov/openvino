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
            execute();
        } else {
            if (requires_update())
                mutate();
            execute();
        }
    }

    return nullptr;
}

void ExecutionGroup::build_list() {

}
bool ExecutionGroup::requires_update() {

}
void ExecutionGroup::mutate() {

}
void ExecutionGroup::execute() {

}

}  // namespace cldnn
