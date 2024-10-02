// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

namespace cldnn {
namespace ze {

class ze_command_list {
public:



private:
    // const ze_engine& _engine;
    mutable ze_command_list_handle_t m_command_list = 0;
    // mutable std::atomic<uint64_t> m_queue_counter{0};
    // std::atomic<uint64_t> m_last_barrier{0};
    // std::shared_ptr<ze_event> m_last_barrier_ev = nullptr;
    // ze_events_pool m_pool;
};

}  // namespace ze
}  // namespace cldnn
