// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_command_list.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "ze/ze_kernel.hpp"
#include "ze/ze_memory.hpp"
#include "ze_api.h"

namespace cldnn {
namespace ze {

ze_command_list::ze_command_list(const ze_engine& engine)
    : m_engine(engine) {
}

void ze_command_list::start() {
    if (m_command_list)
        reset();

    ze_mutable_command_list_exp_desc_t mutable_list_desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_LIST_EXP_DESC, nullptr, 0 };
    ze_command_list_desc_t command_list_desc = { ZE_STRUCTURE_TYPE_COMMAND_LIST_DESC, &mutable_list_desc, 0, 0 };
    ZE_CHECK(zeCommandListCreate(m_engine.get_context(), m_engine.get_device(), &command_list_desc, &m_command_list));
}

void ze_command_list::add(kernel& k, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) {
    auto& casted = downcast<ze_kernel>(k);
    casted._cmd_id = get_command_id();
    auto& ze_handle = casted.get_handle();

    auto& lws = args_desc.workGroups.local;
    ze_group_count_t ze_lws {static_cast<uint32_t>(lws[0]), static_cast<uint32_t>(lws[1]), static_cast<uint32_t>(lws[2])};

    ZE_CHECK(zeCommandListAppendLaunchKernel(m_command_list, ze_handle, &ze_lws, nullptr, 0, nullptr));
}

void ze_command_list::close() {
    ZE_CHECK(zeCommandListClose(m_command_list));
}

void ze_command_list::reset() {
    ZE_CHECK(zeCommandListDestroy(m_command_list));
}

ze_command_list::~ze_command_list() {
    reset();
}

uint64_t ze_command_list::get_command_id() {
    if (is_mutable() && 0) {
        ze_mutable_command_exp_flags_t flags =
            ZE_MUTABLE_COMMAND_EXP_FLAG_KERNEL_ARGUMENTS |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_COUNT |
            ZE_MUTABLE_COMMAND_EXP_FLAG_GROUP_SIZE;

        ze_mutable_command_id_exp_desc_t cmd_id_desc = { ZE_STRUCTURE_TYPE_MUTABLE_COMMAND_ID_EXP_DESC, nullptr, flags };
        uint64_t cmd_id = 0;
        ZE_CHECK(zeCommandListGetNextCommandIdExp(m_command_list, &cmd_id_desc, &cmd_id));
        return cmd_id;
    } else {
        thread_local uint64_t cmd_id = 0;
        cmd_id++;

        return cmd_id;
    }
}

}  // namespace ze
}  // namespace cldnn
