// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ze_command_list.hpp"
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

void ze_command_list::close() {
    ZE_CHECK(zeCommandListClose(m_command_list));
}

void ze_command_list::reset() {
    ZE_CHECK(zeCommandListDestroy(m_command_list));
}

ze_command_list::~ze_command_list() {
    reset();
}


}  // namespace ze
}  // namespace cldnn
