// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "openvino/core/node.hpp"

namespace ov {

struct ImplementationParameters {
    explicit ImplementationParameters(const ov::Node* node = nullptr) {}
    std::string some_parameter = "";
    virtual ~ImplementationParameters() = default;
};

}  // namespace ov
