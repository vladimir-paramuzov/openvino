// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "layout_assignment.hpp"
#include "gpu_opset/node_extension.hpp"

#include <memory>

namespace ov {
namespace intel_gpu {

bool LayoutAssignment::run_on_model(const std::shared_ptr<ov::Model>& model) {
    const auto& ops = model->get_ordered_ops();
    for (const auto& op : ops) {
        std::cerr << "LayoutAssignment: " << op->get_friendly_name() << " " << op->get_type_name() << std::endl;
        auto node = std::dynamic_pointer_cast<NodeExtension>(op);
        node->select_preferred_formats();

        std::cerr << node->get_memory_desc() << std::endl;
    }

    return false;
}

}  // namespace intel_gpu
}  // namespace ov
