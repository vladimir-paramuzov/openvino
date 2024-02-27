// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_to_extended_opset.hpp"

#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/sink.hpp"

#include "joint_impl/extended_opset.hpp"
#include "joint_impl/node_extension.hpp"

#include <memory>

namespace {
void replace_node_unsafe(const std::shared_ptr<ov::Node>& target, const std::shared_ptr<ov::Node>& replacement) {
    OPENVINO_ASSERT(target->get_output_size() == replacement->get_output_size(),
                    "Target output size: ",
                    target->get_output_size(),
                    " must be equal replacement output size: ",
                    replacement->get_output_size());

    std::unordered_set<std::shared_ptr<ov::Node>> replacement_nodes;
    // For each of target's output O with replacement output O_rep:
    //     For each O's connected downstream input I:
    //         Change I's connected upstream output to O_rep
    for (size_t i = 0; i < target->get_output_size(); i++) {
        auto replacement_value = replacement->output(i);
        auto replacement_node = replacement_value.get_node_shared_ptr();
        if (replacement_nodes.find(replacement_node) == replacement_nodes.end()) {
            replacement_node->add_node_control_dependents(target);
            replacement_node->add_node_control_dependencies(target);
            replacement_nodes.insert(replacement_node);
        }
        target->output(i).replace(replacement_value);
    }
    target->clear_control_dependents();
    target->clear_control_dependents();
}
}  // namespace

namespace ov {
ConvertToExtendedOpset::ConvertToExtendedOpset() {
    static std::once_flag flag;
    std::call_once(flag, []() {
        OpConverter::instance().register_ops();
    });
}

bool ConvertToExtendedOpset::run_on_model(const std::shared_ptr<ov::Model>& m) {
    for (auto& op : m->get_ordered_ops()) {
        std::cerr << "Convert: " << op->get_friendly_name() << "(" << op->get_type_name() << ")\n";
        auto converted_op = OpConverter::instance().convert_to_extended_opset(op);

        OPENVINO_ASSERT(std::dynamic_pointer_cast<NodeExtension>(converted_op) != nullptr);

        replace_node_unsafe(op, converted_op);
        if (auto param = std::dynamic_pointer_cast<ov::op::v0::Parameter>(op)) {
            m->remove_parameter(param);
            m->add_parameters({std::dynamic_pointer_cast<ov::op::v0::Parameter>(converted_op)});
        } else if (auto result = std::dynamic_pointer_cast<ov::op::v0::Result>(op)) {
            m->remove_result(result);
            m->add_results({std::dynamic_pointer_cast<ov::op::v0::Result>(converted_op)});
        } else if (auto sink = std::dynamic_pointer_cast<ov::op::Sink>(op)) {
            m->remove_sink(sink);
            m->add_sinks({std::dynamic_pointer_cast<ov::op::Sink>(converted_op)});
        }
    }

    return true;
}

}  // namespace ov
