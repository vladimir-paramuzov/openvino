// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "snippets/lowered/pass/extract_loop_invariants.hpp"

#include "snippets/itt.hpp"
#include "snippets/lowered/linear_ir.hpp"
#include "snippets/snippets_isa.hpp"
#include "snippets/utils.hpp"

namespace ov {
namespace snippets {
namespace lowered {
namespace pass {
namespace {
void remove_last_loop_id(const std::shared_ptr<Expression>& expr) {
    auto loop_ids = expr->get_loop_ids();
    OPENVINO_ASSERT(!loop_ids.empty(), "Expr loop_ids should not be empty when remove last loop id.");
    loop_ids.pop_back();
    expr->set_loop_ids(loop_ids);
}

int64_t get_stride_after_move_outer(const LoopPort& loop_port) {
    const auto& expr_port = loop_port.expr_port;
    const auto& shape = expr_port->get_descriptor_ptr()->get_shape();
    size_t shape_dim_idx = utils::get_dim_idx(*expr_port, loop_port.dim_idx);
    int64_t stride = utils::get_stride(shape_dim_idx, shape);
    if (utils::is_dynamic_value(stride) || utils::is_dynamic_value(shape[shape_dim_idx])) {
        return utils::get_dynamic_value<int64_t>();
    } else {
        return stride * static_cast<int64_t>(shape[shape_dim_idx]);
    }
}

bool is_extraction_applicable(const ExpressionPtr& expr, const UnifiedLoopInfoPtr& inner_loop_info) {
    const auto& expr_input_ports = expr->get_input_ports();
    const auto& input_port_size = expr_input_ports.size();
    if (input_port_size == 0)
        return false;

    for (size_t i = 0; i < input_port_size; ++i) {
        const auto& parent = expr->get_input_port_connector(i)->get_source().get_expr();
        bool parent_scalar_with_single_consumer = ov::is_type<snippets::op::Scalar>(parent->get_node()) &&
                                                  parent->get_output_port_connector(0)->get_consumers().size() == 1;
        const auto& is_loop_port = inner_loop_info->is_loop_port(expr_input_ports[i]);
        // If expr input port is not a loop input port, then should not extract. In this case expr depend on result of another expr in inner loop,
        // i.e. move expr to top(outside) of inner loop does not keep data dependency.
        // If expr has parent scalar which has single consumer, expr and parent scalar could be extracted together. If parent scalar has multiple
        // consumers, the scalar has chance to move with other consumers, which maybe break data dependency as well.
        if (!is_loop_port && !parent_scalar_with_single_consumer) {
            return false;
        }
        if (is_loop_port) {
            // stride is not 1 after move to outside, then should not extract.
            const auto& loop_port = inner_loop_info->get_loop_port(expr_input_ports[i]);
            if (get_stride_after_move_outer(loop_port) != 1) {
                return false;
            }
        }
    }
    return true;
}

void extract_expr(const ExpressionPtr& expr, LinearIR& linear_ir,
                  LinearIR::constExprIt& inner_loop_begin_pos, const LinearIR::constExprIt& inner_loop_end_pos) {
    // update expr loop id
    remove_last_loop_id(expr);
    // move if it is not the first
    if (expr != *inner_loop_begin_pos) {
        auto port_expr_iter = std::find(inner_loop_begin_pos, inner_loop_end_pos, expr);
        OPENVINO_ASSERT(port_expr_iter != inner_loop_end_pos, "Identified extractable expr is not found in loop.");
        linear_ir.move(port_expr_iter, inner_loop_begin_pos);
    } else {
        inner_loop_begin_pos++;
    }
}

void update_loop_ports(const ExpressionPtr& expr, const LoopManagerPtr& loop_manager, size_t inner_loop_id,
                       const LinearIR::constExprIt& inner_loop_begin_pos, const LinearIR::constExprIt& inner_loop_end_pos) {
    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loop_id);
    // delete expr input ports from loop input points, add expr output ports' consumers if
    // consumed in inner loop to loop input ports.
    std::vector<ExpressionPort> new_loop_input_ports;
    for (size_t i = 0; i < expr->get_output_count(); i++) {
        const auto& consumers = expr->get_output_port_connector(i)->get_consumers();
        for (const auto& consumer : consumers) {
            const auto& loop_ids = consumer.get_expr()->get_loop_ids();
            if (std::find(loop_ids.cbegin(), loop_ids.cend(), inner_loop_id) != loop_ids.cend()) {
                new_loop_input_ports.push_back(consumer);
            }
        }
    }
    const auto& expr_input_ports = expr->get_input_ports();
    inner_loop_info->update_loop_ports(expr_input_ports, new_loop_input_ports);

    // delete expr out ports from loop out ports directly if it's in loop output ports
    std::vector<ExpressionPort> out_ports_to_delete;
    for (size_t i = 0; i < expr->get_output_count(); ++i) {
        const auto& out_port = expr->get_output_port(i);
        if (inner_loop_info->is_loop_port(out_port)) {
            out_ports_to_delete.push_back(out_port);
        }
    }
    if (!out_ports_to_delete.empty()) {
        std::vector<ExpressionPort> new_ports;
        inner_loop_info->update_loop_ports(out_ports_to_delete, new_ports);
    }
    // TODO: 142990.
    // Need sort after update loop ports. There are possibility that all exprs are moved to outer loop.
    if (!inner_loop_info->get_input_ports().empty() && !inner_loop_info->get_output_ports().empty()) {
        loop_manager->sort_loop_ports(inner_loop_begin_pos, inner_loop_end_pos, inner_loop_id);
    }
}

std::vector<ExpressionPtr> get_loop_input_exprs(const std::vector<LoopPort>& loop_in_ports) {
    std::vector<ExpressionPtr> input_exprs;
    std::unordered_set<ExpressionPtr> seen_exprs;
    for (size_t port_num = 0; port_num < loop_in_ports.size(); ++port_num) {
        const auto& expr = loop_in_ports[port_num].expr_port->get_expr();
        if (seen_exprs.count(expr) == 0) {
            input_exprs.push_back(expr);
            seen_exprs.insert(expr);
        }
    }
    return input_exprs;
}

bool extract_from_loop(const size_t& inner_loop_id, LinearIR& linear_ir) {
    const auto& loop_manager = linear_ir.get_loop_manager();
    bool status = false;
    bool continue_to_extract = true;
    const auto& inner_loop_info = loop_manager->get_loop_info<UnifiedLoopInfo>(inner_loop_id);
    while (continue_to_extract) {
        const auto& inner_loop_input_ports = inner_loop_info->get_input_ports();
        const auto& potential_extractable_exprs = get_loop_input_exprs(inner_loop_input_ports);
        bool expr_extracted = false;
        for (const auto& port_expr : potential_extractable_exprs) {
            if (is_extraction_applicable(port_expr, inner_loop_info)) {
                status = true;
                LinearIR::constExprIt inner_loop_begin_pos, inner_loop_end_pos;
                std::tie(inner_loop_begin_pos, inner_loop_end_pos) =
                    loop_manager->get_loop_bounds(linear_ir, inner_loop_id);
                // extract scalar on inputs if there are
                for (size_t i = 0; i < port_expr->get_input_count(); ++i) {
                    auto parent = port_expr->get_input_port_connector(i)->get_source().get_expr();
                    if (ov::is_type<snippets::op::Scalar>(parent->get_node())) {
                        extract_expr(parent, linear_ir, inner_loop_begin_pos, inner_loop_end_pos);
                    }
                }
                extract_expr(port_expr, linear_ir, inner_loop_begin_pos, inner_loop_end_pos);
                update_loop_ports(port_expr, loop_manager, inner_loop_id, inner_loop_begin_pos, inner_loop_end_pos);
                expr_extracted = true;
                break;  // extracted and refreshed loop_input_ports. break potential_extractable_exprs loop, and go while() to start again.
            }
        }
        if (inner_loop_input_ports.size() == 0 && inner_loop_info->get_output_ports().size() == 0) {
            // If the loop becomes empty (inner_loop_input_ports is ref) after extraction, remove it from loop_manager
            loop_manager->remove_loop_info(inner_loop_id);
            break;
        }
        // no more extractable expr in this loop after go through all potential_extractable_exprs, done for this loop.
        if (!expr_extracted)
            continue_to_extract = false;
    }
    return status;
}
}  // namespace

bool ExtractLoopInvariants::run(LinearIR& linear_ir, lowered::LinearIR::constExprIt begin, lowered::LinearIR::constExprIt end) {
    OV_ITT_SCOPED_TASK(ov::pass::itt::domains::SnippetsTransform, "Snippets::ExtractLoopInvariants")
    bool modified = false;

    const auto& loop_depth = linear_ir.get_config().m_loop_depth;
    std::vector<std::set<size_t>> loop_ids_need_extract(loop_depth);
    const auto& loop_map = linear_ir.get_loop_manager()->get_map();
    for (const auto& loop : loop_map) {
        const auto& loop_dim = loop.second->get_dim_idx();
        if (loop_dim != LoopInfo::UNDEFINED_DIM_IDX) {
            OPENVINO_ASSERT(loop_dim < loop_depth, "dim_idx of loop should be smaller than loop_depth");
            loop_ids_need_extract[loop_dim].insert(loop.first);
        }
    }
    // move invariant expr to top(outside) of current loop
    for (size_t d = 0; d < loop_depth; d++) {
        const auto& loops_in_this_depth = loop_ids_need_extract[d];
        for (const auto& loop_id : loops_in_this_depth) {
            modified |= extract_from_loop(loop_id, linear_ir);
        }
    }

    return modified;
}

}  // namespace pass
}  // namespace lowered
}  // namespace snippets
}  // namespace ov
