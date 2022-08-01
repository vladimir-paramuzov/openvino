// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/plugin/program.hpp"
#include "intel_gpu/plugin/common_utils.hpp"

#include "ngraph/op/cum_sum.hpp"
#include "ngraph/op/constant.hpp"

#include "intel_gpu/primitives/cum_sum.hpp"

namespace ov {
namespace intel_gpu {

static void CreateCumSumOp(Program& p, const std::shared_ptr<ngraph::op::v0::CumSum>& op) {
    p.ValidateInputs(op, {1, 2});
    auto inputPrimitives = p.GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    auto exclusive = op->is_exclusive();
    auto reverse = op->is_reverse();

    int64_t axis = 0;
    if (op->get_input_size() == 2) {
        auto axes_constant = std::dynamic_pointer_cast<ngraph::op::Constant>(op->get_input_node_shared_ptr(1));
        if (!axes_constant) {
            IE_THROW() << "Unsupported parameter nodes type in " << op->get_friendly_name() << " (" << op->get_type_name() << ")";
        }
        axis = axes_constant->cast_vector<int64_t>()[0];
    }
    axis = ov::normalize_axis(op.get(), axis, op->get_input_partial_shape(0).rank());

    auto primitive = cldnn::cum_sum(layerName,
                                    inputPrimitives[0],
                                    axis,
                                    exclusive,
                                    reverse,
                                    op->get_friendly_name());

    p.AddPrimitive(primitive);
    p.AddPrimitiveToProfiler(op);
}

REGISTER_FACTORY_IMPL(v0, CumSum);

}  // namespace intel_gpu
}  // namespace ov
