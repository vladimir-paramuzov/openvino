// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn_program.h"
#include "cldnn_common_utils.h"

#include "ngraph/op/extractimagepatches.hpp"

#include "api/extract_image_patches.hpp"

namespace CLDNNPlugin {

static inline std::string PadToString(ngraph::op::PadType pad) {
    switch (pad) {
        case ngraph::op::PadType::SAME_UPPER: return "same_upper";
        case ngraph::op::PadType::SAME_LOWER: return "same_lower";
        case ngraph::op::PadType::VALID: return "valid";
        default: THROW_IE_EXCEPTION << "Unsupported pad type in ExtractImagePatches primitive " << pad;
    }

    return "";
}

void Program::CreateExtractImagePatchesOp(cldnn::topology& topology, const std::shared_ptr<ngraph::Node>& node) {
    auto op = std::dynamic_pointer_cast<ngraph::op::v3::ExtractImagePatches>(node);
    if (!op)
        THROW_IE_EXCEPTION << INVALID_OP_MESSAGE;

    ValidateInputs(op, {1});
    auto inputPrimitives = GetInputPrimitiveIDs(op);
    std::string layerName = layer_type_name_ID(op);

    std::vector<uint32_t> sizes = std::vector<uint32_t>(op->get_sizes().begin(), op->get_sizes().end());
    std::vector<uint32_t> strides = std::vector<uint32_t>(op->get_strides().begin(), op->get_strides().end());
    std::vector<uint32_t> rates = std::vector<uint32_t>(op->get_rates().begin(), op->get_rates().end());
    std::string auto_pad = PadToString(op->get_auto_pad());

    auto extractImagePatchesPrim = cldnn::extract_image_patches(layerName,
                                                                inputPrimitives[0],
                                                                sizes,
                                                                strides,
                                                                rates,
                                                                auto_pad,
                                                                CldnnTensorFromIEDims(op->get_output_shape(0)));

    topology.add(extractImagePatchesPrim);
    AddPrimitiveToProfiler(op);
}

}  // namespace CLDNNPlugin
