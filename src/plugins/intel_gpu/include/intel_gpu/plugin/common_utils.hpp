// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_layouts.h>
#include "intel_gpu/runtime/layout.hpp"
#include "openvino/core/layout.hpp"

#include "ngraph/type/element_type.hpp"

namespace ov {
namespace runtime {
namespace intel_gpu {

#define TensorValue(val) static_cast<cldnn::tensor::value_type>(val)

inline cldnn::tensor tensor_from_dims(const InferenceEngine::SizeVector& dims, int def = 1) {
    switch (dims.size()) {
    case 0: return cldnn::tensor(cldnn::batch(def), cldnn::feature(def), cldnn::spatial(def, def));
    case 1: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(def), cldnn::spatial(def, def));
    case 2: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, def));
    case 3: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(def, dims[2]));
    case 4: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[3], dims[2]));
    case 5: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[4], dims[3], dims[2]));
    case 6: return cldnn::tensor(cldnn::batch(dims[0]), cldnn::feature(dims[1]), cldnn::spatial(dims[5], dims[4], dims[3], dims[2]));
    default: IE_THROW() << "Invalid dimensions size(" << dims.size() << ") for gpu tensor";
    }
}

inline cldnn::data_types DataTypeFromPrecision(InferenceEngine::Precision p) {
    switch (p) {
    case InferenceEngine::Precision::I16:
    case InferenceEngine::Precision::U16:
    case InferenceEngine::Precision::FP32:
    case InferenceEngine::Precision::FP64:
        return cldnn::data_types::f32;
    case InferenceEngine::Precision::FP16:
        return cldnn::data_types::f16;
    case InferenceEngine::Precision::U8:
        return cldnn::data_types::u8;
    case InferenceEngine::Precision::I8:
        return cldnn::data_types::i8;
    case InferenceEngine::Precision::I32:
    case InferenceEngine::Precision::U32:
    case InferenceEngine::Precision::U64:
        return cldnn::data_types::i32;
    case InferenceEngine::Precision::I64:
        return cldnn::data_types::i64;
    case InferenceEngine::Precision::BIN:
        return cldnn::data_types::bin;
    case InferenceEngine::Precision::BOOL:
        return cldnn::data_types::i8;
    default:
        IE_THROW(ParameterMismatch)
            << "The plugin does not support " << p.name() << " precision";
    }
}

inline ngraph::element::Type element_type_from_data_type(cldnn::data_types t) {
    switch (t) {
    case cldnn::data_types::f32:
        return ngraph::element::Type_t::f32;
    case cldnn::data_types::f16:
        return ngraph::element::Type_t::f16;
    case cldnn::data_types::u8:
        return ngraph::element::Type_t::u8;
    case cldnn::data_types::i8:
        return ngraph::element::Type_t::i8;
    default:
        IE_THROW(ParameterMismatch)<< "Unsupported data_type";
    }
}

inline cldnn::data_types DataTypeFromPrecision(ngraph::element::Type t) {
    switch (t) {
    case ngraph::element::Type_t::i16:
    case ngraph::element::Type_t::u16:
    case ngraph::element::Type_t::f32:
    case ngraph::element::Type_t::f64:
        return cldnn::data_types::f32;
    case ngraph::element::Type_t::f16:
        return cldnn::data_types::f16;
    case ngraph::element::Type_t::u8:
        return cldnn::data_types::u8;
    case ngraph::element::Type_t::i8:
        return cldnn::data_types::i8;
    case ngraph::element::Type_t::i32:
    case ngraph::element::Type_t::u32:
    case ngraph::element::Type_t::u64:
        return cldnn::data_types::i32;
    case ngraph::element::Type_t::i64:
        return cldnn::data_types::i64;
    case ngraph::element::Type_t::boolean:
        return cldnn::data_types::i8;
    case ngraph::element::Type_t::u1:
        return cldnn::data_types::bin;
    default:
        IE_THROW(ParameterMismatch)
            << "The plugin does not support " << t.get_type_name()<< " precision";
    }
}

inline cldnn::format FormatFromLayout(InferenceEngine::Layout l) {
    switch (l) {
        // TODO: change 6d case once new layout added in IE
    case InferenceEngine::Layout::BLOCKED:
        return cldnn::format::bfwzyx;
    case InferenceEngine::Layout::NCDHW:
        return cldnn::format::bfzyx;
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
        return cldnn::format::bfyx;
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::byxf;
    default:
        IE_THROW(ParameterMismatch) << "The plugin does not support " << l << " layout";
    }
}

inline cldnn::format FormatFromTensorDesc(InferenceEngine::TensorDesc desc) {
    switch (desc.getLayout()) {
    case InferenceEngine::Layout::BLOCKED: {
        if (desc.getDims().size() == 6)
            return cldnn::format::bfwzyx;
        else if (desc.getDims().size() == 5)
            return cldnn::format::bfzyx;
        else if (desc.getDims().size() <= 4)
            return cldnn::format::bfyx;
    }
    case InferenceEngine::Layout::NCDHW:
        return cldnn::format::bfzyx;
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
        return cldnn::format::bfyx;
    case InferenceEngine::Layout::SCALAR:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::byxf;
    default:
        IE_THROW(ParameterMismatch)
            << "The plugin does not support " << desc.getLayout() << " layout";
    }
}

inline cldnn::format ImageFormatFromLayout(InferenceEngine::Layout l) {
    switch (l) {
    // currently, nv12 is the only supported image layout
    case InferenceEngine::Layout::BLOCKED:
    case InferenceEngine::Layout::NCDHW:
    case InferenceEngine::Layout::NCHW:
    case InferenceEngine::Layout::NC:
    case InferenceEngine::Layout::CHW:
    case InferenceEngine::Layout::C:
    case InferenceEngine::Layout::NHWC:
        return cldnn::format::nv12;
    default:
        IE_THROW(ParameterMismatch)
            << "The plugin does not support " << l << " image layout";
    }
}

inline cldnn::format DefaultFormatForDims(size_t dimensions) {
    switch (dimensions) {
    case 0:
    case 1:
    case 2:
    case 3:
    case 4:
        return cldnn::format::bfyx;
    case 5:
        return cldnn::format::bfzyx;
    case 6:
        return cldnn::format::bfwzyx;
    default:
        IE_THROW() << "Unsupported number of dimensions: " << dimensions;
    }

    return cldnn::format::bfyx;  // Should not get here
}

// This helper function is needed to convert permute order from IE format (bfyx) into cldnn format (bfxy)
inline std::vector<uint16_t> ConvertPermuteOrder(const std::vector<uint16_t>& ie_order, size_t rank = 0) {
    std::vector<uint16_t> ie_order_aligned = ie_order;
    // if order size is less than 4 - fill the rest with just copy
    rank = std::max(rank, (size_t)4);
    for (auto o = ie_order_aligned.size(); o < rank; o++)
        ie_order_aligned.push_back((uint16_t)o);

    std::vector<uint16_t> cldnn_order;
    // 1. Switch permute order values for spatial dims
    for (auto const& o : ie_order_aligned) {
        if (o >= 2)
            cldnn_order.push_back(1 + ie_order_aligned.size() - o);
        else
            cldnn_order.push_back(o);
    }

    // 2. Swap spatial positions
    for (int i = 0; i < (cldnn_order.size() - 2) / 2; i++) {
        std::swap(cldnn_order[2 + i], cldnn_order[1 + cldnn_order.size() - (2 + i)]);
    }

    return cldnn_order;
}

inline InferenceEngine::Layout InferenceEngineLayoutFromOVLayout(ov::Layout l) {
    if (l == ov::Layout("C")) return InferenceEngine::Layout::C;
    if (l == ov::Layout("CN")) return InferenceEngine::Layout::CN;
    if (l == ov::Layout("HW")) return InferenceEngine::Layout::HW;
    if (l == ov::Layout("NC")) return InferenceEngine::Layout::NC;
    if (l == ov::Layout("CHW")) return InferenceEngine::Layout::CHW;
    if (l == ov::Layout("HWC")) return InferenceEngine::Layout::HWC;
    if (l == ov::Layout("NCHW")) return InferenceEngine::Layout::NCHW;
    if (l == ov::Layout("NC??")) return InferenceEngine::Layout::NCHW;
    if (l == ov::Layout("NHWC")) return InferenceEngine::Layout::NHWC;
    if (l == ov::Layout("NCDHW")) return InferenceEngine::Layout::NCDHW;
    if (l == ov::Layout("NDHWC")) return InferenceEngine::Layout::NDHWC;
    IE_THROW() << "The plugin does not support " << l.to_string() << " layout";
}

}  // namespace intel_gpu
}  // namespace runtime
}  // namespace ov
