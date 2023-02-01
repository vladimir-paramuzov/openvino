// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/primitive.hpp"
#include "intel_gpu/runtime/memory.hpp"

#include <vector>

namespace cldnn {

struct WeightsReorderParams {
    WeightsReorderParams(layout in_layout, layout out_layout) : _in_layout(in_layout), _out_layout(out_layout) {}
    virtual ~WeightsReorderParams() = default;
    virtual size_t hash() const { return hash_combine(_in_layout.hash(), _out_layout.hash()); }
    layout get_input_layout() const { return _in_layout; }
    layout get_output_layout() const { return _out_layout; }

protected:
    layout _in_layout;
    layout _out_layout;
};

/// @brief Changes how data is ordered in memory. Value type is not changed & all information is preserved.
/// @details Corresponding values are bitwise equal before/after reorder.
struct generic_layer : public primitive_base<generic_layer> {
    CLDNN_DECLARE_PRIMITIVE(generic_layer)

    /// @brief Constructs generic_layer primitive which takes mean subtract values from another primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_layout Requested memory layout.
    /// @param mean Primitive id to get mean subtract values.
    generic_layer(const primitive_id& id,
                  const primitive_id& input,
                  std::shared_ptr<WeightsReorderParams> params,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}), params(params) {}

    std::shared_ptr<WeightsReorderParams> params;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_combine(seed, id);
        return seed;
    }

protected:
    std::vector<std::reference_wrapper<const primitive_id>> get_dependencies() const override { return {}; }
};
/// @}
/// @}
/// @}
}  // namespace cldnn
