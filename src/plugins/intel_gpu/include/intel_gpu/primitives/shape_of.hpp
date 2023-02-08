// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"
#include <vector>

namespace cldnn {

/// @brief Returns shape of input primitive.
struct shape_of : public primitive_base<shape_of> {
    CLDNN_DECLARE_PRIMITIVE(shape_of)

    /// @brief Constructs shape_of primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_data_type type of output values. can be i32 and i64.
    shape_of(const primitive_id& id,
             const input_info& input,
             size_t output_rank,
             const data_types output_data_type,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}, {optional_data_type{output_data_type}})
        , output_rank(output_rank) {}

    /// @brief Constructs shape_of primitive.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param output_data_type type of output values. can be i32 and i64.
    shape_of(const primitive_id& id,
             const input_info& input,
             const data_types output_data_type,
             const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}, {optional_data_type{output_data_type}})
        , output_rank(0) {}

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const shape_of>(rhs);

        return output_rank == rhs_casted.output_rank;
    }

    size_t output_rank;
};
}  // namespace cldnn
