// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "primitive.hpp"

#include "openvino/core/strides.hpp"
#include "openvino/core/coordinate_diff.hpp"
#include <vector>

namespace cldnn {

/// @brief Performs transposed convolution.
/// Also supports built-in Relu @ref activation available by setting it in arguments.
/// @details Deconvolution is similar to convolution layer with the weights flipped on the axis
/// and stride and input padding parameters used in opposite sense as in convolution.
struct deconvolution : public primitive_base<deconvolution> {
    CLDNN_DECLARE_PRIMITIVE(deconvolution)

    deconvolution() : primitive_base("", {}) {}

    /// @brief Constructs deconvolution primitive with dynamic shape.
    /// @param id This primitive id.
    /// @param input Input primitive id.
    /// @param weights List of primitive ids containing weights data.
    /// @param bias List of primitive ids containing bias data. Provide empty vector if using next parameters without bias.
    /// @param groups Number of filter groups.
    /// @param pad Defines logical pad value added to input tensor
    /// @param stride Defines shift in input buffer between adjacent calculations of output values.
    /// @param with_activation Enables Relu activation.
    /// @param activation_slp Relu activation slope.
    /// @param output_size User-defined output data size of the primitive (w/o padding).
    deconvolution(const primitive_id& id,
                  const input_info& input,
                  const primitive_id& weights,
                  const primitive_id& bias,
                  uint32_t groups,
                  ov::Strides stride,
                  ov::Strides dilations,
                  ov::CoordinateDiff pads_begin,
                  ov::CoordinateDiff pads_end,
                  ov::CoordinateDiff out_padding,
                  bool grouped_weights_shape,
                  const padding& output_padding = padding())
        : primitive_base(id, {input}, {output_padding}),
          stride(stride),
          dilations(dilations),
          groups(groups),
          pads_begin(pads_begin),
          pads_end(pads_end),
          out_padding(out_padding),
          grouped_weights_shape(grouped_weights_shape),
          output_partial_shape({}),
          output_shape_id(""),
          weights(weights),
          bias(bias) {}
    /// @brief Defines shift in input buffer between adjacent calculations of output values.
    ov::Strides stride;
    /// @brief Defines the distance in width and height between elements in the filter.
    ov::Strides dilations;
    /// @brief Number of feature groups (grouped convolution). If more than 1 then weights/bias count needs to be 1.
    uint32_t groups = 1;
    /// @brief Defines a padding added to input image on left (x axis) and top (y axis).
    ov::CoordinateDiff pads_begin;
    /// @brief Defines a padding added to input image on right (x axis) and bottom (y axis).
    ov::CoordinateDiff pads_end;
    /// @brief Defines additional amount of paddings per each spatial axis added to output tensor.
    ov::CoordinateDiff out_padding;
    /// @param grouped_weights_shape Defines if weights tensor has explicit group dimension.
    bool grouped_weights_shape = false;
    /// @brief Defines spatial shape of the output.
    ov::PartialShape output_partial_shape;
    /// @brief Data primitive id containing spatial shape of the output.
    primitive_id output_shape_id;
    /// @brief List of primitive ids containing weights data.
    const primitive_id weights;
    /// @brief List of primitive ids containing bias data.
    const primitive_id bias;

    size_t hash() const override {
        size_t seed = primitive::hash();
        seed = hash_range(seed, pads_begin.begin(), pads_begin.end());
        seed = hash_range(seed, pads_end.begin(), pads_end.end());
        seed = hash_range(seed, out_padding.begin(), out_padding.end());
        seed = hash_range(seed, stride.begin(), stride.end());
        seed = hash_combine(seed, groups);
        seed = hash_combine(seed, grouped_weights_shape);
        seed = hash_combine(seed, !weights.empty());
        seed = hash_combine(seed, !bias.empty());
        seed = hash_combine(seed, output_shape_id.empty());
        return seed;
    }

    bool operator==(const primitive& rhs) const override {
        if (!compare_common_params(rhs))
            return false;

        auto rhs_casted = downcast<const deconvolution>(rhs);

        #define cmp_fields(name) name == rhs_casted.name
        return cmp_fields(stride) &&
               cmp_fields(dilations) &&
               cmp_fields(groups) &&
               cmp_fields(pads_begin) &&
               cmp_fields(pads_end) &&
               cmp_fields(out_padding) &&
               cmp_fields(grouped_weights_shape) &&
               cmp_fields(weights.empty()) &&
               cmp_fields(bias.empty()) &&
               cmp_fields(output_shape_id.empty());
        #undef cmp_fields
    }

    void save(BinaryOutputBuffer& ob) const override {
        primitive_base<deconvolution>::save(ob);
        ob << stride;
        ob << dilations;
        ob << groups;
        ob << pads_begin;
        ob << pads_end;
        ob << out_padding;
        ob << grouped_weights_shape;
        ob << output_partial_shape;
        ob << output_shape_id;
        ob << weights;
        ob << bias;
    }

    void load(BinaryInputBuffer& ib) override {
        primitive_base<deconvolution>::load(ib);
        ib >> stride;
        ib >> dilations;
        ib >> groups;
        ib >> pads_begin;
        ib >> pads_end;
        ib >> out_padding;
        ib >> grouped_weights_shape;
        ib >> output_partial_shape;
        ib >> output_shape_id;
        ib >> *const_cast<primitive_id*>(&weights);
        ib >> *const_cast<primitive_id*>(&bias);
    }

protected:
    std::vector<input_info> get_dependencies() const override {
        std::vector<input_info> ret = {weights};
        if (!bias.empty()) {
            ret.push_back(bias);
        }
        if (!output_shape_id.empty()) {
            ret.push_back(output_shape_id);
        }

        return ret;
    }
};
}  // namespace cldnn
