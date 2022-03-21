// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "intel_gpu/primitives/custom_dpcpp_primitive.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

    template <>
    struct typed_program_node<custom_dpcpp_primitive> : public typed_program_node_base<custom_dpcpp_primitive> {
        using parent = typed_program_node_base<custom_dpcpp_primitive>;

    public:
        using parent::parent;

        program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    };

    using custom_dpcpp_primitive_node = typed_program_node<custom_dpcpp_primitive>;

    template <>
    class typed_primitive_inst<custom_dpcpp_primitive> : public typed_primitive_inst_base<custom_dpcpp_primitive> {
        using parent = typed_primitive_inst_base<custom_dpcpp_primitive>;

    public:
        static layout calc_output_layout(custom_dpcpp_primitive_node const& node) {
            assert(static_cast<bool>(node.get_primitive()->output_data_type) == false &&
                   "Output data type forcing is not supported for "
                   "custom_dpcpp_primitive_node!");
            layout output_layout = node.get_primitive()->output_layout;

            // if the output layout format was set to any,
            // it means the layer output format will be the same as the first input
            if (output_layout.format == format::any) {
                output_layout.format = node.get_dependency(0).get_output_layout().format;
            }
            return output_layout;
        }

        static std::string to_string(custom_dpcpp_primitive_node const& node);

    public:
        typed_primitive_inst(network& network, custom_dpcpp_primitive_node const& node);
    };

    using custom_dpcpp_primitive_inst = typed_primitive_inst<custom_dpcpp_primitive>;

}  // namespace cldnn
