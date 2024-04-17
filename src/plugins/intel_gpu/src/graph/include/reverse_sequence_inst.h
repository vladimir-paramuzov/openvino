// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include "intel_gpu/primitives/reverse_sequence.hpp"
#include "primitive_inst.h"

#include <string>

namespace cldnn {

using reverse_sequence_node = typed_program_node<reverse_sequence>;

template <>
class typed_primitive_inst<reverse_sequence> : public typed_primitive_inst_base<reverse_sequence> {
    using parent = typed_primitive_inst_base<reverse_sequence>;
    using parent::parent;

public:
    template<typename ShapeType>
    static std::vector<layout> calc_output_layouts(reverse_sequence_node const& /*node*/, const kernel_impl_params& impl_param) {
        return forward_input0_shape<ShapeType>(impl_param);
    }
    static std::string to_string(reverse_sequence_node const& node);

    typed_primitive_inst(network& network, reverse_sequence_node const& desc);
};

using reverse_sequence_inst = typed_primitive_inst<reverse_sequence>;
}  // namespace cldnn
