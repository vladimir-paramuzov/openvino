// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include "cldnn/primitives/reorder.hpp"
#include "primitive_inst.h"

#include <string>
#include <memory>

namespace cldnn {

template <>
struct typed_program_node<reorder> : public typed_program_node_base<reorder> {
    using parent = typed_program_node_base<reorder>;

public:
    typed_program_node(const std::shared_ptr<reorder> prim, program& prog) : parent(prim, prog) {
        support_padding_all(true);
    }

    size_t inputs_count() const { return get_primitive()->input_ids.size(); }
    program_node& mean_nv12() const { return get_dependency(2); }
    program_node& input(size_t idx = 0) const { return get_dependency(idx); }
    program_node& mean() const { return get_dependency(1); }

    bool has_mean() const { return !typed_desc()->mean.empty(); }

    bool requires_reinterpret() const { return req_reinterpr; }
    void requires_reinterpret(bool val) { req_reinterpr = (optimized && val); }

    void set_input_offset(tensor const& io) { input_offset = io; }
    tensor get_input_offset() const { return input_offset; }

private:
    bool req_reinterpr = false;
    tensor input_offset = tensor{0};  // used by reorder to winograd domain
};

using reorder_node = typed_program_node<reorder>;

template <>
class typed_primitive_inst<reorder> : public typed_primitive_inst_base<reorder> {
    using parent = typed_primitive_inst_base<reorder>;

public:
    static layout calc_output_layout(reorder_node const& node);
    static std::string to_string(reorder_node const& node);

public:
    typed_primitive_inst(network& network, reorder_node const& node);
    memory::ptr mean_nv12_memory() const { return dep_memory_ptr(2); }
    memory::ptr mean_memory() const { return dep_memory_ptr(1); }

    bool has_mean() const { return !argument.mean.empty(); }

private:
    void on_execute() override;
    void reuse_input();
};

using reorder_inst = typed_primitive_inst<reorder>;

}  // namespace cldnn
