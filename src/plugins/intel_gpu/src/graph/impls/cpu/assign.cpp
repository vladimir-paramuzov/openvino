// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "assign_inst.h"
#include "implementation_map.hpp"
#include "register.hpp"
#include "intel_gpu/runtime/error_handler.hpp"

namespace cldnn {
namespace cpu {

struct assign_impl : public typed_primitive_impl<assign> {
    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<assign_impl>(*this);
    }

    event::ptr execute_impl(const std::vector<event::ptr>& events, assign_inst& instance) override {
        const auto arg = instance.argument;
        const auto variable_id = arg->variable_id;
        auto& variable = instance.get_network().get_variable_memory(variable_id);

        if (variable.memory->get_layout() != arg->output_layout) {
            CLDNN_ERROR_MESSAGE(instance.id(), "Layout mismatch");
        }

        auto& stream = instance.get_network().get_stream();

        for (auto e : events) {
            e->wait();
        }

        const auto ev_set_memory = variable.memory->copy_from(stream, instance.input_memory());
        variable.is_set = true;

        return ev_set_memory;
    }

    void init_kernels(const kernels_cache&) override {}

public:
    static std::unique_ptr<primitive_impl> create(const assign_node& arg, const kernel_impl_params& impl_param) {
        return make_unique<assign_impl>();
    }
};


namespace detail {

attach_assign_impl::attach_assign_impl() {
    implementation_map<assign>::add(impl_types::cpu, assign_impl::create, {});
}

}  // namespace detail
}  // namespace cpu
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::cpu::assign_impl)
