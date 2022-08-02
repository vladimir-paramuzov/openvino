// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cum_sum_inst.h"
#include "primitive_base.hpp"
#include "impls/implementation_map.hpp"
#include "kernel_selector_helper.h"
#include "cum_sum/cum_sum_kernel_selector.h"
#include "cum_sum/cum_sum_kernel_ref.h"
#include "intel_gpu/runtime/error_handler.hpp"

using namespace cldnn;
namespace cldnn {
namespace ocl {

namespace {
kernel_selector::cum_sum_axis convert_axis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::cum_sum_axis::BATCH;
        case 1: return kernel_selector::cum_sum_axis::FEATURE;
        case 2:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::W;
            else if (rank == 5)
                return kernel_selector::cum_sum_axis::Z;
            else
                return kernel_selector::cum_sum_axis::Y;
        case 3:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::Z;
            else if (rank == 5)
                return kernel_selector::cum_sum_axis::Y;
            else
                return kernel_selector::cum_sum_axis::X;
        case 4:
            if (rank == 6)
                return kernel_selector::cum_sum_axis::Y;
            else
                return kernel_selector::cum_sum_axis::X;
        case 5: return kernel_selector::cum_sum_axis::X;
        default: return kernel_selector::cum_sum_axis::BATCH;
    }
}
}  // namespace

struct cum_sum_impl : typed_primitive_impl_ocl<cum_sum> {
    using parent = typed_primitive_impl_ocl<cum_sum>;
    using parent::parent;

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<cum_sum_impl>(*this);
    }

public:
    static primitive_impl* create(const cum_sum_node& arg, const kernel_impl_params& impl_param) {
        const auto& prim = arg.get_primitive();

        auto cum_sum_params = get_default_params<kernel_selector::cum_sum_params>(impl_param);
        auto cum_sum_optional_params =
            get_default_optional_params<kernel_selector::cum_sum_optional_params>(arg.get_program());

        size_t rank = arg.get_output_layout().get_rank();
        cum_sum_params.axis = convert_axis(prim->axis, rank);
        cum_sum_params.exclusive = arg.get_primitive()->exclusive;
        cum_sum_params.reverse = arg.get_primitive()->reverse;

        auto& kernel_selector = kernel_selector::cum_sum_kernel_selector::Instance();
        auto best_kernels = kernel_selector.GetBestKernels(cum_sum_params, cum_sum_optional_params);

        CLDNN_ERROR_BOOL(arg.id(),
                         "Best_kernel.empty()",
                         best_kernels.empty(),
                         "Cannot find a proper kernel with this arguments");

        auto cum_sum = new cum_sum_impl(arg, best_kernels[0]);

        return cum_sum;
    }
};

namespace detail {

attach_cum_sum_impl::attach_cum_sum_impl() {
    implementation_map<cum_sum>::add(impl_types::ocl, cum_sum_impl::create, {
        std::make_tuple(data_types::i32, format::bfyx),
        std::make_tuple(data_types::i32, format::bfzyx),
        std::make_tuple(data_types::i32, format::bfwzyx),
        std::make_tuple(data_types::i64, format::bfyx),
        std::make_tuple(data_types::i64, format::bfzyx),
        std::make_tuple(data_types::i64, format::bfwzyx),
        std::make_tuple(data_types::f16, format::bfyx),
        std::make_tuple(data_types::f16, format::bfzyx),
        std::make_tuple(data_types::f16, format::bfwzyx),
        std::make_tuple(data_types::f32, format::bfyx),
        std::make_tuple(data_types::f32, format::bfzyx),
        std::make_tuple(data_types::f32, format::bfwzyx),
    });
}

}  // namespace detail
}  // namespace ocl
}  // namespace cldnn
