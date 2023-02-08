// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "arg_max_min_inst.h"
#include "arg_max_min/arg_max_min_kernel_selector.h"
#include "arg_max_min/arg_max_min_kernel_base.h"

namespace cldnn {
namespace ocl {

static inline kernel_selector::argm_axis GetArgMaxMinAxis(int64_t axis, size_t rank) {
    if (axis < 0) {
        axis += rank;
    }
    switch (axis) {
        case 0: return kernel_selector::argm_axis::BATCH;
        case 1: return kernel_selector::argm_axis::FEATURE;
        case 2:
            if (rank > 4)
                return kernel_selector::argm_axis::Z;
            else
                return kernel_selector::argm_axis::Y;
        case 3:
            if (rank > 4)
                return kernel_selector::argm_axis::Y;
            else
                return kernel_selector::argm_axis::X;
        case 4: return kernel_selector::argm_axis::X;
        default: IE_THROW() << "Invalid arg_max_min axis " << axis;
    }
}

struct arg_max_min_impl : typed_primitive_impl_ocl<arg_max_min> {
    using parent = typed_primitive_impl_ocl<arg_max_min>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::arg_max_min_kernel_selector;
    using kernel_params_t = std::pair<kernel_selector::arg_max_min_params, kernel_selector::arg_max_min_optional_params>;

    DECLARE_OBJECT_TYPE_SERIALIZATION

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<arg_max_min_impl>(*this);
    }

protected:
    kernel_arguments_data get_arguments(const typed_primitive_inst<arg_max_min>& instance) const override {
        kernel_arguments_data args = parent::get_arguments(instance);

        if (instance.node->has_second_output()) {
            args.inputs.erase(args.inputs.begin() + 1);  // erase constant input in case of TOP_K
        }

        return args;
    }

public:
    static std::unique_ptr<primitive_impl> create(const arg_max_min_node& arg, const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<arg_max_min>();
        const auto& axis = primitive->axis;
        const auto& top_k = primitive->top_k;
        const auto& mode = primitive->mode;
        const auto& sort_type = primitive->sort;
        const auto& values_first = primitive->values_first;
        const auto& outputs_num = arg.get_output_nums();  // second output passed as input for TOP_K layer

        auto argm_params = get_default_params<kernel_selector::arg_max_min_params>(impl_param);
        auto argm_optional_params =
            get_default_optional_params<kernel_selector::arg_max_min_optional_params>(impl_param.get_program());

        argm_params.outputs_num = outputs_num;
        argm_params.topK = top_k;
        argm_params.argMaxMinAxis = GetArgMaxMinAxis(axis, arg.get_output_layout().get_rank());

        if (mode == ov::op::TopKMode::MAX)
            argm_params.argMaxMinOut = kernel_selector::argm_output::MAX;
        else
            argm_params.argMaxMinOut = kernel_selector::argm_output::MIN;

        if (sort_type == ov::op::TopKSortType::SORT_VALUES)
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::VALUE;
        else
            argm_params.argMaxMinSortType = kernel_selector::argm_sort::INDEX;

        if (arg.has_second_output()) {  // for backward compatibility
            argm_params.has_second_output = true;
            if (arg.use_multiple_outputs()) {
                argm_params.use_multiple_outputs = true;
                argm_params.outputs.push_back(convert_data_tensor(impl_param.get_output_layout(1)));
            } else {
                argm_params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(2)));
            }
        }

        argm_params.values_first = values_first;

        auto& kernel_selector = kernel_selector::arg_max_min_kernel_selector::Instance();
        auto best_kernel = kernel_selector.get_best_kernel(argm_params, argm_optional_params);

        return make_unique<arg_max_min_impl>(best_kernel);
    }
};

namespace detail {
attach_arg_max_min_impl::attach_arg_max_min_impl() {
    auto types = {data_types::f16, data_types::f32, data_types::i8, data_types::i32};

    auto formats = {format::bfyx,
                    format::yxfb,
                    format::b_fs_yx_fsv16,
                    format::b_fs_yx_fsv32,
                    format::bs_fs_yx_bsv16_fsv16,
                    format::bs_fs_yx_bsv32_fsv16,
                    format::bs_fs_yx_bsv32_fsv32,

                    format::bfzyx};

    implementation_map<arg_max_min>::add(impl_types::ocl, arg_max_min_impl::create, types, formats);
}
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::arg_max_min_impl)
