// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "primitive_base.hpp"

#include "generate_proposals_inst.h"
#include "generate_proposals/generate_proposals_kernel_selector.h"
#include "generate_proposals/generate_proposals_kernel_ref.h"

namespace cldnn {
namespace ocl {
struct generate_proposals_impl
        : public typed_primitive_impl_ocl<generate_proposals> {
    using parent = typed_primitive_impl_ocl<generate_proposals>;
    using parent::parent;
    using kernel_selector_t = kernel_selector::generate_proposals_kernel_selector;
    using kernel_params_t = kernel_selector::generate_proposals_params;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::ocl::generate_proposals_impl)

    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<generate_proposals_impl>(*this);
    }

public:
    static kernel_params_t get_kernel_params(const kernel_impl_params& impl_param) {
        const auto& primitive = impl_param.typed_desc<generate_proposals>();
        auto params = get_default_params<kernel_selector::generate_proposals_params>(impl_param);

        params.min_size = primitive->attrs.min_size;
        params.nms_threshold  = primitive->attrs.nms_threshold;
        params.pre_nms_count = primitive->attrs.pre_nms_count;
        params.post_nms_count = primitive->attrs.post_nms_count;
        params.normalized = primitive->attrs.normalized;
        params.nms_eta = primitive->attrs.nms_eta;
        params.roi_num_type = to_data_type(primitive->output_data_types[2].value());

        const size_t num_inputs = primitive->input_size();
        for (size_t i = 1; i < num_inputs; i++) {
            params.inputs.push_back(convert_data_tensor(impl_param.get_input_layout(i)));
        }

        params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[1]));
        params.outputs.push_back(convert_data_tensor(impl_param.output_layouts[2]));

        return params;
    }
};

namespace detail {
    attach_generate_proposals_impl::attach_generate_proposals_impl() {
        implementation_map<generate_proposals>::add(impl_types::ocl,
                                                    typed_primitive_impl_ocl<generate_proposals>::create<generate_proposals_impl>, {
                                                            std::make_tuple(data_types::f16, format::bfyx),
                                                            std::make_tuple(data_types::f16, format::b_fs_yx_fsv16),
                                                            std::make_tuple(data_types::f16, format::b_fs_yx_fsv32),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv16_fsv16),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv16),
                                                            std::make_tuple(data_types::f16, format::bs_fs_yx_bsv32_fsv32),

                                                            std::make_tuple(data_types::f32, format::bfyx),
                                                            std::make_tuple(data_types::f32, format::b_fs_yx_fsv16),
                                                            std::make_tuple(data_types::f32, format::b_fs_yx_fsv32),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv16_fsv16),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv16),
                                                            std::make_tuple(data_types::f32, format::bs_fs_yx_bsv32_fsv32)});
    }
}  // namespace detail
}  // namespace ocl
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::ocl::generate_proposals_impl)
BIND_BINARY_BUFFER_WITH_TYPE(cldnn::generate_proposals)
