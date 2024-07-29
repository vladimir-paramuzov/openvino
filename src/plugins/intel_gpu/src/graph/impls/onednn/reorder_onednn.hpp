// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "utils.hpp"
#include "reorder_inst.h"
#include "impls/registry/implementation_manager.hpp"

#include <memory>
namespace cldnn {
namespace onednn {

struct ReorderImplementationManager : public ImplementationManager {
    ReorderImplementationManager() : ImplementationManager(impl_types::onednn, shape_types::static_shape) {}
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override;

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<reorder>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad)
            return false;

        static const std::vector<format::type> supported_formats = {
            format::custom,
            format::bfyx,
            format::byxf,
            format::b_fs_zyx_fsv16,
            format::b_fs_yx_fsv16,
            format::b_fs_yx_fsv32,
            format::bs_fs_zyx_bsv8_fsv4,
            format::bs_fs_yx_bsv8_fsv4,
            format::bs_fs_yx_bsv16_fsv4,
            format::bs_fs_zyx_bsv16_fsv4,
            format::bs_fs_yx_bsv16_fsv2,
            format::bs_fs_zyx_bsv16_fsv2,
            format::bs_fs_zyx_bsv8_fsv2,
            format::bs_fs_yx_bsv8_fsv2,
            format::bs_fs_zyx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv16,
            format::bs_fs_yx_bsv16_fsv32,
            format::bs_fs_zyx_bsv32_fsv16,
            format::bs_fs_yx_bsv32_fsv16,
            format::bs_fs_zyx_bsv32_fsv32,
            format::bs_fs_yx_bsv32_fsv32,
        };

        const auto& input_layout = node.get_input_layout(0);
        const auto& output_layout = node.get_output_layout(0);

        auto input_fmt = input_layout.format;
        auto output_fmt = output_layout.format;

        auto in_dt = input_layout.data_type;
        auto out_dt = output_layout.data_type;

        if (!one_of(input_fmt.value, supported_formats) || !one_of(output_fmt.value, supported_formats))
            return false;

        // onednn doesn't support paddings
        if (input_layout.data_padding || output_layout.data_padding)
            return false;

        // Native impl works faster for this type of reorder
        if (input_fmt == format::bfyx && output_fmt == format::bfyx)
            return false;

        // onednn reorder doesn't support different number of dimensions in input and output layouts
        if (input_fmt.dimension() != output_fmt.dimension())
            return false;

        if (in_dt == data_types::i64 || out_dt == data_types::i64)
            return false;

        // For mixed precision case, oneDNN is slower than clDNN
        if (input_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::b_fs_yx_fsv16 && data_type_traits::is_i8_u8(in_dt))
            return false;
        if (output_fmt == format::bfyx && out_dt == data_types::f32)
            return false;

        return ImplementationManager::validate(node);
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        return get_shape_type(params) == shape_types::static_shape;
    }
};

}  // namespace onednn
}  // namespace cldnn
