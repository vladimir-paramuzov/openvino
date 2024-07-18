// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_inst.h"
#include "primitive_onednn_base.h"
#include "implementation_map.hpp"

#include <oneapi/dnnl/dnnl.hpp>

#include <algorithm>
#include <memory>
namespace cldnn {
namespace onednn {

struct pooling_onednn : typed_primitive_onednn_impl<pooling> {
    using parent = typed_primitive_onednn_impl<pooling>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::pooling_onednn)

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<pooling_onednn>(*this);
    }

    static std::shared_ptr<dnnl::pooling_forward::primitive_desc> get_pooling_primitive_descriptor(const kernel_impl_params& impl_params,
                                                                                                   const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto& engine = impl_params.prog->get_engine();
        auto prim = impl_params.typed_desc<pooling>();

        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();

        auto kernel_shape = prim->size;
        auto stride_shape = prim->stride;
        auto pads_begin_shape = prim->pads_begin;
        auto pads_end_shape = prim->pads_end;
        auto dilation_shape = prim->dilation;
        if (dilation_shape.empty()) dilation_shape.resize(stride_shape.size(), 0);

        kernel_shape.resize(std::max<size_t>(2, prim->size.size()), 1);
        stride_shape.resize(std::max<size_t>(2, prim->stride.size()), 1);
        pads_begin_shape.resize(std::max<size_t>(2, prim->pads_begin.size()), 0);
        pads_end_shape.resize(std::max<size_t>(2, prim->pads_end.size()), 0);
        dilation_shape.resize(std::max<size_t>(2, dilation_shape.size()), 0);

        dnnl::memory::dims stride(stride_shape.begin(), stride_shape.end());
        dnnl::memory::dims kernel(kernel_shape.begin(), kernel_shape.end());
        dnnl::memory::dims pad_l(pads_begin_shape.begin(), pads_begin_shape.end());
        dnnl::memory::dims pad_r(pads_end_shape.begin(), pads_end_shape.end());
        dnnl::memory::dims dilation(dilation_shape.begin(), dilation_shape.end());

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        for (size_t i = 0; i < kernel.size(); i++) {
            pad_r[i] = (output_md.get_dims()[2 + i] - 1) * stride[i] - input_md.get_dims()[2 + i] + kernel[i] - pad_l[i];
        }

        dnnl::algorithm alg;
        switch (prim->mode) {
            case pooling_mode::average: alg = dnnl::algorithm::pooling_avg_include_padding; break;
            case pooling_mode::max: alg = dnnl::algorithm::pooling_max; break;
            case pooling_mode::average_no_padding: alg = dnnl::algorithm::pooling_avg_exclude_padding; break;
            default: throw std::runtime_error("unsupported pool mode");
        }

        return std::make_shared<dnnl::pooling_forward::primitive_desc>(
            engine.get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            alg,
            input_md,
            output_md,
            stride,
            kernel,
            dilation,
            pad_l,
            pad_r,
            attr);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);

        const dnnl::pooling_forward::primitive_desc *typed_pd
            = reinterpret_cast<const dnnl::pooling_forward::primitive_desc *>(&_pd);

        dnnl::algorithm alg = typed_pd->get_algorithm();
        ob << make_data(&alg, sizeof(dnnl::algorithm));
        ob << typed_pd->get_strides();
        ob << typed_pd->get_kernel();
        ob << typed_pd->get_dilations();
        ob << typed_pd->get_padding_l();
        ob << typed_pd->get_padding_r();

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());

        dnnl::algorithm alg;
        ib >> make_data(&alg, sizeof(dnnl::algorithm));

        auto input_md = onednn::layout_to_memory_desc(impl_params->get_input_layout(0));
        auto output_md = onednn::layout_to_memory_desc(impl_params->get_output_layout());

        dnnl::memory::dims stride;
        dnnl::memory::dims kernel;
        dnnl::memory::dims dilation;
        dnnl::memory::dims pad_l;
        dnnl::memory::dims pad_r;
        ib >> stride;
        ib >> kernel;
        ib >> dilation;
        ib >> pad_l;
        ib >> pad_r;

        auto prim_desc = std::make_shared<dnnl::pooling_forward::primitive_desc>(
            ib.get_engine().get_onednn_engine(),
            dnnl::prop_kind::forward_inference,
            alg,
            input_md,
            output_md,
            stride,
            kernel,
            dilation,
            pad_l,
            pad_r,
            *_attrs.get());
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static bool validate(const pooling_node& node) {
        if (!is_supported_format(node.get_preferred_input_fmt(0)))
            return false;

        auto in_dt = node.get_input_layout(0).data_type;
        auto out_dt = node.get_output_layout(false).data_type;

        bool fp_case = data_type_traits::is_floating_point(in_dt) && in_dt == out_dt;
        bool u8s8_case = one_of(in_dt, {data_types::i8, data_types::u8}) && one_of(out_dt, {data_types::i8, data_types::u8});

        if (!fp_case && !u8s8_case)
            return false;

        if (!is_supported_post_ops(node))
            return false;

        return true;
    }

    static std::unique_ptr<primitive_impl> create(const pooling_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim_desc = get_pooling_primitive_descriptor(impl_params, *attr);

        return cldnn::make_unique<pooling_onednn>(engine, config, attr, *prim_desc);
    }
};

struct pooling_factory : public cldnn::implementation_factory<pooling> {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        OPENVINO_ASSERT(node.is_type<pooling>());
        return onednn::pooling_onednn::create(static_cast<const pooling_node&>(node), params);
    }

    bool validate(const program_node& node) const override {
        OPENVINO_ASSERT(node.is_type<pooling>());
        return onednn::pooling_onednn::validate(static_cast<const pooling_node&>(node));
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        OPENVINO_NOT_IMPLEMENTED;
    }
};

namespace detail {

attach_pooling_onednn::attach_pooling_onednn() {
    std::vector<data_types> dt = {
        data_types::f32,
        data_types::f16,
        data_types::u8,
        data_types::i8,
    };
    std::vector<format::type> fmt = {
        format::bfyx,
        format::b_fs_yx_fsv16,
        format::b_fs_zyx_fsv16,
        format::b_fs_yx_fsv32,
        format::b_fs_zyx_fsv32,
        format::bs_fs_yx_bsv16_fsv16,
        format::bs_fs_yx_bsv16_fsv32,
        format::bs_fs_yx_bsv32_fsv16,
        format::bs_fs_yx_bsv32_fsv32,
        format::bs_fs_zyx_bsv16_fsv16,
        format::bs_fs_zyx_bsv16_fsv32,
        format::bs_fs_zyx_bsv32_fsv16,
        format::bs_fs_zyx_bsv32_fsv32,
    };

    implementation_map<pooling>::add(impl_types::onednn, cldnn::make_unique<pooling_factory>(), dt, fmt);
}

}  // namespace detail
}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::pooling_onednn)
