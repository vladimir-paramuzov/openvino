// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "openvino/core/except.hpp"

#include <functional>
#include <memory>
#include <tuple>

namespace cldnn {

using in_out_fmts_t = std::pair<std::vector<format::type>, std::vector<format::type>>;

struct primitive_impl;

struct program_node;
template <class PType>
struct typed_program_node;

using key_type = std::tuple<data_types, format::type>;
struct implementation_key {
    key_type operator()(const layout& proposed_layout) {
        return std::make_tuple(proposed_layout.data_type, proposed_layout.format);
    }
};

struct ImplementationManager {
public:
    virtual std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const = 0;
    virtual std::unique_ptr<primitive_impl> create(const kernel_impl_params& params) const { OPENVINO_NOT_IMPLEMENTED; }
    virtual bool validate(const program_node& node) const = 0;
    virtual bool support_shapes(const kernel_impl_params& param) const = 0;
    virtual in_out_fmts_t query_formats(const program_node& node) const = 0;
    explicit ImplementationManager(impl_types impl_type, shape_types shape_type) : m_impl_type(impl_type), m_shape_type(shape_type) {}
    virtual ~ImplementationManager() = default;

    static shape_types get_shape_type(const program_node& node);
    static shape_types get_shape_type(const kernel_impl_params& params);

    impl_types get_impl_type() const { return m_impl_type; }
    shape_types get_shape_type() const { return m_shape_type; }

protected:
    static bool is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types shape_type);
    impl_types m_impl_type;
    shape_types m_shape_type;
};

template <typename primitive_kind>
struct ImplementationManagerLegacy : public ImplementationManager {
    std::unique_ptr<primitive_impl> create(const program_node& node, const kernel_impl_params& params) const override {
        if (m_factory) {
            auto res = m_factory(static_cast<const typed_program_node<primitive_kind>&>(node), params);
            res->set_dynamic(get_shape_type(params) == shape_types::dynamic_shape);
            return res;
        }

        OPENVINO_NOT_IMPLEMENTED;
    }
    bool validate(const program_node& node) const override {
       return ImplementationManager::is_supported(node, m_keys, m_shape_type);
    }

    bool support_shapes(const kernel_impl_params& params) const override {
        auto shape_type = get_shape_type(params);
        return (m_shape_type & shape_type) == shape_type;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        return {};
    }

    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    ImplementationManagerLegacy(simple_factory_type factory, impl_types impl_type, shape_types shape_type, std::set<key_type> keys)
        : ImplementationManager(impl_type, shape_type)
        , m_factory(factory)
        , m_keys(keys) {}

    ImplementationManagerLegacy() = default;

private:
    simple_factory_type m_factory;
    std::set<key_type> m_keys;
};

}  // namespace cldnn
