// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/graph/kernel_impl_params.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "openvino/core/except.hpp"

#include <functional>
#include <string>
#include <tuple>


namespace cldnn {

template <typename T, typename primitive_type>
class singleton_list : public std::vector<T> {
    singleton_list() : std::vector<T>() {}
    singleton_list(singleton_list const&) = delete;
    void operator=(singleton_list const&) = delete;

public:
    using type = primitive_type;
    static singleton_list& instance() {
        static singleton_list instance_;
        return instance_;
    }
};

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
    virtual bool validate(const program_node& node) const = 0;
    virtual bool support_shapes(const kernel_impl_params& param) const = 0;
    virtual in_out_fmts_t query_formats(const program_node& node) const = 0;
    virtual ~ImplementationManager() = default;

    static shape_types get_shape_type(const program_node& node);
    static shape_types get_shape_type(const kernel_impl_params& params);
protected:
    static bool is_supported(const program_node& node, const std::set<key_type>& supported_keys, shape_types shape_type);
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
    ImplementationManagerLegacy(simple_factory_type factory, shape_types shape_type, std::set<key_type> keys)
        : m_factory(factory)
        , m_shape_type(shape_type)
        , m_keys(keys) {}

    ImplementationManagerLegacy() = default;

private:
    simple_factory_type m_factory;
    shape_types m_shape_type;
    std::set<key_type> m_keys;
};

template <typename primitive_kind>
class ImplementationsRegistry {
public:
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const typed_program_node<primitive_kind>&, const kernel_impl_params&)>;
    using key_type = cldnn::key_type;
    using list_type = singleton_list<std::tuple<impl_types, shape_types, std::unique_ptr<ImplementationManager>>, primitive_kind>;

    static const ImplementationManager* get(impl_types preferred_impl_type, shape_types target_shape_type) {
        for (auto& entry : list_type::instance()) {
            impl_types impl_type = std::get<0>(entry);
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;

            shape_types supported_shape_type = std::get<1>(entry);
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;

            auto& factory = std::get<2>(entry);
            return factory.get();
        }
        return nullptr;
    }

    static std::set<impl_types> get_available_impls(shape_types target_shape_type) {
        std::set<impl_types> impls;
        for (auto& entry : list_type::instance()) {
            shape_types supported_shape_type = std::get<1>(entry);
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;

            impls.insert(std::get<0>(entry));
        }

        return impls;
    }

    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, shape_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, simple_factory_type factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, simple_factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        add(impl_type, shape_types::static_shape, std::move(factory), keys);
    }

    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory, std::set<key_type> keys) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        auto f = cldnn::make_unique<ImplementationManagerLegacy<primitive_kind>>(factory, shape_type, keys);
        list_type::instance().push_back({impl_type, shape_type, std::move(f)});
    }

    static void add(impl_types impl_type, shape_types shape_type, std::unique_ptr<ImplementationManager> factory,
                    const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        add(impl_type, shape_type, std::move(factory), combine(types, formats));
    }

    static void add(impl_types impl_type, std::unique_ptr<ImplementationManager> factory) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register impl with type any");
        // shape_type::any is used as factory::validate will check everything
        list_type::instance().push_back({impl_type, shape_types::any, std::move(factory)});
    }

    static std::set<key_type> combine(const std::vector<data_types>& types, const std::vector<format::type>& formats) {
        std::set<key_type> keys;
        for (const auto& type : types) {
            for (const auto& format : formats) {
                keys.emplace(type, format);
            }
        }
        return keys;
    }
};

template <typename PrimitiveType>
using implementation_map = ImplementationsRegistry<PrimitiveType>;

struct DummyReorderType {};
struct WeightsReordersFactory {
    using simple_factory_type = std::function<std::unique_ptr<primitive_impl>(const kernel_impl_params&)>;
    using list_type = singleton_list<std::tuple<impl_types, shape_types, simple_factory_type>, DummyReorderType>;
    static void add(impl_types impl_type, shape_types shape_type, simple_factory_type factory) {
        OPENVINO_ASSERT(impl_type != impl_types::any, "[GPU] Can't register WeightsReordersFactory with type any");
        list_type::instance().push_back({impl_type, shape_type, factory});
    }

    static simple_factory_type get(impl_types preferred_impl_type, shape_types target_shape_type) {
        for (auto& kv : list_type::instance()) {
            impl_types impl_type = std::get<0>(kv);
            shape_types supported_shape_type = std::get<1>(kv);
            if ((preferred_impl_type & impl_type) != impl_type)
                continue;
            if ((target_shape_type & supported_shape_type) != target_shape_type)
                continue;

            return std::get<2>(kv);
        }
        OPENVINO_THROW("[GPU] WeightsReordersFactory doesn't have any implementation for "
                       " impl_type: ", preferred_impl_type, ", shape_type: ", target_shape_type);
    }
};
}  // namespace cldnn
