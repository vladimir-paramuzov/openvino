// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include "extension/executor.hpp"
#include "extension/op_implementation.hpp"
#include "implementation_params.hpp"
#include "implementation_selector.hpp"
#include "openvino/core/node.hpp"


namespace ov {

// Factory handles available primitives implementations
// and allows to create executor for a given operation
class ImplementationsFactory {
public:
    virtual ~ImplementationsFactory() = default;
    virtual OpImplementation::Ptr select_best_implementation(const ov::Node* node) = 0;
    virtual OpExecutor::Ptr create_executor(OpImplementation::Ptr impl, const ImplementationBuilder& builder) = 0;

    std::shared_ptr<ImplementationParameters> m_params = nullptr;
    std::shared_ptr<ImplSelector> m_impl_selector = nullptr;

    void initialize_selector(const ov::Node* node);

    ImplementationsList m_available_impls;
};

template<typename NodeType, typename ParametersType>
class TypedFactory : public ImplementationsFactory{
public:
    ParametersType get_params() { return *std::dynamic_pointer_cast<ParametersType>(m_params); }

    TypedFactory(const ov::Node* node, const ImplementationsList& all_impls)
        : TypedFactory(dynamic_cast<const NodeType*>(node), all_impls) { }
    TypedFactory(const NodeType* node, const ImplementationsList& all_impls) {
        m_params = make_params(node);
        initialize_selector(node);
        m_available_impls = filter_unsupported(m_params.get(), all_impls);
    }

    OpImplementation::Ptr select_best_implementation(const ov::Node* node) override {
        return m_impl_selector->select_best_implementation(m_available_impls, node);
    }

    OpExecutor::Ptr create_executor(OpImplementation::Ptr impl, const ImplementationBuilder& builder) override {
        return impl->get_executor(m_params.get()/*, builder*/);
    }

protected:
    std::shared_ptr<ParametersType> make_params(const NodeType* node) const {
        return std::make_shared<ParametersType>(node);
    }

    ImplementationsList filter_unsupported(const ImplementationParameters* params, const ImplementationsList& impls) const {
        ImplementationsList supported;
        for (const auto& impl : impls) {
            if (impl->supports(params))
                supported.push_back(impl);
        }

        return supported;
    }
};


}  // namespace ov
