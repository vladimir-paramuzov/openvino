// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "op_implementation.hpp"

namespace ov {

template<typename ImplementationParams>
struct ImplementationsRegistry {
public:
    virtual ~ImplementationsRegistry() = default;
    const BuildersList& all() const { return m_impls; }

protected:
    ImplementationsRegistry() { }
    template <typename ImplType, typename std::enable_if<std::is_base_of<OpImplementation, ImplType>::value, bool>::type = true>
    void register_impl() {
        m_impls.push_back(
            [](const FactoryParameters& params) {
                return std::make_shared<ImplType>(static_cast<const ImplementationParams&>(params));
            });
    }

    BuildersList m_impls;
};

}  // namespace ov
