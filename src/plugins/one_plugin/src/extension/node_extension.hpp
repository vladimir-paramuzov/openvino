// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "extension/executor.hpp"
#include "openvino/core/node.hpp"

#include "extension/implementation_factory.hpp"
#include "extension/implementation_args.hpp"
#include "extension/implementation_params.hpp"
#include "extension/memory_descriptor.hpp"
#include "extension/op_implementation.hpp"
#include "extension/layout_optimizer.hpp"
#include "extension/optimization_attributes.hpp"


#include <memory>


namespace ov {

class NodeExtension {
public:
    virtual ~NodeExtension() = default;
    NodeExtension() = default;

    void visit_attributes(AttributeVisitor& visitor);

    const MemoryDescs& get_memory_desc() const;
    void set_memory_desc(const Argument& arg, const MemoryDesc& desc);
    void set_memory_descs(const MemoryDescs& descs);

    void set_inplace();
    bool is_inplace() const;

    virtual void select_preferred_formats();
    virtual void select_best_implementation() = 0;
    std::shared_ptr<OpImplementation> get_impl() const;
    std::shared_ptr<OpExecutor> get_executor() const;
    void create_executor(const ImplementationBuilders& builder);

    void add_fused_op(std::shared_ptr<ov::Node> op);
    void set_fused_ops(std::shared_ptr<ov::Model> fused_ops);

    const ov::Node* get_node_ptr() const;

    void set_layout_optimizer(std::shared_ptr<const LayoutOptimizer> layout_optimizer);
    void set_affinity(const NodeAffinity& affinity);
    void set_affinity(const DeviceType& device_type);
    NodeAffinity get_affinity() const;

protected:
    const ov::Node* m_node;
    MemoryDescs m_memory_desc;
    std::shared_ptr<ov::Model> m_fused_ops = nullptr;
    std::shared_ptr<ImplementationsFactory> m_factory;
    std::shared_ptr<OpImplementation> m_best_implementation = nullptr;
    std::shared_ptr<OptimizationAttributes> m_opt_attributes = nullptr;
    std::shared_ptr<const LayoutOptimizer> m_layout_optimizer = nullptr;
    std::shared_ptr<OpExecutor> m_executor = nullptr;


    NodeAffinity m_affinity;

    virtual void initialize_descriptors();
};

template <typename NodeType, typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtensionBase : public NodeExtension {
public:
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    void init_factory(const ov::Node* ptr) {
        m_node = ptr;
        m_factory = std::make_shared<FactoryType>(ptr);
    }
    template<typename FactoryType, typename std::enable_if<std::is_base_of<ImplementationsFactory, FactoryType>::value, bool>::type = true>
    FactoryType& get_factory() const {
        return static_cast<FactoryType&>(m_factory);
    }

    void select_best_implementation() override {
        m_best_implementation = m_factory->select_best_implementation(m_node);
    }
};

template <typename NodeType, typename std::enable_if<std::is_base_of<ov::Node, NodeType>::value, bool>::type = true>
class TypedNodeExtension : public TypedNodeExtensionBase<NodeType> { };

}  // namespace ov
