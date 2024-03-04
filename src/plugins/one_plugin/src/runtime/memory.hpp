// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include "extension/implementation_args.hpp"
#include "memory_descriptor.hpp"
#include "event.hpp"
#include "openvino/runtime/tensor.hpp"
#include "stream.hpp"

namespace ov {
class Engine;

struct Memory {
    using Ptr = std::shared_ptr<Memory>;
    using cptr = std::shared_ptr<const Memory>;
    Memory(Engine* engine, const MemoryDesc& desc);
    Memory(Engine* engine, const MemoryDesc& desc, void* p);

    virtual ~Memory();

    // virtual void* lock(const Stream& stream) = 0;
    // virtual void unlock(const Stream& stream) = 0;

    // virtual Event::Ptr fill(Stream& stream, unsigned char pattern) = 0;
    // virtual Event::Ptr fill(Stream& stream) = 0;

    // only supports gpu_usm
    virtual void* buffer_ptr() const { return nullptr; }

    size_t size() const { return ov::shape_size(m_desc.m_shape.to_shape()) * m_desc.m_data_type.size(); }
    size_t count() const { return ov::shape_size(m_desc.m_shape.to_shape()); }

    ov::element::Type_t element_type() const { return m_desc.m_data_type; }
    ov::Shape shape() const { return m_desc.m_shape.to_shape(); }

    ov::Tensor to_tensor() const {
        return ov::Tensor(element_type(), shape(), ptr);
    }

    void* ptr = nullptr;
    bool external = false;

    // virtual bool is_allocated_by(const Engine& engine) const { return &engine == m_engine; }

    // Engine* get_engine() const { return m_engine; }

    // virtual Event::Ptr copy_from(Stream& /* stream */, const Memory& /* other */, bool blocking = true) = 0;
    // virtual Event::Ptr copy_from(Stream& /* stream */, const void* /* host_ptr */, bool blocking = true, size_t dst_offset = 0, size_t data_size = 0) = 0;

    // virtual Event::Ptr copy_to(Stream& stream, Memory& other, bool blocking = true) { return other.copy_from(stream, *this, blocking); }
    // virtual Event::Ptr copy_to(Stream& /* stream */, void* /* host_ptr */, bool blocking = true) = 0;

protected:
    Engine* m_engine;
    MemoryDesc m_desc;
};

class MemoryArgs : public std::map<ov::Argument, Memory::Ptr> {

};

}  // namespace ov
