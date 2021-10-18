// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_engine.hpp"
#include "ocl_stream.hpp"
#include "cldnn/runtime/memory.hpp"

#include <cassert>
#include <iterator>
#include <mutex>
#include <memory>

namespace cldnn {
namespace ocl {
struct lockable_gpu_mem {
    lockable_gpu_mem() :
        _lock_count(0),
        _mapped_ptr(nullptr) {}

    std::mutex _mutex;
    unsigned _lock_count;
    void* _mapped_ptr;
};

struct gpu_buffer : public lockable_gpu_mem, public memory {
    gpu_buffer(ocl_engine* engine, const layout& new_layout, const cl::Buffer& buffer);
    gpu_buffer(ocl_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern) override;
    event::ptr fill(stream& stream) override;
    shared_mem_params get_internal_params() const override;
    const cl::Buffer& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

    event::ptr copy_from(stream& stream, const memory& other) override;
    event::ptr copy_from(stream& stream, const void* host_ptr) override;
    event::ptr copy_to(stream& stream, void* host_ptr) override;
#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc /* desc */) override;
#endif

protected:
    cl::Buffer _buffer;
};

struct gpu_image2d : public lockable_gpu_mem, public memory {
    gpu_image2d(ocl_engine* engine, const layout& new_layout, const cl::Image2D& buffer);
    gpu_image2d(ocl_engine* engine, const layout& layout);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    event::ptr fill(stream& stream, unsigned char pattern) override;
    event::ptr fill(stream& stream) override;
    shared_mem_params get_internal_params() const override;
    const cl::Image2D& get_buffer() const {
        assert(0 == _lock_count);
        return _buffer;
    }

    event::ptr copy_from(stream& /* stream */, const memory& /* other */) override;
    event::ptr copy_from(stream& /* stream */, const void* /* other */) override;
    event::ptr copy_to(stream& /* stream */, void* /* other */) override;

protected:
    cl::Image2D _buffer;
    size_t _width;
    size_t _height;
    size_t _row_pitch;
    size_t _slice_pitch;
};

struct gpu_media_buffer : public gpu_image2d {
    gpu_media_buffer(ocl_engine* engine, const layout& new_layout, shared_mem_params params);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
#ifdef _WIN32
    void* surface;
#else
    uint32_t surface;
#endif
    uint32_t plane;
};

#ifdef _WIN32
struct gpu_dx_buffer : public gpu_buffer {
    gpu_dx_buffer(ocl_engine* engine, const layout& new_layout, shared_mem_params VAEncMiscParameterTypeSubMbPartPel);
    shared_mem_params get_internal_params() const override;
private:
    void* device;
    void* resource;
};
#endif

struct gpu_usm : public lockable_gpu_mem, public memory {
    gpu_usm(ocl_engine* engine, const layout& new_layout, const cl::UsmMemory& usm_buffer, allocation_type type);
    gpu_usm(ocl_engine* engine, const layout& layout, allocation_type type);

    void* lock(const stream& stream, mem_lock_type type = mem_lock_type::read_write) override;
    void unlock(const stream& stream) override;
    const cl::UsmMemory& get_buffer() const { return _buffer; }
    cl::UsmMemory& get_buffer() { return _buffer; }

    event::ptr fill(stream& stream, unsigned char pattern) override;
    event::ptr fill(stream& stream) override;
    shared_mem_params get_internal_params() const override;

    event::ptr copy_from(stream& stream, const memory& other) override;
    event::ptr copy_from(stream& stream, const void* host_ptr) override;
    event::ptr copy_to(stream& stream, void* host_ptr) override;

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::memory get_onednn_memory(dnnl::memory::desc desc) override;
#endif

protected:
    cl::UsmMemory _buffer;
};

struct ocl_surfaces_lock : public surfaces_lock {
    ocl_surfaces_lock(std::vector<memory::ptr> mem, const stream& stream);

    ~ocl_surfaces_lock() = default;
private:
    std::vector<cl_mem> get_handles(std::vector<memory::ptr> mem) const;
    const stream& _stream;
    std::vector<cl_mem> _handles;
    std::unique_ptr<cl::SharedSurfLock> _lock;
};
}  // namespace ocl
}  // namespace cldnn
