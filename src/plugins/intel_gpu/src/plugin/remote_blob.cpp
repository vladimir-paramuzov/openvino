// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/remote_blob.hpp"
#include "intel_gpu/plugin/remote_allocators.hpp"
#include "intel_gpu/plugin/itt.hpp"
#include "intel_gpu/plugin/plugin.hpp"
#include "intel_gpu/runtime/device_query.hpp"

using namespace InferenceEngine;
using namespace InferenceEngine::gpu;
using namespace InferenceEngine::details;

namespace ov {
namespace intel_gpu {

RemoteBlobImpl::RemoteBlobImpl(InferenceEngine::gpu::ClContext::Ptr context,
                               cldnn::stream& stream,
                               const cldnn::layout& layout,
                               cldnn::shared_handle mem,
                               cldnn::shared_surface surf,
                               uint32_t plane,
                               BlobType mem_type,
                               cldnn::memory::ptr mem_handle)
    : m_allocator(std::make_shared<RemoteAllocator>())
    , m_context(context)
    , m_stream(stream)
    , m_mem(mem)
    , m_surf(surf)
    , m_plane(plane)
    , m_layout(layout)
    , m_mem_type(mem_type)
    , m_memory_object(mem_handle)
    , m_reused_memory_object(mem_handle != nullptr)
    , lockedCounter(0)
    , lockedHolder(nullptr)
    , _handle(nullptr) {
    OPENVINO_ASSERT(m_context != nullptr, "[GPU] Invalid context object passed to RemoteBlobImpl contructro");
    auto& engine = get_context_impl(m_context)->get_engine();

    if (!m_reused_memory_object) {
        // Verify shared buffer/usm memory and ensure that requested byte size is not greater than allocated one
        switch (m_mem_type) {
        case BlobType::BT_BUF_SHARED: {
            engine.share_buffer(m_layout, m_mem);
            break;
        }
        case BlobType::BT_USM_SHARED: {
            engine.share_usm(m_layout, m_mem);
            break;
        }
        default: break;
        }
    }
}

AnyMap RemoteBlobImpl::getParams() const {
    assert(m_memory_object != nullptr);
    auto params = m_memory_object->get_internal_params();

    switch (m_mem_type) {
    case BlobType::BT_BUF_INTERNAL:
    case BlobType::BT_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BlobType::BT_USM_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(USM_USER_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BlobType::BT_USM_HOST_INTERNAL:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(USM_HOST_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BlobType::BT_USM_DEVICE_INTERNAL:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(USM_DEVICE_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
#ifdef _WIN32
    case BlobType::BT_DX_BUF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(DX_BUFFER) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface }
        };
#endif
    case BlobType::BT_IMG_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(OCL_IMAGE2D) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem }
        };
    case BlobType::BT_SURF_SHARED:
        return{
            { GPU_PARAM_KEY(SHARED_MEM_TYPE), GPU_PARAM_VALUE(VA_SURFACE) },
            { GPU_PARAM_KEY(OCL_CONTEXT), params.context },
            { GPU_PARAM_KEY(VA_DEVICE),   params.user_device },
            { GPU_PARAM_KEY(MEM_HANDLE),  params.mem },
            { GPU_PARAM_KEY(DEV_OBJECT_HANDLE), params.surface },
            { GPU_PARAM_KEY(VA_PLANE),  params.plane }
        };
    default:
        IE_THROW() << "Unsupported shared object type " << static_cast<int>(m_mem_type);
    }
}

bool RemoteBlobImpl::deallocate() noexcept {
    m_memory_object.reset();
    return m_memory_object == nullptr;
}

bool RemoteBlobImpl::is_allocated() const noexcept {
    return m_memory_object != nullptr;
}

bool RemoteBlobImpl::is_locked() const noexcept {
    return lockedHolder != nullptr;
}

void RemoteBlobImpl::allocate() {
    OV_ITT_SCOPED_TASK(itt::domains::intel_gpu_plugin, "RemoteBlobImpl::Allocate");
    assert(m_memory_object == nullptr || m_reused_memory_object);

    if (m_reused_memory_object)
        return;

    auto& engine = get_context_impl(m_context)->get_engine();

    switch (m_mem_type) {
    case BlobType::BT_BUF_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::cl_mem);
        break;
    }
    case BlobType::BT_USM_HOST_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::usm_host);
        break;
    }
    case BlobType::BT_USM_DEVICE_INTERNAL: {
        m_memory_object = engine.allocate_memory(m_layout, cldnn::allocation_type::usm_device);
        break;
    }
    case BlobType::BT_BUF_SHARED: {
        m_memory_object = engine.share_buffer(m_layout, m_mem);
        break;
    }
    case BlobType::BT_USM_SHARED: {
        m_memory_object = engine.share_usm(m_layout, m_mem);
        break;
    }
#ifdef _WIN32
    case BlobType::BT_SURF_SHARED: {
        m_memory_object = engine.share_surface(m_layout, m_mem, m_plane);
        break;
    }
    case BlobType::BT_DX_BUF_SHARED: {
        m_memory_object = engine.share_dx_buffer(m_layout, m_mem);
        break;
    }
#else
    case BlobType::BT_SURF_SHARED: {
        m_memory_object = engine.share_surface(m_layout, m_surf, m_plane);
        break;
    }
#endif
    case BlobType::BT_IMG_SHARED: {
        m_memory_object = engine.share_image(m_layout, m_mem);
        break;
    }
    default:
        m_memory_object.reset();
    }
}

const std::shared_ptr<IAllocator>& RemoteBlobImpl::getAllocator() const noexcept {
    return m_allocator;
};

std::string RemoteBlobImpl::getDeviceName() const noexcept {
    return m_context->getDeviceName();
};

std::shared_ptr<InferenceEngine::RemoteContext> RemoteBlobImpl::getContext() const noexcept {
    return m_context;
}

void RemoteBlobImpl::reinterpret(cldnn::layout new_layout) {
    OPENVINO_ASSERT(m_layout.bytes_count() >= new_layout.bytes_count(),
                    "[GPU] Can't reinterpret blob to the size bigger than allocated memory buffer");
    m_layout = new_layout;
    auto engine = m_memory_object->get_engine();
    m_memory_object = engine->reinterpret_buffer(*m_memory_object, new_layout);
}

void RemoteBlobImpl::lock() const {
    if (!is_allocated()) {
        IE_THROW(NotAllocated) << "[GPU] Remote blob can't be locked as it's not allocated";
    }

    std::lock_guard<std::mutex> locker(lockedMutex);
    if (lockedCounter == 0) {
        lockedHolder = std::unique_ptr<cldnn::mem_lock<uint8_t>>(new cldnn::mem_lock<uint8_t>(m_memory_object, m_stream));
        auto ptr = lockedHolder->data();
        _handle = reinterpret_cast<void*>(ptr);
        auto casted_allocator = std::dynamic_pointer_cast<RemoteAllocator>(m_allocator);
        OPENVINO_ASSERT(casted_allocator, "[GPU] Invalid remote allocator type");
        casted_allocator->regLockedBlob(_handle, this);
    }
    lockedCounter++;
}

void RemoteBlobImpl::unlock() const {
    std::lock_guard<std::mutex> locker(lockedMutex);
    lockedCounter--;
    if (lockedCounter == 0)
        lockedHolder.reset();
}

LockedMemory<void> RemoteBlobImpl::buffer() noexcept {
    try {
        lock();
        return LockedMemory<void>(m_allocator.get(), _handle, 0);
    } catch (...) {
        return LockedMemory<void>(nullptr, nullptr, 0);
    }
}

LockedMemory<const void> RemoteBlobImpl::cbuffer() const noexcept {
    try {
        lock();
        return LockedMemory<const void>(m_allocator.get(), _handle, 0);
    } catch (...) {
        return LockedMemory<const void>(nullptr, nullptr, 0);
    }
}

LockedMemory<void> RemoteBlobImpl::rwmap() noexcept {
    try {
        lock();
        return LockedMemory<void>(m_allocator.get(), _handle, 0);
    } catch (...) {
        return LockedMemory<void>(nullptr, nullptr, 0);
    }
}

LockedMemory<const void> RemoteBlobImpl::rmap() const noexcept {
    try {
        lock();
        return LockedMemory<const void>(m_allocator.get(), _handle, 0);
    } catch (...) {
        return LockedMemory<const void>(nullptr, nullptr, 0);
    }
}

LockedMemory<void> RemoteBlobImpl::wmap() noexcept {
    try {
        lock();
        return LockedMemory<void>(m_allocator.get(), _handle, 0);
    } catch (...) {
        return LockedMemory<void>(nullptr, nullptr, 0);
    }
}

}  // namespace intel_gpu
}  // namespace ov
