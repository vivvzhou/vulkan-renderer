#pragma once

#include "vk/vma.hpp"

// RAII wrapper around a VkBuffer + its VMA allocation. Move-only: ownership of the underlying
// GPU memory transfers with the object, and the destructor frees both the buffer and its
// backing allocation in one vmaDestroyBuffer call.
class Buffer {
public:
    Buffer() = default;
    Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
           VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags = 0);
    ~Buffer();

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }
    VmaAllocation allocation() const { return allocation_; }

    // Non-null only when the buffer was created with VMA_ALLOCATION_CREATE_MAPPED_BIT.
    void* mapped() const { return info_.pMappedData; }

private:
    void reset();

    VmaAllocator allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VmaAllocationInfo info_{};
    VkDeviceSize size_ = 0;
};
