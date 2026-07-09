#pragma once

#include "vk/DeviceAllocator.hpp"

#include <vulkan/vulkan.h>

// RAII wrapper around a VkBuffer plus its sub-allocation from the custom DeviceAllocator.
// Move-only: the destructor destroys the buffer and returns its memory range to the allocator.
class Buffer {
public:
    Buffer() = default;
    Buffer(DeviceAllocator& allocator, VkDeviceSize size, VkBufferUsageFlags usage,
           VkMemoryPropertyFlags properties);
    ~Buffer();

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    VkBuffer handle() const { return buffer_; }
    VkDeviceSize size() const { return size_; }

    // Non-null only when the buffer was allocated from host-visible memory.
    void* mapped() const { return allocation_.mapped; }

private:
    void reset();

    DeviceAllocator* allocator_ = nullptr;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    DeviceAllocator::Allocation allocation_{};
    VkDeviceSize size_ = 0;
};
