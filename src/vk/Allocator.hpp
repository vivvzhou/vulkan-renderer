#pragma once

#include "vk/vma.hpp"

// Owns the VMA allocator: a single object that sub-allocates all buffers and images out of
// large VkDeviceMemory heaps, so we don't hit the driver's (small) allocation-count limit.
// This is the reference allocator; Phase 7 replaces it with a hand-written one.
class Allocator {
public:
    Allocator(VkInstance instance, VkPhysicalDevice physical, VkDevice device);
    ~Allocator();

    Allocator(const Allocator&) = delete;
    Allocator& operator=(const Allocator&) = delete;

    VmaAllocator handle() const { return allocator_; }

private:
    VmaAllocator allocator_ = nullptr;
};
