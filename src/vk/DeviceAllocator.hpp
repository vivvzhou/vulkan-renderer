#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

// A hand-written GPU memory allocator (the Phase 7 replacement for VMA, which Phase 2 used as a
// reference). It requests a few large VkDeviceMemory blocks and sub-allocates from them with an
// aligned free-list, honoring VkMemoryRequirements and choosing memory types by property flags.
//
// Linear resources (buffers) and non-linear resources (optimal-tiling images) are kept in
// separate blocks so bufferImageGranularity never applies within a block. Host-visible blocks
// are persistently mapped, so an allocation's `mapped` pointer is ready to use immediately.
class DeviceAllocator {
public:
    struct Allocation {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize offset = 0;
        VkDeviceSize size = 0;
        void* mapped = nullptr; // non-null only for host-visible memory
        void* block = nullptr;  // opaque owning Block*, used by free()
    };

    DeviceAllocator(VkPhysicalDevice physical, VkDevice device);
    ~DeviceAllocator();

    DeviceAllocator(const DeviceAllocator&) = delete;
    DeviceAllocator& operator=(const DeviceAllocator&) = delete;

    // Allocate memory satisfying `reqs`, from a memory type that has all `required` property
    // flags. `linear` selects the buffer (true) vs optimal-image (false) block pool.
    Allocation allocate(const VkMemoryRequirements& reqs, VkMemoryPropertyFlags required,
                        bool linear);
    void free(const Allocation& allocation);

    VkDevice device() const { return device_; }

private:
    struct FreeRange {
        VkDeviceSize offset;
        VkDeviceSize size;
    };
    struct Block {
        VkDeviceMemory memory = VK_NULL_HANDLE;
        VkDeviceSize size = 0;
        uint32_t memoryTypeIndex = 0;
        bool linear = false;
        void* mapped = nullptr;
        std::vector<FreeRange> freeList; // kept sorted by offset
    };

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const;
    Block* createBlock(uint32_t memoryTypeIndex, VkDeviceSize size, bool linear, bool hostVisible);
    bool tryAllocateFromBlock(Block& block, const VkMemoryRequirements& reqs, Allocation& out);

    VkDevice device_;
    VkPhysicalDeviceMemoryProperties memProps_{};
    std::vector<std::unique_ptr<Block>> blocks_;
    std::mutex mutex_;

    static constexpr VkDeviceSize kBlockSize = 64ull * 1024 * 1024; // 64 MB default block
    VkDeviceSize reservedBytes_ = 0; // total VkDeviceMemory requested, for the teardown report
};
