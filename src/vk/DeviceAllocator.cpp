#include "vk/DeviceAllocator.hpp"

#include "vk/Common.hpp"

#include <algorithm>
#include <cstdio>

namespace {
// Round `value` up to a multiple of `alignment` (always a power of two in Vulkan).
VkDeviceSize alignUp(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) {
        return value;
    }
    return (value + alignment - 1) & ~(alignment - 1);
}
} // namespace

DeviceAllocator::DeviceAllocator(VkPhysicalDevice physical, VkDevice device) : device_(device) {
    vkGetPhysicalDeviceMemoryProperties(physical, &memProps_);
}

DeviceAllocator::~DeviceAllocator() {
    for (const auto& block : blocks_) {
        if (block->mapped != nullptr) {
            vkUnmapMemory(device_, block->memory);
        }
        vkFreeMemory(device_, block->memory, nullptr);
    }
    std::printf("DeviceAllocator: %zu block(s), %.1f MB of VkDeviceMemory reserved\n",
                blocks_.size(), static_cast<double>(reservedBytes_) / (1024.0 * 1024.0));
}

uint32_t DeviceAllocator::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags required) const {
    for (uint32_t i = 0; i < memProps_.memoryTypeCount; ++i) {
        const bool typeAllowed = (typeBits & (1u << i)) != 0;
        const bool hasProps =
            (memProps_.memoryTypes[i].propertyFlags & required) == required;
        if (typeAllowed && hasProps) {
            return i;
        }
    }
    throw std::runtime_error("DeviceAllocator: no memory type matches the requirements");
}

DeviceAllocator::Block* DeviceAllocator::createBlock(uint32_t memoryTypeIndex, VkDeviceSize size,
                                                     bool linear, bool hostVisible) {
    auto block = std::make_unique<Block>();
    block->size = size;
    block->memoryTypeIndex = memoryTypeIndex;
    block->linear = linear;

    VkMemoryAllocateInfo ai{};
    ai.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    ai.allocationSize = size;
    ai.memoryTypeIndex = memoryTypeIndex;
    VK_CHECK(vkAllocateMemory(device_, &ai, nullptr, &block->memory));

    if (hostVisible) {
        VK_CHECK(vkMapMemory(device_, block->memory, 0, VK_WHOLE_SIZE, 0, &block->mapped));
    }
    block->freeList.push_back({0, size});
    reservedBytes_ += size;

    blocks_.push_back(std::move(block));
    return blocks_.back().get();
}

bool DeviceAllocator::tryAllocateFromBlock(Block& block, const VkMemoryRequirements& reqs,
                                           Allocation& out) {
    for (size_t i = 0; i < block.freeList.size(); ++i) {
        const FreeRange range = block.freeList[i];
        const VkDeviceSize aligned = alignUp(range.offset, reqs.alignment);
        const VkDeviceSize padding = aligned - range.offset;
        if (range.size < padding + reqs.size) {
            continue; // doesn't fit even after alignment padding
        }

        // Carve [aligned, aligned + reqs.size) out of this free range. Any padding before it and
        // any remainder after it go back on the free list.
        const VkDeviceSize rangeEnd = range.offset + range.size;
        block.freeList.erase(block.freeList.begin() + static_cast<std::ptrdiff_t>(i));
        const VkDeviceSize allocEnd = aligned + reqs.size;
        if (allocEnd < rangeEnd) {
            block.freeList.insert(block.freeList.begin() + static_cast<std::ptrdiff_t>(i),
                                  {allocEnd, rangeEnd - allocEnd});
        }
        if (padding > 0) {
            block.freeList.insert(block.freeList.begin() + static_cast<std::ptrdiff_t>(i),
                                  {range.offset, padding});
        }

        out.memory = block.memory;
        out.offset = aligned;
        out.size = reqs.size;
        out.mapped =
            block.mapped != nullptr ? static_cast<char*>(block.mapped) + aligned : nullptr;
        out.block = &block;
        return true;
    }
    return false;
}

DeviceAllocator::Allocation DeviceAllocator::allocate(const VkMemoryRequirements& reqs,
                                                      VkMemoryPropertyFlags required, bool linear) {
    std::lock_guard<std::mutex> lock(mutex_);

    const uint32_t typeIndex = findMemoryType(reqs.memoryTypeBits, required);
    const bool hostVisible =
        (memProps_.memoryTypes[typeIndex].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    Allocation out{};
    for (const auto& block : blocks_) {
        if (block->memoryTypeIndex == typeIndex && block->linear == linear) {
            if (tryAllocateFromBlock(*block, reqs, out)) {
                return out;
            }
        }
    }

    // No existing block had room: create one (large enough for an oversized request).
    const VkDeviceSize blockSize = std::max(kBlockSize, alignUp(reqs.size, reqs.alignment));
    Block* block = createBlock(typeIndex, blockSize, linear, hostVisible);
    if (!tryAllocateFromBlock(*block, reqs, out)) {
        throw std::runtime_error("DeviceAllocator: allocation failed in a fresh block");
    }
    return out;
}

void DeviceAllocator::free(const Allocation& allocation) {
    if (allocation.block == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> lock(mutex_);

    Block* block = static_cast<Block*>(allocation.block);
    // Insert the freed range in offset order, then coalesce with adjacent free ranges.
    FreeRange freed{allocation.offset, allocation.size};
    auto pos = std::lower_bound(
        block->freeList.begin(), block->freeList.end(), freed,
        [](const FreeRange& a, const FreeRange& b) { return a.offset < b.offset; });
    block->freeList.insert(pos, freed);

    std::vector<FreeRange> merged;
    merged.reserve(block->freeList.size());
    for (const FreeRange& range : block->freeList) {
        if (!merged.empty() && merged.back().offset + merged.back().size == range.offset) {
            merged.back().size += range.size;
        } else {
            merged.push_back(range);
        }
    }
    block->freeList = std::move(merged);
}
