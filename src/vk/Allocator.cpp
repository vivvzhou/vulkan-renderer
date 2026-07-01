#include "vk/Allocator.hpp"

#include "vk/Common.hpp"

Allocator::Allocator(VkInstance instance, VkPhysicalDevice physical, VkDevice device) {
    VmaAllocatorCreateInfo ci{};
    ci.instance = instance;
    ci.physicalDevice = physical;
    ci.device = device;
    ci.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&ci, &allocator_));
}

Allocator::~Allocator() {
    if (allocator_ != nullptr) {
        vmaDestroyAllocator(allocator_);
    }
}
