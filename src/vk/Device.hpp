#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <optional>
#include <vector>

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    bool isComplete() const { return graphics.has_value() && present.has_value(); }
};

struct SwapchainSupport {
    VkSurfaceCapabilitiesKHR capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR> presentModes;
};

// Selects a physical device (prefers a discrete GPU) and creates the logical device + queues.
class Device {
public:
    Device(VkInstance instance, VkSurfaceKHR surface, bool enableValidation);
    ~Device();

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    VkPhysicalDevice physical() const { return physical_; }
    VkDevice handle() const { return device_; }
    VkQueue graphicsQueue() const { return graphicsQueue_; }
    VkQueue presentQueue() const { return presentQueue_; }
    const QueueFamilyIndices& queueFamilies() const { return indices_; }

    SwapchainSupport querySwapchainSupport() const;

private:
    VkSurfaceKHR surface_;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    QueueFamilyIndices indices_;
    VkQueue graphicsQueue_ = VK_NULL_HANDLE;
    VkQueue presentQueue_ = VK_NULL_HANDLE;
};
