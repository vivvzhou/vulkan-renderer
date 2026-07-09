#include "vk/Device.hpp"
#include "vk/Common.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

const std::array<const char*, 1> kDeviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

QueueFamilyIndices findQueueFamilies(VkPhysicalDevice dev, VkSurfaceKHR surface) {
    QueueFamilyIndices indices;
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &count, families.data());

    for (uint32_t i = 0; i < count; ++i) {
        if (families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
            indices.graphics = i;
        }
        VkBool32 present = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(dev, i, surface, &present);
        if (present == VK_TRUE) {
            indices.present = i;
        }
        if (indices.isComplete()) {
            break;
        }
    }
    return indices;
}

bool deviceExtensionsSupported(VkPhysicalDevice dev) {
    uint32_t count = 0;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, available.data());
    for (const char* needed : kDeviceExtensions) {
        bool found = false;
        for (const auto& ext : available) {
            if (std::strcmp(ext.extensionName, needed) == 0) {
                found = true;
                break;
            }
        }
        if (!found) {
            return false;
        }
    }
    return true;
}

// Returns a suitability score, or -1 if the device cannot present our graphics.
int scoreDevice(VkPhysicalDevice dev, VkSurfaceKHR surface) {
    if (!findQueueFamilies(dev, surface).isComplete() || !deviceExtensionsSupported(dev)) {
        return -1;
    }
    uint32_t formatCount = 0;
    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev, surface, &formatCount, nullptr);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev, surface, &presentModeCount, nullptr);
    if (formatCount == 0 || presentModeCount == 0) {
        return -1;
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(dev, &props);
    int score = 0;
    if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 1000; // strongly prefer the discrete GPU over the integrated one
    }
    score += static_cast<int>(props.limits.maxImageDimension2D);
    return score;
}

} // namespace

Device::Device(VkInstance instance, VkSurfaceKHR surface, bool /*enableValidation*/)
    : surface_(surface) {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) {
        throw std::runtime_error("no Vulkan physical devices found");
    }
    std::vector<VkPhysicalDevice> devices(count);
    vkEnumeratePhysicalDevices(instance, &count, devices.data());

    int bestScore = -1;
    for (VkPhysicalDevice dev : devices) {
        const int score = scoreDevice(dev, surface);
        if (score > bestScore) {
            bestScore = score;
            physical_ = dev;
        }
    }
    if (physical_ == VK_NULL_HANDLE || bestScore < 0) {
        throw std::runtime_error("no suitable GPU found");
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_, &props);
    std::printf("Selected GPU: %s\n", props.deviceName);

    indices_ = findQueueFamilies(physical_, surface);

    const std::set<uint32_t> uniqueFamilies = {*indices_.graphics, *indices_.present};
    std::vector<VkDeviceQueueCreateInfo> queueInfos;
    const float priority = 1.0f;
    for (uint32_t family : uniqueFamilies) {
        VkDeviceQueueCreateInfo qi{};
        qi.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi.queueFamilyIndex = family;
        qi.queueCount = 1;
        qi.pQueuePriorities = &priority;
        queueInfos.push_back(qi);
    }

    VkPhysicalDeviceFeatures features{};

    VkDeviceCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    ci.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
    ci.pQueueCreateInfos = queueInfos.data();
    ci.pEnabledFeatures = &features;
    ci.enabledExtensionCount = static_cast<uint32_t>(kDeviceExtensions.size());
    ci.ppEnabledExtensionNames = kDeviceExtensions.data();

    VK_CHECK(vkCreateDevice(physical_, &ci, nullptr, &device_));

    vkGetDeviceQueue(device_, *indices_.graphics, 0, &graphicsQueue_);
    vkGetDeviceQueue(device_, *indices_.present, 0, &presentQueue_);
}

Device::~Device() {
    if (device_ != VK_NULL_HANDLE) {
        vkDestroyDevice(device_, nullptr);
    }
}

SwapchainSupport Device::querySwapchainSupport() const {
    SwapchainSupport support;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &support.capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, nullptr);
    support.formats.resize(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &formatCount, support.formats.data());

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &presentModeCount, nullptr);
    support.presentModes.resize(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &presentModeCount,
                                              support.presentModes.data());

    return support;
}
