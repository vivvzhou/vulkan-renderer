#include "vk/Swapchain.hpp"
#include "vk/Common.hpp"
#include "vk/Device.hpp"

#include <algorithm>
#include <limits>

namespace {

VkSurfaceFormatKHR chooseFormat(const std::vector<VkSurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_B8G8R8A8_SRGB &&
            f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
            return f;
        }
    }
    return formats[0];
}

VkPresentModeKHR choosePresentMode(const std::vector<VkPresentModeKHR>& modes) {
    for (const auto& m : modes) {
        if (m == VK_PRESENT_MODE_MAILBOX_KHR) {
            return m; // low-latency triple buffering when available
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR; // guaranteed present (vsync)
}

VkExtent2D chooseExtent(const VkSurfaceCapabilitiesKHR& caps, uint32_t width, uint32_t height) {
    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        return caps.currentExtent;
    }
    VkExtent2D extent{width, height};
    extent.width = std::clamp(extent.width, caps.minImageExtent.width, caps.maxImageExtent.width);
    extent.height = std::clamp(extent.height, caps.minImageExtent.height, caps.maxImageExtent.height);
    return extent;
}

} // namespace

Swapchain::Swapchain(const Device& device, VkSurfaceKHR surface, uint32_t width, uint32_t height)
    : device_(device), surface_(surface) {
    create(width, height);
}

Swapchain::~Swapchain() { destroy(); }

void Swapchain::recreate(uint32_t width, uint32_t height) {
    destroy();
    create(width, height);
}

void Swapchain::create(uint32_t width, uint32_t height) {
    const SwapchainSupport support = device_.querySwapchainSupport();

    const VkSurfaceFormatKHR surfaceFormat = chooseFormat(support.formats);
    const VkPresentModeKHR presentMode = choosePresentMode(support.presentModes);
    const VkExtent2D extent = chooseExtent(support.capabilities, width, height);

    uint32_t imageCount = support.capabilities.minImageCount + 1;
    if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount) {
        imageCount = support.capabilities.maxImageCount;
    }

    VkSwapchainCreateInfoKHR ci{};
    ci.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = surfaceFormat.format;
    ci.imageColorSpace = surfaceFormat.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    const QueueFamilyIndices& indices = device_.queueFamilies();
    const uint32_t familyIndices[] = {*indices.graphics, *indices.present};
    if (indices.graphics != indices.present) {
        ci.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = familyIndices;
    } else {
        ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }

    ci.preTransform = support.capabilities.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = presentMode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(device_.handle(), &ci, nullptr, &swapchain_));

    uint32_t count = 0;
    vkGetSwapchainImagesKHR(device_.handle(), swapchain_, &count, nullptr);
    images_.resize(count);
    vkGetSwapchainImagesKHR(device_.handle(), swapchain_, &count, images_.data());

    format_ = surfaceFormat.format;
    extent_ = extent;

    imageViews_.resize(images_.size());
    for (size_t i = 0; i < images_.size(); ++i) {
        VkImageViewCreateInfo vi{};
        vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        vi.image = images_[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = format_;
        vi.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY,
                         VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
        vi.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        vi.subresourceRange.baseMipLevel = 0;
        vi.subresourceRange.levelCount = 1;
        vi.subresourceRange.baseArrayLayer = 0;
        vi.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(device_.handle(), &vi, nullptr, &imageViews_[i]));
    }
}

void Swapchain::destroy() {
    for (VkImageView view : imageViews_) {
        vkDestroyImageView(device_.handle(), view, nullptr);
    }
    imageViews_.clear();
    if (swapchain_ != VK_NULL_HANDLE) {
        vkDestroySwapchainKHR(device_.handle(), swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    images_.clear();
}
