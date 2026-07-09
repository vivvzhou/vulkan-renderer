#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class Device;

// Owns the swapchain, its images, and image views. Recreatable on resize.
class Swapchain {
public:
    Swapchain(const Device& device, VkSurfaceKHR surface, uint32_t width, uint32_t height);
    ~Swapchain();

    Swapchain(const Swapchain&) = delete;
    Swapchain& operator=(const Swapchain&) = delete;

    void recreate(uint32_t width, uint32_t height);

    VkSwapchainKHR handle() const { return swapchain_; }
    VkFormat imageFormat() const { return format_; }
    VkExtent2D extent() const { return extent_; }
    const std::vector<VkImageView>& imageViews() const { return imageViews_; }
    uint32_t imageCount() const { return static_cast<uint32_t>(images_.size()); }

private:
    void create(uint32_t width, uint32_t height);
    void destroy();

    const Device& device_;
    VkSurfaceKHR surface_;
    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    std::vector<VkImage> images_;
    std::vector<VkImageView> imageViews_;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
    VkExtent2D extent_{};
};
