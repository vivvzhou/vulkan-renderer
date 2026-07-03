#pragma once

#include "vk/DeviceAllocator.hpp"

#include <vulkan/vulkan.h>

// RAII wrapper around a VkImage (optimal tiling), its DeviceAllocator sub-allocation, and a
// matching VkImageView. Move-only. Used for the depth buffer, G-buffer targets, and textures.
class Image {
public:
    Image() = default;
    Image(DeviceAllocator& allocator, uint32_t width, uint32_t height, VkFormat format,
          VkImageUsageFlags usage, VkImageAspectFlags aspect);
    ~Image();

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;
    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    VkImage handle() const { return image_; }
    VkImageView view() const { return view_; }
    VkFormat format() const { return format_; }

private:
    void reset();

    DeviceAllocator* allocator_ = nullptr;
    VkImage image_ = VK_NULL_HANDLE;
    DeviceAllocator::Allocation allocation_{};
    VkImageView view_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};
