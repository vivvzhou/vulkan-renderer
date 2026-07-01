#pragma once

#include "vk/vma.hpp"

// RAII wrapper around a VkImage + its VMA allocation + a matching VkImageView. Move-only.
// Used for both the depth buffer and sampled textures; the caller picks the format, usage,
// and aspect mask (color vs depth).
class Image {
public:
    Image() = default;
    Image(VmaAllocator allocator, VkDevice device, uint32_t width, uint32_t height,
          VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect);
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

    VmaAllocator allocator_ = nullptr;
    VkDevice device_ = VK_NULL_HANDLE;
    VkImage image_ = VK_NULL_HANDLE;
    VmaAllocation allocation_ = nullptr;
    VkImageView view_ = VK_NULL_HANDLE;
    VkFormat format_ = VK_FORMAT_UNDEFINED;
};
