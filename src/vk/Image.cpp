#include "vk/Image.hpp"

#include "vk/Common.hpp"

Image::Image(VmaAllocator allocator, VkDevice device, uint32_t width, uint32_t height,
             VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspect)
    : allocator_(allocator), device_(device), format_(format) {
    VkImageCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.extent = {width, height, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.format = format;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    ici.usage = usage;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    // Images are large; ask VMA to give them a dedicated VkDeviceMemory block when appropriate.
    aci.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;

    VK_CHECK(vmaCreateImage(allocator_, &ici, &aci, &image_, &allocation_, nullptr));

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange.aspectMask = aspect;
    vi.subresourceRange.baseMipLevel = 0;
    vi.subresourceRange.levelCount = 1;
    vi.subresourceRange.baseArrayLayer = 0;
    vi.subresourceRange.layerCount = 1;
    VK_CHECK(vkCreateImageView(device_, &vi, nullptr, &view_));
}

Image::~Image() { reset(); }

void Image::reset() {
    if (view_ != VK_NULL_HANDLE) {
        vkDestroyImageView(device_, view_, nullptr);
    }
    if (image_ != VK_NULL_HANDLE) {
        vmaDestroyImage(allocator_, image_, allocation_);
    }
    allocator_ = nullptr;
    device_ = VK_NULL_HANDLE;
    image_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    view_ = VK_NULL_HANDLE;
    format_ = VK_FORMAT_UNDEFINED;
}

Image::Image(Image&& other) noexcept
    : allocator_(other.allocator_),
      device_(other.device_),
      image_(other.image_),
      allocation_(other.allocation_),
      view_(other.view_),
      format_(other.format_) {
    other.allocator_ = nullptr;
    other.device_ = VK_NULL_HANDLE;
    other.image_ = VK_NULL_HANDLE;
    other.allocation_ = nullptr;
    other.view_ = VK_NULL_HANDLE;
    other.format_ = VK_FORMAT_UNDEFINED;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = other.allocator_;
        device_ = other.device_;
        image_ = other.image_;
        allocation_ = other.allocation_;
        view_ = other.view_;
        format_ = other.format_;
        other.allocator_ = nullptr;
        other.device_ = VK_NULL_HANDLE;
        other.image_ = VK_NULL_HANDLE;
        other.allocation_ = nullptr;
        other.view_ = VK_NULL_HANDLE;
        other.format_ = VK_FORMAT_UNDEFINED;
    }
    return *this;
}
