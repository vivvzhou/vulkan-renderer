#include "vk/Image.hpp"

#include "vk/Common.hpp"

Image::Image(DeviceAllocator& allocator, uint32_t width, uint32_t height, VkFormat format,
             VkImageUsageFlags usage, VkImageAspectFlags aspect)
    : allocator_(&allocator), format_(format) {
    const VkDevice device = allocator.device();

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
    VK_CHECK(vkCreateImage(device, &ici, nullptr, &image_));

    VkMemoryRequirements reqs{};
    vkGetImageMemoryRequirements(device, image_, &reqs);
    // Optimal-tiling images are non-linear resources, kept in device-local memory.
    allocation_ = allocator.allocate(reqs, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, /*linear=*/false);
    VK_CHECK(vkBindImageMemory(device, image_, allocation_.memory, allocation_.offset));

    VkImageViewCreateInfo vi{};
    vi.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    vi.image = image_;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = format;
    vi.subresourceRange = {aspect, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device, &vi, nullptr, &view_));
}

Image::~Image() { reset(); }

void Image::reset() {
    if (image_ != VK_NULL_HANDLE) {
        const VkDevice device = allocator_->device();
        vkDestroyImageView(device, view_, nullptr);
        vkDestroyImage(device, image_, nullptr);
        allocator_->free(allocation_);
    }
    allocator_ = nullptr;
    image_ = VK_NULL_HANDLE;
    allocation_ = {};
    view_ = VK_NULL_HANDLE;
    format_ = VK_FORMAT_UNDEFINED;
}

Image::Image(Image&& other) noexcept
    : allocator_(other.allocator_),
      image_(other.image_),
      allocation_(other.allocation_),
      view_(other.view_),
      format_(other.format_) {
    other.allocator_ = nullptr;
    other.image_ = VK_NULL_HANDLE;
    other.allocation_ = {};
    other.view_ = VK_NULL_HANDLE;
    other.format_ = VK_FORMAT_UNDEFINED;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = other.allocator_;
        image_ = other.image_;
        allocation_ = other.allocation_;
        view_ = other.view_;
        format_ = other.format_;
        other.allocator_ = nullptr;
        other.image_ = VK_NULL_HANDLE;
        other.allocation_ = {};
        other.view_ = VK_NULL_HANDLE;
        other.format_ = VK_FORMAT_UNDEFINED;
    }
    return *this;
}
