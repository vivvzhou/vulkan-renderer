#include "vk/Buffer.hpp"

#include "vk/Common.hpp"

#include <utility>

Buffer::Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
               VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags)
    : allocator_(allocator), size_(size) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci{};
    aci.usage = memoryUsage;
    aci.flags = flags;

    VK_CHECK(vmaCreateBuffer(allocator_, &bci, &aci, &buffer_, &allocation_, &info_));
}

Buffer::~Buffer() { reset(); }

void Buffer::reset() {
    if (buffer_ != VK_NULL_HANDLE) {
        vmaDestroyBuffer(allocator_, buffer_, allocation_);
    }
    allocator_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    allocation_ = nullptr;
    info_ = {};
    size_ = 0;
}

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      info_(other.info_),
      size_(other.size_) {
    other.allocator_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = nullptr;
    other.info_ = {};
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        info_ = other.info_;
        size_ = other.size_;
        other.allocator_ = nullptr;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = nullptr;
        other.info_ = {};
        other.size_ = 0;
    }
    return *this;
}
