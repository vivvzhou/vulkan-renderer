#include "vk/Buffer.hpp"

#include "vk/Common.hpp"

Buffer::Buffer(DeviceAllocator& allocator, VkDeviceSize size, VkBufferUsageFlags usage,
               VkMemoryPropertyFlags properties)
    : allocator_(&allocator), size_(size) {
    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(allocator.device(), &bci, nullptr, &buffer_));

    VkMemoryRequirements reqs{};
    vkGetBufferMemoryRequirements(allocator.device(), buffer_, &reqs);
    allocation_ = allocator.allocate(reqs, properties, /*linear=*/true);
    VK_CHECK(vkBindBufferMemory(allocator.device(), buffer_, allocation_.memory,
                                allocation_.offset));
}

Buffer::~Buffer() { reset(); }

void Buffer::reset() {
    if (buffer_ != VK_NULL_HANDLE) {
        vkDestroyBuffer(allocator_->device(), buffer_, nullptr);
        allocator_->free(allocation_);
    }
    allocator_ = nullptr;
    buffer_ = VK_NULL_HANDLE;
    allocation_ = {};
    size_ = 0;
}

Buffer::Buffer(Buffer&& other) noexcept
    : allocator_(other.allocator_),
      buffer_(other.buffer_),
      allocation_(other.allocation_),
      size_(other.size_) {
    other.allocator_ = nullptr;
    other.buffer_ = VK_NULL_HANDLE;
    other.allocation_ = {};
    other.size_ = 0;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        reset();
        allocator_ = other.allocator_;
        buffer_ = other.buffer_;
        allocation_ = other.allocation_;
        size_ = other.size_;
        other.allocator_ = nullptr;
        other.buffer_ = VK_NULL_HANDLE;
        other.allocation_ = {};
        other.size_ = 0;
    }
    return *this;
}
