#pragma once

#include <vulkan/vulkan.h>

// Owns the VkInstance and (when validation is enabled) a debug-utils messenger.
class Instance {
public:
    explicit Instance(bool enableValidation);
    ~Instance();

    Instance(const Instance&) = delete;
    Instance& operator=(const Instance&) = delete;

    VkInstance handle() const { return instance_; }
    bool validationEnabled() const { return validationEnabled_; }

private:
    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT debugMessenger_ = VK_NULL_HANDLE;
    bool validationEnabled_ = false;
};
