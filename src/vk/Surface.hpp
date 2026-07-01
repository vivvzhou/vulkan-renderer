#pragma once

#include <vulkan/vulkan.h>

#include "core/Window.hpp"

// RAII wrapper for the window surface. Declared after Instance and before Device in main()
// so it is destroyed after the device but before the instance (correct Vulkan teardown order).
class Surface {
public:
    Surface(VkInstance instance, const Window& window)
        : instance_(instance), surface_(window.createSurface(instance)) {}

    ~Surface() {
        if (surface_ != VK_NULL_HANDLE) {
            vkDestroySurfaceKHR(instance_, surface_, nullptr);
        }
    }

    Surface(const Surface&) = delete;
    Surface& operator=(const Surface&) = delete;

    VkSurfaceKHR handle() const { return surface_; }

private:
    VkInstance instance_;
    VkSurfaceKHR surface_;
};
