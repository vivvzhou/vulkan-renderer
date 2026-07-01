#pragma once

#include <vulkan/vulkan.h>

struct GLFWwindow;

// Thin GLFW wrapper: owns the window, creates the Vulkan surface, and tracks resize.
class Window {
public:
    Window(int width, int height, const char* title);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    bool shouldClose() const;
    void pollEvents() const;
    void waitEvents() const;

    VkSurfaceKHR createSurface(VkInstance instance) const;
    void framebufferSize(int& width, int& height) const;

    GLFWwindow* handle() const { return window_; }

    bool framebufferResized = false;

private:
    GLFWwindow* window_ = nullptr;
};
