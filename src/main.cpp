// Phase 1: window + swapchain + a first triangle.
//
// Wires together the RAII Vulkan objects in the correct construction/destruction order
// (surface after instance, device after surface, everything torn down in reverse) and hands
// them to the Renderer, which owns the render pass, pipeline, and the main draw loop.

#include "core/Window.hpp"
#include "render/Renderer.hpp"
#include "vk/Device.hpp"
#include "vk/Instance.hpp"
#include "vk/Surface.hpp"
#include "vk/Swapchain.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>

namespace {
constexpr int kWidth = 1280;
constexpr int kHeight = 720;

#ifdef NDEBUG
constexpr bool kEnableValidation = false;
#else
constexpr bool kEnableValidation = true;
#endif
} // namespace

int main() {
    try {
        Window window(kWidth, kHeight, "vulkan-renderer");
        Instance instance(kEnableValidation);
        Surface surface(instance.handle(), window);
        Device device(instance.handle(), surface.handle(), instance.validationEnabled());

        int fbWidth = 0;
        int fbHeight = 0;
        window.framebufferSize(fbWidth, fbHeight);
        Swapchain swapchain(device, surface.handle(), static_cast<uint32_t>(fbWidth),
                            static_cast<uint32_t>(fbHeight));

        Renderer renderer(window, device, swapchain);
        renderer.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "fatal: %s\n", e.what());
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
