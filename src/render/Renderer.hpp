#pragma once

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

class Window;
class Device;
class Swapchain;

// Owns the render pass, graphics pipeline, framebuffers, command buffers, and per-frame
// sync objects; runs the main draw loop and handles swapchain recreation on resize.
class Renderer {
public:
    Renderer(Window& window, Device& device, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void run();

private:
    void createRenderPass();
    void createPipeline();
    void createFramebuffers();
    void createCommandResources();
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void drawFrame();
    void recreateSwapchain();

    VkShaderModule loadShaderModule(const char* path);

    static constexpr int kFramesInFlight = 2;

    Window& window_;
    Device& device_;
    Swapchain& swapchain_;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_; // one per frame in flight

    std::vector<VkSemaphore> imageAvailable_; // per frame in flight
    std::vector<VkSemaphore> renderFinished_; // per swapchain image
    std::vector<VkFence> inFlight_;           // per frame in flight

    uint32_t currentFrame_ = 0;
};
