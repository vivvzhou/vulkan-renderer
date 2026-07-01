#pragma once

#include <vulkan/vulkan.h>

#include "vk/Buffer.hpp"
#include "vk/Image.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <functional>
#include <vector>

class Window;
class Device;
class Swapchain;
class Allocator;
struct MeshData;

// Owns the render pass, graphics pipeline, framebuffers, depth buffer, per-frame uniform
// buffers + descriptor sets, the mesh (vertex/index buffers), a texture, and per-frame sync
// objects; runs the main draw loop and handles swapchain recreation on resize.
class Renderer {
public:
    Renderer(Window& window, Device& device, Allocator& allocator, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void run();

private:
    void createRenderPass();
    void createDescriptorSetLayout();
    void createPipeline();
    void createDepthResources();
    void createFramebuffers();
    void createCommandResources();
    void createTexture(const MeshData& model);
    void createMesh(const MeshData& model);
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t frame);
    void drawFrame();
    void recreateSwapchain();

    VkShaderModule loadShaderModule(const char* path);
    VkFormat findDepthFormat() const;

    // Records a one-off command buffer, submits it, and blocks until the GPU finishes.
    // Used for staging uploads (buffer copies, image layout transitions).
    void immediateSubmit(const std::function<void(VkCommandBuffer)>& record);
    Buffer createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);

    static constexpr int kFramesInFlight = 2;

    Window& window_;
    Device& device_;
    Allocator& allocator_;
    Swapchain& swapchain_;

    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    VkRenderPass renderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;

    Image depthImage_;
    std::vector<VkFramebuffer> framebuffers_;

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_; // one per frame in flight

    // Mesh + texture.
    Buffer vertexBuffer_;
    Buffer indexBuffer_;
    uint32_t indexCount_ = 0;
    Image texture_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    // Used to auto-fit the loaded mesh (any authored scale) into view.
    glm::vec3 modelCenter_{0.0f};
    float modelRadius_ = 1.0f;

    // Per-frame uniform buffers (persistently mapped host-visible memory) + descriptor sets.
    std::vector<Buffer> uniformBuffers_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> descriptorSets_;

    std::vector<VkSemaphore> imageAvailable_; // per frame in flight
    std::vector<VkSemaphore> renderFinished_; // per swapchain image
    std::vector<VkFence> inFlight_;           // per frame in flight

    uint32_t currentFrame_ = 0;
};
