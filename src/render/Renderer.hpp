#pragma once

#include <vulkan/vulkan.h>

#include "vk/Buffer.hpp"
#include "vk/Image.hpp"

#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

class Window;
class Device;
class Swapchain;
class Allocator;
class Ibl;
struct MeshData;
struct TextureData;

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
    void createIbl();
    void createTextures(const MeshData& model);
    void createMesh(const MeshData& model);
    void createUniformBuffers();
    void createDescriptorPool();
    void createDescriptorSets();
    void createSkyboxPipeline();
    void createSkyboxDescriptors();
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
    Image uploadTexture(const TextureData& tex, VkFormat format);

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

    // Mesh + PBR material.
    Buffer vertexBuffer_;
    Buffer indexBuffer_;
    uint32_t indexCount_ = 0;

    // Five material maps, in binding order: base color, metallic-roughness, normal, emissive,
    // occlusion (see kTextureCount). One sampler is shared across all of them. The three IBL
    // maps follow at bindings 6, 7, 8.
    static constexpr int kTextureCount = 5;
    static constexpr int kIblTextureCount = 3;
    std::array<Image, kTextureCount> textures_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    std::unique_ptr<Ibl> ibl_;

    // Skybox: its own pipeline sampling the environment map, reusing the camera UBO.
    VkDescriptorSetLayout skyboxSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout skyboxPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline skyboxPipeline_ = VK_NULL_HANDLE;
    VkDescriptorPool skyboxDescriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> skyboxDescriptorSets_;

    // Material factors pushed as push constants; layout matches the shader's Material block.
    struct MaterialPush {
        glm::vec4 baseColorFactor{1.0f};
        glm::vec4 emissiveFactor{0.0f};
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
    } material_;

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
