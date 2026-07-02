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

// Deferred renderer. Each frame runs three passes: a shadow pass (scene depth from the light),
// a geometry pass that writes surface attributes into a G-buffer, and a fullscreen lighting
// pass that reads the G-buffer and shades with the shadowed directional light plus IBL.
class Renderer {
public:
    Renderer(Window& window, Device& device, Allocator& allocator, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void run();

private:
    void createShadowResources();
    void createGeometryRenderPass();
    void createLightingRenderPass();
    void createDescriptorSetLayouts();
    void createGeometryPipeline();
    void createShadowPipeline();
    void createLightingPipeline();
    void createGBuffers();
    void createFramebuffers();
    void createCommandResources();
    void createIbl();
    void createTextures(const MeshData& model);
    void createMesh(const MeshData& model);
    void createGround();
    void createUniformBuffers();
    void createDescriptorPool();
    void createGeomDescriptors();
    void createLightDescriptors();
    void writeLightDescriptors(); // (re)points the light sets at the current G-buffer views
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void updateUniformBuffer(uint32_t frame);
    void drawFrame();
    void recreateSwapchain();

    VkShaderModule loadShaderModule(const char* path);
    VkFormat findDepthFormat() const;

    void immediateSubmit(const std::function<void(VkCommandBuffer)>& record);
    Buffer createDeviceLocalBuffer(const void* data, VkDeviceSize size, VkBufferUsageFlags usage);
    Image uploadTexture(const TextureData& tex, VkFormat format);

    static constexpr int kFramesInFlight = 2;
    static constexpr int kTextureCount = 5;    // material maps
    static constexpr int kIblTextureCount = 3; // irradiance, prefilter, brdf LUT
    static constexpr int kGBufferCount = 4;    // position, normal, albedo, emissive
    static constexpr uint32_t kShadowMapSize = 2048;

    Window& window_;
    Device& device_;
    Allocator& allocator_;
    Swapchain& swapchain_;

    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    // Shadow pass: per-frame offscreen depth map rendered from the light.
    VkRenderPass shadowRenderPass_ = VK_NULL_HANDLE;
    VkPipelineLayout shadowPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline shadowPipeline_ = VK_NULL_HANDLE;
    std::array<Image, kFramesInFlight> shadowMaps_;
    std::array<VkFramebuffer, kFramesInFlight> shadowFramebuffers_{};
    VkSampler shadowSampler_ = VK_NULL_HANDLE;

    // Geometry pass: writes the G-buffer. Per-frame G-buffer so frames in flight don't race.
    VkRenderPass geomRenderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout geomSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout geomPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline geomPipeline_ = VK_NULL_HANDLE;
    struct GBuffer {
        Image position; // rgb world position
        Image normal;   // rgb normal, a roughness
        Image albedo;   // rgb albedo, a metallic
        Image emissive; // rgb emissive, a ao
        Image depth;
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
    };
    std::array<GBuffer, kFramesInFlight> gbuffers_;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;

    // Lighting pass: fullscreen, reads the G-buffer + shadow + IBL, writes the swapchain.
    VkRenderPass lightRenderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline lightPipeline_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_; // one per swapchain image

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    // Material maps (bindings 1..5 of the geometry set). One sampler shared across them.
    std::array<Image, kTextureCount> textures_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    std::unique_ptr<Ibl> ibl_;

    // Per-object material factors; emissiveFactor.w is the "use textures" flag (0 = flat ground).
    struct MaterialPush {
        glm::vec4 baseColorFactor{1.0f};
        glm::vec4 emissiveFactor{0.0f};
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
    };
    MaterialPush material_;
    MaterialPush groundMaterial_;

    // Mesh + ground geometry.
    Buffer vertexBuffer_;
    Buffer indexBuffer_;
    uint32_t indexCount_ = 0;
    Buffer groundVertexBuffer_;
    Buffer groundIndexBuffer_;
    uint32_t groundIndexCount_ = 0;

    glm::vec3 modelCenter_{0.0f};
    float modelRadius_ = 1.0f;

    // Per-frame transforms computed in updateUniformBuffer and reused when recording.
    glm::mat4 meshModel_{1.0f};
    glm::mat4 groundModel_{1.0f};
    glm::mat4 lightSpace_{1.0f};

    std::vector<Buffer> uniformBuffers_;
    VkDescriptorPool descriptorPool_ = VK_NULL_HANDLE;
    std::vector<VkDescriptorSet> geomDescriptorSets_;  // per frame
    std::vector<VkDescriptorSet> lightDescriptorSets_; // per frame

    std::vector<VkSemaphore> imageAvailable_; // per frame in flight
    std::vector<VkSemaphore> renderFinished_; // per swapchain image
    std::vector<VkFence> inFlight_;           // per frame in flight

    uint32_t currentFrame_ = 0;
};
