#pragma once

#include <vulkan/vulkan.h>

#include "core/ThreadPool.hpp"
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
class DeviceAllocator;
class Ibl;
struct MeshData;
struct TextureData;

// Deferred renderer. Each frame runs three passes: a shadow pass (scene depth from the light),
// a geometry pass that writes surface attributes into a G-buffer, and a fullscreen lighting
// pass that reads the G-buffer and shades with the shadowed directional light plus IBL.
class Renderer {
public:
    Renderer(Window& window, Device& device, DeviceAllocator& allocator, Swapchain& swapchain);
    ~Renderer();

    Renderer(const Renderer&) = delete;
    Renderer& operator=(const Renderer&) = delete;

    void run();

private:
    void createPipelineCache();
    void savePipelineCache() const;
    void createTimestampPool();
    void readTimestamps(uint32_t frame);
    void createShadowResources();
    void createGeometryRenderPass();
    void createLightingRenderPass();
    void createDescriptorSetLayouts();
    void createGeometryPipeline();
    void createShadowPipeline();
    void createLightingPipeline();
    void createSsaoResources();
    void createSsaoDescriptors();
    void writeSsaoDescriptors();
    void createGBuffers();
    void createFramebuffers();
    void createCommandResources();
    void createThreadResources();
    void createIbl();
    void createTextures(const MeshData& model);
    void createMesh(const MeshData& model);
    void buildMeshSubDraws(const MeshData& model);
    void createGround();
    void createUniformBuffers();
    void createDescriptorPool();
    void createGeomDescriptors();
    void createLightDescriptors();
    void writeLightDescriptors(); // (re)points the light sets at the current G-buffer views
    void createSyncObjects();

    void recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex);
    void buildInstances(float time);
    void recordGeometrySecondary(int threadIndex, uint32_t frame, VkExtent2D extent);
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
    static constexpr int kThreadCount = 4;     // parallel geometry-recording workers

    Window& window_;
    Device& device_;
    DeviceAllocator& allocator_;
    Swapchain& swapchain_;

    VkFormat depthFormat_ = VK_FORMAT_UNDEFINED;

    // Serialized across runs so pipeline creation is warm; fed to every pipeline build.
    VkPipelineCache pipelineCache_ = VK_NULL_HANDLE;

    // GPU timestamp profiling: 5 timestamps per frame bracket the shadow / geometry / neural-AO
    // / lighting passes; results are read back one frame later and shown in the window title.
    static constexpr uint32_t kTimestampsPerFrame = 5;
    VkQueryPool timestampPool_ = VK_NULL_HANDLE;
    bool timestampsSupported_ = false;
    double timestampPeriodNs_ = 0.0;
    uint64_t timestampMask_ = ~0ull;
    double gpuShadowMs_ = 0.0;
    double gpuGeomMs_ = 0.0;
    double gpuSsaoMs_ = 0.0;
    double gpuLightMs_ = 0.0;
    uint64_t frameIndex_ = 0;
    uint32_t titleThrottle_ = 0;

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
        Image ssao;     // r16f neural ambient occlusion (compute output)
        VkFramebuffer framebuffer = VK_NULL_HANDLE;
    };
    std::array<GBuffer, kFramesInFlight> gbuffers_;
    VkSampler gbufferSampler_ = VK_NULL_HANDLE;

    // Neural AO compute pass: an MLP (weights in an SSBO) reads the G-buffer and writes ssao.
    VkDescriptorSetLayout ssaoSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout ssaoPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline ssaoPipeline_ = VK_NULL_HANDLE;
    Buffer ssaoWeights_;
    std::vector<VkDescriptorSet> ssaoDescriptorSets_; // per frame

    // Lighting pass: fullscreen, reads the G-buffer + shadow + IBL, writes the swapchain.
    VkRenderPass lightRenderPass_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout lightSetLayout_ = VK_NULL_HANDLE;
    VkPipelineLayout lightPipelineLayout_ = VK_NULL_HANDLE;
    VkPipeline lightPipeline_ = VK_NULL_HANDLE;
    std::vector<VkFramebuffer> framebuffers_; // one per swapchain image

    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    std::vector<VkCommandBuffer> commandBuffers_;

    // Multithreaded geometry recording: one worker + command pool per thread, each producing a
    // secondary command buffer per frame in flight. Command pools are not thread-safe, so each
    // worker owns its own.
    ThreadPool threadPool_{kThreadCount};
    std::array<VkCommandPool, kThreadCount> threadCommandPools_{};
    std::array<std::array<VkCommandBuffer, kFramesInFlight>, kThreadCount> geomSecondaries_{};

    // Per-object material factors; emissiveFactor.w is the "use textures" flag (0 = flat).
    struct MaterialPush {
        glm::vec4 baseColorFactor{1.0f};
        glm::vec4 emissiveFactor{0.0f};
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
    };
    MaterialPush groundMaterial_;

    // A material-homogeneous slice of a mesh's index buffer (one glTF material).
    struct SubDraw {
        uint32_t firstIndex;
        uint32_t indexCount;
        MaterialPush material;
    };
    std::vector<SubDraw> meshSubDraws_;   // the loaded model, split by material
    std::vector<SubDraw> groundSubDraws_; // the ground plane (single flat material)

    // One drawable object; the per-frame instance list is partitioned across the workers. The
    // shadow pass draws totalIndexCount at once; the geometry pass draws each subDraw separately.
    struct DrawInstance {
        glm::mat4 model;
        const Buffer* vertexBuffer;
        const Buffer* indexBuffer;
        uint32_t totalIndexCount;
        const std::vector<SubDraw>* subDraws;
    };
    std::vector<DrawInstance> instances_;

    // Material maps (bindings 1..5 of the geometry set). One sampler shared across them.
    std::array<Image, kTextureCount> textures_;
    VkSampler sampler_ = VK_NULL_HANDLE;

    std::unique_ptr<Ibl> ibl_;

    // Mesh + ground geometry.
    Buffer vertexBuffer_;
    Buffer indexBuffer_;
    uint32_t indexCount_ = 0;
    Buffer groundVertexBuffer_;
    Buffer groundIndexBuffer_;
    uint32_t groundIndexCount_ = 0;

    glm::vec3 modelCenter_{0.0f};
    float modelRadius_ = 1.0f;
    float modelMinY_ = 0.0f; // lowest vertex Y, for resting the mesh on the ground

    // Per-frame transforms computed in updateUniformBuffer and reused when recording.
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
