#pragma once

#include "vk/DeviceAllocator.hpp"

#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

// Precomputes the image-based-lighting resources from an HDR environment map, all stored as
// equirectangular 2D images (no cubemaps): the environment itself (for the skybox), a diffuse
// irradiance map, a roughness-mipped prefiltered specular map, and the split-sum BRDF LUT.
// Everything is generated once at construction via compute shaders.
class Ibl {
public:
    Ibl(VkDevice device, DeviceAllocator& allocator, VkQueue queue, VkCommandPool pool,
        const std::string& hdrPath);
    ~Ibl();

    Ibl(const Ibl&) = delete;
    Ibl& operator=(const Ibl&) = delete;

    VkImageView environmentView() const { return env_.view; }
    VkImageView irradianceView() const { return irradiance_.view; }
    VkImageView prefilterView() const { return prefilter_.view; }
    VkImageView brdfLutView() const { return brdf_.view; }
    VkSampler environmentSampler() const { return envSampler_; }
    VkSampler lutSampler() const { return lutSampler_; }
    float prefilterMaxLod() const { return static_cast<float>(kPrefilterMips - 1); }

private:
    struct Tex {
        VkImage image = VK_NULL_HANDLE;
        DeviceAllocator::Allocation alloc{};
        VkImageView view = VK_NULL_HANDLE; // full-mip sampling view
    };

    Tex createTex(uint32_t width, uint32_t height, uint32_t mips, VkFormat format,
                  VkImageUsageFlags usage);
    VkImageView createView(VkImage image, VkFormat format, uint32_t baseMip, uint32_t mipCount);
    VkShaderModule loadModule(const char* name);
    void immediateSubmit(const std::function<void(VkCommandBuffer)>& fn);

    void loadEnvironment(const std::string& hdrPath);
    void computeIrradiance();
    void computePrefilter();
    void computeBrdf();

    VkDevice device_;
    DeviceAllocator& allocator_;
    VkQueue queue_;
    VkCommandPool pool_;

    Tex env_;        // RGBA32F equirectangular environment
    Tex irradiance_; // RGBA16F diffuse irradiance
    Tex prefilter_;  // RGBA16F prefiltered specular (mip = roughness)
    Tex brdf_;       // RG16F BRDF integration LUT
    std::vector<VkImageView> prefilterMipViews_; // per-mip storage views for compute writes

    VkSampler envSampler_ = VK_NULL_HANDLE;
    VkSampler lutSampler_ = VK_NULL_HANDLE;

    static constexpr uint32_t kPrefilterMips = 5;
    static constexpr uint32_t kIrradianceW = 64; // equirect: height = W / 2
    static constexpr uint32_t kPrefilterW = 128;
    static constexpr uint32_t kBrdfSize = 512;
};
