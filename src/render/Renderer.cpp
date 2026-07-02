#include "render/Renderer.hpp"

#include "core/Window.hpp"
#include "render/GltfLoader.hpp"
#include "render/Ibl.hpp"
#include "render/Vertex.hpp"
#include "vk/Allocator.hpp"
#include "vk/Common.hpp"
#include "vk/Device.hpp"
#include "vk/Swapchain.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif
#ifndef ASSET_PATH
#define ASSET_PATH "DamagedHelmet.glb"
#endif
#ifndef ENV_HDR_PATH
#define ENV_HDR_PATH "environment.hdr"
#endif

namespace {

// Matches the CameraUBO block in the shaders (std140-compatible).
struct CameraUBO {
    glm::mat4 view;
    glm::mat4 proj;
    glm::vec4 camPos;    // world-space eye position (xyz)
    glm::vec4 iblParams; // x = prefilter max LOD
    glm::mat4 lightSpace;
    glm::vec4 lightDir;   // directional light travel direction (xyz)
    glm::vec4 lightColor; // rgb intensity
};

// Matches the PushConstants block in mesh.vert/frag.
struct MeshPush {
    glm::mat4 model;
    glm::vec4 baseColorFactor;
    glm::vec4 emissiveFactor; // w = useTextures flag
    float metallicFactor;
    float roughnessFactor;
};

// Fixed eye position; also fed to the fragment shader for the view vector.
constexpr glm::vec3 kEye = glm::vec3(2.0f, 1.5f, 2.5f);

// Directional key light: travel direction and radiance. Casts the shadow.
constexpr glm::vec3 kLightDir = glm::vec3(-0.5f, -1.0f, -0.4f);
constexpr glm::vec3 kLightColor = glm::vec3(3.0f);

constexpr float kGroundY = -1.2f; // ground plane sits just below the fitted mesh

std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("failed to open shader file: " + path);
    }
    const auto size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}

} // namespace

Renderer::Renderer(Window& window, Device& device, Allocator& allocator, Swapchain& swapchain)
    : window_(window), device_(device), allocator_(allocator), swapchain_(swapchain) {
    const MeshData model = loadGltf(ASSET_PATH);
    modelCenter_ = model.center;
    modelRadius_ = model.radius;
    material_.baseColorFactor = model.baseColorFactor;
    material_.emissiveFactor = glm::vec4(model.emissiveFactor, 1.0f); // w = 1: sample textures
    material_.metallicFactor = model.metallicFactor;
    material_.roughnessFactor = model.roughnessFactor;

    // Flat, slightly rough dielectric ground; w = 0 selects the non-textured shading path.
    groundMaterial_.baseColorFactor = glm::vec4(0.22f, 0.22f, 0.25f, 1.0f);
    groundMaterial_.emissiveFactor = glm::vec4(0.0f);
    groundMaterial_.metallicFactor = 0.0f;
    groundMaterial_.roughnessFactor = 0.85f;

    depthFormat_ = findDepthFormat();
    createRenderPass();
    createShadowResources();
    createDescriptorSetLayout();
    createPipeline();
    createShadowPipeline();
    createDepthResources();
    createFramebuffers();
    createCommandResources();
    createIbl();
    createTextures(model);
    createMesh(model);
    createGround();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createSkyboxPipeline();
    createSkyboxDescriptors();
    createSyncObjects();
}

Renderer::~Renderer() {
    const VkDevice dev = device_.handle();
    vkDeviceWaitIdle(dev);

    for (VkSemaphore s : renderFinished_) {
        vkDestroySemaphore(dev, s, nullptr);
    }
    for (VkSemaphore s : imageAvailable_) {
        vkDestroySemaphore(dev, s, nullptr);
    }
    for (VkFence f : inFlight_) {
        vkDestroyFence(dev, f, nullptr);
    }
    if (skyboxDescriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, skyboxDescriptorPool_, nullptr);
    }
    if (skyboxPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, skyboxPipeline_, nullptr);
    }
    if (skyboxPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, skyboxPipelineLayout_, nullptr);
    }
    if (skyboxSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, skyboxSetLayout_, nullptr);
    }
    if (descriptorPool_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(dev, descriptorPool_, nullptr);
    }
    if (sampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, sampler_, nullptr);
    }
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev, commandPool_, nullptr);
    }
    if (shadowSampler_ != VK_NULL_HANDLE) {
        vkDestroySampler(dev, shadowSampler_, nullptr);
    }
    for (VkFramebuffer fb : shadowFramebuffers_) {
        if (fb != VK_NULL_HANDLE) {
            vkDestroyFramebuffer(dev, fb, nullptr);
        }
    }
    if (shadowPipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, shadowPipeline_, nullptr);
    }
    if (shadowPipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, shadowPipelineLayout_, nullptr);
    }
    if (shadowRenderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(dev, shadowRenderPass_, nullptr);
    }
    for (VkFramebuffer fb : framebuffers_) {
        vkDestroyFramebuffer(dev, fb, nullptr);
    }
    if (pipeline_ != VK_NULL_HANDLE) {
        vkDestroyPipeline(dev, pipeline_, nullptr);
    }
    if (pipelineLayout_ != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(dev, pipelineLayout_, nullptr);
    }
    if (descriptorSetLayout_ != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(dev, descriptorSetLayout_, nullptr);
    }
    if (renderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(dev, renderPass_, nullptr);
    }
    // Buffer/Image members (mesh, texture, depth, uniforms) free themselves via VMA here.
}

VkFormat Renderer::findDepthFormat() const {
    // Prefer a pure-depth 32-bit float format; fall back to the common combined formats.
    const std::array<VkFormat, 3> candidates = {
        VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    for (VkFormat format : candidates) {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(device_.physical(), format, &props);
        if (props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) {
            return format;
        }
    }
    throw std::runtime_error("no supported depth format found");
}

VkShaderModule Renderer::loadShaderModule(const char* path) {
    const std::vector<char> code = readFile(std::string(SHADER_DIR) + "/" + path);

    VkShaderModuleCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    ci.codeSize = code.size();
    ci.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device_.handle(), &ci, nullptr, &module));
    return module;
}

void Renderer::createRenderPass() {
    VkAttachmentDescription color{};
    color.format = swapchain_.imageFormat();
    color.samples = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE; // not needed after the frame
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 1;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;
    subpass.pDepthStencilAttachment = &depthRef;

    // Wait for the previous frame's color output and depth tests before we write either.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    const std::array<VkAttachmentDescription, 2> attachments = {color, depth};

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = static_cast<uint32_t>(attachments.size());
    ci.pAttachments = attachments.data();
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(device_.handle(), &ci, nullptr, &renderPass_));
}

void Renderer::createShadowResources() {
    // Depth-only render pass. finalLayout is READ_ONLY so the main pass can sample it directly.
    VkAttachmentDescription depth{};
    depth.format = depthFormat_;
    depth.samples = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL;

    VkAttachmentReference depthRef{};
    depthRef.attachment = 0;
    depthRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 0;
    subpass.pDepthStencilAttachment = &depthRef;

    // Serialize the depth write against the previous frame's sampling and the following main
    // pass's sampling of this map.
    std::array<VkSubpassDependency, 2> deps{};
    deps[0].srcSubpass = VK_SUBPASS_EXTERNAL;
    deps[0].dstSubpass = 0;
    deps[0].srcStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[0].dstStageMask = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    deps[0].srcAccessMask = VK_ACCESS_SHADER_READ_BIT;
    deps[0].dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].srcSubpass = 0;
    deps[1].dstSubpass = VK_SUBPASS_EXTERNAL;
    deps[1].srcStageMask = VK_PIPELINE_STAGE_LATE_FRAGMENT_TESTS_BIT;
    deps[1].dstStageMask = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    deps[1].srcAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    deps[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    VkRenderPassCreateInfo rpci{};
    rpci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    rpci.attachmentCount = 1;
    rpci.pAttachments = &depth;
    rpci.subpassCount = 1;
    rpci.pSubpasses = &subpass;
    rpci.dependencyCount = static_cast<uint32_t>(deps.size());
    rpci.pDependencies = deps.data();
    VK_CHECK(vkCreateRenderPass(device_.handle(), &rpci, nullptr, &shadowRenderPass_));

    for (int i = 0; i < kFramesInFlight; ++i) {
        shadowMaps_[i] = Image(allocator_.handle(), device_.handle(), kShadowMapSize,
                               kShadowMapSize, depthFormat_,
                               VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT |
                                   VK_IMAGE_USAGE_SAMPLED_BIT,
                               VK_IMAGE_ASPECT_DEPTH_BIT);

        VkImageView view = shadowMaps_[i].view();
        VkFramebufferCreateInfo fci{};
        fci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        fci.renderPass = shadowRenderPass_;
        fci.attachmentCount = 1;
        fci.pAttachments = &view;
        fci.width = kShadowMapSize;
        fci.height = kShadowMapSize;
        fci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_.handle(), &fci, nullptr, &shadowFramebuffers_[i]));
    }

    // Nearest sampling with edge clamp; the fragment shader does its own PCF filtering.
    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_NEAREST;
    si.minFilter = VK_FILTER_NEAREST;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    VK_CHECK(vkCreateSampler(device_.handle(), &si, nullptr, &shadowSampler_));
}

void Renderer::createDescriptorSetLayout() {
    // Binding 0: camera UBO (vertex builds clip pos, fragment needs the eye for the view vector).
    // Bindings 1..5: material maps. Bindings 6..8: IBL maps. Binding 9: the shadow map.
    constexpr int kImageBindings = kTextureCount + kIblTextureCount + 1;
    std::array<VkDescriptorSetLayoutBinding, 1 + kImageBindings> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

    for (int i = 0; i < kImageBindings; ++i) {
        bindings[i + 1].binding = static_cast<uint32_t>(i + 1);
        bindings[i + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i + 1].descriptorCount = 1;
        bindings[i + 1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
    }

    VkDescriptorSetLayoutCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    ci.bindingCount = static_cast<uint32_t>(bindings.size());
    ci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_.handle(), &ci, nullptr, &descriptorSetLayout_));
}

void Renderer::createPipeline() {
    VkShaderModule vert = loadShaderModule("mesh.vert.spv");
    VkShaderModule frag = loadShaderModule("mesh.frag.spv");

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vert;
    vertStage.pName = "main";

    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = frag;
    fragStage.pName = "main";

    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {vertStage, fragStage};

    const auto binding = Vertex::bindingDescription();
    const auto attributes = Vertex::attributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                         VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    // We flip Y in the projection matrix (Vulkan clip space), which reverses winding, so our
    // counter-clockwise cube faces present as front-facing under this setting.
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS; // smaller depth = closer, wins
    depthStencil.depthBoundsTestEnable = VK_FALSE;
    depthStencil.stencilTestEnable = VK_FALSE;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(MeshPush);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &descriptorSetLayout_;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layoutInfo, nullptr, &pipelineLayout_));

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = static_cast<uint32_t>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState = &multisampling;
    ci.pDepthStencilState = &depthStencil;
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.layout = pipelineLayout_;
    ci.renderPass = renderPass_;
    ci.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                       &pipeline_));

    vkDestroyShaderModule(device_.handle(), frag, nullptr);
    vkDestroyShaderModule(device_.handle(), vert, nullptr);
}

void Renderer::createShadowPipeline() {
    // Push constant: lightSpace * model (per object). No descriptor sets needed.
    VkPushConstantRange pushRange{};
    pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    pushRange.offset = 0;
    pushRange.size = sizeof(glm::mat4);

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushRange;
    VK_CHECK(vkCreatePipelineLayout(device_.handle(), &layoutInfo, nullptr,
                                    &shadowPipelineLayout_));

    VkShaderModule vert = loadShaderModule("shadow.vert.spv");
    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vert;
    vertStage.pName = "main";

    // Only the position attribute is needed for a depth-only pass.
    const auto binding = Vertex::bindingDescription();
    const auto attributes = Vertex::attributeDescriptions();
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertexInput.vertexBindingDescriptionCount = 1;
    vertexInput.pVertexBindingDescriptions = &binding;
    vertexInput.vertexAttributeDescriptionCount = 1; // position only
    vertexInput.pVertexAttributeDescriptions = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                         VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    // Depth bias combats shadow acne; keep the same winding/cull as the main pass.
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;
    rasterizer.depthBiasEnable = VK_TRUE;
    rasterizer.depthBiasConstantFactor = 1.25f;
    rasterizer.depthBiasSlopeFactor = 1.75f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_TRUE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;

    // No color attachments in the shadow render pass.
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 0;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = 1;
    ci.pStages = &vertStage;
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState = &multisampling;
    ci.pDepthStencilState = &depthStencil;
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.layout = shadowPipelineLayout_;
    ci.renderPass = shadowRenderPass_;
    ci.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                       &shadowPipeline_));

    vkDestroyShaderModule(device_.handle(), vert, nullptr);
}

void Renderer::createDepthResources() {
    const VkExtent2D extent = swapchain_.extent();
    depthImage_ = Image(allocator_.handle(), device_.handle(), extent.width, extent.height,
                        depthFormat_, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                        VK_IMAGE_ASPECT_DEPTH_BIT);
}

void Renderer::createFramebuffers() {
    const auto& views = swapchain_.imageViews();
    framebuffers_.resize(views.size());
    const VkExtent2D extent = swapchain_.extent();

    for (size_t i = 0; i < views.size(); ++i) {
        // Attachment order must match the render pass: color first, depth second. The depth
        // view is shared across all framebuffers (only one frame renders at a time per image).
        const std::array<VkImageView, 2> attachments = {views[i], depthImage_.view()};

        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = renderPass_;
        ci.attachmentCount = static_cast<uint32_t>(attachments.size());
        ci.pAttachments = attachments.data();
        ci.width = extent.width;
        ci.height = extent.height;
        ci.layers = 1;
        VK_CHECK(vkCreateFramebuffer(device_.handle(), &ci, nullptr, &framebuffers_[i]));
    }
}

void Renderer::createCommandResources() {
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = *device_.queueFamilies().graphics;
    VK_CHECK(vkCreateCommandPool(device_.handle(), &poolInfo, nullptr, &commandPool_));

    commandBuffers_.resize(kFramesInFlight);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(commandBuffers_.size());
    VK_CHECK(vkAllocateCommandBuffers(device_.handle(), &allocInfo, commandBuffers_.data()));
}

void Renderer::createIbl() {
    // Precompute irradiance / prefilter / BRDF LUT from the HDR environment (uses commandPool_).
    ibl_ = std::make_unique<Ibl>(device_.handle(), allocator_.handle(), device_.graphicsQueue(),
                                 commandPool_, ENV_HDR_PATH);
}

void Renderer::immediateSubmit(const std::function<void(VkCommandBuffer)>& record) {
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = commandPool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_.handle(), &allocInfo, &cmd));

    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &begin));
    record(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(device_.handle(), &fenceInfo, nullptr, &fence));

    VK_CHECK(vkQueueSubmit(device_.graphicsQueue(), 1, &submit, fence));
    VK_CHECK(vkWaitForFences(device_.handle(), 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(device_.handle(), fence, nullptr);
    vkFreeCommandBuffers(device_.handle(), commandPool_, 1, &cmd);
}

Buffer Renderer::createDeviceLocalBuffer(const void* data, VkDeviceSize size,
                                         VkBufferUsageFlags usage) {
    // Staging buffer: host-visible, we memcpy into it, then copy to a device-local buffer.
    Buffer staging(allocator_.handle(), size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_MEMORY_USAGE_AUTO,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(staging.mapped(), data, static_cast<size_t>(size));

    Buffer result(allocator_.handle(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
                  VMA_MEMORY_USAGE_AUTO);

    immediateSubmit([&](VkCommandBuffer cmd) {
        VkBufferCopy copy{};
        copy.size = size;
        vkCmdCopyBuffer(cmd, staging.handle(), result.handle(), 1, &copy);
    });
    return result;
}

Image Renderer::uploadTexture(const TextureData& tex, VkFormat format) {
    const VkDeviceSize imageBytes = tex.pixels.size();

    Buffer staging(allocator_.handle(), imageBytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                   VMA_MEMORY_USAGE_AUTO,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);
    std::memcpy(staging.mapped(), tex.pixels.data(), static_cast<size_t>(imageBytes));

    Image image(allocator_.handle(), device_.handle(), tex.width, tex.height, format,
                VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT);

    immediateSubmit([&](VkCommandBuffer cmd) {
        // 1) UNDEFINED -> TRANSFER_DST_OPTIMAL so we can copy into it.
        VkImageMemoryBarrier toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        toDst.image = image.handle();
        toDst.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        toDst.srcAccessMask = 0;
        toDst.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &toDst);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1;
        copy.imageExtent = {tex.width, tex.height, 1};
        vkCmdCopyBufferToImage(cmd, staging.handle(), image.handle(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);

        // 2) TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL for sampling in the frag shader.
        VkImageMemoryBarrier toRead = toDst;
        toRead.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toRead.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        toRead.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toRead.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1,
                             &toRead);
    });

    return image;
}

void Renderer::createTextures(const MeshData& model) {
    // Color/emissive maps are authored in sRGB (hardware linearizes on sample); the data maps
    // (metallic-roughness, normal, occlusion) are raw linear values, so use UNORM formats.
    constexpr VkFormat kSrgb = VK_FORMAT_R8G8B8A8_SRGB;
    constexpr VkFormat kUnorm = VK_FORMAT_R8G8B8A8_UNORM;
    textures_[0] = uploadTexture(model.baseColor, kSrgb);
    textures_[1] = uploadTexture(model.metallicRoughness, kUnorm);
    textures_[2] = uploadTexture(model.normal, kUnorm);
    textures_[3] = uploadTexture(model.emissive, kSrgb);
    textures_[4] = uploadTexture(model.occlusion, kUnorm);

    VkSamplerCreateInfo si{};
    si.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    si.magFilter = VK_FILTER_LINEAR;
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.anisotropyEnable = VK_FALSE; // device feature not enabled yet
    si.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    si.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    VK_CHECK(vkCreateSampler(device_.handle(), &si, nullptr, &sampler_));
}

void Renderer::createMesh(const MeshData& model) {
    indexCount_ = static_cast<uint32_t>(model.indices.size());

    vertexBuffer_ =
        createDeviceLocalBuffer(model.vertices.data(), sizeof(Vertex) * model.vertices.size(),
                                VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    indexBuffer_ =
        createDeviceLocalBuffer(model.indices.data(), sizeof(uint32_t) * model.indices.size(),
                                VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

void Renderer::createGround() {
    // A large flat quad at kGroundY, facing up. Indexed both windings so it's visible and casts
    // depth regardless of the pipeline's back-face culling.
    constexpr float h = 15.0f;
    const std::array<Vertex, 4> vertices = {{
        {{-h, kGroundY, -h}, {0, 1, 0}, {0, 0}, {1, 0, 0, 1}},
        {{h, kGroundY, -h}, {0, 1, 0}, {1, 0}, {1, 0, 0, 1}},
        {{h, kGroundY, h}, {0, 1, 0}, {1, 1}, {1, 0, 0, 1}},
        {{-h, kGroundY, h}, {0, 1, 0}, {0, 1}, {1, 0, 0, 1}},
    }};
    const std::array<uint32_t, 12> indices = {0, 1, 2, 2, 3, 0, 0, 3, 2, 2, 1, 0};
    groundIndexCount_ = static_cast<uint32_t>(indices.size());

    groundVertexBuffer_ = createDeviceLocalBuffer(vertices.data(), sizeof(Vertex) * vertices.size(),
                                                  VK_BUFFER_USAGE_VERTEX_BUFFER_BIT);
    groundIndexBuffer_ = createDeviceLocalBuffer(indices.data(), sizeof(uint32_t) * indices.size(),
                                                 VK_BUFFER_USAGE_INDEX_BUFFER_BIT);
}

void Renderer::createUniformBuffers() {
    uniformBuffers_.clear();
    uniformBuffers_.reserve(kFramesInFlight);
    for (int i = 0; i < kFramesInFlight; ++i) {
        // Host-visible + persistently mapped: we overwrite it from the CPU every frame, so
        // there's no benefit to device-local memory or a staging copy here.
        uniformBuffers_.emplace_back(allocator_.handle(), sizeof(CameraUBO),
                                     VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_AUTO,
                                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                         VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
}

void Renderer::createDescriptorPool() {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = kFramesInFlight;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount =
        kFramesInFlight * (kTextureCount + kIblTextureCount + 1); // material + IBL + shadow

    VkDescriptorPoolCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes = sizes.data();
    ci.maxSets = kFramesInFlight;
    VK_CHECK(vkCreateDescriptorPool(device_.handle(), &ci, nullptr, &descriptorPool_));
}

void Renderer::createDescriptorSets() {
    const std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, descriptorSetLayout_);

    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = descriptorPool_;
    allocInfo.descriptorSetCount = kFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets_.resize(kFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device_.handle(), &allocInfo, descriptorSets_.data()));

    for (int i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers_[i].handle();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(CameraUBO);

        constexpr int kImageBindings = kTextureCount + kIblTextureCount + 1;
        std::array<VkDescriptorImageInfo, kImageBindings> imageInfos{};
        for (int t = 0; t < kTextureCount; ++t) {
            imageInfos[t].imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfos[t].imageView = textures_[t].view();
            imageInfos[t].sampler = sampler_;
        }
        // Bindings 6, 7, 8: irradiance, prefilter (env sampler), BRDF LUT (clamp sampler).
        imageInfos[5] = {ibl_->environmentSampler(), ibl_->irradianceView(),
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imageInfos[6] = {ibl_->environmentSampler(), ibl_->prefilterView(),
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        imageInfos[7] = {ibl_->lutSampler(), ibl_->brdfLutView(),
                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        // Binding 9: this frame's shadow map (in depth read-only layout after the shadow pass).
        imageInfos[8] = {shadowSampler_, shadowMaps_[i].view(),
                         VK_IMAGE_LAYOUT_DEPTH_STENCIL_READ_ONLY_OPTIMAL};

        std::array<VkWriteDescriptorSet, 1 + kImageBindings> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = descriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;

        for (int t = 0; t < kImageBindings; ++t) {
            writes[t + 1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[t + 1].dstSet = descriptorSets_[i];
            writes[t + 1].dstBinding = static_cast<uint32_t>(t + 1);
            writes[t + 1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            writes[t + 1].descriptorCount = 1;
            writes[t + 1].pImageInfo = &imageInfos[t];
        }

        vkUpdateDescriptorSets(device_.handle(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void Renderer::createSkyboxPipeline() {
    // Set 0: camera UBO (binding 0) + environment map (binding 1).
    std::array<VkDescriptorSetLayoutBinding, 2> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    VkDescriptorSetLayoutCreateInfo slci{};
    slci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    slci.bindingCount = static_cast<uint32_t>(bindings.size());
    slci.pBindings = bindings.data();
    VK_CHECK(vkCreateDescriptorSetLayout(device_.handle(), &slci, nullptr, &skyboxSetLayout_));

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &skyboxSetLayout_;
    VK_CHECK(vkCreatePipelineLayout(device_.handle(), &plci, nullptr, &skyboxPipelineLayout_));

    VkShaderModule vert = loadShaderModule("skybox.vert.spv");
    VkShaderModule frag = loadShaderModule("skybox.frag.spv");

    VkPipelineShaderStageCreateInfo vertStage{};
    vertStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vertStage.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vertStage.module = vert;
    vertStage.pName = "main";
    VkPipelineShaderStageCreateInfo fragStage{};
    fragStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    fragStage.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    fragStage.module = frag;
    fragStage.pName = "main";
    const std::array<VkPipelineShaderStageCreateInfo, 2> stages = {vertStage, fragStage};

    // No vertex input: the skybox is a shader-generated fullscreen triangle.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    const std::array<VkDynamicState, 2> dynamicStates = {VK_DYNAMIC_STATE_VIEWPORT,
                                                         VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamicState.dynamicStateCount = static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    VkPipelineViewportStateCreateInfo viewportState{};
    viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_NONE;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    // Draw at the far plane; keep it behind anything already in the depth buffer, don't write depth.
    VkPipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depthStencil.depthTestEnable = VK_TRUE;
    depthStencil.depthWriteEnable = VK_FALSE;
    depthStencil.depthCompareOp = VK_COMPARE_OP_LESS_OR_EQUAL;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;
    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkGraphicsPipelineCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    ci.stageCount = static_cast<uint32_t>(stages.size());
    ci.pStages = stages.data();
    ci.pVertexInputState = &vertexInput;
    ci.pInputAssemblyState = &inputAssembly;
    ci.pViewportState = &viewportState;
    ci.pRasterizationState = &rasterizer;
    ci.pMultisampleState = &multisampling;
    ci.pDepthStencilState = &depthStencil;
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.layout = skyboxPipelineLayout_;
    ci.renderPass = renderPass_;
    ci.subpass = 0;
    VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                       &skyboxPipeline_));

    vkDestroyShaderModule(device_.handle(), frag, nullptr);
    vkDestroyShaderModule(device_.handle(), vert, nullptr);
}

void Renderer::createSkyboxDescriptors() {
    std::array<VkDescriptorPoolSize, 2> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    sizes[0].descriptorCount = kFramesInFlight;
    sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sizes[1].descriptorCount = kFramesInFlight;

    VkDescriptorPoolCreateInfo pci{};
    pci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    pci.pPoolSizes = sizes.data();
    pci.maxSets = kFramesInFlight;
    VK_CHECK(vkCreateDescriptorPool(device_.handle(), &pci, nullptr, &skyboxDescriptorPool_));

    const std::vector<VkDescriptorSetLayout> layouts(kFramesInFlight, skyboxSetLayout_);
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = skyboxDescriptorPool_;
    allocInfo.descriptorSetCount = kFramesInFlight;
    allocInfo.pSetLayouts = layouts.data();
    skyboxDescriptorSets_.resize(kFramesInFlight);
    VK_CHECK(vkAllocateDescriptorSets(device_.handle(), &allocInfo, skyboxDescriptorSets_.data()));

    for (int i = 0; i < kFramesInFlight; ++i) {
        VkDescriptorBufferInfo bufferInfo{};
        bufferInfo.buffer = uniformBuffers_[i].handle();
        bufferInfo.offset = 0;
        bufferInfo.range = sizeof(CameraUBO);

        VkDescriptorImageInfo envInfo{};
        envInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        envInfo.imageView = ibl_->environmentView();
        envInfo.sampler = ibl_->environmentSampler();

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = skyboxDescriptorSets_[i];
        writes[0].dstBinding = 0;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        writes[0].descriptorCount = 1;
        writes[0].pBufferInfo = &bufferInfo;
        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = skyboxDescriptorSets_[i];
        writes[1].dstBinding = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        writes[1].descriptorCount = 1;
        writes[1].pImageInfo = &envInfo;
        vkUpdateDescriptorSets(device_.handle(), static_cast<uint32_t>(writes.size()),
                               writes.data(), 0, nullptr);
    }
}

void Renderer::createSyncObjects() {
    imageAvailable_.resize(kFramesInFlight);
    inFlight_.resize(kFramesInFlight);
    renderFinished_.resize(swapchain_.imageCount());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (int i = 0; i < kFramesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(device_.handle(), &semInfo, nullptr, &imageAvailable_[i]));
        VK_CHECK(vkCreateFence(device_.handle(), &fenceInfo, nullptr, &inFlight_[i]));
    }
    for (uint32_t i = 0; i < swapchain_.imageCount(); ++i) {
        VK_CHECK(vkCreateSemaphore(device_.handle(), &semInfo, nullptr, &renderFinished_[i]));
    }
}

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    const VkExtent2D extent = swapchain_.extent();

    auto setViewportScissor = [&](uint32_t w, uint32_t h) {
        VkViewport viewport{};
        viewport.width = static_cast<float>(w);
        viewport.height = static_cast<float>(h);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{};
        scissor.extent = {w, h};
        vkCmdSetScissor(cmd, 0, 1, &scissor);
    };

    auto bindMesh = [&](const Buffer& vb, const Buffer& ib) {
        const VkBuffer buffers[] = {vb.handle()};
        const VkDeviceSize offsets[] = {0};
        vkCmdBindVertexBuffers(cmd, 0, 1, buffers, offsets);
        vkCmdBindIndexBuffer(cmd, ib.handle(), 0, VK_INDEX_TYPE_UINT32);
    };

    // --- Shadow pass: render scene depth from the light into this frame's shadow map. ---
    VkClearValue shadowClear{};
    shadowClear.depthStencil = {1.0f, 0};
    VkRenderPassBeginInfo shadowBegin{};
    shadowBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    shadowBegin.renderPass = shadowRenderPass_;
    shadowBegin.framebuffer = shadowFramebuffers_[currentFrame_];
    shadowBegin.renderArea.extent = {kShadowMapSize, kShadowMapSize};
    shadowBegin.clearValueCount = 1;
    shadowBegin.pClearValues = &shadowClear;

    vkCmdBeginRenderPass(cmd, &shadowBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, shadowPipeline_);
    setViewportScissor(kShadowMapSize, kShadowMapSize);

    const glm::mat4 meshLightMvp = lightSpace_ * meshModel_;
    vkCmdPushConstants(cmd, shadowPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(glm::mat4), &meshLightMvp);
    bindMesh(vertexBuffer_, indexBuffer_);
    vkCmdDrawIndexed(cmd, indexCount_, 1, 0, 0, 0);

    const glm::mat4 groundLightMvp = lightSpace_ * groundModel_;
    vkCmdPushConstants(cmd, shadowPipelineLayout_, VK_SHADER_STAGE_VERTEX_BIT, 0,
                       sizeof(glm::mat4), &groundLightMvp);
    bindMesh(groundVertexBuffer_, groundIndexBuffer_);
    vkCmdDrawIndexed(cmd, groundIndexCount_, 1, 0, 0, 0);
    vkCmdEndRenderPass(cmd);

    // --- Main pass: shade the scene, sampling the shadow map just written. ---
    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{0.01f, 0.01f, 0.015f, 1.0f}};
    clears[1].depthStencil = {1.0f, 0};

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = renderPass_;
    rpBegin.framebuffer = framebuffers_[imageIndex];
    rpBegin.renderArea.extent = extent;
    rpBegin.clearValueCount = static_cast<uint32_t>(clears.size());
    rpBegin.pClearValues = clears.data();

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);
    setViewportScissor(extent.width, extent.height);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout_, 0, 1,
                            &descriptorSets_[currentFrame_], 0, nullptr);

    auto drawObject = [&](const glm::mat4& model, const MaterialPush& mat, const Buffer& vb,
                          const Buffer& ib, uint32_t indexCount) {
        MeshPush push{};
        push.model = model;
        push.baseColorFactor = mat.baseColorFactor;
        push.emissiveFactor = mat.emissiveFactor;
        push.metallicFactor = mat.metallicFactor;
        push.roughnessFactor = mat.roughnessFactor;
        vkCmdPushConstants(cmd, pipelineLayout_,
                           VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0,
                           sizeof(MeshPush), &push);
        bindMesh(vb, ib);
        vkCmdDrawIndexed(cmd, indexCount, 1, 0, 0, 0);
    };

    drawObject(meshModel_, material_, vertexBuffer_, indexBuffer_, indexCount_);
    drawObject(groundModel_, groundMaterial_, groundVertexBuffer_, groundIndexBuffer_,
               groundIndexCount_);

    // Skybox last: it fills only the background (depth test LESS_OR_EQUAL vs the cleared far
    // plane), so it never overdraws the scene but avoids shading pixels already covered.
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipeline_);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, skyboxPipelineLayout_, 0, 1,
                            &skyboxDescriptorSets_[currentFrame_], 0, nullptr);
    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::updateUniformBuffer(uint32_t frame) {
    static const auto start = std::chrono::high_resolution_clock::now();
    const auto now = std::chrono::high_resolution_clock::now();
    const float t = std::chrono::duration<float>(now - start).count();

    const VkExtent2D extent = swapchain_.extent();
    const float aspect = static_cast<float>(extent.width) /
                         static_cast<float>(extent.height == 0 ? 1 : extent.height);

    // Fit the mesh to a unit sphere at the origin (center it, then scale by 1/radius), spin it
    // about the up axis, and view it from a fixed 3/4 angle.
    const float fit = 1.0f / modelRadius_;
    meshModel_ = glm::rotate(glm::mat4(1.0f), t * glm::radians(30.0f), glm::vec3(0, 1, 0));
    meshModel_ = glm::scale(meshModel_, glm::vec3(fit));
    meshModel_ = glm::translate(meshModel_, -modelCenter_);
    groundModel_ = glm::mat4(1.0f); // ground vertices are already in world space

    // Light-space matrix: orthographic projection from the light, framing the scene near origin.
    const glm::vec3 L = glm::normalize(kLightDir);
    const glm::vec3 target(0.0f, -0.2f, 0.0f);
    const glm::vec3 lightEye = target - L * 5.0f;
    const glm::mat4 lightView = glm::lookAt(lightEye, target, glm::vec3(0.0f, 1.0f, 0.0f));
    const float r = 2.2f; // half-extent of the shadowed region around the origin
    const glm::mat4 lightProj = glm::ortho(-r, r, -r, r, 0.1f, 10.0f);
    lightSpace_ = lightProj * lightView; // no Y flip: render and sample use the same matrix

    CameraUBO ubo{};
    ubo.view = glm::lookAt(kEye, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    ubo.proj[1][1] *= -1.0f; // GLM targets OpenGL's flipped-Y clip space; undo it for Vulkan
    ubo.camPos = glm::vec4(kEye, 1.0f);
    ubo.iblParams = glm::vec4(ibl_->prefilterMaxLod(), 0.0f, 0.0f, 0.0f);
    ubo.lightSpace = lightSpace_;
    ubo.lightDir = glm::vec4(L, 0.0f);
    ubo.lightColor = glm::vec4(kLightColor, 0.0f);

    std::memcpy(uniformBuffers_[frame].mapped(), &ubo, sizeof(ubo));
}

void Renderer::recreateSwapchain() {
    int width = 0;
    int height = 0;
    window_.framebufferSize(width, height);
    while (width == 0 || height == 0) {
        window_.framebufferSize(width, height);
        window_.waitEvents();
    }

    vkDeviceWaitIdle(device_.handle());

    for (VkFramebuffer fb : framebuffers_) {
        vkDestroyFramebuffer(device_.handle(), fb, nullptr);
    }
    framebuffers_.clear();

    swapchain_.recreate(static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    createDepthResources(); // depth buffer must match the new extent
    createFramebuffers();
}

void Renderer::drawFrame() {
    const VkDevice dev = device_.handle();

    VK_CHECK(vkWaitForFences(dev, 1, &inFlight_[currentFrame_], VK_TRUE, UINT64_MAX));

    uint32_t imageIndex = 0;
    VkResult acquire = vkAcquireNextImageKHR(dev, swapchain_.handle(), UINT64_MAX,
                                             imageAvailable_[currentFrame_], VK_NULL_HANDLE,
                                             &imageIndex);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR) {
        recreateSwapchain();
        return;
    }
    if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR) {
        throw std::runtime_error("vkAcquireNextImageKHR failed");
    }

    updateUniformBuffer(currentFrame_);

    VK_CHECK(vkResetFences(dev, 1, &inFlight_[currentFrame_]));

    VkCommandBuffer cmd = commandBuffers_[currentFrame_];
    VK_CHECK(vkResetCommandBuffer(cmd, 0));
    recordCommandBuffer(cmd, imageIndex);

    const VkPipelineStageFlags waitStage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;

    VkSubmitInfo submit{};
    submit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount = 1;
    submit.pWaitSemaphores = &imageAvailable_[currentFrame_];
    submit.pWaitDstStageMask = &waitStage;
    submit.commandBufferCount = 1;
    submit.pCommandBuffers = &cmd;
    submit.signalSemaphoreCount = 1;
    submit.pSignalSemaphores = &renderFinished_[imageIndex];
    VK_CHECK(vkQueueSubmit(device_.graphicsQueue(), 1, &submit, inFlight_[currentFrame_]));

    VkPresentInfoKHR present{};
    present.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1;
    present.pWaitSemaphores = &renderFinished_[imageIndex];
    VkSwapchainKHR swapchains[] = {swapchain_.handle()};
    present.swapchainCount = 1;
    present.pSwapchains = swapchains;
    present.pImageIndices = &imageIndex;

    VkResult presentResult = vkQueuePresentKHR(device_.presentQueue(), &present);
    if (presentResult == VK_ERROR_OUT_OF_DATE_KHR || presentResult == VK_SUBOPTIMAL_KHR ||
        window_.framebufferResized) {
        window_.framebufferResized = false;
        recreateSwapchain();
    } else if (presentResult != VK_SUCCESS) {
        throw std::runtime_error("vkQueuePresentKHR failed");
    }

    currentFrame_ = (currentFrame_ + 1) % kFramesInFlight;
}

void Renderer::run() {
    while (!window_.shouldClose()) {
        window_.pollEvents();
        drawFrame();
    }
    vkDeviceWaitIdle(device_.handle());
}
