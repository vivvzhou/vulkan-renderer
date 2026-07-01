#include "render/Renderer.hpp"

#include "core/Window.hpp"
#include "vk/Common.hpp"
#include "vk/Device.hpp"
#include "vk/Swapchain.hpp"

#include <array>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#ifndef SHADER_DIR
#define SHADER_DIR "shaders"
#endif

namespace {

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

Renderer::Renderer(Window& window, Device& device, Swapchain& swapchain)
    : window_(window), device_(device), swapchain_(swapchain) {
    createRenderPass();
    createPipeline();
    createFramebuffers();
    createCommandResources();
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
    if (commandPool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev, commandPool_, nullptr);
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
    if (renderPass_ != VK_NULL_HANDLE) {
        vkDestroyRenderPass(dev, renderPass_, nullptr);
    }
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

    VkAttachmentReference colorRef{};
    colorRef.attachment = 0;
    colorRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorRef;

    // Ensure the swapchain image is done being read (presented) before we write it.
    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo ci{};
    ci.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    ci.attachmentCount = 1;
    ci.pAttachments = &color;
    ci.subpassCount = 1;
    ci.pSubpasses = &subpass;
    ci.dependencyCount = 1;
    ci.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(device_.handle(), &ci, nullptr, &renderPass_));
}

void Renderer::createPipeline() {
    VkShaderModule vert = loadShaderModule("triangle.vert.spv");
    VkShaderModule frag = loadShaderModule("triangle.frag.spv");

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

    // No vertex buffers yet: the triangle is baked into the vertex shader.
    VkPipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;

    VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

    // Viewport and scissor are dynamic so we don't rebuild the pipeline on resize.
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
    rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineColorBlendAttachmentState blendAttachment{};
    blendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                                     VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
    blendAttachment.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    colorBlend.attachmentCount = 1;
    colorBlend.pAttachments = &blendAttachment;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
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
    ci.pColorBlendState = &colorBlend;
    ci.pDynamicState = &dynamicState;
    ci.layout = pipelineLayout_;
    ci.renderPass = renderPass_;
    ci.subpass = 0;

    VK_CHECK(vkCreateGraphicsPipelines(device_.handle(), VK_NULL_HANDLE, 1, &ci, nullptr,
                                       &pipeline_));

    // Shader modules can be destroyed once the pipeline is built.
    vkDestroyShaderModule(device_.handle(), frag, nullptr);
    vkDestroyShaderModule(device_.handle(), vert, nullptr);
}

void Renderer::createFramebuffers() {
    const auto& views = swapchain_.imageViews();
    framebuffers_.resize(views.size());
    const VkExtent2D extent = swapchain_.extent();

    for (size_t i = 0; i < views.size(); ++i) {
        VkFramebufferCreateInfo ci{};
        ci.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        ci.renderPass = renderPass_;
        ci.attachmentCount = 1;
        ci.pAttachments = &views[i];
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

void Renderer::createSyncObjects() {
    imageAvailable_.resize(kFramesInFlight);
    inFlight_.resize(kFramesInFlight);
    renderFinished_.resize(swapchain_.imageCount());

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT; // so the first wait returns immediately

    for (int i = 0; i < kFramesInFlight; ++i) {
        VK_CHECK(vkCreateSemaphore(device_.handle(), &semInfo, nullptr, &imageAvailable_[i]));
        VK_CHECK(vkCreateFence(device_.handle(), &fenceInfo, nullptr, &inFlight_[i]));
    }
    // One "render finished" semaphore per swapchain image: it is signaled by the submit that
    // draws into that image and waited on by the present, so it must be tied to the image, not
    // the frame-in-flight slot.
    for (uint32_t i = 0; i < swapchain_.imageCount(); ++i) {
        VK_CHECK(vkCreateSemaphore(device_.handle(), &semInfo, nullptr, &renderFinished_[i]));
    }
}

void Renderer::recordCommandBuffer(VkCommandBuffer cmd, uint32_t imageIndex) {
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

    VkClearValue clear{};
    clear.color = {{0.01f, 0.01f, 0.015f, 1.0f}};

    const VkExtent2D extent = swapchain_.extent();

    VkRenderPassBeginInfo rpBegin{};
    rpBegin.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpBegin.renderPass = renderPass_;
    rpBegin.framebuffer = framebuffers_[imageIndex];
    rpBegin.renderArea.offset = {0, 0};
    rpBegin.renderArea.extent = extent;
    rpBegin.clearValueCount = 1;
    rpBegin.pClearValues = &clear;

    vkCmdBeginRenderPass(cmd, &rpBegin, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(extent.width);
    viewport.height = static_cast<float>(extent.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(cmd, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = extent;
    vkCmdSetScissor(cmd, 0, 1, &scissor);

    vkCmdDraw(cmd, 3, 1, 0, 0);

    vkCmdEndRenderPass(cmd);
    VK_CHECK(vkEndCommandBuffer(cmd));
}

void Renderer::recreateSwapchain() {
    // Block while minimized (zero-size framebuffer) so we never build a 0x0 swapchain.
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

    // Only reset the fence once we know we are submitting work, to avoid a deadlock on the
    // early-out swapchain-recreation path above.
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
