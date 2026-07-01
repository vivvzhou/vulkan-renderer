#pragma once

#include <vulkan/vulkan.h>

#include <glm/glm.hpp>

#include <array>
#include <cstddef>

// One interleaved vertex. The binding/attribute descriptions below tell the pipeline how to
// pull these fields out of the vertex buffer and feed them to the vertex shader's inputs.
struct Vertex {
    glm::vec3 pos;
    glm::vec3 normal;
    glm::vec2 uv;

    static VkVertexInputBindingDescription bindingDescription() {
        VkVertexInputBindingDescription binding{};
        binding.binding = 0;
        binding.stride = sizeof(Vertex);
        binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX; // advance per-vertex, not per-instance
        return binding;
    }

    static std::array<VkVertexInputAttributeDescription, 3> attributeDescriptions() {
        std::array<VkVertexInputAttributeDescription, 3> attrs{};
        attrs[0].location = 0; // layout(location = 0) in vec3 inPos
        attrs[0].binding = 0;
        attrs[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[0].offset = offsetof(Vertex, pos);

        attrs[1].location = 1; // layout(location = 1) in vec3 inNormal
        attrs[1].binding = 0;
        attrs[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attrs[1].offset = offsetof(Vertex, normal);

        attrs[2].location = 2; // layout(location = 2) in vec2 inUV
        attrs[2].binding = 0;
        attrs[2].format = VK_FORMAT_R32G32_SFLOAT;
        attrs[2].offset = offsetof(Vertex, uv);
        return attrs;
    }
};
