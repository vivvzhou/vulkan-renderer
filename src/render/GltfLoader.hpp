#pragma once

#include "render/Vertex.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

// A decoded image: tightly packed RGBA8, row-major. Missing glTF textures come back as a 1x1
// neutral fallback so the shader can sample all five maps unconditionally.
struct TextureData {
    std::vector<uint8_t> pixels;
    uint32_t width = 1;
    uint32_t height = 1;
};

// The CPU-side result of loading a glTF file: a single merged mesh (all primitives baked into
// world space and concatenated) plus a full metallic-roughness material (five maps + factors).
// Assumes one material for the whole model, which holds for the DamagedHelmet sample.
struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    // PBR metallic-roughness material maps.
    TextureData baseColor;         // sRGB
    TextureData metallicRoughness; // linear: G = roughness, B = metallic
    TextureData normal;            // linear tangent-space normal map
    TextureData emissive;          // sRGB
    TextureData occlusion;         // linear: R = ambient occlusion

    // Scalar/vector factors that multiply the sampled maps.
    glm::vec4 baseColorFactor{1.0f};
    glm::vec3 emissiveFactor{0.0f};
    float metallicFactor = 1.0f;
    float roughnessFactor = 1.0f;

    // Bounds, used to auto-fit the mesh to the camera and rest it on the ground regardless of
    // its authored scale.
    glm::vec3 center{0.0f};
    float radius = 1.0f;
    glm::vec3 aabbMin{0.0f};
    glm::vec3 aabbMax{0.0f};
};

// Loads a .glb (or .gltf) file, applying node transforms. Throws std::runtime_error on failure.
MeshData loadGltf(const std::string& path);
