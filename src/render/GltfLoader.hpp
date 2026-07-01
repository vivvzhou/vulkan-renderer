#pragma once

#include "render/Vertex.hpp"

#include <glm/glm.hpp>

#include <cstdint>
#include <string>
#include <vector>

// The CPU-side result of loading a glTF file: a single merged mesh (all primitives baked into
// world space and concatenated) plus one base-color texture. Phase 3 will split this back out
// into per-material draws; for now we render one mesh with one albedo map.
struct MeshData {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;

    std::vector<uint8_t> texturePixels; // tightly packed RGBA8, row-major
    uint32_t textureWidth = 0;
    uint32_t textureHeight = 0;

    // Bounding sphere, used to auto-fit the mesh to the camera regardless of its authored scale.
    glm::vec3 center{0.0f};
    float radius = 1.0f;
};

// Loads a .glb (or .gltf) file, applying node transforms. Throws std::runtime_error on failure.
MeshData loadGltf(const std::string& path);
