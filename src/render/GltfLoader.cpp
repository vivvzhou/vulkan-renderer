#include "render/GltfLoader.hpp"

// tinygltf pulls in stb_image / stb_image_write / json; instantiate all implementations here,
// in this one translation unit, with warnings silenced (it's third-party and noisy under /W4).
#if defined(_MSC_VER)
#pragma warning(push, 0)
#endif
#define TINYGLTF_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <tiny_gltf.h>
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {

// Base pointer to an accessor's first element within its backing buffer.
const unsigned char* accessorPtr(const tinygltf::Model& model, const tinygltf::Accessor& acc) {
    const tinygltf::BufferView& view = model.bufferViews[acc.bufferView];
    const tinygltf::Buffer& buffer = model.buffers[view.buffer];
    return buffer.data.data() + view.byteOffset + acc.byteOffset;
}

// Local transform of a node: either an explicit matrix, or composed from TRS.
glm::mat4 nodeLocalMatrix(const tinygltf::Node& node) {
    if (node.matrix.size() == 16) {
        return glm::make_mat4(node.matrix.data()); // glTF matrices are column-major, like glm
    }
    glm::mat4 m(1.0f);
    if (node.translation.size() == 3) {
        m = glm::translate(m, glm::vec3(node.translation[0], node.translation[1],
                                        node.translation[2]));
    }
    if (node.rotation.size() == 4) {
        // glTF quaternion is (x, y, z, w); glm::quat is (w, x, y, z).
        const glm::quat q(static_cast<float>(node.rotation[3]), static_cast<float>(node.rotation[0]),
                          static_cast<float>(node.rotation[1]), static_cast<float>(node.rotation[2]));
        m *= glm::mat4_cast(q);
    }
    if (node.scale.size() == 3) {
        m = glm::scale(m, glm::vec3(node.scale[0], node.scale[1], node.scale[2]));
    }
    return m;
}

void appendPrimitive(const tinygltf::Model& model, const tinygltf::Primitive& prim,
                     const glm::mat4& world, MeshData& out) {
    if (prim.attributes.count("POSITION") == 0) {
        return;
    }
    const tinygltf::Accessor& posAcc = model.accessors[prim.attributes.at("POSITION")];
    const size_t vertexCount = posAcc.count;
    const unsigned char* posData = accessorPtr(model, posAcc);
    const size_t posStride = posAcc.ByteStride(model.bufferViews[posAcc.bufferView]);

    const unsigned char* normData = nullptr;
    size_t normStride = 0;
    if (prim.attributes.count("NORMAL")) {
        const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("NORMAL")];
        normData = accessorPtr(model, acc);
        normStride = acc.ByteStride(model.bufferViews[acc.bufferView]);
    }

    const unsigned char* uvData = nullptr;
    size_t uvStride = 0;
    if (prim.attributes.count("TEXCOORD_0")) {
        const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("TEXCOORD_0")];
        uvData = accessorPtr(model, acc);
        uvStride = acc.ByteStride(model.bufferViews[acc.bufferView]);
    }

    const unsigned char* tanData = nullptr;
    size_t tanStride = 0;
    if (prim.attributes.count("TANGENT")) {
        const tinygltf::Accessor& acc = model.accessors[prim.attributes.at("TANGENT")];
        tanData = accessorPtr(model, acc);
        tanStride = acc.ByteStride(model.bufferViews[acc.bufferView]);
    }

    // Normals/tangents are transformed by the upper-left 3x3 (fine for rotations/uniform scale).
    const glm::mat3 normalMatrix(world);
    const auto base = static_cast<uint32_t>(out.vertices.size());

    for (size_t i = 0; i < vertexCount; ++i) {
        Vertex v{};
        const auto* p = reinterpret_cast<const float*>(posData + i * posStride);
        const glm::vec4 worldPos = world * glm::vec4(p[0], p[1], p[2], 1.0f);
        v.pos = glm::vec3(worldPos);

        if (normData) {
            const auto* n = reinterpret_cast<const float*>(normData + i * normStride);
            v.normal = glm::normalize(normalMatrix * glm::vec3(n[0], n[1], n[2]));
        } else {
            v.normal = glm::vec3(0.0f, 1.0f, 0.0f);
        }

        if (uvData) {
            const auto* uv = reinterpret_cast<const float*>(uvData + i * uvStride);
            v.uv = glm::vec2(uv[0], uv[1]);
        }

        if (tanData) {
            const auto* t = reinterpret_cast<const float*>(tanData + i * tanStride);
            v.tangent = glm::vec4(normalMatrix * glm::vec3(t[0], t[1], t[2]), t[3]);
        } else {
            v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f); // no normal-map basis available
        }
        out.vertices.push_back(v);
    }

    // Indices (widen whatever integer width glTF used to uint32, offset into the merged buffer).
    const tinygltf::Accessor& idxAcc = model.accessors[prim.indices];
    const unsigned char* idxData = accessorPtr(model, idxAcc);
    for (size_t i = 0; i < idxAcc.count; ++i) {
        uint32_t index = 0;
        switch (idxAcc.componentType) {
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                index = reinterpret_cast<const uint32_t*>(idxData)[i];
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                index = reinterpret_cast<const uint16_t*>(idxData)[i];
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                index = idxData[i];
                break;
            default:
                throw std::runtime_error("unsupported glTF index component type");
        }
        out.indices.push_back(base + index);
    }
}

void traverseNode(const tinygltf::Model& model, int nodeIndex, const glm::mat4& parent,
                  MeshData& out) {
    const tinygltf::Node& node = model.nodes[nodeIndex];
    const glm::mat4 world = parent * nodeLocalMatrix(node);

    if (node.mesh >= 0) {
        for (const tinygltf::Primitive& prim : model.meshes[node.mesh].primitives) {
            appendPrimitive(model, prim, world, out);
        }
    }
    for (int child : node.children) {
        traverseNode(model, child, world, out);
    }
}

// A 1x1 RGBA fallback texture.
TextureData solidTexture(uint8_t r, uint8_t g, uint8_t b, uint8_t a) {
    TextureData tex;
    tex.pixels = {r, g, b, a};
    tex.width = 1;
    tex.height = 1;
    return tex;
}

// Decode the glTF texture at textureIndex into RGBA8, or return the given fallback if it is
// absent. tinygltf has already decoded the image bytes; we just widen RGB -> RGBA here.
TextureData loadTexture(const tinygltf::Model& model, int textureIndex,
                        const TextureData& fallback) {
    if (textureIndex < 0 || model.textures[textureIndex].source < 0) {
        return fallback;
    }
    const tinygltf::Image& image = model.images[model.textures[textureIndex].source];
    if (image.width <= 0 || image.height <= 0) {
        return fallback;
    }

    TextureData tex;
    tex.width = static_cast<uint32_t>(image.width);
    tex.height = static_cast<uint32_t>(image.height);
    tex.pixels.resize(static_cast<size_t>(image.width) * image.height * 4);

    if (image.component == 4) {
        std::memcpy(tex.pixels.data(), image.image.data(), tex.pixels.size());
    } else if (image.component == 3) {
        for (size_t i = 0; i < static_cast<size_t>(image.width) * image.height; ++i) {
            tex.pixels[i * 4 + 0] = image.image[i * 3 + 0];
            tex.pixels[i * 4 + 1] = image.image[i * 3 + 1];
            tex.pixels[i * 4 + 2] = image.image[i * 3 + 2];
            tex.pixels[i * 4 + 3] = 255;
        }
    } else {
        throw std::runtime_error("unsupported glTF image component count");
    }
    return tex;
}

// Load the full metallic-roughness material (five maps + factors) from the first material.
void loadMaterial(const tinygltf::Model& model, MeshData& out) {
    // Neutral fallbacks: white base/MR/AO, black emissive, flat (+Z) tangent-space normal.
    out.baseColor = solidTexture(255, 255, 255, 255);
    out.metallicRoughness = solidTexture(255, 255, 255, 255);
    out.normal = solidTexture(128, 128, 255, 255);
    out.emissive = solidTexture(0, 0, 0, 255);
    out.occlusion = solidTexture(255, 255, 255, 255);

    if (model.materials.empty()) {
        return;
    }
    const tinygltf::Material& mat = model.materials[0];
    const auto& pbr = mat.pbrMetallicRoughness;

    out.baseColor = loadTexture(model, pbr.baseColorTexture.index, out.baseColor);
    out.metallicRoughness =
        loadTexture(model, pbr.metallicRoughnessTexture.index, out.metallicRoughness);
    out.normal = loadTexture(model, mat.normalTexture.index, out.normal);
    out.emissive = loadTexture(model, mat.emissiveTexture.index, out.emissive);
    out.occlusion = loadTexture(model, mat.occlusionTexture.index, out.occlusion);

    out.baseColorFactor = glm::vec4(pbr.baseColorFactor[0], pbr.baseColorFactor[1],
                                    pbr.baseColorFactor[2], pbr.baseColorFactor[3]);
    out.metallicFactor = static_cast<float>(pbr.metallicFactor);
    out.roughnessFactor = static_cast<float>(pbr.roughnessFactor);
    out.emissiveFactor =
        glm::vec3(mat.emissiveFactor[0], mat.emissiveFactor[1], mat.emissiveFactor[2]);
}

void computeBounds(MeshData& out) {
    glm::vec3 lo(std::numeric_limits<float>::max());
    glm::vec3 hi(std::numeric_limits<float>::lowest());
    for (const Vertex& v : out.vertices) {
        lo = glm::min(lo, v.pos);
        hi = glm::max(hi, v.pos);
    }
    out.center = 0.5f * (lo + hi);
    out.radius = std::max(0.5f * glm::length(hi - lo), 1e-4f);
}

} // namespace

MeshData loadGltf(const std::string& path) {
    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    std::string err;
    std::string warn;

    const bool isBinary = path.size() >= 4 && path.compare(path.size() - 4, 4, ".glb") == 0;
    const bool ok = isBinary ? loader.LoadBinaryFromFile(&model, &err, &warn, path)
                             : loader.LoadASCIIFromFile(&model, &err, &warn, path);
    if (!ok) {
        throw std::runtime_error("failed to load glTF '" + path + "': " + err);
    }

    MeshData out;
    const glm::mat4 identity(1.0f);
    const int sceneIndex = model.defaultScene >= 0 ? model.defaultScene : 0;
    if (!model.scenes.empty()) {
        for (int nodeIndex : model.scenes[sceneIndex].nodes) {
            traverseNode(model, nodeIndex, identity, out);
        }
    } else {
        // No scene graph: just draw every mesh at identity.
        for (size_t n = 0; n < model.nodes.size(); ++n) {
            traverseNode(model, static_cast<int>(n), identity, out);
        }
    }

    if (out.vertices.empty()) {
        throw std::runtime_error("glTF '" + path + "' contained no geometry");
    }

    loadMaterial(model, out);
    computeBounds(out);
    return out;
}
