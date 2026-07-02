#version 450

// Per-frame camera + light state. The per-object model matrix arrives via push constants so a
// single UBO/pipeline can draw multiple objects (the mesh and the ground plane).
layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;    // world-space eye position (xyz)
    vec4 iblParams; // x = prefilter max LOD
    mat4 lightSpace;
    vec4 lightDir;   // directional light travel direction (xyz)
    vec4 lightColor; // rgb intensity
} cam;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 baseColorFactor;
    vec4 emissiveFactor; // w = useTextures flag
    float metallicFactor;
    float roughnessFactor;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec4 fragTangent;

void main() {
    vec4 world = pc.model * vec4(inPos, 1.0);
    fragWorldPos = world.xyz;
    gl_Position = cam.proj * cam.view * world;

    mat3 normalMatrix = mat3(pc.model);
    fragNormal = normalMatrix * inNormal;
    fragTangent = vec4(normalMatrix * inTangent.xyz, inTangent.w);
    fragUV = inUV;
}
