#version 450

// Per-frame camera transform. Binding 0 matches the descriptor set layout on the C++ side.
layout(binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 camPos;    // world-space eye position (xyz)
    vec4 iblParams; // x = prefilter max LOD
} cam;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;
layout(location = 3) in vec4 inTangent;

layout(location = 0) out vec3 fragWorldPos;
layout(location = 1) out vec3 fragNormal;
layout(location = 2) out vec2 fragUV;
layout(location = 3) out vec4 fragTangent;

void main() {
    vec4 world = cam.model * vec4(inPos, 1.0);
    fragWorldPos = world.xyz;
    gl_Position = cam.proj * cam.view * world;

    // Uniform scale only, so the upper-left 3x3 suffices for normals and tangents.
    mat3 normalMatrix = mat3(cam.model);
    fragNormal = normalMatrix * inNormal;
    fragTangent = vec4(normalMatrix * inTangent.xyz, inTangent.w);
    fragUV = inUV;
}
