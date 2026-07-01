#version 450

// Per-frame camera transform. Binding 0 matches the descriptor set layout on the C++ side.
layout(binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
} cam;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUV;

layout(location = 0) out vec3 fragNormal; // world-space normal
layout(location = 1) out vec2 fragUV;

void main() {
    gl_Position = cam.proj * cam.view * cam.model * vec4(inPos, 1.0);

    // Rotating the normal by the model matrix keeps lighting correct as the mesh spins.
    // (Uniform scale only here, so the upper-left 3x3 is sufficient; non-uniform scale would
    // need the inverse-transpose.)
    fragNormal = mat3(cam.model) * inNormal;
    fragUV = inUV;
}
