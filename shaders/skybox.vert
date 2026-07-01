#version 450

// Fullscreen-triangle skybox. Reconstructs a world-space view ray per vertex from the inverse
// camera matrices and emits it at the far plane (z = 1), so the mesh depth-tests over it.

layout(binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 iblParams;
} cam;

layout(location = 0) out vec3 viewDir;

void main() {
    // Covers the screen with a single oversized triangle (indices 0,1,2).
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    vec2 ndc = pos * 2.0 - 1.0;

    vec4 clip = vec4(ndc, 1.0, 1.0);
    vec3 viewSpace = (inverse(cam.proj) * clip).xyz;
    viewDir = mat3(inverse(cam.view)) * viewSpace;

    gl_Position = vec4(ndc, 1.0, 1.0); // z = w -> far plane
}
