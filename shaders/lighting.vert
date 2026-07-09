#version 450

// Fullscreen triangle for the deferred lighting pass. Emits a G-buffer sample UV and a
// world-space view ray (for shading background pixels with the environment).
layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 iblParams;
    mat4 lightSpace;
    vec4 lightDir;
    vec4 lightColor;
} cam;

layout(location = 0) out vec2 fragUV;
layout(location = 1) out vec3 viewDir;

void main() {
    vec2 pos = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2); // (0,0) (2,0) (0,2)
    vec2 ndc = pos * 2.0 - 1.0;
    fragUV = pos; // spans 0..1 across the screen

    vec4 clip = vec4(ndc, 1.0, 1.0);
    vec3 viewSpace = (inverse(cam.proj) * clip).xyz;
    viewDir = mat3(inverse(cam.view)) * viewSpace;

    gl_Position = vec4(ndc, 0.0, 1.0);
}
