#version 450

// Depth-only pass: transform each vertex into the light's clip space. No fragment shader is
// bound; the render pass writes only depth, producing the shadow map.

layout(push_constant) uniform PushConstants {
    mat4 lightMVP; // lightSpace * model for the current object
} pc;

layout(location = 0) in vec3 inPos;

void main() {
    gl_Position = pc.lightMVP * vec4(inPos, 1.0);
}
