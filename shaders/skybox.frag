#version 450

// Sample the equirectangular environment along the view ray and tonemap to match the mesh.

layout(binding = 1) uniform sampler2D envMap;

layout(location = 0) in vec3 viewDir;
layout(location = 0) out vec4 outColor;

const vec2 invAtan = vec2(0.1591, 0.3183);

vec2 dirToUv(vec3 d) {
    vec2 uv = vec2(atan(d.z, d.x), asin(clamp(d.y, -1.0, 1.0)));
    return uv * invAtan + 0.5;
}

void main() {
    vec3 color = texture(envMap, dirToUv(normalize(viewDir))).rgb;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14); // ACES
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
