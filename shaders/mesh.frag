#version 450

// Combined image sampler at binding 1: the base-color (albedo) texture.
layout(binding = 1) uniform sampler2D albedo;

layout(location = 0) in vec3 fragNormal;
layout(location = 1) in vec2 fragUV;

layout(location = 0) out vec4 outColor;

void main() {
    vec3 N = normalize(fragNormal);
    // A single hard-coded directional light. Real PBR lighting arrives in Phase 3.
    vec3 L = normalize(vec3(0.5, 1.0, 0.3));
    float diffuse = max(dot(N, L), 0.0);
    float ambient = 0.15;

    vec3 base = texture(albedo, fragUV).rgb;
    outColor = vec4(base * (ambient + diffuse), 1.0);
}
