#version 450

// Geometry pass: write surface attributes into the G-buffer. No lighting happens here. The
// four targets pack: world position, world normal + roughness, albedo + metallic, and
// emissive + ambient occlusion. Objects with useTextures == 0 (the ground) shade flat.

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform sampler2D metalRoughMap;
layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform sampler2D emissiveMap;
layout(binding = 5) uniform sampler2D occlusionMap;

layout(push_constant) uniform PushConstants {
    mat4 model;
    vec4 baseColorFactor;
    vec4 emissiveFactor; // w = useTextures flag
    float metallicFactor;
    float roughnessFactor;
} pc;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragTangent;

layout(location = 0) out vec4 outPosition; // xyz world position
layout(location = 1) out vec4 outNormal;   // xyz world normal, w = roughness
layout(location = 2) out vec4 outAlbedo;   // rgb albedo, a = metallic
layout(location = 3) out vec4 outEmissive; // rgb emissive, a = ambient occlusion

vec3 getNormal() {
    vec3 tangentNormal = texture(normalMap, fragUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent.xyz);
    vec3 B = cross(N, T) * fragTangent.w;
    return normalize(mat3(T, B, N) * tangentNormal);
}

void main() {
    vec3 albedo;
    float roughness;
    float metallic;
    float ao;
    vec3 emissive;
    vec3 N;
    if (pc.emissiveFactor.w > 0.5) {
        albedo = (texture(baseColorMap, fragUV) * pc.baseColorFactor).rgb;
        vec3 mr = texture(metalRoughMap, fragUV).rgb;
        roughness = clamp(mr.g * pc.roughnessFactor, 0.04, 1.0);
        metallic = mr.b * pc.metallicFactor;
        ao = texture(occlusionMap, fragUV).r;
        emissive = texture(emissiveMap, fragUV).rgb * pc.emissiveFactor.xyz;
        N = getNormal();
    } else {
        albedo = pc.baseColorFactor.rgb;
        roughness = clamp(pc.roughnessFactor, 0.04, 1.0);
        metallic = pc.metallicFactor;
        ao = 1.0;
        emissive = vec3(0.0);
        N = normalize(fragNormal);
    }

    outPosition = vec4(fragWorldPos, 1.0);
    outNormal = vec4(N, roughness);
    outAlbedo = vec4(albedo, metallic);
    outEmissive = vec4(emissive, ao);
}
