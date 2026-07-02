#version 450

// Metallic-roughness PBR: IBL ambient + one shadowed analytic directional light. Objects with
// useTextures == 0 (the ground plane) skip the material maps and shade from the push-constant
// factors instead.

layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 iblParams; // x = prefilter max LOD
    mat4 lightSpace;
    vec4 lightDir;
    vec4 lightColor;
} cam;

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform sampler2D metalRoughMap;
layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform sampler2D emissiveMap;
layout(binding = 5) uniform sampler2D occlusionMap;
layout(binding = 6) uniform sampler2D irradianceMap;
layout(binding = 7) uniform sampler2D prefilterMap;
layout(binding = 8) uniform sampler2D brdfLut;
layout(binding = 9) uniform sampler2D shadowMap;

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

layout(location = 0) out vec4 outColor;

const float PI = 3.14159265359;
const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 dirToUv(vec3 d) {
    vec2 uv = vec2(atan(d.z, d.x), asin(clamp(d.y, -1.0, 1.0)));
    return uv * invAtan + 0.5;
}

float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a2 = roughness * roughness * roughness * roughness;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}
float geometrySchlickGGX(float NdotX, float roughness) {
    float k = (roughness + 1.0);
    k = k * k / 8.0;
    return NdotX / (NdotX * (1.0 - k) + k);
}
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0);
    return F0 + (Fr - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 getNormal() {
    vec3 tangentNormal = texture(normalMap, fragUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent.xyz);
    vec3 B = cross(N, T) * fragTangent.w;
    return normalize(mat3(T, B, N) * tangentNormal);
}

// 3x3 PCF. Returns 1.0 (fully lit) .. 0.0 (fully shadowed).
float computeShadow(vec3 N, vec3 L) {
    vec4 lightClip = cam.lightSpace * vec4(fragWorldPos, 1.0);
    vec3 proj = lightClip.xyz / lightClip.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0; // outside the light frustum -> treat as lit
    }
    float bias = max(0.0025 * (1.0 - max(dot(N, L), 0.0)), 0.0008);
    vec2 texel = 1.0 / vec2(textureSize(shadowMap, 0));
    float sum = 0.0;
    for (int x = -1; x <= 1; ++x) {
        for (int y = -1; y <= 1; ++y) {
            float closest = texture(shadowMap, uv + vec2(x, y) * texel).r;
            sum += (proj.z - bias > closest) ? 1.0 : 0.0;
        }
    }
    return 1.0 - sum / 9.0;
}

void main() {
    bool useTextures = pc.emissiveFactor.w > 0.5;

    vec3 albedo;
    float roughness;
    float metallic;
    float ao;
    vec3 emissive;
    vec3 N;
    if (useTextures) {
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

    vec3 V = normalize(cam.camPos.xyz - fragWorldPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Shadowed directional light (Cook-Torrance).
    vec3 L = normalize(-cam.lightDir.xyz);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 Fdir = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * Fdir) / (4.0 * NdotV * NdotL + 0.0001);
    vec3 kDdir = (1.0 - Fdir) * (1.0 - metallic);
    float shadow = computeShadow(N, L);
    vec3 Lo = (kDdir * albedo / PI + specular) * cam.lightColor.rgb * NdotL * shadow;

    // Ambient IBL (split sum).
    vec3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuseIBL = texture(irradianceMap, dirToUv(N)).rgb * albedo;
    float lod = roughness * cam.iblParams.x;
    vec3 prefiltered = textureLod(prefilterMap, dirToUv(R), lod).rgb;
    vec2 envBRDF = texture(brdfLut, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefiltered * (F * envBRDF.x + envBRDF.y);
    vec3 ambient = (kD * diffuseIBL + specularIBL) * ao;

    vec3 color = ambient + Lo + emissive;
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14); // ACES
    outColor = vec4(clamp(color, 0.0, 1.0), 1.0);
}
