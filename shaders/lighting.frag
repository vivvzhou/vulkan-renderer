#version 450

// Deferred lighting pass: read the G-buffer and shade each pixel with the shadowed directional
// light plus split-sum IBL. Background pixels (no geometry written) show the environment.

layout(binding = 0) uniform CameraUBO {
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 iblParams; // x = prefilter max LOD
    mat4 lightSpace;
    vec4 lightDir;
    vec4 lightColor;
} cam;

layout(binding = 1) uniform sampler2D gPosition; // xyz world position
layout(binding = 2) uniform sampler2D gNormal;   // xyz normal, w roughness
layout(binding = 3) uniform sampler2D gAlbedo;   // rgb albedo, a metallic
layout(binding = 4) uniform sampler2D gEmissive; // rgb emissive, a ao
layout(binding = 5) uniform sampler2D shadowMap;
layout(binding = 6) uniform sampler2D irradianceMap;
layout(binding = 7) uniform sampler2D prefilterMap;
layout(binding = 8) uniform sampler2D brdfLut;
layout(binding = 9) uniform sampler2D environmentMap;
layout(binding = 10) uniform sampler2D ssaoMap; // neural ambient occlusion

layout(location = 0) in vec2 fragUV;
layout(location = 1) in vec3 viewDir;

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

vec3 aces(vec3 c) {
    return clamp((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14), 0.0, 1.0);
}

float computeShadow(vec3 worldPos, vec3 N, vec3 L) {
    vec4 lightClip = cam.lightSpace * vec4(worldPos, 1.0);
    vec3 proj = lightClip.xyz / lightClip.w;
    vec2 uv = proj.xy * 0.5 + 0.5;
    if (proj.z > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return 1.0;
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
    vec4 packedNormal = texture(gNormal, fragUV);
    vec3 N = packedNormal.xyz;

    // Background: the geometry pass cleared the normal to zero here. Keep the environment very
    // dark on screen (product-shot look) while the object still reflects the full-brightness
    // environment through IBL below.
    if (dot(N, N) < 0.5) {
        vec3 bg = texture(environmentMap, dirToUv(normalize(viewDir))).rgb;
        outColor = vec4(aces(bg * 0.08), 1.0);
        return;
    }

    N = normalize(N);
    float roughness = packedNormal.w;
    vec3 worldPos = texture(gPosition, fragUV).xyz;
    vec4 albedoMetallic = texture(gAlbedo, fragUV);
    vec3 albedo = albedoMetallic.rgb;
    float metallic = albedoMetallic.a;
    vec4 emissiveAo = texture(gEmissive, fragUV);
    vec3 emissive = emissiveAo.rgb;
    float ao = emissiveAo.a;

    vec3 V = normalize(cam.camPos.xyz - worldPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Shadowed directional light.
    vec3 L = normalize(-cam.lightDir.xyz);
    vec3 H = normalize(V + L);
    float NdotL = max(dot(N, L), 0.0);
    float NDF = distributionGGX(N, H, roughness);
    float G = geometrySmith(N, V, L, roughness);
    vec3 Fdir = fresnelSchlick(max(dot(H, V), 0.0), F0);
    vec3 specular = (NDF * G * Fdir) / (4.0 * NdotV * NdotL + 0.0001);
    vec3 kDdir = (1.0 - Fdir) * (1.0 - metallic);
    float shadow = computeShadow(worldPos, N, L);
    vec3 Lo = (kDdir * albedo / PI + specular) * cam.lightColor.rgb * NdotL * shadow;

    // Ambient IBL (split sum).
    vec3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = (1.0 - F) * (1.0 - metallic);
    vec3 diffuseIBL = texture(irradianceMap, dirToUv(N)).rgb * albedo;
    float lod = roughness * cam.iblParams.x;
    vec3 prefiltered = textureLod(prefilterMap, dirToUv(R), lod).rgb;
    vec2 envBRDF = texture(brdfLut, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefiltered * (F * envBRDF.x + envBRDF.y);

    // Neural screen-space AO modulates the (indirect) ambient term along with the baked AO map.
    // The power is an intensity control that deepens contact/crease darkening.
    float ssao = pow(clamp(texture(ssaoMap, fragUV).r, 0.0, 1.0), 2.5);

    // Dim the environment lighting so the scene reads dark; the object still catches the
    // environment as reflections and the directional key light does most of the shaping.
    const float kDiffuseIbl = 0.5;
    const float kSpecularIbl = 0.9;
    vec3 ambient = (kD * diffuseIBL * kDiffuseIbl + specularIBL * kSpecularIbl) * ao * ssao;

    outColor = vec4(aces(ambient + Lo + emissive), 1.0);
}
