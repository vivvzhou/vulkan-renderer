#version 450

// Metallic-roughness PBR with a Cook-Torrance specular BRDF and two analytic directional
// lights. Image-based lighting replaces the flat ambient term in Phase 3b.

layout(binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 camPos;
    vec4 iblParams; // x = prefilter max LOD
} cam;

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform sampler2D metalRoughMap;
layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform sampler2D emissiveMap;
layout(binding = 5) uniform sampler2D occlusionMap;

// Precomputed image-based lighting (equirectangular).
layout(binding = 6) uniform sampler2D irradianceMap; // diffuse IBL
layout(binding = 7) uniform sampler2D prefilterMap;  // specular IBL (roughness in mips)
layout(binding = 8) uniform sampler2D brdfLut;       // environment BRDF integration

layout(push_constant) uniform Material {
    vec4 baseColorFactor;
    vec4 emissiveFactor; // xyz used
    float metallicFactor;
    float roughnessFactor;
} mat;

layout(location = 0) in vec3 fragWorldPos;
layout(location = 1) in vec3 fragNormal;
layout(location = 2) in vec2 fragUV;
layout(location = 3) in vec4 fragTangent;

layout(location = 0) out vec4 outColor;

// Roughness-aware Fresnel for the IBL ambient term (rough surfaces reflect less at grazing).
vec3 fresnelSchlickRoughness(float cosTheta, vec3 F0, float roughness) {
    vec3 Fr = max(vec3(1.0 - roughness), F0);
    return F0 + (Fr - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

const vec2 invAtan = vec2(0.1591, 0.3183);
vec2 dirToUv(vec3 d) {
    vec2 uv = vec2(atan(d.z, d.x), asin(clamp(d.y, -1.0, 1.0)));
    return uv * invAtan + 0.5;
}

// Perturb the geometric normal by the tangent-space normal map.
vec3 getNormal() {
    vec3 tangentNormal = texture(normalMap, fragUV).xyz * 2.0 - 1.0;
    vec3 N = normalize(fragNormal);
    vec3 T = normalize(fragTangent.xyz);
    vec3 B = cross(N, T) * fragTangent.w;
    return normalize(mat3(T, B, N) * tangentNormal);
}

void main() {
    vec4 base = texture(baseColorMap, fragUV) * mat.baseColorFactor;
    vec3 albedo = base.rgb;
    vec3 mr = texture(metalRoughMap, fragUV).rgb;
    float roughness = clamp(mr.g * mat.roughnessFactor, 0.04, 1.0);
    float metallic = mr.b * mat.metallicFactor;
    float ao = texture(occlusionMap, fragUV).r;
    vec3 emissive = texture(emissiveMap, fragUV).rgb * mat.emissiveFactor.xyz;

    vec3 N = getNormal();
    vec3 V = normalize(cam.camPos.xyz - fragWorldPos);
    vec3 R = reflect(-V, N);
    float NdotV = max(dot(N, V), 0.0);

    // Dielectrics reflect ~4%; metals tint their specular with the albedo.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    // Image-based ambient: split diffuse (irradiance) and specular (prefilter * BRDF LUT).
    vec3 F = fresnelSchlickRoughness(NdotV, F0, roughness);
    vec3 kD = (1.0 - F) * (1.0 - metallic);

    vec3 irradiance = texture(irradianceMap, dirToUv(N)).rgb;
    vec3 diffuseIBL = irradiance * albedo;

    float lod = roughness * cam.iblParams.x;
    vec3 prefiltered = textureLod(prefilterMap, dirToUv(R), lod).rgb;
    vec2 envBRDF = texture(brdfLut, vec2(NdotV, roughness)).rg;
    vec3 specularIBL = prefiltered * (F * envBRDF.x + envBRDF.y);

    vec3 color = (kD * diffuseIBL + specularIBL) * ao + emissive;

    // ACES filmic tonemap, then let the sRGB swapchain apply the OETF.
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    outColor = vec4(color, base.a);
}
