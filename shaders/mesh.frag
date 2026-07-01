#version 450

// Metallic-roughness PBR with a Cook-Torrance specular BRDF and two analytic directional
// lights. Image-based lighting replaces the flat ambient term in Phase 3b.

layout(binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec4 camPos;
} cam;

layout(binding = 1) uniform sampler2D baseColorMap;
layout(binding = 2) uniform sampler2D metalRoughMap;
layout(binding = 3) uniform sampler2D normalMap;
layout(binding = 4) uniform sampler2D emissiveMap;
layout(binding = 5) uniform sampler2D occlusionMap;

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

const float PI = 3.14159265359;

// Trowbridge-Reitz GGX normal distribution: concentration of microfacets aligned with H.
float distributionGGX(vec3 N, vec3 H, float roughness) {
    float a = roughness * roughness;
    float a2 = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float denom = NdotH * NdotH * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom);
}

// Smith geometry term with Schlick-GGX, accounting for both view and light occlusion.
float geometrySchlickGGX(float NdotX, float roughness) {
    float r = roughness + 1.0;
    float k = (r * r) / 8.0; // direct-lighting remap of roughness
    return NdotX / (NdotX * (1.0 - k) + k);
}
float geometrySmith(vec3 N, vec3 V, vec3 L, float roughness) {
    return geometrySchlickGGX(max(dot(N, V), 0.0), roughness) *
           geometrySchlickGGX(max(dot(N, L), 0.0), roughness);
}

// Fresnel-Schlick: how reflectivity rises toward grazing angles.
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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

    // Dielectrics reflect ~4%; metals tint their specular with the albedo.
    vec3 F0 = mix(vec3(0.04), albedo, metallic);

    const int LIGHT_COUNT = 2;
    vec3 lightDirs[2] = vec3[](normalize(vec3(-0.5, -1.0, -0.3)), normalize(vec3(0.6, 0.3, 0.8)));
    vec3 lightColors[2] = vec3[](vec3(3.2), vec3(1.4, 1.4, 1.8));

    vec3 Lo = vec3(0.0);
    for (int i = 0; i < LIGHT_COUNT; ++i) {
        vec3 L = -lightDirs[i]; // direction toward the light
        vec3 H = normalize(V + L);
        vec3 radiance = lightColors[i];

        float NDF = distributionGGX(N, H, roughness);
        float G = geometrySmith(N, V, L, roughness);
        vec3 F = fresnelSchlick(max(dot(H, V), 0.0), F0);

        vec3 specular = (NDF * G * F) /
                        (4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.0001);

        // Energy conservation: light not reflected specularly is diffusely scattered, and
        // metals have no diffuse response.
        vec3 kD = (vec3(1.0) - F) * (1.0 - metallic);

        float NdotL = max(dot(N, L), 0.0);
        Lo += (kD * albedo / PI + specular) * radiance * NdotL;
    }

    // Flat ambient stand-in for IBL (Phase 3b), modulated by the occlusion map.
    vec3 ambient = vec3(0.03) * albedo * ao;
    vec3 color = ambient + Lo + emissive;

    // ACES filmic tonemap, then let the sRGB swapchain apply the OETF.
    color = (color * (2.51 * color + 0.03)) / (color * (2.43 * color + 0.59) + 0.14);
    color = clamp(color, 0.0, 1.0);

    outColor = vec4(color, base.a);
}
