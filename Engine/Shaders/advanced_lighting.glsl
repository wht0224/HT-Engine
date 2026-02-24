#version 430

/**
 * Advanced Cinematic Lighting Compute Shader
 * 
 * Features:
 * - Global Illumination (SSGI) approximation
 * - Atmospheric Scattering (Rayleigh & Mie)
 * - 3D LUT Color Grading
 * - ACES Filmic Tone Mapping
 * - Dynamic Time of Day (Sunrise/Sunset)
 * 
 * Performance Target: 1.0-1.5ms @ 1080p
 */

layout(local_size_x = 16, local_size_y = 16) in;

// ============================================================================
// Input Textures
// ============================================================================
layout(r32f, binding = 0) readonly uniform image2D height_map;
layout(rgba32f, binding = 1) readonly uniform image2D albedo_map;
layout(rgba32f, binding = 2) readonly uniform image2D normal_map;
layout(rgba32f, binding = 3) readonly uniform image2D roughness_metallic_ao;
layout(rgba32f, binding = 4) readonly uniform image2D emission_map;

// ============================================================================
// Output Textures
// ============================================================================
layout(r32f, binding = 5) writeonly uniform image2D shadow_map;
layout(r32f, binding = 6) writeonly uniform image2D ao_map;
layout(rgba32f, binding = 7) writeonly uniform image2D gi_map;
layout(rgba32f, binding = 8) writeonly uniform image2D atmosphere_map;
layout(rgba32f, binding = 9) writeonly uniform image2D output_map;

// ============================================================================
// 3D LUT Texture for Color Grading
// ============================================================================
layout(rgba32f, binding = 10) readonly uniform image3D lut_texture;

// ============================================================================
// Uniforms - Lighting
// ============================================================================
uniform vec3 sun_dir;
uniform float sun_intensity;
uniform vec3 sun_color;
uniform vec3 ambient_color;
uniform float ambient_intensity;

// ============================================================================
// Uniforms - Shadow & AO
// ============================================================================
uniform float step_size;
uniform int max_steps;
uniform float ao_strength;
uniform float ao_radius;
uniform int ao_samples;

// ============================================================================
// Uniforms - Global Illumination
// ============================================================================
uniform float gi_strength;
uniform int gi_samples;
uniform float gi_radius;
uniform int enable_gi;

// ============================================================================
// Uniforms - Atmospheric Scattering
// ============================================================================
uniform int enable_atmosphere;
uniform vec3 rayleigh_beta;
uniform vec3 mie_beta;
uniform float mie_g;
uniform float rayleigh_height;
uniform float mie_height;
uniform float planet_radius;
uniform float atmosphere_height;
uniform vec3 camera_position;

// ============================================================================
// Uniforms - Color Grading & Tone Mapping
// ============================================================================
uniform int enable_lut;
uniform int enable_aces;
uniform float lut_intensity;
uniform float contrast;
uniform float saturation;
uniform float exposure;
uniform vec3 lift;
uniform vec3 gamma;
uniform vec3 gain;
uniform float film_grain;
uniform float vignette_strength;

// ============================================================================
// Uniforms - Time
// ============================================================================
uniform float time_of_day;  // 0.0 - 1.0 (0=midnight, 0.5=noon, 1.0=midnight)
uniform float time_scale;

// ============================================================================
// Constants
// ============================================================================
const float PI = 3.14159265359;
const float EPSILON = 0.0001;
const float GOLDEN_ANGLE = 2.399963229728653;

// ============================================================================
// ACES Filmic Tone Mapping
// ============================================================================

// ACES Input Transform (sRGB to ACES AP0)
const mat3 ACES_INPUT_MAT = mat3(
    0.59719, 0.35458, 0.04823,
    0.07600, 0.90834, 0.01566,
    0.02840, 0.13383, 0.83777
);

// ACES Output Transform (ACES AP0 to sRGB)
const mat3 ACES_OUTPUT_MAT = mat3(
    1.60475, -0.53108, -0.07367,
    -0.10208, 1.10813, -0.00605,
    -0.00327, -0.07276, 1.07602
);

vec3 RRTAndODTFit(vec3 v) {
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}

// Full ACES tone mapping pipeline
vec3 ACESFitted(vec3 color) {
    color = ACES_INPUT_MAT * color;
    color = RRTAndODTFit(color);
    color = ACES_OUTPUT_MAT * color;
    return clamp(color, 0.0, 1.0);
}

// Simplified ACES (faster, slightly less accurate)
vec3 ACESFilmic(vec3 x) {
    float a = 2.51;
    float b = 0.03;
    float c = 2.43;
    float d = 0.59;
    float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

// Alternative: Uncharted 2 tone mapping (cinematic look)
vec3 Uncharted2Tonemap(vec3 x) {
    float A = 0.15;
    float B = 0.50;
    float C = 0.10;
    float D = 0.20;
    float E = 0.02;
    float F = 0.30;
    return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F;
}

// ============================================================================
// Atmospheric Scattering (Rayleigh & Mie)
// ============================================================================

// Rayleigh phase function (isotropic-ish, affects blue light)
vec3 rayleigh_phase(float cos_theta) {
    return vec3(3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta));
}

// Mie phase function (anisotropic, affects sun glow/halo)
float mie_phase(float cos_theta, float g) {
    float g2 = g * g;
    float num = (1.0 - g2);
    float denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
    return num / denom;
}

// Density functions for atmospheric layers
float rayleigh_density(float h) {
    return exp(-h / rayleigh_height);
}

float mie_density(float h) {
    return exp(-h / mie_height);
}

// Sphere intersection for atmospheric calculations
bool intersect_sphere(vec3 ro, vec3 rd, float radius, out float t_near, out float t_far) {
    vec3 center = vec3(0.0, -planet_radius, 0.0);
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    
    if (discriminant < 0.0) return false;
    
    float sqrt_d = sqrt(discriminant);
    t_near = -b - sqrt_d;
    t_far = -b + sqrt_d;
    return t_far > 0.0;
}

// Compute light transmittance through atmosphere
vec3 compute_transmittance(vec3 pos, vec3 dir, float t_max, int samples) {
    float segment = t_max / float(samples);
    vec3 optical_depth = vec3(0.0);
    
    for (int i = 0; i < samples; i++) {
        float t = (float(i) + 0.5) * segment;
        vec3 p = pos + dir * t;
        float h = length(p) - planet_radius;
        
        if (h < 0.0) break;
        
        optical_depth += rayleigh_beta * rayleigh_density(h) * segment;
        optical_depth += mie_beta * mie_density(h) * segment;
    }
    
    return exp(-optical_depth);
}

// Main atmospheric scattering computation
vec3 compute_atmospheric_scattering(vec3 ray_origin, vec3 ray_dir, vec3 sun_direction, float t_max) {
    int samples = 16;
    int transmittance_samples = 8;
    
    float segment = t_max / float(samples);
    vec3 total_rayleigh = vec3(0.0);
    vec3 total_mie = vec3(0.0);
    vec3 optical_depth = vec3(0.0);
    
    float cos_theta = dot(ray_dir, sun_direction);
    vec3 rayleigh_phase_val = rayleigh_phase(cos_theta);
    float mie_phase_val = mie_phase(cos_theta, mie_g);
    
    for (int i = 0; i < samples; i++) {
        float t = (float(i) + 0.5) * segment;
        vec3 p = ray_origin + ray_dir * t;
        float h = length(p) - planet_radius;
        
        if (h < 0.0) break;
        
        float rayleigh_d = rayleigh_density(h);
        float mie_d = mie_density(h);
        
        optical_depth += rayleigh_beta * rayleigh_d * segment;
        optical_depth += mie_beta * mie_d * segment;
        
        float t_near, t_far;
        if (intersect_sphere(p, sun_direction, planet_radius + atmosphere_height, t_near, t_far)) {
            float t_sun = t_far;
            vec3 transmittance_sun = compute_transmittance(p, sun_direction, t_sun, transmittance_samples);
            vec3 transmittance_view = exp(-optical_depth);
            
            total_rayleigh += rayleigh_beta * rayleigh_d * transmittance_view * transmittance_sun * segment;
            total_mie += mie_beta * mie_d * transmittance_view * transmittance_sun * segment;
        }
    }
    
    vec3 result = total_rayleigh * rayleigh_phase_val + total_mie * mie_phase_val;
    
    // Sun disk and glow effects
    float sun_disk = smoothstep(0.9995, 0.9999, cos_theta);
    float sun_glow = pow(max(0.0, cos_theta), 64.0);
    float sun_halo = pow(max(0.0, cos_theta), 8.0) * 0.3;
    
    vec3 sun_color_base = vec3(1.0, 0.95, 0.8);
    vec3 sun_contribution = sun_color_base * (sun_disk * 15.0 + sun_glow * 2.0 + sun_halo);
    result += sun_contribution * exp(-optical_depth.r * 5.0);
    
    return result * sun_intensity;
}

// ============================================================================
// Global Illumination (SSGI - Screen Space Global Illumination)
// ============================================================================

// Hammersley sequence for quasi-random sampling
vec2 hammersley2d(int i, int N) {
    uint bits = uint(i);
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    float radicalInverseVDC = float(bits) * 2.3283064365386963e-10;
    return vec2(float(i) / float(N), radicalInverseVDC);
}

// Cosine-weighted hemisphere sampling
vec3 hemisphere_sample(int index, int sample_count) {
    float phi = float(index) * GOLDEN_ANGLE;
    float cos_theta = 1.0 - (float(index) + 0.5) / float(sample_count);
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    return vec3(
        cos(phi) * sin_theta,
        cos_theta,
        sin(phi) * sin_theta
    );
}

// Screen Space Global Illumination
vec3 compute_ssgi(ivec2 pos, ivec2 size, vec3 normal, float roughness) {
    if (enable_gi == 0) return vec3(0.0);
    
    vec3 gi_accum = vec3(0.0);
    float total_weight = 0.0;
    
    // Build tangent space
    vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, normal));
    vec3 bitangent = cross(normal, tangent);
    
    for (int i = 0; i < gi_samples; i++) {
        vec3 sample_dir = hemisphere_sample(i, gi_samples);
        
        // Transform to world space
        vec3 world_dir = tangent * sample_dir.x + normal * sample_dir.y + bitangent * sample_dir.z;
        
        // Screen space sampling
        vec2 offset = world_dir.xz * gi_radius;
        ivec2 sample_pos = pos + ivec2(offset);
        
        if (sample_pos.x >= 0 && sample_pos.x < size.x &&
            sample_pos.y >= 0 && sample_pos.y < size.y) {
            
            vec4 sample_color = imageLoad(albedo_map, sample_pos);
            float sample_height = imageLoad(height_map, sample_pos).r;
            
            // Weight by cosine and distance
            float weight = max(0.0, dot(normal, world_dir));
            float dist_falloff = 1.0 / (1.0 + length(offset) * 0.1);
            
            gi_accum += sample_color.rgb * weight * dist_falloff;
            total_weight += weight * dist_falloff;
        }
    }
    
    if (total_weight > 0.0) {
        gi_accum /= total_weight;
    }
    
    // Modulate by roughness (smoother surfaces have less GI variation)
    return gi_accum * gi_strength * (1.0 - roughness * 0.5);
}

// ============================================================================
// Shadow Calculation (Ray Marching)
// ============================================================================

float compute_shadow(ivec2 pos, ivec2 size, float base_height) {
    float shadow = 1.0;
    
    vec2 sun_xz = normalize(vec2(sun_dir.x, sun_dir.z));
    float sun_height_factor = -sun_dir.y;
    
    // Sun below horizon = full shadow
    if (sun_dir.y >= 0.0) {
        return 0.0;
    }
    
    float current_dist = step_size;
    float ray_height = base_height + sun_height_factor * step_size;
    float height_gain = sun_height_factor * step_size;
    
    vec2 current_pos = vec2(pos) + sun_xz * step_size;
    
    for (int i = 0; i < max_steps; i++) {
        current_pos += sun_xz * step_size;
        ray_height += height_gain;
        
        // Boundary check
        if (current_pos.x < 0.0 || current_pos.x >= float(size.x) ||
            current_pos.y < 0.0 || current_pos.y >= float(size.y)) {
            break;
        }
        
        float h = imageLoad(height_map, ivec2(current_pos)).r;
        
        if (h > ray_height) {
            shadow = 0.0;
            break;
        }
    }
    
    return shadow;
}

// Soft shadow with penumbra
float compute_soft_shadow(ivec2 pos, ivec2 size, float base_height) {
    float shadow = 1.0;
    float penumbra = 1.0;
    
    vec2 sun_xz = normalize(vec2(sun_dir.x, sun_dir.z));
    float sun_height_factor = -sun_dir.y;
    
    if (sun_dir.y >= 0.0) {
        return 0.0;
    }
    
    float current_dist = step_size;
    float ray_height = base_height + sun_height_factor * step_size;
    float height_gain = sun_height_factor * step_size;
    
    vec2 current_pos = vec2(pos) + sun_xz * step_size;
    float blocker_dist = 0.0;
    bool blocked = false;
    
    for (int i = 0; i < max_steps; i++) {
        current_pos += sun_xz * step_size;
        ray_height += height_gain;
        current_dist += step_size;
        
        if (current_pos.x < 0.0 || current_pos.x >= float(size.x) ||
            current_pos.y < 0.0 || current_pos.y >= float(size.y)) {
            break;
        }
        
        float h = imageLoad(height_map, ivec2(current_pos)).r;
        float height_diff = h - ray_height;
        
        if (height_diff > 0.0 && !blocked) {
            blocked = true;
            blocker_dist = current_dist;
            // Soft penumbra based on distance
            penumbra = 1.0 - smoothstep(0.0, 50.0, height_diff);
        }
        
        if (height_diff > 2.0) {
            shadow = 0.0;
            break;
        }
    }
    
    return shadow * penumbra;
}

// ============================================================================
// Ambient Occlusion (SSAO-like)
// ============================================================================

float compute_ao(ivec2 pos, ivec2 size, float base_height, vec3 normal) {
    // Sample neighbors
    float h_l = imageLoad(height_map, pos + ivec2(-1, 0)).r;
    float h_r = imageLoad(height_map, pos + ivec2(1, 0)).r;
    float h_u = imageLoad(height_map, pos + ivec2(0, 1)).r;
    float h_d = imageLoad(height_map, pos + ivec2(0, -1)).r;
    
    // Laplacian approximation for curvature
    float laplacian = (h_l + h_r + h_u + h_d) - 4.0 * base_height;
    float curvature = laplacian / (abs(base_height) + 1.0);
    
    // Sigmoid function for smooth AO
    float ao = 1.0 / (1.0 + exp(-curvature * 5.0 * ao_strength));
    
    // Horizon-based AO enhancement
    float horizon_ao = 1.0 - max(0.0, -normal.y) * 0.5;
    
    return clamp(ao * horizon_ao, 0.2, 1.0);
}

// ============================================================================
// Color Grading & LUT
// ============================================================================

// Lift/Gamma/Gain color correction
vec3 apply_lift_gamma_gain(vec3 color, vec3 lift, vec3 gamma, vec3 gain) {
    color = pow(max(color, vec3(0.0)), gamma);
    color = gain * color + lift * (1.0 - color);
    return color;
}

// Contrast adjustment
vec3 apply_contrast(vec3 color, float contrast) {
    return (color - 0.5) * contrast + 0.5;
}

// Luminance calculation (Rec. 709)
float luminance(vec3 color) {
    return dot(color, vec3(0.2126, 0.7152, 0.0722));
}

// Saturation adjustment
vec3 apply_saturation(vec3 color, float saturation) {
    float lum = luminance(color);
    return mix(vec3(lum), color, saturation);
}

// Sample 3D LUT with trilinear interpolation
vec3 sample_3d_lut(vec3 color, int lut_size) {
    // Map color to LUT coordinates
    vec3 lut_coord = clamp(color, 0.0, 1.0) * float(lut_size - 1);
    
    // Trilinear interpolation
    ivec3 lut_min = ivec3(floor(lut_coord));
    vec3 weights = fract(lut_coord);
    
    vec3 result = vec3(0.0);
    
    for (int x = 0; x <= 1; x++) {
        for (int y = 0; y <= 1; y++) {
            for (int z = 0; z <= 1; z++) {
                ivec3 idx = ivec3(
                    clamp(lut_min.x + x, 0, lut_size - 1),
                    clamp(lut_min.y + y, 0, lut_size - 1),
                    clamp(lut_min.z + z, 0, lut_size - 1)
                );
                
                vec3 w = vec3(
                    x == 0 ? 1.0 - weights.x : weights.x,
                    y == 0 ? 1.0 - weights.y : weights.y,
                    z == 0 ? 1.0 - weights.z : weights.z
                );
                
                float weight = w.x * w.y * w.z;
                vec4 lut_val = imageLoad(lut_texture, idx);
                result += lut_val.rgb * weight;
            }
        }
    }
    
    return result;
}

// Full color grading pipeline
vec3 apply_color_grading(vec3 color, int lut_size) {
    if (enable_lut == 0) return color;
    
    // Exposure adjustment
    color *= exposure;
    
    // Lift/Gamma/Gain
    color = apply_lift_gamma_gain(color, lift, gamma, gain);
    
    // Contrast
    color = apply_contrast(color, contrast);
    
    // Saturation
    color = apply_saturation(color, saturation);
    
    // 3D LUT lookup
    vec3 lut_color = sample_3d_lut(color, lut_size);
    color = mix(color, lut_color, lut_intensity);
    
    return max(color, vec3(0.0));
}

// ============================================================================
// Film Effects
// ============================================================================

// Vignette effect
vec3 apply_vignette(vec3 color, vec2 uv, float strength) {
    vec2 center = uv - vec2(0.5);
    float dist = length(center);
    float vignette = 1.0 - dist * dist * strength;
    return color * clamp(vignette, 0.0, 1.0);
}

// Film grain (simple noise)
float random(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 apply_film_grain(vec3 color, vec2 uv, float strength) {
    float noise = random(uv + time_of_day * 100.0);
    return color + (noise - 0.5) * strength;
}

// ============================================================================
// PBR Lighting Calculation
// ============================================================================

// Normal Distribution Function (GGX/Trowbridge-Reitz)
float D_GGX(float NoH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NoH2 = NoH * NoH;
    float denom = NoH2 * (alpha2 - 1.0) + 1.0;
    return alpha2 / (PI * denom * denom);
}

// Geometry function (Smith's method with GGX)
float G_Smith(float NoV, float NoL, float roughness) {
    float k = (roughness * roughness) / 2.0;
    float G1_V = NoV / (NoV * (1.0 - k) + k);
    float G1_L = NoL / (NoL * (1.0 - k) + k);
    return G1_V * G1_L;
}

// Fresnel function (Schlick's approximation)
vec3 F_Schlick(float cosTheta, vec3 F0) {
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

// Cook-Torrance BRDF
vec3 cook_torrance(vec3 N, vec3 V, vec3 L, vec3 albedo, float roughness, float metallic) {
    vec3 H = normalize(V + L);
    
    float NoV = max(dot(N, V), 0.0);
    float NoL = max(dot(N, L), 0.0);
    float NoH = max(dot(N, H), 0.0);
    float HoV = max(dot(H, V), 0.0);
    
    // F0 for dielectrics and metals
    vec3 F0 = mix(vec3(0.04), albedo, metallic);
    
    // BRDF components
    float D = D_GGX(NoH, roughness);
    vec3 F = F_Schlick(HoV, F0);
    float G = G_Smith(NoV, NoL, roughness);
    
    // Specular
    vec3 specular = (D * F * G) / (4.0 * NoV * NoL + EPSILON);
    
    // Diffuse (Lambert for dielectrics, 0 for metals)
    vec3 diffuse = albedo * (1.0 - metallic) / PI;
    
    return (diffuse + specular) * NoL;
}

// ============================================================================
// Main Function
// ============================================================================

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(height_map);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    // Sample input data
    float base_height = imageLoad(height_map, pos).r;
    vec4 albedo = imageLoad(albedo_map, pos);
    vec4 normal_data = imageLoad(normal_map, pos);
    vec4 rma = imageLoad(roughness_metallic_ao, pos);
    vec4 emission = imageLoad(emission_map, pos);
    
    vec3 normal = normalize(normal_data.xyz * 2.0 - 1.0);
    float roughness = rma.r;
    float metallic = rma.g;
    float material_ao = rma.b;
    
    // UV for effects
    vec2 uv = vec2(float(pos.x) / float(size.x), float(pos.y) / float(size.y));
    
    // =========================================================================
    // 1. Shadow Calculation
    // =========================================================================
    float shadow = compute_shadow(pos, size, base_height);
    imageStore(shadow_map, pos, vec4(shadow, 0.0, 0.0, 0.0));
    
    // =========================================================================
    // 2. Ambient Occlusion
    // =========================================================================
    float ao = compute_ao(pos, size, base_height, normal);
    ao *= material_ao;
    imageStore(ao_map, pos, vec4(ao, 0.0, 0.0, 0.0));
    
    // =========================================================================
    // 3. Global Illumination (SSGI)
    // =========================================================================
    vec3 gi = compute_ssgi(pos, size, normal, roughness);
    imageStore(gi_map, pos, vec4(gi, 1.0));
    
    // =========================================================================
    // 4. PBR Lighting
    // =========================================================================
    vec3 view_dir = normalize(vec3(0.0, 1.0, 1.0));  // Simplified view direction
    vec3 light_dir = -sun_dir;
    
    // Direct lighting
    vec3 direct_lighting = cook_torrance(normal, view_dir, light_dir, albedo.rgb, roughness, metallic);
    direct_lighting *= sun_color * sun_intensity * shadow;
    
    // Ambient lighting
    vec3 ambient = ambient_color * ambient_intensity * albedo.rgb * ao;
    
    // Indirect lighting (GI)
    vec3 indirect = gi * ao * albedo.rgb;
    
    // Emission
    vec3 emission_light = emission.rgb * emission.a;
    
    // Combine lighting
    vec3 final_color = ambient + direct_lighting + indirect + emission_light;
    
    // =========================================================================
    // 5. Atmospheric Scattering
    // =========================================================================
    vec3 atmosphere = vec3(0.0);
    if (enable_atmosphere != 0) {
        vec3 ray_origin = camera_position;
        ray_origin.y += planet_radius + base_height * 100.0;
        
        vec3 ray_dir = normalize(vec3(
            (uv.x - 0.5) * 2.0,
            0.0,
            (uv.y - 0.5) * 2.0
        ));
        
        float t_near, t_far;
        float t_max = 100000.0;
        if (intersect_sphere(ray_origin, ray_dir, planet_radius + atmosphere_height, t_near, t_far)) {
            t_max = t_far;
        }
        
        atmosphere = compute_atmospheric_scattering(ray_origin, ray_dir, light_dir, t_max);
        atmosphere = pow(atmosphere, vec3(1.0 / 2.2));  // Gamma correction
    }
    imageStore(atmosphere_map, pos, vec4(atmosphere, 1.0));
    
    // Blend atmosphere at horizon
    float horizon_factor = 1.0 - abs(normal.y);
    horizon_factor = pow(horizon_factor, 2.0);
    final_color = mix(final_color, atmosphere, horizon_factor * 0.2);
    
    // =========================================================================
    // 6. ACES Tone Mapping
    // =========================================================================
    if (enable_aces != 0) {
        final_color = ACESFilmic(final_color);
    }
    
    // =========================================================================
    // 7. Color Grading
    // =========================================================================
    final_color = apply_color_grading(final_color, 64);
    
    // =========================================================================
    // 8. Film Effects
    // =========================================================================
    // Vignette
    final_color = apply_vignette(final_color, uv, vignette_strength);
    
    // Film grain
    final_color = apply_film_grain(final_color, uv, film_grain);
    
    // Final clamp
    final_color = clamp(final_color, 0.0, 1.0);
    
    // Output
    imageStore(output_map, pos, vec4(final_color, 1.0));
}
