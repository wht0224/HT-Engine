#version 430

/*
 * Advanced Water Rendering Compute Shaders
 * 
 * 实现业内顶级水面效果的Compute Shader集合：
 * 1. Gerstner波浪模拟
 * 2. 屏幕空间反射 (SSR)
 * 3. 水下焦散效果
 * 4. 动态涟漪系统
 * 5. 水下视野扭曲和模糊
 * 
 * 针对GTX 1650 Max-Q优化
 */

// ============================================================================
// 常量定义
// ============================================================================
#define PI 3.14159265359
#define G 9.81
#define MAX_WAVES 8
#define MAX_RIPPLES 32

// ============================================================================
// Gerstner波浪计算 (Pass 0)
// ============================================================================
#ifdef GERSTNER_PASS

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba16f, binding = 0) writeonly uniform image2D position_buffer;
layout(rgba16f, binding = 1) writeonly uniform image2D normal_buffer;

uniform float u_time;
uniform int u_num_waves;
uniform float u_wave_speed;
uniform float u_world_size;

// 波浪参数结构
struct WaveParams {
    float frequency;
    float amplitude;
    float phase;
    float direction_x;
    float direction_y;
    float q;  // 陡峭度参数
};

layout(std430, binding = 2) readonly buffer WaveBuffer {
    WaveParams waves[MAX_WAVES];
};

// 改进的Gerstner波函数
vec3 gerstner_wave(vec2 pos, WaveParams wave) {
    float k = wave.frequency;
    float w = sqrt(G * k) * u_wave_speed;
    vec2 d = normalize(vec2(wave.direction_x, wave.direction_y));
    
    float theta = dot(d, pos) * k - w * u_time + wave.phase;
    
    float c = cos(theta);
    float s = sin(theta);
    
    // 标准Gerstner波公式
    float qa = wave.q * wave.amplitude;
    float x = d.x * qa * c;
    float z = d.y * qa * c;
    float y = wave.amplitude * s;
    
    return vec3(x, y, z);
}

// 计算Gerstner波法线和切线
vec3 gerstner_normal(vec2 pos, WaveParams wave) {
    float k = wave.frequency;
    float w = sqrt(G * k) * u_wave_speed;
    vec2 d = normalize(vec2(wave.direction_x, wave.direction_y));
    
    float theta = dot(d, pos) * k - w * u_time + wave.phase;
    
    float c = cos(theta);
    float s = sin(theta);
    
    float wa = wave.frequency * wave.amplitude;
    float qwa = wave.q * wa;
    
    // 法线计算
    float nx = -d.x * wa * c;
    float nz = -d.y * wa * c;
    float ny = 1.0 - qwa * s;
    
    return vec3(nx, ny, nz);
}

void main_gerstner() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(position_buffer);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec2 uv = vec2(pos) / vec2(size);
    vec2 world_pos = (uv - 0.5) * u_world_size;
    
    vec3 displacement = vec3(0.0);
    vec3 normal = vec3(0.0, 1.0, 0.0);
    vec3 binormal = vec3(1.0, 0.0, 0.0);
    vec3 tangent = vec3(0.0, 0.0, 1.0);
    
    // 叠加所有波浪
    for (int i = 0; i < u_num_waves && i < MAX_WAVES; i++) {
        WaveParams wave = waves[i];
        
        vec3 wave_disp = gerstner_wave(world_pos, wave);
        displacement += wave_disp;
        
        // 累加法线扰动
        vec3 wave_normal = gerstner_normal(world_pos, wave);
        normal.x += wave_normal.x;
        normal.z += wave_normal.z;
        normal.y += wave_normal.y - 1.0;
    }
    
    // 归一化
    normal = normalize(normal);
    
    // 最终位置
    vec3 final_pos = vec3(world_pos.x + displacement.x, displacement.y, world_pos.y + displacement.z);
    
    // 存储结果
    imageStore(position_buffer, pos, vec4(final_pos, 1.0));
    imageStore(normal_buffer, pos, vec4(normal * 0.5 + 0.5, 1.0));
}

#endif

// ============================================================================
// 屏幕空间反射 (Pass 1)
// ============================================================================
#ifdef SSR_PASS

layout(local_size_x = 8, local_size_y = 8) in;

layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
layout(r32f, binding = 1) readonly uniform image2D depth_buffer;
layout(rgba16f, binding = 2) readonly uniform image2D normal_buffer;
layout(r32f, binding = 3) readonly uniform image2D roughness_buffer;
layout(rgba16f, binding = 4) writeonly uniform image2D reflection_buffer;

uniform mat4 u_view_matrix;
uniform mat4 u_projection_matrix;
uniform mat4 u_inv_view_matrix;
uniform mat4 u_inv_projection_matrix;
uniform vec3 u_camera_position;
uniform int u_max_steps;
uniform int u_binary_search_steps;
uniform float u_intensity;
uniform float u_max_distance;
uniform float u_pixel_stride;

// 坐标转换函数
vec3 screen_to_view(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 view = u_inv_projection_matrix * ndc;
    return view.xyz / view.w;
}

vec3 view_to_world(vec3 view_pos) {
    vec4 world = u_inv_view_matrix * vec4(view_pos, 1.0);
    return world.xyz;
}

vec2 world_to_screen(vec3 world_pos) {
    vec4 view_pos = u_view_matrix * vec4(world_pos, 1.0);
    vec4 ndc = u_projection_matrix * view_pos;
    vec2 screen = ndc.xy / ndc.w;
    return screen * 0.5 + 0.5;
}

float linear_depth(float depth_buffer_val) {
    float near = 0.1;
    float far = 1000.0;
    float z = depth_buffer_val * 2.0 - 1.0;
    return (2.0 * near * far) / (far + near - z * (far - near));
}

// 改进的Hierarchical Z射线步进
bool hierarchical_ray_march(vec3 origin, vec3 direction, out vec2 hit_uv, out float hit_depth) {
    vec3 current_pos = origin;
    float step_size = u_max_distance / float(u_max_steps);
    float thickness = 0.5;
    
    for (int i = 0; i < u_max_steps; i++) {
        current_pos += direction * step_size;
        
        vec2 screen_uv = world_to_screen(current_pos);
        
        if (screen_uv.x < 0.0 || screen_uv.x > 1.0 ||
            screen_uv.y < 0.0 || screen_uv.y > 1.0) {
            return false;
        }
        
        ivec2 tex_pos = ivec2(screen_uv * vec2(imageSize(depth_buffer)));
        float scene_depth = imageLoad(depth_buffer, tex_pos).r;
        float ray_depth = linear_depth(length(current_pos - u_camera_position));
        float scene_linear = linear_depth(scene_depth);
        
        float depth_diff = ray_depth - scene_linear;
        
        if (depth_diff > 0.0 && depth_diff < thickness) {
            // 二分搜索细化
            vec3 start = current_pos - direction * step_size;
            vec3 end = current_pos;
            
            for (int j = 0; j < u_binary_search_steps; j++) {
                vec3 mid = (start + end) * 0.5;
                vec2 mid_uv = world_to_screen(mid);
                
                if (mid_uv.x < 0.0 || mid_uv.x > 1.0 ||
                    mid_uv.y < 0.0 || mid_uv.y > 1.0) {
                    break;
                }
                
                ivec2 mid_tex_pos = ivec2(mid_uv * vec2(imageSize(depth_buffer)));
                float mid_scene_depth = imageLoad(depth_buffer, mid_tex_pos).r;
                float mid_ray_depth = linear_depth(length(mid - u_camera_position));
                
                float mid_diff = mid_ray_depth - linear_depth(mid_scene_depth);
                
                if (abs(mid_diff) < 0.01) {
                    hit_uv = mid_uv;
                    hit_depth = mid_ray_depth;
                    return true;
                } else if (mid_diff > 0.0) {
                    end = mid;
                } else {
                    start = mid;
                }
            }
            
            hit_uv = screen_uv;
            hit_depth = ray_depth;
            return true;
        }
        
        // 自适应步长
        step_size *= 1.05;
    }
    
    return false;
}

// 重要性采样（用于粗糙表面）
vec3 importance_sample_ggx(vec2 xi, vec3 n, float roughness) {
    float a = roughness * roughness;
    
    float phi = 2.0 * PI * xi.x;
    float cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a * a - 1.0) * xi.y));
    float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
    
    vec3 h;
    h.x = sin_theta * cos(phi);
    h.y = sin_theta * sin(phi);
    h.z = cos_theta;
    
    vec3 up = abs(n.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, n));
    vec3 bitangent = cross(n, tangent);
    
    return normalize(tangent * h.x + bitangent * h.y + n * h.z);
}

void main_ssr() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(color_buffer);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    float depth = imageLoad(depth_buffer, pos).r;
    if (depth >= 1.0) {
        imageStore(reflection_buffer, pos, vec4(0.0));
        return;
    }
    
    vec4 normal_data = imageLoad(normal_buffer, pos);
    vec3 normal = normalize(normal_data.xyz * 2.0 - 1.0);
    float roughness = imageLoad(roughness_buffer, pos).r;
    
    // 粗糙表面减少反射计算
    if (roughness > 0.85) {
        imageStore(reflection_buffer, pos, vec4(0.0));
        return;
    }
    
    vec2 uv = vec2(pos) / vec2(size);
    vec3 view_pos = screen_to_view(uv, depth);
    vec3 world_pos = view_to_world(view_pos);
    
    vec3 view_dir = normalize(u_camera_position - world_pos);
    vec3 reflect_dir = reflect(-view_dir, normal);
    
    if (dot(reflect_dir, normal) < 0.0) {
        imageStore(reflection_buffer, pos, vec4(0.0));
        return;
    }
    
    vec2 hit_uv;
    float hit_depth;
    
    bool hit = hierarchical_ray_march(world_pos + normal * 0.1, reflect_dir, hit_uv, hit_depth);
    
    if (hit) {
        ivec2 hit_pos = ivec2(hit_uv * vec2(size));
        hit_pos = clamp(hit_pos, ivec2(0), size - 1);
        
        vec4 reflection_color = imageLoad(color_buffer, hit_pos);
        
        // 边缘淡出
        vec2 edge_dist = abs(hit_uv - 0.5) * 2.0;
        float edge_fade = 1.0 - smoothstep(0.8, 1.0, max(edge_dist.x, edge_dist.y));
        
        // 粗糙度影响（使用GGX近似）
        float roughness_factor = 1.0 - roughness;
        float reflection_strength = u_intensity * edge_fade * roughness_factor;
        
        // 距离衰减
        float dist_factor = 1.0 - smoothstep(0.0, u_max_distance, hit_depth);
        reflection_strength *= dist_factor;
        
        vec3 result = reflection_color.rgb * reflection_strength;
        imageStore(reflection_buffer, pos, vec4(result, reflection_strength));
    } else {
        imageStore(reflection_buffer, pos, vec4(0.0));
    }
}

#endif

// ============================================================================
// 水下焦散效果 (Pass 2)
// ============================================================================
#ifdef CAUSTICS_PASS

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) readonly uniform image2D depth_buffer;
layout(rgba16f, binding = 1) readonly uniform image2D position_buffer;
layout(rgba16f, binding = 2) writeonly uniform image2D caustics_buffer;

uniform float u_time;
uniform float u_intensity;
uniform float u_scale;
uniform vec3 u_light_direction;
uniform float u_water_level;
uniform float u_depth_attenuation;

// 伪随机函数
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

// 2D噪声
float noise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    
    float a = hash(i);
    float b = hash(i + vec2(1.0, 0.0));
    float c = hash(i + vec2(0.0, 1.0));
    float d = hash(i + vec2(1.0, 1.0));
    
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// 分形布朗运动
float fbm(vec2 p) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    
    for (int i = 0; i < 5; i++) {
        value += amplitude * noise(p * frequency);
        amplitude *= 0.5;
        frequency *= 2.0;
    }
    
    return value;
}

// Voronoi图案（用于焦散）
float voronoi(vec2 x) {
    vec2 n = floor(x);
    vec2 f = fract(x);
    
    float min_dist = 8.0;
    
    for (int j = -1; j <= 1; j++) {
        for (int i = -1; i <= 1; i++) {
            vec2 g = vec2(float(i), float(j));
            vec2 o = hash(n + g) * 0.5 + 0.5;
            o = 0.5 + 0.5 * sin(u_time * 0.5 + 6.2831 * o);
            vec2 r = g + o - f;
            float d = dot(r, r);
            min_dist = min(min_dist, d);
        }
    }
    
    return sqrt(min_dist);
}

// 焦散图案生成
float caustics_pattern(vec2 uv, float t) {
    vec2 p = uv * u_scale;
    
    // 多层波浪叠加
    float c1 = sin(p.x * 10.0 + t * 2.0) * cos(p.y * 8.0 + t * 1.5);
    float c2 = sin(p.x * 7.0 - t * 1.2) * sin(p.y * 12.0 + t * 2.3);
    float c3 = sin((p.x + p.y) * 5.0 + t);
    
    // Voronoi焦散
    float v = voronoi(p * 3.0 + t * 0.3);
    v = 1.0 - smoothstep(0.0, 0.5, v);
    
    // 添加噪声
    float n = fbm(p * 2.0 + t * 0.5);
    
    float caustics = (c1 + c2 + c3) * 0.33;
    caustics = abs(caustics);
    caustics = pow(caustics, 2.0);
    caustics += v * 0.5;
    caustics += n * 0.2;
    
    return caustics;
}

void main_caustics() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(depth_buffer);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec4 pos_data = imageLoad(position_buffer, pos);
    float height = pos_data.y;
    
    // 只在水下生成焦散
    if (height > u_water_level) {
        imageStore(caustics_buffer, pos, vec4(0.0));
        return;
    }
    
    vec2 uv = vec2(pos) / vec2(size);
    
    // 计算焦散强度
    float caustics = caustics_pattern(uv, u_time);
    
    // 深度衰减（越深越弱）
    float underwater_depth = u_water_level - height;
    float depth_factor = exp(-underwater_depth * u_depth_attenuation);
    
    // 光照方向影响
    vec3 normal = normalize(pos_data.xyz - vec3(uv * 200.0 - 100.0, 0.0));
    float light_factor = max(0.0, dot(normal, -u_light_direction));
    
    float final_caustics = caustics * u_intensity * depth_factor * light_factor;
    
    // 焦散颜色（偏向蓝绿色）
    vec3 caustics_color = vec3(0.8, 0.95, 1.0) * final_caustics;
    
    imageStore(caustics_buffer, pos, vec4(caustics_color, final_caustics));
}

#endif

// ============================================================================
// 动态涟漪系统 (Pass 3)
// ============================================================================
#ifdef RIPPLE_PASS

layout(local_size_x = 16, local_size_y = 16) in;

layout(r32f, binding = 0) readonly uniform image2D ripple_in;
layout(r32f, binding = 1) writeonly uniform image2D ripple_out;

uniform int u_num_ripples;
uniform float u_time;
uniform float u_decay;
uniform float u_ripple_speed;
uniform float u_damping;
uniform float u_world_size;

// 涟漪数据结构
struct Ripple {
    vec2 position;
    float strength;
    float birth_time;
    float frequency;
    float amplitude;
};

layout(std430, binding = 2) readonly buffer RippleBuffer {
    Ripple ripples[MAX_RIPPLES];
};

// 单个涟漪的波形（使用Bessel函数近似）
float ripple_wave(vec2 pos, Ripple ripple) {
    float dist = length(pos - ripple.position);
    float age = u_time - ripple.birth_time;
    float wave_front = age * u_ripple_speed;
    
    // 涟漪波形：衰减的正弦波
    float wave = sin((dist - wave_front) * ripple.frequency) * exp(-dist * 0.3);
    
    // 时间和距离衰减
    float time_attenuation = exp(-age * u_decay);
    float dist_attenuation = exp(-dist * 0.2);
    
    // 只在波前附近产生效果
    float front_factor = smoothstep(wave_front - 2.0, wave_front, dist) * 
                         smoothstep(wave_front + 2.0, wave_front, dist);
    
    return wave * ripple.strength * time_attenuation * dist_attenuation * front_factor;
}

// 波动方程求解（简化版）
float solve_wave_equation(ivec2 pos, ivec2 size) {
    float center = imageLoad(ripple_in, pos).r;
    
    // 采样邻居
    float left = imageLoad(ripple_in, clamp(pos + ivec2(-1, 0), ivec2(0), size - 1)).r;
    float right = imageLoad(ripple_in, clamp(pos + ivec2(1, 0), ivec2(0), size - 1)).r;
    float up = imageLoad(ripple_in, clamp(pos + ivec2(0, 1), ivec2(0), size - 1)).r;
    float down = imageLoad(ripple_in, clamp(pos + ivec2(0, -1), ivec2(0), size - 1)).r;
    
    // 拉普拉斯算子
    float laplacian = (left + right + up + down) * 0.25 - center;
    
    // 波动方程：加速度与位移成正比
    float velocity = laplacian * 0.5;
    
    return center + velocity;
}

void main_ripple() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(ripple_in);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec2 uv = vec2(pos) / vec2(size);
    vec2 world_pos = (uv - 0.5) * u_world_size;
    
    float height = 0.0;
    
    // 叠加所有活跃涟漪
    for (int i = 0; i < u_num_ripples && i < MAX_RIPPLES; i++) {
        Ripple ripple = ripples[i];
        float age = u_time - ripple.birth_time;
        
        if (age > 0.0 && age < 5.0) {  // 5秒生命周期
            height += ripple_wave(world_pos, ripple);
        }
    }
    
    // 读取上一帧的涟漪并应用波动方程
    float prev_height = solve_wave_equation(pos, size);
    
    // 混合新涟漪和波动传播
    height = height * 0.3 + prev_height * u_damping * 0.7;
    
    // 限制范围
    height = clamp(height, -1.0, 1.0);
    
    imageStore(ripple_out, pos, vec4(height, 0.0, 0.0, 1.0));
}

#endif

// ============================================================================
// 水下视野效果 (Pass 4)
// ============================================================================
#ifdef UNDERWATER_PASS

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
layout(r32f, binding = 1) readonly uniform image2D depth_buffer;
layout(rgba16f, binding = 2) writeonly uniform image2D output_buffer;

uniform float u_time;
uniform float u_blur_amount;
uniform float u_distortion_strength;
uniform float u_fog_density;
uniform vec3 u_fog_color;
uniform float u_water_level;
uniform vec3 u_camera_position;
uniform mat4 u_inv_view_matrix;
uniform mat4 u_inv_projection_matrix;

// 简单的伪随机
float rand(vec2 co) {
    return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
}

// 3D噪声
float noise3d(vec3 p) {
    return fract(sin(dot(p, vec3(127.1, 311.7, 74.7))) * 43758.5453);
}

// 2D旋转
vec2 rotate(vec2 v, float a) {
    float s = sin(a);
    float c = cos(a);
    return mat2(c, -s, s, c) * v;
}

// 获取世界空间位置
vec3 get_world_position(vec2 uv, float depth) {
    vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
    vec4 view = u_inv_projection_matrix * ndc;
    view /= view.w;
    vec4 world = u_inv_view_matrix * view;
    return world.xyz;
}

// 高斯模糊采样
vec4 gaussian_blur(ivec2 center, ivec2 size, float radius) {
    vec4 color = vec4(0.0);
    float total_weight = 0.0;
    
    int samples = int(radius * 2.0);
    samples = clamp(samples, 1, 8);
    
    for (int x = -samples; x <= samples; x++) {
        for (int y = -samples; y <= samples; y++) {
            ivec2 offset = ivec2(x, y);
            float dist = length(vec2(offset));
            float weight = exp(-dist * dist / (2.0 * radius * radius));
            
            ivec2 sample_pos = clamp(center + offset, ivec2(0), size - 1);
            color += imageLoad(color_buffer, sample_pos) * weight;
            total_weight += weight;
        }
    }
    
    return color / total_weight;
}

void main_underwater() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(color_buffer);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec2 uv = vec2(pos) / vec2(size);
    float depth = imageLoad(depth_buffer, pos).r;
    
    // 计算世界空间位置
    vec3 world_pos = get_world_position(uv, depth);
    
    // 检查是否在水下
    if (world_pos.y > u_water_level) {
        imageStore(output_buffer, pos, imageLoad(color_buffer, pos));
        return;
    }
    
    // 水下扭曲（使用噪声）
    float distortion_phase = u_time * 2.0;
    vec2 distortion = vec2(
        sin(uv.y * 20.0 + distortion_phase + world_pos.y * 0.5) * 0.01 +
        cos(uv.x * 15.0 + distortion_phase * 0.7) * 0.005,
        cos(uv.x * 15.0 + distortion_phase * 0.7 + world_pos.y * 0.3) * 0.01 +
        sin(uv.y * 20.0 + distortion_phase) * 0.005
    ) * u_distortion_strength;
    
    // 深度影响扭曲强度
    float underwater_depth = u_water_level - world_pos.y;
    distortion *= (1.0 + underwater_depth * 0.1);
    
    vec2 distorted_uv = uv + distortion;
    distorted_uv = clamp(distorted_uv, 0.0, 1.0);
    
    ivec2 distorted_pos = ivec2(distorted_uv * vec2(size));
    distorted_pos = clamp(distorted_pos, ivec2(0), size - 1);
    
    vec4 color = imageLoad(color_buffer, distorted_pos);
    
    // 高斯模糊
    if (u_blur_amount > 0.0) {
        float blur_radius = u_blur_amount * (1.0 + underwater_depth * 0.05);
        vec4 blur_color = gaussian_blur(distorted_pos, size, blur_radius);
        color = mix(color, blur_color, u_blur_amount);
    }
    
    // 水下雾效（指数雾）
    float fog_factor = 1.0 - exp(-underwater_depth * u_fog_density);
    
    // 体积光效果（简化）
    float volumetric = sin(u_time * 0.5 + world_pos.x * 0.1) * 0.1 + 0.9;
    fog_factor *= volumetric;
    
    color.rgb = mix(color.rgb, u_fog_color, fog_factor);
    
    // 水下颜色偏移（蓝绿色调）
    float depth_factor = clamp(underwater_depth / 20.0, 0.0, 1.0);
    color.r *= mix(0.9, 0.6, depth_factor);
    color.g *= mix(0.98, 0.85, depth_factor);
    color.b *= mix(1.0, 1.1, depth_factor);
    
    // 对比度调整
    color.rgb = pow(color.rgb, vec3(1.1));
    
    imageStore(output_buffer, pos, color);
}

#endif

// ============================================================================
// 最终合成 (Pass 5)
// ============================================================================
#ifdef COMPOSE_PASS

layout(local_size_x = 16, local_size_y = 16) in;

layout(rgba16f, binding = 0) readonly uniform image2D base_color;
layout(rgba16f, binding = 1) readonly uniform image2D reflection;
layout(rgba16f, binding = 2) readonly uniform image2D caustics;
layout(r32f, binding = 3) readonly uniform image2D ripple;
layout(rgba16f, binding = 4) readonly uniform image2D underwater;
layout(r32f, binding = 5) readonly uniform image2D depth;
layout(rgba16f, binding = 6) readonly uniform image2D water_normal;
layout(rgba16f, binding = 7) writeonly uniform image2D output_buffer;

uniform float u_water_level;
uniform vec3 u_camera_position;
uniform bool u_use_planar_reflection;
uniform float u_planar_reflection_blend;
uniform float u_fresnel_power;
uniform vec3 u_water_color;
uniform float u_water_transparency;

// Fresnel计算
float fresnel(vec3 view_dir, vec3 normal, float power) {
    float cos_theta = max(0.0, dot(view_dir, normal));
    return pow(1.0 - cos_theta, power);
}

// 色调映射（ACES近似）
vec3 aces_tonemap(vec3 x) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

void main_compose() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(base_color);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    vec4 base = imageLoad(base_color, pos);
    vec4 refl = imageLoad(reflection, pos);
    vec4 caus = imageLoad(caustics, pos);
    float rip = imageLoad(ripple, pos).r;
    vec4 under = imageLoad(underwater, pos);
    float d = imageLoad(depth, pos).r;
    vec4 normal_data = imageLoad(water_normal, pos);
    
    vec3 normal = normalize(normal_data.xyz * 2.0 - 1.0);
    
    // 计算view方向
    vec2 uv = vec2(pos) / vec2(size);
    vec3 view_dir = normalize(vec3(uv - 0.5, -1.0));
    
    // Fresnel效果
    float fresnel_factor = fresnel(view_dir, normal, u_fresnel_power);
    
    // 基础水色
    vec3 water_base = u_water_color * (0.5 + 0.5 * normal.y);
    
    // 添加反射
    vec3 reflection_color = refl.rgb;
    if (u_use_planar_reflection) {
        // 混合平面反射和SSR
        reflection_color = mix(reflection_color, refl.rgb, u_planar_reflection_blend);
    }
    
    // 组合水面颜色
    vec3 water_color = mix(water_base, reflection_color, fresnel_factor * refl.a);
    
    // 添加焦散（水下）
    water_color += caus.rgb * caus.a * 0.3;
    
    // 添加涟漪扰动
    float ripple_highlight = max(0.0, rip) * 0.2;
    float ripple_shadow = min(0.0, rip) * 0.1;
    water_color += vec3(ripple_highlight - ripple_shadow);
    
    // 判断相机是否在水下
    bool camera_underwater = u_camera_position.y < u_water_level;
    
    vec3 final_color;
    if (camera_underwater) {
        // 相机在水下：使用水下效果
        final_color = under.rgb;
        // 添加焦散
        final_color += caus.rgb * caus.a * 0.5;
    } else {
        // 相机在水上：混合基础场景和水面
        float water_mask = smoothstep(u_water_level - 0.1, u_water_level + 0.1, 
                                      u_camera_position.y - d * 100.0);
        final_color = mix(water_color, base.rgb, water_mask * u_water_transparency);
    }
    
    // 色调映射
    final_color = aces_tonemap(final_color);
    
    // Gamma校正
    final_color = pow(final_color, vec3(1.0 / 2.2));
    
    imageStore(output_buffer, pos, vec4(final_color, base.a));
}

#endif

// ============================================================================
// 主入口点
// ============================================================================
void main() {
    #ifdef GERSTNER_PASS
    main_gerstner();
    #endif
    
    #ifdef SSR_PASS
    main_ssr();
    #endif
    
    #ifdef CAUSTICS_PASS
    main_caustics();
    #endif
    
    #ifdef RIPPLE_PASS
    main_ripple();
    #endif
    
    #ifdef UNDERWATER_PASS
    main_underwater();
    #endif
    
    #ifdef COMPOSE_PASS
    main_compose();
    #endif
}
