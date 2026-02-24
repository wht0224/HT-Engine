#version 430
/**
 * 体积云渲染 Compute Shader
 * Volumetric Clouds Ray Marching Shader
 * 
 * 特性:
 * - 3D Perlin-Worley 混合噪声
 * - 光线步进体积渲染
 * - Beer-Lambert 光衰减
 * - Henyey-Greenstein 相位函数
 * - 丁达尔效应 (光束散射)
 * 
 * 针对 GTX 750 Ti 优化:
 * - 降采样渲染
 * - 自适应步进
 * - 早期终止
 */

layout(local_size_x = 8, local_size_y = 8) in;

// 输出云层缓冲
layout(rgba16f, binding = 0) writeonly uniform image2D cloud_buffer;
// 输入深度缓冲
layout(r32f, binding = 1) readonly uniform image2D depth_buffer;

// 3D 噪声纹理
layout(binding = 2) uniform sampler3D noise_texture;         // 128x128x128 Perlin-Worley
layout(binding = 3) uniform sampler3D detail_noise_texture;  // 64x64x64 Worley
layout(binding = 4) uniform sampler2D weather_texture;       // 512x512 Weather Map

// 变换矩阵
uniform mat4 view_matrix;
uniform mat4 projection_matrix;
uniform mat4 inv_view_matrix;
uniform mat4 inv_projection_matrix;

// 场景参数
uniform vec3 camera_position;
uniform vec3 light_direction;
uniform vec3 light_color;
uniform float light_intensity;

// 云参数
uniform float time;
uniform float cloud_scale;
uniform float cloud_density;
uniform float wind_speed;
uniform int max_steps;
uniform int light_steps;
uniform vec2 resolution;
uniform vec2 jitter;

// ============================================================================
// 常量定义
// ============================================================================
const float PI = 3.14159265359;
const float CLOUD_MIN_HEIGHT = 1500.0;      // 云层底部高度 (米)
const float CLOUD_MAX_HEIGHT = 4000.0;      // 云层顶部高度 (米)
const float CLOUD_THICKNESS = CLOUD_MAX_HEIGHT - CLOUD_MIN_HEIGHT;
const float EARTH_RADIUS = 6371000.0;       // 地球半径
const vec3 EARTH_CENTER = vec3(0.0, -EARTH_RADIUS, 0.0);

// 散射参数
const float ABSORPTION = 0.003;
const float SCATTERING = 0.08;
const float DENSITY_MULTIPLIER = 0.1;

// ============================================================================
// 工具函数
// ============================================================================

/**
 * Henyey-Greenstein 相位函数
 * 模拟光在介质中的散射方向分布
 * 
 * @param cos_theta: 光线与观察方向的夹角余弦
 * @param g: 不对称参数 (-1到1, 0为各向同性)
 * @return: 相位函数值
 */
float henyey_greenstein(float cos_theta, float g) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

/**
 * Beer-Lambert 定律
 * 计算光在吸收介质中的衰减
 * 
 * @param density: 介质密度
 * @param step_size: 步进距离
 * @return: 透射率 (0-1)
 */
float beer_lambert(float density, float step_size) {
    return exp(-density * ABSORPTION * step_size);
}

/**
 * 计算双重 Henyey-Greenstein 相位
 * 结合前向和后向散射，模拟真实的云光照
 */
float dual_henyey_greenstein(float cos_theta) {
    float forward = henyey_greenstein(cos_theta, 0.3);   // 前向散射
    float backward = henyey_greenstein(cos_theta, -0.3); // 后向散射
    return forward * 0.7 + backward * 0.3;
}

// ============================================================================
// 噪声采样
// ============================================================================

/**
 * 采样基础形状噪声 (Perlin-Worley 混合)
 * 用于生成大型云团形状
 */
float sample_shape_noise(vec3 pos) {
    vec4 noise = texture(noise_texture, pos);
    // FBM 合成
    return noise.r * 0.5 + noise.g * 0.25 + noise.b * 0.125 + noise.a * 0.0625;
}

/**
 * 采样细节侵蚀噪声 (Worley)
 * 用于添加云团边缘细节
 */
float sample_detail_noise(vec3 pos) {
    vec4 noise = texture(detail_noise_texture, pos);
    return noise.r * 0.5 + noise.g * 0.25 + noise.b * 0.125 + noise.a * 0.0625;
}

/**
 * 采样天气图
 * 获取云层覆盖率和类型信息
 */
vec4 sample_weather(vec2 pos) {
    return texture(weather_texture, pos);
}

// ============================================================================
// 密度计算
// ============================================================================

/**
 * 计算云密度
 * 核心函数，整合噪声和天气数据
 * 
 * @param pos: 世界空间位置
 * @return: 密度值 (0-1)
 */
float sample_density(vec3 pos) {
    // 应用风场偏移 (模拟云移动)
    vec3 wind_offset = vec3(
        time * wind_speed * 10.0, 
        0.0, 
        time * wind_speed * 5.0
    );
    
    // 缩放采样坐标
    vec3 sample_pos = (pos + wind_offset) * cloud_scale * 0.0001;
    
    // 采样天气图获取覆盖率和云类型
    vec2 weather_uv = sample_pos.xz * 0.1;
    vec4 weather = sample_weather(weather_uv);
    float coverage = weather.r;      // 云层覆盖率
    float cloud_type = weather.b;    // 云类型 (0=层云, 1=积云)
    
    // 高度归一化
    float height = clamp((pos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS, 0.0, 1.0);
    
    // 高度梯度 (云在垂直方向上的分布)
    float height_gradient;
    if (cloud_type < 0.5) {
        // 层云: 均匀分布
        height_gradient = 1.0 - abs(height * 2.0 - 1.0);
    } else {
        // 积云: 底部平坦，顶部蓬松
        height_gradient = smoothstep(0.0, 0.3, height) * (1.0 - height * 0.5);
    }
    height_gradient = pow(height_gradient, 0.5);
    
    // 基础形状
    float shape = sample_shape_noise(sample_pos);
    
    // 应用覆盖率和高度梯度
    float base_density = shape * coverage * height_gradient;
    
    // 细节侵蚀 (高频噪声减去边缘)
    vec3 detail_pos = sample_pos * 4.0;
    float detail = sample_detail_noise(detail_pos);
    base_density -= detail * 0.3;
    
    // 确保非负
    base_density = max(base_density, 0.0);
    
    // 应用密度乘数
    return base_density * cloud_density * DENSITY_MULTIPLIER;
}

// ============================================================================
// 光线追踪
// ============================================================================

/**
 * 光线与球体相交测试
 * 
 * @param ro: 射线起点
 * @param rd: 射线方向
 * @param center: 球心
 * @param radius: 半径
 * @return: (t_min, t_max) 相交距离，负值表示无相交
 */
vec2 ray_sphere_intersect(vec3 ro, vec3 rd, vec3 center, float radius) {
    vec3 oc = ro - center;
    float b = dot(oc, rd);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    
    if (h < 0.0) return vec2(-1.0);
    
    h = sqrt(h);
    return vec2(-b - h, -b + h);
}

/**
 * 计算光线与云层范围的相交
 * 考虑地球曲率
 */
vec2 ray_cloud_intersect(vec3 ro, vec3 rd) {
    // 云层底部球面
    vec2 t_min = ray_sphere_intersect(
        ro, rd, EARTH_CENTER, EARTH_RADIUS + CLOUD_MIN_HEIGHT
    );
    
    // 云层顶部球面
    vec2 t_max = ray_sphere_intersect(
        ro, rd, EARTH_CENTER, EARTH_RADIUS + CLOUD_MAX_HEIGHT
    );
    
    float start = max(0.0, t_min.x);
    float end = t_max.y;
    
    if (start < end && end > 0.0) {
        return vec2(start, end);
    }
    return vec2(-1.0);
}

// ============================================================================
// 光照计算
// ============================================================================

/**
 * 向光源步进计算光照透射率
 * 实现丁达尔效应 (光束散射)
 * 
 * @param pos: 当前位置
 * @param light_dir: 光源方向
 * @return: 透射率 (1.0 = 完全照亮, 0.0 = 完全遮挡)
 */
float light_march(vec3 pos, vec3 light_dir) {
    float step_size = (CLOUD_MAX_HEIGHT - pos.y) / float(light_steps);
    float transmittance = 1.0;
    
    for (int i = 0; i < light_steps; i++) {
        pos += light_dir * step_size;
        
        // 超出云层顶部
        if (pos.y > CLOUD_MAX_HEIGHT) break;
        
        float density = sample_density(pos);
        
        // Beer-Lambert 衰减
        transmittance *= beer_lambert(density, step_size);
        
        // 早期终止优化
        if (transmittance < 0.01) break;
    }
    
    return transmittance;
}

/**
 * 计算环境光 (天空散射)
 */
vec3 ambient_light(float height) {
    // 高度越高，天空颜色越亮
    vec3 sky_color = vec3(0.3, 0.5, 0.8);
    vec3 ground_color = vec3(0.1, 0.1, 0.15);
    return mix(ground_color, sky_color, height) * 0.1;
}

// ============================================================================
// 主渲染函数
// ============================================================================

void main() {
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
    ivec2 size = imageSize(cloud_buffer);
    
    if (pos.x >= size.x || pos.y >= size.y) return;
    
    // UV 坐标 (添加抖动抗锯齿)
    vec2 uv = (vec2(pos) + jitter) / vec2(size);
    
    // 重建世界空间射线
    vec4 ndc = vec4(uv * 2.0 - 1.0, 1.0, 1.0);
    vec4 view_pos = inv_projection_matrix * ndc;
    view_pos /= view_pos.w;
    vec4 world_pos = inv_view_matrix * view_pos;
    
    vec3 ray_origin = camera_position;
    vec3 ray_dir = normalize(world_pos.xyz - ray_origin);
    
    // 云层相交测试
    vec2 cloud_hit = ray_cloud_intersect(ray_origin, ray_dir);
    
    vec3 cloud_color = vec3(0.0);
    float cloud_alpha = 0.0;
    
    if (cloud_hit.x >= 0.0) {
        // 计算步进参数
        float ray_length = cloud_hit.y - cloud_hit.x;
        float step_size = ray_length / float(max_steps);
        
        // 光源方向
        vec3 light_dir = normalize(-light_direction);
        
        // 计算相位函数
        float cos_theta = dot(ray_dir, light_dir);
        float phase = dual_henyey_greenstein(cos_theta);
        
        // 累积变量
        float transmittance = 1.0;
        vec3 accumulated_light = vec3(0.0);
        
        // 蓝噪声抖动 (减少条纹伪影)
        float blue_noise = fract(sin(dot(uv, vec2(12.9898, 78.233))) * 43758.5453);
        float t = cloud_hit.x + step_size * blue_noise;
        
        // ===== 光线步进主循环 =====
        for (int i = 0; i < max_steps; i++) {
            vec3 sample_pos = ray_origin + ray_dir * t;
            
            // 高度边界检查
            if (sample_pos.y < CLOUD_MIN_HEIGHT || sample_pos.y > CLOUD_MAX_HEIGHT) {
                t += step_size;
                continue;
            }
            
            // 采样密度
            float density = sample_density(sample_pos);
            
            if (density > 0.001) {
                // 计算向光源的透射率 (丁达尔效应)
                float light_trans = light_march(sample_pos, light_dir);
                
                // Beer-Lambert 衰减
                float step_trans = beer_lambert(density, step_size);
                
                // 计算散射光
                float height_norm = (sample_pos.y - CLOUD_MIN_HEIGHT) / CLOUD_THICKNESS;
                vec3 ambient = ambient_light(height_norm);
                vec3 sun_light = light_color * light_intensity * light_trans;
                
                // 内散射
                vec3 scatter = (ambient + sun_light * phase) * density * step_size * SCATTERING;
                
                // 累积光照
                accumulated_light += scatter * transmittance;
                transmittance *= step_trans;
                
                // 早期终止 (优化性能)
                if (transmittance < 0.01) break;
            }
            
            t += step_size;
            if (t > cloud_hit.y) break;
        }
        
        // 计算最终不透明度
        cloud_alpha = 1.0 - transmittance;
        cloud_color = accumulated_light;
        
        // 色调映射 (防止过曝)
        cloud_color = cloud_color / (cloud_color + vec3(1.0));
    }
    
    // 输出结果: RGB = 云颜色, A = 不透明度
    imageStore(cloud_buffer, pos, vec4(cloud_color, cloud_alpha));
}
