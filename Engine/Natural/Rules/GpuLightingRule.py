import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuLightingRule(Rule):
    """
    GPU 加速电影级光照规则 (GPU Cinematic Lighting Rule)
    
    使用 ModernGL Compute Shader 实现电影级光照效果:
    - 全局光照（GI）近似（屏幕空间GI + 环境光）
    - 大气散射计算（Mie散射和Rayleigh散射）
    - LUT色彩分级系统（支持3D LUT）
    - 电影级色调映射（ACES Filmic Tone Mapping）
    - 动态时间系统（日出日落，太阳位置变化）
    
    性能目标: 1.0-1.5ms
    """
    
    def __init__(self, shadow_step_size=1.0, max_shadow_steps=100, 
                 ao_strength=1.0, context=None, manager=None, readback=True,
                 table_name: str = "terrain_main", use_shared_textures: bool = True,
                 quality="high", enable_gi=True, enable_atmosphere=True,
                 enable_lut=True, enable_aces=True):
        super().__init__("Lighting.CinematicGPU", priority=80)
        self.shadow_step_size = shadow_step_size
        self.max_shadow_steps = max_shadow_steps
        self.ao_strength = ao_strength
        self.manager = manager
        self.readback = readback
        self.table_name = table_name
        self.use_shared_textures = use_shared_textures
        self.quality = quality
        self.enable_gi = enable_gi
        self.enable_atmosphere = enable_atmosphere
        self.enable_lut = enable_lut
        self.enable_aces = enable_aces
        
        # 初始化 OpenGL Context
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuLightingRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # 质量参数
        self.quality_params = {
            "low": {"gi_samples": 4, "ssr_samples": 8, "lut_size": 16},
            "medium": {"gi_samples": 8, "ssr_samples": 16, "lut_size": 32},
            "high": {"gi_samples": 16, "ssr_samples": 32, "lut_size": 64}
        }
        
        # 动态时间系统参数
        self.time_params = {
            "current_time": 12.0,  # 24小时制
            "time_scale": 1.0,     # 时间流速
            "day_duration": 60.0,  # 游戏内一天对应的实际秒数
            "latitude": 45.0,      # 纬度
        }
        
        # 大气散射参数
        self.atmosphere_params = {
            "planet_radius": 6371000.0,
            "atmosphere_height": 100000.0,
            "rayleigh_height": 8000.0,
            "mie_height": 1200.0,
            "rayleigh_beta": np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float32),
            "mie_beta": np.array([21e-6, 21e-6, 21e-6], dtype=np.float32),
            "mie_g": 0.76,
            "sun_intensity": 20.0
        }
        
        # LUT参数
        self.lut_params = {
            "lut_size": 64,
            "lut_intensity": 1.0,
            "contrast": 1.0,
            "saturation": 1.1,
            "lift": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "gamma": np.array([1.0, 1.0, 1.0], dtype=np.float32),
            "gain": np.array([1.0, 1.0, 1.0], dtype=np.float32)
        }
        
        # Compute Shader: 电影级光照
        self.compute_shader_source = self._build_compute_shader()
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_height = None
        self.texture_shadow = None
        self.texture_ao = None
        self.texture_gi = None
        self.texture_atmosphere = None
        self.texture_lut = None
        self.texture_output = None
        
        # 初始化LUT
        self._init_default_lut()

    def _build_compute_shader(self):
        """构建Compute Shader源代码"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        // 输入纹理
        layout(r32f, binding = 0) readonly uniform image2D height_map;
        layout(rgba32f, binding = 1) readonly uniform image2D albedo_map;
        layout(rgba32f, binding = 2) readonly uniform image2D normal_map;
        
        // 输出纹理
        layout(r32f, binding = 3) writeonly uniform image2D shadow_map;
        layout(r32f, binding = 4) writeonly uniform image2D ao_map;
        layout(rgba32f, binding = 5) writeonly uniform image2D gi_map;
        layout(rgba32f, binding = 6) writeonly uniform image2D atmosphere_map;
        layout(rgba32f, binding = 7) writeonly uniform image2D output_map;
        
        // LUT 3D纹理 - 使用安全的 binding 编号
        layout(rgba32f, binding = 10) readonly uniform image3D lut_texture;
        
        //  uniforms
        uniform vec3 sun_dir;
        uniform float sun_intensity;
        uniform vec3 sun_color;
        uniform float step_size;
        uniform int max_steps;
        uniform float ao_strength;
        uniform float time_of_day;
        uniform float gi_strength;
        uniform int gi_samples;
        uniform int enable_gi;
        uniform int enable_atmosphere;
        uniform int enable_lut;
        uniform int enable_aces;
        uniform float lut_intensity;
        uniform float contrast;
        uniform float saturation;
        uniform vec3 lift;
        uniform vec3 gamma;
        uniform vec3 gain;
        
        // 大气散射参数
        uniform vec3 rayleigh_beta;
        uniform vec3 mie_beta;
        uniform float mie_g;
        uniform float rayleigh_height;
        uniform float mie_height;
        uniform float planet_radius;
        uniform float atmosphere_height;
        
        const float PI = 3.14159265359;
        
        // ==================== ACES Filmic Tone Mapping ====================
        
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
        
        vec3 ACESFitted(vec3 color) {
            color = ACES_INPUT_MAT * color;
            color = RRTAndODTFit(color);
            color = ACES_OUTPUT_MAT * color;
            return clamp(color, 0.0, 1.0);
        }
        
        // 简化版ACES
        vec3 ACESFilmic(vec3 x) {
            float a = 2.51;
            float b = 0.03;
            float c = 2.43;
            float d = 0.59;
            float e = 0.14;
            return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
        }
        
        // ==================== 大气散射 ====================
        
        vec3 rayleigh_phase(float cos_theta) {
            return vec3(3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta));
        }
        
        float mie_phase(float cos_theta, float g) {
            float g2 = g * g;
            float num = (1.0 - g2);
            float denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
            return num / denom;
        }
        
        float rayleigh_density(float h) {
            return exp(-h / rayleigh_height);
        }
        
        float mie_density(float h) {
            return exp(-h / mie_height);
        }
        
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
            
            // 太阳光晕
            float sun_disk = smoothstep(0.9995, 0.9999, cos_theta);
            float sun_glow = pow(max(0.0, cos_theta), 64.0);
            float sun_halo = pow(max(0.0, cos_theta), 8.0) * 0.3;
            
            vec3 sun_color = vec3(1.0, 0.95, 0.8);
            vec3 sun_contribution = sun_color * (sun_disk * 15.0 + sun_glow * 2.0 + sun_halo);
            result += sun_contribution * exp(-optical_depth.r * 5.0);
            
            return result * sun_intensity;
        }
        
        // ==================== 全局光照近似 ====================
        
        vec3 hemisphere_sample(int index, int sample_count) {
            float phi = float(index) * 2.399963229728653;  // 黄金角
            float cos_theta = 1.0 - (float(index) + 0.5) / float(sample_count);
            float sin_theta = sqrt(1.0 - cos_theta * cos_theta);
            return vec3(
                cos(phi) * sin_theta,
                cos_theta,
                sin(phi) * sin_theta
            );
        }
        
        vec3 compute_ssgi(ivec2 pos, ivec2 size, vec3 normal) {
            if (enable_gi == 0) return vec3(0.0);
            
            vec3 gi_accum = vec3(0.0);
            float total_weight = 0.0;
            
            float radius = 16.0;
            
            for (int i = 0; i < gi_samples; i++) {
                vec3 sample_dir = hemisphere_sample(i, gi_samples);
                
                // 转换到切线空间
                vec3 up = abs(normal.y) < 0.999 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
                vec3 tangent = normalize(cross(up, normal));
                vec3 bitangent = cross(normal, tangent);
                
                vec3 world_dir = tangent * sample_dir.x + normal * sample_dir.y + bitangent * sample_dir.z;
                
                // 简单的屏幕空间采样
                vec2 offset = world_dir.xz * radius;
                ivec2 sample_pos = pos + ivec2(offset);
                
                if (sample_pos.x >= 0 && sample_pos.x < size.x &&
                    sample_pos.y >= 0 && sample_pos.y < size.y) {
                    
                    vec4 sample_color = imageLoad(albedo_map, sample_pos);
                    float sample_height = imageLoad(height_map, sample_pos).r;
                    
                    float weight = max(0.0, dot(normal, world_dir));
                    gi_accum += sample_color.rgb * weight;
                    total_weight += weight;
                }
            }
            
            if (total_weight > 0.0) {
                gi_accum /= total_weight;
            }
            
            return gi_accum * gi_strength;
        }
        
        // ==================== LUT色彩分级 ====================
        
        vec3 apply_lift_gamma_gain(vec3 color, vec3 lift, vec3 gamma, vec3 gain) {
            color = pow(color, gamma);
            color = gain * color + lift * (1.0 - color);
            return color;
        }
        
        vec3 apply_contrast(vec3 color, float contrast) {
            return (color - 0.5) * contrast + 0.5;
        }
        
        float luminance(vec3 color) {
            return dot(color, vec3(0.2126, 0.7152, 0.0722));
        }
        
        vec3 apply_saturation(vec3 color, float saturation) {
            float lum = luminance(color);
            return mix(vec3(lum), color, saturation);
        }
        
        vec3 sample_3d_lut(vec3 color, int lut_size) {
            // 将颜色映射到LUT坐标
            vec3 lut_coord = clamp(color, 0.0, 1.0) * float(lut_size - 1);
            
            // 三线性插值
            ivec3 lut_min = ivec3(floor(lut_coord));
            ivec3 lut_max = ivec3(ceil(lut_coord));
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
        
        vec3 apply_color_grading(vec3 color, int lut_size) {
            if (enable_lut == 0) return color;
            
            // Lift/Gamma/Gain
            color = apply_lift_gamma_gain(color, lift, gamma, gain);
            
            // Contrast
            color = apply_contrast(color, contrast);
            
            // Saturation
            color = apply_saturation(color, saturation);
            
            // 3D LUT
            vec3 lut_color = sample_3d_lut(color, lut_size);
            color = mix(color, lut_color, lut_intensity);
            
            return color;
        }
        
        // ==================== 阴影计算 ====================
        
        float compute_shadow(ivec2 pos, ivec2 size, float base_height) {
            float shadow = 1.0;
            
            vec2 sun_xz = normalize(vec2(sun_dir.x, sun_dir.z));
            float sun_height_factor = -sun_dir.y;
            
            if (sun_dir.y >= 0.0) {
                shadow = 0.0;
            } else {
                float current_dist = 0.0;
                float ray_height = base_height;
                float height_gain = sun_height_factor * step_size;
                
                current_dist += step_size;
                ray_height += height_gain;
                
                vec2 current_pos = vec2(pos);
                
                for (int i = 0; i < max_steps; i++) {
                    current_pos += sun_xz * step_size;
                    ray_height += height_gain;
                    
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
            }
            
            return shadow;
        }
        
        // ==================== AO计算 ====================
        
        float compute_ao(ivec2 pos, ivec2 size, float base_height) {
            float h_l = imageLoad(height_map, pos + ivec2(-1, 0)).r;
            float h_r = imageLoad(height_map, pos + ivec2(1, 0)).r;
            float h_u = imageLoad(height_map, pos + ivec2(0, 1)).r;
            float h_d = imageLoad(height_map, pos + ivec2(0, -1)).r;
            
            float laplacian = (h_l + h_r + h_u + h_d) - 4.0 * base_height;
            float curvature = laplacian / (abs(base_height) + 1.0);
            
            float ao = 1.0 / (1.0 + exp(-curvature * 5.0 * ao_strength));
            return clamp(ao, 0.3, 1.0);
        }
        
        // ==================== 主函数 ====================
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_map);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float base_height = imageLoad(height_map, pos).r;
            vec4 albedo = imageLoad(albedo_map, pos);
            vec4 normal_data = imageLoad(normal_map, pos);
            vec3 normal = normalize(normal_data.xyz * 2.0 - 1.0);
            
            // 1. 阴影计算
            float shadow = compute_shadow(pos, size, base_height);
            imageStore(shadow_map, pos, vec4(shadow, 0.0, 0.0, 0.0));
            
            // 2. AO计算
            float ao = compute_ao(pos, size, base_height);
            imageStore(ao_map, pos, vec4(ao, 0.0, 0.0, 0.0));
            
            // 3. 全局光照近似
            vec3 gi = compute_ssgi(pos, size, normal);
            imageStore(gi_map, pos, vec4(gi, 1.0));
            
            // 4. 基础光照计算
            float NdotL = max(0.0, dot(normal, -sun_dir));
            vec3 ambient = vec3(0.1, 0.12, 0.15) * (1.0 - shadow * 0.5);
            vec3 direct_light = sun_color * sun_intensity * NdotL * shadow;
            vec3 indirect_light = gi * ao;
            
            vec3 final_color = albedo.rgb * (ambient + direct_light + indirect_light);
            
            // 5. 大气散射
            vec3 atmosphere = vec3(0.0);
            if (enable_atmosphere != 0) {
                vec3 ray_origin = vec3(0.0, planet_radius + base_height * 100.0, 0.0);
                vec3 ray_dir = normalize(vec3(
                    (float(pos.x) / float(size.x) - 0.5) * 2.0,
                    0.0,
                    (float(pos.y) / float(size.y) - 0.5) * 2.0
                ));
                
                float t_near, t_far;
                float t_max = 100000.0;
                if (intersect_sphere(ray_origin, ray_dir, planet_radius + atmosphere_height, t_near, t_far)) {
                    t_max = t_far;
                }
                
                atmosphere = compute_atmospheric_scattering(ray_origin, ray_dir, -sun_dir, t_max);
                atmosphere = pow(atmosphere, vec3(1.0 / 2.2));  // Gamma校正
            }
            imageStore(atmosphere_map, pos, vec4(atmosphere, 1.0));
            
            // 混合大气散射
            float horizon_factor = 1.0 - abs(normal.y);
            final_color = mix(final_color, atmosphere, horizon_factor * 0.3);
            
            // 6. ACES色调映射
            if (enable_aces != 0) {
                final_color = ACESFilmic(final_color);
            }
            
            // 7. LUT色彩分级
            final_color = apply_color_grading(final_color, 64);
            
            // 最终输出
            imageStore(output_map, pos, vec4(final_color, 1.0));
        }
        """
    
    def _init_default_lut(self):
        """初始化默认3D LUT（中性LUT）"""
        lut_size = self.lut_params["lut_size"]
        lut_data = np.zeros((lut_size, lut_size, lut_size, 4), dtype=np.float32)
        
        for b in range(lut_size):
            for g in range(lut_size):
                for r in range(lut_size):
                    lut_data[r, g, b, 0] = r / (lut_size - 1)
                    lut_data[r, g, b, 1] = g / (lut_size - 1)
                    lut_data[r, g, b, 2] = b / (lut_size - 1)
                    lut_data[r, g, b, 3] = 1.0
        
        self.default_lut_data = lut_data
    
    def update_time(self, delta_time):
        """更新时间系统"""
        self.time_params["current_time"] += delta_time * self.time_params["time_scale"] / self.time_params["day_duration"] * 24.0
        if self.time_params["current_time"] >= 24.0:
            self.time_params["current_time"] -= 24.0
    
    def get_sun_direction(self):
        """根据时间计算太阳方向"""
        time = self.time_params["current_time"]
        latitude = np.radians(self.time_params["latitude"])
        
        # 太阳角度计算
        hour_angle = np.radians((time - 12.0) * 15.0)  # 每小时15度
        declination = np.radians(23.45 * np.sin(np.radians((time / 24.0) * 360.0 - 80.0)))  # 简化黄赤交角
        
        # 计算太阳高度角和方位角
        sin_alt = np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle)
        altitude = np.arcsin(np.clip(sin_alt, -1.0, 1.0))
        
        cos_az = (np.sin(declination) - np.sin(latitude) * sin_alt) / (np.cos(latitude) * np.cos(altitude))
        cos_az = np.clip(cos_az, -1.0, 1.0)
        azimuth = np.arccos(cos_az)
        
        if hour_angle > 0:
            azimuth = 2 * np.pi - azimuth
        
        # 转换为方向向量
        x = np.cos(altitude) * np.sin(azimuth)
        y = -np.sin(altitude)
        z = np.cos(altitude) * np.cos(azimuth)
        
        return np.array([x, y, z], dtype=np.float32)
    
    def get_sun_color(self):
        """根据时间获取太阳颜色"""
        time = self.time_params["current_time"]
        
        # 日出/日落颜色
        sunrise_color = np.array([1.0, 0.6, 0.3], dtype=np.float32)
        noon_color = np.array([1.0, 0.98, 0.95], dtype=np.float32)
        sunset_color = np.array([1.0, 0.4, 0.2], dtype=np.float32)
        night_color = np.array([0.1, 0.1, 0.2], dtype=np.float32)
        
        if 6.0 <= time < 10.0:  # 日出
            t = (time - 6.0) / 4.0
            return sunrise_color * (1 - t) + noon_color * t
        elif 10.0 <= time < 16.0:  # 正午
            return noon_color
        elif 16.0 <= time < 20.0:  # 日落
            t = (time - 16.0) / 4.0
            return noon_color * (1 - t) + sunset_color * t
        else:  # 夜晚
            return night_color
    
    def get_sun_intensity(self):
        """根据时间获取太阳强度"""
        time = self.time_params["current_time"]
        
        if 6.0 <= time < 18.0:  # 白天
            # 正午最强
            return 1.0 - abs(time - 12.0) / 6.0 * 0.3
        else:  # 夜晚
            return 0.05
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return

        table_name = self.table_name
        try:
            # 全面检查所有必需资源是否就绪
            if not hasattr(self, 'texture_lut') or self.texture_lut is None:
                return
            # 更新时间
            delta_time = facts.get_global("delta_time") or 0.016
            self.update_time(delta_time)
            
            # 获取太阳参数
            sun_dir = self.get_sun_direction()
            sun_color = self.get_sun_color()
            sun_intensity = self.get_sun_intensity() * self.atmosphere_params["sun_intensity"]
            
            # 检查共享纹理
            shared_height = self.manager.get_texture("height") if (self.use_shared_textures and self.manager) else None
            shared_albedo = self.manager.get_texture("albedo") if (self.use_shared_textures and self.manager) else None
            shared_normal = self.manager.get_texture("normal") if (self.use_shared_textures and self.manager) else None
            
            if not shared_height:
                flat_height = facts.get_column(table_name, "height")
                grid_len = len(flat_height)
                size = int(np.sqrt(grid_len))
            else:
                size = shared_height.width
                grid_len = size * size

            if size * size != grid_len:
                return
                
            if self.texture_size != (size, size):
                self._init_textures(size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            # 绑定输入纹理
            if shared_height:
                shared_height.bind_to_image(0, read=True, write=False)
            else:
                self.texture_height.write(flat_height.astype(np.float32).tobytes())
                self.texture_height.bind_to_image(0, read=True, write=False)
            
            if shared_albedo:
                shared_albedo.bind_to_image(1, read=True, write=False)
            else:
                # 使用默认白色
                default_albedo = np.ones((size, size, 4), dtype=np.float32) * 0.8
                self.texture_albedo.write(default_albedo.tobytes())
                self.texture_albedo.bind_to_image(1, read=True, write=False)
            
            if shared_normal:
                shared_normal.bind_to_image(2, read=True, write=False)
            else:
                # 使用默认法线（朝上）
                default_normal = np.zeros((size, size, 4), dtype=np.float32)
                default_normal[:, :, 0:3] = 0.5  # 中性法线
                default_normal[:, :, 2] = 1.0
                self.texture_normal.write(default_normal.tobytes())
                self.texture_normal.bind_to_image(2, read=True, write=False)
            
            # 绑定LUT
            self.texture_lut.bind_to_image(10, read=True, write=False)
            
            # 设置uniforms - 先检查 program 是否存在
            if self.program is None:
                return
                
            params = self.quality_params[self.quality]
            
            self.program['sun_dir'].value = tuple(sun_dir)
            self.program['sun_intensity'].value = sun_intensity
            self.program['sun_color'].value = tuple(sun_color)
            self.program['step_size'].value = self.shadow_step_size
            self.program['max_steps'].value = self.max_shadow_steps
            self.program['ao_strength'].value = self.ao_strength
            self.program['time_of_day'].value = self.time_params["current_time"] / 24.0
            self.program['gi_strength'].value = 0.5
            self.program['gi_samples'].value = params["gi_samples"]
            self.program['enable_gi'].value = 1 if self.enable_gi else 0
            self.program['enable_atmosphere'].value = 1 if self.enable_atmosphere else 0
            self.program['enable_lut'].value = 1 if self.enable_lut else 0
            self.program['enable_aces'].value = 1 if self.enable_aces else 0
            self.program['lut_intensity'].value = self.lut_params["lut_intensity"]
            self.program['contrast'].value = self.lut_params["contrast"]
            self.program['saturation'].value = self.lut_params["saturation"]
            self.program['lift'].value = tuple(self.lut_params["lift"])
            self.program['gamma'].value = tuple(self.lut_params["gamma"])
            self.program['gain'].value = tuple(self.lut_params["gain"])
            
            # 大气参数
            self.program['rayleigh_beta'].value = tuple(self.atmosphere_params["rayleigh_beta"])
            self.program['mie_beta'].value = tuple(self.atmosphere_params["mie_beta"])
            self.program['mie_g'].value = self.atmosphere_params["mie_g"]
            self.program['rayleigh_height'].value = self.atmosphere_params["rayleigh_height"]
            self.program['mie_height'].value = self.atmosphere_params["mie_height"]
            self.program['planet_radius'].value = self.atmosphere_params["planet_radius"]
            self.program['atmosphere_height'].value = self.atmosphere_params["atmosphere_height"]
            
            # 绑定输出纹理
            self.texture_shadow.bind_to_image(3, read=False, write=True)
            self.texture_ao.bind_to_image(4, read=False, write=True)
            self.texture_gi.bind_to_image(5, read=False, write=True)
            self.texture_atmosphere.bind_to_image(6, read=False, write=True)
            self.texture_output.bind_to_image(7, read=False, write=True)
            
            # 运行Compute Shader
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            # 注册输出纹理
            if self.manager:
                self.manager.register_texture("shadow_mask", self.texture_shadow)
                self.manager.register_texture("ao_map", self.texture_ao)
                self.manager.register_texture("gi_map", self.texture_gi)
                self.manager.register_texture("atmosphere_map", self.texture_atmosphere)
                self.manager.register_texture("cinematic_output", self.texture_output)
            
            # 回读数据
            if self.readback:
                shadow_data = np.frombuffer(self.texture_shadow.read(), dtype=np.float32)
                ao_data = np.frombuffer(self.texture_ao.read(), dtype=np.float32)
                gi_data = np.frombuffer(self.texture_gi.read(), dtype=np.float32)
                output_data = np.frombuffer(self.texture_output.read(), dtype=np.float32)
                
                facts.set_column(table_name, "shadow_mask", shadow_data)
                facts.set_column(table_name, "ao_map", ao_data)
                facts.set_column(table_name, "gi_map", gi_data)
                facts.set_column(table_name, "cinematic_color", output_data)
            
            # 设置全局光照信息
            facts.set_global("sun_direction", sun_dir)
            facts.set_global("sun_color", sun_color)
            facts.set_global("sun_intensity", sun_intensity)
            facts.set_global("time_of_day", self.time_params["current_time"])
            
        except KeyError:
            pass
        except Exception as e:
            print(f"[GpuLightingRule] Error: {e}")
            
    def _init_textures(self, size):
        """初始化所有纹理"""
        if self.texture_height:
            self.texture_height.release()
            self.texture_shadow.release()
            self.texture_ao.release()
            self.texture_gi.release()
            self.texture_atmosphere.release()
            self.texture_output.release()
            self.texture_albedo.release()
            self.texture_normal.release()
            self.texture_lut.release()
            
        self.texture_size = (size, size)
        
        # 基础纹理
        self.texture_height = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_shadow = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_ao = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_gi = self.ctx.texture((size, size), 4, dtype='f4')
        self.texture_atmosphere = self.ctx.texture((size, size), 4, dtype='f4')
        self.texture_output = self.ctx.texture((size, size), 4, dtype='f4')
        
        # 输入纹理
        self.texture_albedo = self.ctx.texture((size, size), 4, dtype='f4')
        self.texture_normal = self.ctx.texture((size, size), 4, dtype='f4')
        
        # 3D LUT纹理
        lut_size = self.lut_params["lut_size"]
        self.texture_lut = self.ctx.texture3d((lut_size, lut_size, lut_size), 4, dtype='f4')
        self.texture_lut.write(self.default_lut_data.tobytes())
    
    def set_lut_data(self, lut_data):
        """设置自定义LUT数据"""
        if self.texture_lut is not None:
            self.texture_lut.write(lut_data.astype(np.float32).tobytes())
    
    def set_time(self, hour):
        """手动设置时间（0-24小时）"""
        self.time_params["current_time"] = np.clip(hour, 0.0, 24.0)
    
    def set_color_grading_params(self, contrast=None, saturation=None, 
                                  lift=None, gamma=None, gain=None, lut_intensity=None):
        """设置色彩分级参数"""
        if contrast is not None:
            self.lut_params["contrast"] = contrast
        if saturation is not None:
            self.lut_params["saturation"] = saturation
        if lift is not None:
            self.lut_params["lift"] = np.array(lift, dtype=np.float32)
        if gamma is not None:
            self.lut_params["gamma"] = np.array(gamma, dtype=np.float32)
        if gain is not None:
            self.lut_params["gain"] = np.array(gain, dtype=np.float32)
        if lut_intensity is not None:
            self.lut_params["lut_intensity"] = lut_intensity
    
    def __del__(self):
        """清理资源"""
        if hasattr(self, 'texture_height') and self.texture_height:
            self.texture_height.release()
            self.texture_shadow.release()
            self.texture_ao.release()
            self.texture_gi.release()
            self.texture_atmosphere.release()
            self.texture_output.release()
            self.texture_albedo.release()
            self.texture_normal.release()
            self.texture_lut.release()
