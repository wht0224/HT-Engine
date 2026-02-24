"""
GPU高级水面规则 (GPU Advanced Water Rule)

实现业内顶级水面效果，包括：
- Gerstner波浪模拟（多方向波浪叠加）
- 屏幕空间反射（SSR）
- 平面反射（用于静态水面）
- 水下焦散效果（Caustics）
- 动态涟漪系统（角色交互）
- 水下视野扭曲和模糊

使用ModernGL Compute Shader实现高性能GPU并行计算。
针对GTX 1650 Max-Q优化，使用降采样和自适应步进。

优先级: 50 (水面渲染阶段)
"""

import numpy as np
import moderngl
import logging
from typing import List, Tuple, Optional, Dict
from ..Core.GpuRuleBase import GpuRuleBase
from ..Core.FactBase import FactBase


class GpuAdvancedWaterRule(GpuRuleBase):
    """
    GPU高级水面规则
    
    实现电影级水面渲染效果，结合多种先进技术：
    1. Gerstner波浪：多方向波浪叠加，模拟真实海浪
    2. SSR：动态屏幕空间反射
    3. 平面反射：预计算静态反射
    4. 焦散：水下光线折射效果
    5. 涟漪：角色/物体交互产生的动态波纹
    6. 水下效果：视野扭曲和模糊
    """
    
    def __init__(self, 
                 # Gerstner波浪参数
                 num_waves: int = 6,
                 wave_amplitude: float = 0.8,
                 wave_length: float = 15.0,
                 wave_speed: float = 1.2,
                 
                 # SSR参数
                 ssr_max_steps: int = 32,
                 ssr_binary_search_steps: int = 6,
                 ssr_intensity: float = 0.6,
                 ssr_max_distance: float = 150.0,
                 
                 # 焦散参数
                 caustics_intensity: float = 1.0,
                 caustics_scale: float = 2.0,
                 
                 # 涟漪参数
                 ripple_decay: float = 0.95,
                 ripple_speed: float = 3.0,
                 max_ripples: int = 32,
                 
                 # 水下效果参数
                 underwater_blur: float = 0.3,
                 underwater_distortion: float = 0.1,
                 underwater_fog_density: float = 0.05,
                 
                 # 通用参数
                 context=None, 
                 manager=None, 
                 readback: bool = False,
                 use_shared_textures: bool = True):
        """
        初始化GPU高级水面规则
        
        Args:
            num_waves: Gerstner波浪数量
            wave_amplitude: 波浪振幅
            wave_length: 基础波长
            wave_speed: 波浪速度
            ssr_max_steps: SSR最大步进次数
            ssr_binary_search_steps: SSR二分搜索步数
            ssr_intensity: SSR强度
            ssr_max_distance: SSR最大反射距离
            caustics_intensity: 焦散强度
            caustics_scale: 焦散图案缩放
            ripple_decay: 涟漪衰减系数
            ripple_speed: 涟漪传播速度
            max_ripples: 最大同时存在的涟漪数量
            underwater_blur: 水下模糊强度
            underwater_distortion: 水下扭曲强度
            underwater_fog_density: 水下雾密度
        """
        super().__init__(
            name="Water.AdvancedGPU", 
            priority=50,
            manager=manager,
            readback=readback,
            use_shared_textures=use_shared_textures
        )
        
        self.logger = logging.getLogger("Water.AdvancedGPU")
        
        # Gerstner波浪参数
        self.num_waves = num_waves
        self.wave_amplitude = wave_amplitude
        self.wave_length = wave_length
        self.wave_speed = wave_speed
        
        # SSR参数
        self.ssr_max_steps = ssr_max_steps
        self.ssr_binary_search_steps = ssr_binary_search_steps
        self.ssr_intensity = ssr_intensity
        self.ssr_max_distance = ssr_max_distance
        
        # 焦散参数
        self.caustics_intensity = caustics_intensity
        self.caustics_scale = caustics_scale
        
        # 涟漪参数
        self.ripple_decay = ripple_decay
        self.ripple_speed = ripple_speed
        self.max_ripples = max_ripples
        
        # 水下效果参数
        self.underwater_blur = underwater_blur
        self.underwater_distortion = underwater_distortion
        self.underwater_fog_density = underwater_fog_density
        
        # 初始化OpenGL Context
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
                self.logger.info("Created standalone OpenGL context")
            except Exception as e:
                self.logger.error(f"Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        # 波浪参数（GPU上传）
        self.wave_directions = np.zeros((num_waves, 2), dtype=np.float32)
        self.wave_frequencies = np.zeros(num_waves, dtype=np.float32)
        self.wave_phases = np.random.rand(num_waves).astype(np.float32) * 2 * np.pi
        self.wave_amplitudes = np.zeros(num_waves, dtype=np.float32)
        self._init_wave_params()
        
        # 涟漪系统
        self.ripple_positions: List[Tuple[float, float]] = []
        self.ripple_strengths: List[float] = []
        self.ripple_times: List[float] = []
        self.ripple_buffer = np.zeros((max_ripples, 4), dtype=np.float32)  # x, y, strength, time
        
        # 着色器程序
        self.program_gerstner = None
        self.program_ssr = None
        self.program_caustics = None
        self.program_ripples = None
        self.program_underwater = None
        self.program_compose = None
        
        # 纹理
        self.texture_size = (0, 0)
        self.texture_position = None      # 水面位置
        self.texture_normal = None        # 水面法线
        self.texture_depth = None         # 深度缓冲
        self.texture_color = None         # 场景颜色
        self.texture_reflection = None    # 反射结果
        self.texture_caustics = None      # 焦散图案
        self.texture_ripple = None        # 涟漪高度
        self.texture_underwater = None    # 水下效果
        self.texture_output = None        # 最终输出
        
        # 平面反射纹理（用于静态反射）
        self.texture_planar_reflection = None
        self.use_planar_reflection = False
        
        # 初始化着色器
        self._init_shaders()
        
        # 脏标记
        self._texture_dirty = {
            'position': True,
            'normal': True,
            'depth': True,
            'color': True,
            'caustics': True,
            'ripple': True
        }
        
        self._time = 0.0
        self._frame_count = 0
        
        self.logger.info(f"Initialized GpuAdvancedWaterRule with {num_waves} waves")
    
    def _init_wave_params(self):
        """初始化Gerstner波浪参数"""
        for i in range(self.num_waves):
            # 波浪方向（均匀分布在[0, 2π]）
            angle = (i / self.num_waves) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
            self.wave_directions[i] = [np.cos(angle), np.sin(angle)]
            
            # 频率（对数分布）
            self.wave_frequencies[i] = 2.0 * np.pi / (self.wave_length / (i * 0.5 + 1))
            
            # 振幅（递减）
            self.wave_amplitudes[i] = self.wave_amplitude / (i * 0.6 + 1)
    
    def _init_shaders(self):
        """初始化所有Compute Shader"""
        if not self.ctx:
            return
        
        try:
            # Gerstner波浪计算
            self.program_gerstner = self.ctx.compute_shader(self._get_gerstner_shader())
            # SSR计算
            self.program_ssr = self.ctx.compute_shader(self._get_ssr_shader())
            # 焦散计算
            self.program_caustics = self.ctx.compute_shader(self._get_caustics_shader())
            # 涟漪计算
            self.program_ripples = self.ctx.compute_shader(self._get_ripple_shader())
            # 水下效果
            self.program_underwater = self.ctx.compute_shader(self._get_underwater_shader())
            # 最终合成
            self.program_compose = self.ctx.compute_shader(self._get_compose_shader())
            
            self.logger.info("All compute shaders compiled successfully")
        except Exception as e:
            self.logger.error(f"Failed to compile shaders: {e}")
    
    def _get_gerstner_shader(self) -> str:
        """Gerstner波浪Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) writeonly uniform image2D position_buffer;
        layout(rgba16f, binding = 1) writeonly uniform image2D normal_buffer;
        
        uniform float time;
        uniform int num_waves;
        uniform float wave_speed;
        
        layout(std430, binding = 2) readonly buffer WaveParams {
            vec4 wave_data[];  // x: freq, y: amp, z: phase, w: unused
        };
        
        layout(std430, binding = 3) readonly buffer WaveDirs {
            vec2 directions[];
        };
        
        // Gerstner波函数
        vec3 gerstner_wave(vec2 pos, float freq, float amp, float phase, vec2 dir) {
            float k = freq;
            float w = sqrt(9.81 * k) * wave_speed;
            
            float theta = dot(dir, pos) * k - w * time + phase;
            
            float c = cos(theta);
            float s = sin(theta);
            
            // Gerstner波位移
            float x = dir.x * amp * c;
            float z = dir.y * amp * c;
            float y = amp * s;
            
            return vec3(x, y, z);
        }
        
        // 计算Gerstner波法线
        vec3 gerstner_normal(vec2 pos, float freq, float amp, float phase, vec2 dir) {
            float k = freq;
            float w = sqrt(9.81 * k) * wave_speed;
            
            float theta = dot(dir, pos) * k - w * time + phase;
            
            float c = cos(theta);
            float s = sin(theta);
            
            // 法线近似
            float nx = -dir.x * k * amp * s;
            float nz = -dir.y * k * amp * s;
            float ny = k * amp * c;
            
            return vec3(nx, ny, nz);
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(position_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            // 归一化坐标
            vec2 uv = vec2(pos) / vec2(size);
            vec2 world_pos = (uv - 0.5) * 200.0;  // 假设200单位范围
            
            // 叠加所有波浪
            vec3 displacement = vec3(0.0);
            vec3 normal = vec3(0.0, 1.0, 0.0);
            
            for (int i = 0; i < num_waves; i++) {
                vec4 data = wave_data[i];
                vec2 dir = directions[i];
                
                displacement += gerstner_wave(world_pos, data.x, data.y, data.z, dir);
                normal += gerstner_normal(world_pos, data.x, data.y, data.z, dir);
            }
            
            // 归一化法线
            normal = normalize(normal);
            
            // 最终位置
            vec3 final_pos = vec3(world_pos.x, displacement.y, world_pos.y);
            
            imageStore(position_buffer, pos, vec4(final_pos, 1.0));
            imageStore(normal_buffer, pos, vec4(normal * 0.5 + 0.5, 1.0));
        }
        """
    
    def _get_ssr_shader(self) -> str:
        """屏幕空间反射Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
        layout(r32f, binding = 1) readonly uniform image2D depth_buffer;
        layout(rgba16f, binding = 2) readonly uniform image2D normal_buffer;
        layout(r32f, binding = 3) readonly uniform image2D roughness_buffer;
        layout(rgba16f, binding = 4) writeonly uniform image2D reflection_buffer;
        
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform mat4 inv_view_matrix;
        uniform mat4 inv_projection_matrix;
        uniform vec3 camera_position;
        uniform int max_steps;
        uniform int binary_search_steps;
        uniform float intensity;
        uniform float max_distance;
        
        vec3 screen_to_view(vec2 uv, float depth) {
            vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
            vec4 view = inv_projection_matrix * ndc;
            return view.xyz / view.w;
        }
        
        vec3 view_to_world(vec3 view_pos) {
            vec4 world = inv_view_matrix * vec4(view_pos, 1.0);
            return world.xyz;
        }
        
        vec2 world_to_screen(vec3 world_pos) {
            vec4 view_pos = view_matrix * vec4(world_pos, 1.0);
            vec4 ndc = projection_matrix * view_pos;
            vec2 screen = ndc.xy / ndc.w;
            return screen * 0.5 + 0.5;
        }
        
        float linear_depth(float depth_buffer_val) {
            float near = 0.1;
            float far = 1000.0;
            float z = depth_buffer_val * 2.0 - 1.0;
            return (2.0 * near * far) / (far + near - z * (far - near));
        }
        
        // 改进的光线步进，使用自适应步长
        bool ray_march(vec3 origin, vec3 direction, out vec2 hit_uv, out float hit_depth) {
            vec3 current_pos = origin;
            float step_size = max_distance / float(max_steps);
            float thickness = 0.5;
            
            for (int i = 0; i < max_steps; i++) {
                current_pos += direction * step_size;
                
                vec2 screen_uv = world_to_screen(current_pos);
                
                if (screen_uv.x < 0.0 || screen_uv.x > 1.0 ||
                    screen_uv.y < 0.0 || screen_uv.y > 1.0) {
                    return false;
                }
                
                ivec2 tex_pos = ivec2(screen_uv * vec2(imageSize(depth_buffer)));
                float scene_depth = imageLoad(depth_buffer, tex_pos).r;
                float ray_depth = linear_depth(length(current_pos - camera_position));
                float scene_linear = linear_depth(scene_depth);
                
                float depth_diff = ray_depth - scene_linear;
                
                // 检测相交
                if (depth_diff > 0.0 && depth_diff < thickness) {
                    // 二分搜索细化
                    vec3 start = current_pos - direction * step_size;
                    vec3 end = current_pos;
                    
                    for (int j = 0; j < binary_search_steps; j++) {
                        vec3 mid = (start + end) * 0.5;
                        vec2 mid_uv = world_to_screen(mid);
                        
                        if (mid_uv.x < 0.0 || mid_uv.x > 1.0 ||
                            mid_uv.y < 0.0 || mid_uv.y > 1.0) {
                            break;
                        }
                        
                        ivec2 mid_tex_pos = ivec2(mid_uv * vec2(imageSize(depth_buffer)));
                        float mid_scene_depth = imageLoad(depth_buffer, mid_tex_pos).r;
                        float mid_ray_depth = linear_depth(length(mid - camera_position));
                        
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
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pos).r;
            if (depth >= 1.0) {
                imageStore(reflection_buffer, pos, vec4(0.0));
                return;
            }
            
            vec4 normal_roughness = imageLoad(normal_buffer, pos);
            vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
            float roughness = imageLoad(roughness_buffer, pos).r;
            
            // 粗糙表面减少反射计算
            if (roughness > 0.85) {
                imageStore(reflection_buffer, pos, vec4(0.0));
                return;
            }
            
            vec2 uv = vec2(pos) / vec2(size);
            vec3 view_pos = screen_to_view(uv, depth);
            vec3 world_pos = view_to_world(view_pos);
            
            vec3 view_dir = normalize(camera_position - world_pos);
            vec3 reflect_dir = reflect(-view_dir, normal);
            
            if (dot(reflect_dir, normal) < 0.0) {
                imageStore(reflection_buffer, pos, vec4(0.0));
                return;
            }
            
            vec2 hit_uv;
            float hit_depth;
            
            bool hit = ray_march(world_pos + normal * 0.1, reflect_dir, hit_uv, hit_depth);
            
            if (hit) {
                ivec2 hit_pos = ivec2(hit_uv * vec2(size));
                hit_pos = clamp(hit_pos, ivec2(0), size - 1);
                
                vec4 reflection_color = imageLoad(color_buffer, hit_pos);
                
                // 边缘淡出
                vec2 edge_dist = abs(hit_uv - 0.5) * 2.0;
                float edge_fade = 1.0 - smoothstep(0.8, 1.0, max(edge_dist.x, edge_dist.y));
                
                // 粗糙度影响
                float roughness_factor = 1.0 - roughness;
                float reflection_strength = intensity * edge_fade * roughness_factor;
                
                // 距离衰减
                float dist_factor = 1.0 - smoothstep(0.0, max_distance, hit_depth);
                reflection_strength *= dist_factor;
                
                vec3 result = reflection_color.rgb * reflection_strength;
                imageStore(reflection_buffer, pos, vec4(result, reflection_strength));
            } else {
                imageStore(reflection_buffer, pos, vec4(0.0));
            }
        }
        """
    
    def _get_caustics_shader(self) -> str:
        """水下焦散效果Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D depth_buffer;
        layout(rgba16f, binding = 1) readonly uniform image2D normal_buffer;
        layout(rgba16f, binding = 2) writeonly uniform image2D caustics_buffer;
        
        uniform float time;
        uniform float intensity;
        uniform float scale;
        uniform vec3 light_direction;
        uniform float water_level;
        
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
        
        // 焦散图案生成
        float caustics_pattern(vec2 uv, float t) {
            vec2 p = uv * scale;
            
            // 多层波浪叠加
            float c1 = sin(p.x * 10.0 + t * 2.0) * cos(p.y * 8.0 + t * 1.5);
            float c2 = sin(p.x * 7.0 - t * 1.2) * sin(p.y * 12.0 + t * 2.3);
            float c3 = sin((p.x + p.y) * 5.0 + t);
            
            // 添加噪声
            float n = fbm(p * 2.0 + t * 0.5);
            
            float caustics = (c1 + c2 + c3) * 0.33;
            caustics = abs(caustics);  // 焦散是明亮的线条
            caustics = pow(caustics, 2.0);  // 增强对比
            caustics += n * 0.3;
            
            return caustics;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(depth_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pos).r;
            
            // 只在水下生成焦散
            vec4 normal_data = imageLoad(normal_buffer, pos);
            float height = normal_data.w;  // 存储高度在w通道
            
            if (height > water_level) {
                imageStore(caustics_buffer, pos, vec4(0.0));
                return;
            }
            
            vec2 uv = vec2(pos) / vec2(size);
            
            // 计算焦散强度
            float caustics = caustics_pattern(uv, time);
            
            // 深度衰减（越深越弱）
            float depth_factor = 1.0 - smoothstep(0.0, 20.0, water_level - height);
            
            // 光照方向影响
            vec3 normal = normalize(normal_data.xyz * 2.0 - 1.0);
            float light_factor = max(0.0, dot(normal, -light_direction));
            
            float final_caustics = caustics * intensity * depth_factor * light_factor;
            
            imageStore(caustics_buffer, pos, vec4(final_caustics, 0.0, 0.0, 1.0));
        }
        """
    
    def _get_ripple_shader(self) -> str:
        """动态涟漪Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D ripple_in;
        layout(r32f, binding = 1) writeonly uniform image2D ripple_out;
        
        uniform int num_ripples;
        uniform float time;
        uniform float decay;
        uniform float ripple_speed;
        
        layout(std430, binding = 2) readonly buffer RippleData {
            vec4 ripples[];  // x, y, strength, birth_time
        };
        
        // 单个涟漪的波形
        float ripple_wave(vec2 pos, vec2 center, float strength, float age) {
            float dist = length(pos - center);
            float wave_front = age * ripple_speed;
            
            // 涟漪波形：衰减的正弦波
            float wave = sin((dist - wave_front) * 10.0) * exp(-dist * 0.5);
            
            // 时间和距离衰减
            float attenuation = exp(-age * decay) * exp(-dist * 0.3);
            
            return wave * strength * attenuation;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(ripple_in);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec2 uv = vec2(pos) / vec2(size);
            vec2 world_pos = (uv - 0.5) * 200.0;
            
            float height = 0.0;
            
            // 叠加所有活跃涟漪
            for (int i = 0; i < num_ripples; i++) {
                vec4 ripple = ripples[i];
                float age = time - ripple.w;
                
                if (age > 0.0 && age < 5.0) {  // 5秒生命周期
                    height += ripple_wave(world_pos, ripple.xy, ripple.z, age);
                }
            }
            
            // 读取上一帧的涟漪（用于传播）
            float prev_height = imageLoad(ripple_in, pos).r;
            
            // 简单的波动方程近似
            float damping = 0.98;
            height = height * 0.3 + prev_height * damping * 0.7;
            
            imageStore(ripple_out, pos, vec4(height, 0.0, 0.0, 1.0));
        }
        """
    
    def _get_underwater_shader(self) -> str:
        """水下视野效果Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
        layout(r32f, binding = 1) readonly uniform image2D depth_buffer;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_buffer;
        
        uniform float time;
        uniform float blur_amount;
        uniform float distortion_strength;
        uniform float fog_density;
        uniform vec3 fog_color;
        uniform float water_level;
        uniform vec3 camera_position;
        
        // 简单的伪随机
        float rand(vec2 co) {
            return fract(sin(dot(co.xy, vec2(12.9898, 78.233))) * 43758.5453);
        }
        
        // 2D旋转
        vec2 rotate(vec2 v, float a) {
            float s = sin(a);
            float c = cos(a);
            return mat2(c, -s, s, c) * v;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec2 uv = vec2(pos) / vec2(size);
            float depth = imageLoad(depth_buffer, pos).r;
            
            // 计算世界空间高度
            float linear_depth = depth * 100.0;  // 简化
            vec3 world_pos = camera_position + vec3(uv - 0.5, -1.0) * linear_depth;
            
            // 检查是否在水下
            if (world_pos.y > water_level) {
                imageStore(output_buffer, pos, imageLoad(color_buffer, pos));
                return;
            }
            
            // 水下扭曲
            float distortion_phase = time * 2.0;
            vec2 distortion = vec2(
                sin(uv.y * 20.0 + distortion_phase) * 0.01,
                cos(uv.x * 15.0 + distortion_phase * 0.7) * 0.01
            ) * distortion_strength;
            
            vec2 distorted_uv = uv + distortion;
            distorted_uv = clamp(distorted_uv, 0.0, 1.0);
            
            ivec2 distorted_pos = ivec2(distorted_uv * vec2(size));
            distorted_pos = clamp(distorted_pos, ivec2(0), size - 1);
            
            vec4 color = imageLoad(color_buffer, distorted_pos);
            
            // 简单模糊（采样周围像素）
            if (blur_amount > 0.0) {
                vec4 blur_color = vec4(0.0);
                float total_weight = 0.0;
                
                for (int x = -2; x <= 2; x++) {
                    for (int y = -2; y <= 2; y++) {
                        ivec2 sample_pos = pos + ivec2(x, y) * int(blur_amount * 3.0);
                        sample_pos = clamp(sample_pos, ivec2(0), size - 1);
                        
                        float weight = 1.0 / (1.0 + length(vec2(x, y)));
                        blur_color += imageLoad(color_buffer, sample_pos) * weight;
                        total_weight += weight;
                    }
                }
                
                blur_color /= total_weight;
                color = mix(color, blur_color, blur_amount);
            }
            
            // 水下雾效
            float underwater_depth = water_level - world_pos.y;
            float fog_factor = 1.0 - exp(-underwater_depth * fog_density);
            
            color.rgb = mix(color.rgb, fog_color, fog_factor);
            
            // 蓝绿色调（水下颜色偏移）
            color.r *= 0.8;
            color.g *= 0.95;
            color.b *= 1.1;
            
            imageStore(output_buffer, pos, color);
        }
        """
    
    def _get_compose_shader(self) -> str:
        """最终合成Compute Shader"""
        return """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D base_color;
        layout(rgba16f, binding = 1) readonly uniform image2D reflection;
        layout(rgba16f, binding = 2) readonly uniform image2D caustics;
        layout(r32f, binding = 3) readonly uniform image2D ripple;
        layout(rgba16f, binding = 4) readonly uniform image2D underwater;
        layout(r32f, binding = 5) readonly uniform image2D depth;
        layout(rgba16f, binding = 6) writeonly uniform image2D output_buffer;
        
        uniform float water_level;
        uniform vec3 camera_position;
        uniform bool use_planar_reflection;
        uniform float planar_reflection_blend;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(base_color);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 base = imageLoad(base_color, pos);
            vec4 refl = imageLoad(reflection, pos);
            vec4 caus = imageLoad(caustics, pos);
            float rip = imageLoad(ripple, pos).r;
            vec4 under = imageLoad(underwater, pos);
            float d = imageLoad(depth, pos).r;
            
            // 基础颜色
            vec3 color = base.rgb;
            
            // 添加反射
            color += refl.rgb * refl.a;
            
            // 添加焦散（水下）
            color += caus.rgb * caus.a * 0.5;
            
            // 添加涟漪扰动
            color += vec3(rip * 0.1);
            
            // 混合水下效果
            float underwater_factor = smoothstep(water_level - 0.5, water_level + 0.5, 
                                                  camera_position.y - d * 100.0);
            color = mix(under.rgb, color, underwater_factor);
            
            // 色调映射
            color = color / (color + vec3(1.0));
            
            imageStore(output_buffer, pos, vec4(color, base.a));
        }
        """
    
    def evaluate(self, facts: FactBase):
        """
        执行高级水面渲染
        
        Args:
            facts: FactBase实例，包含场景数据
        """
        if not self.ctx:
            return
        
        self._on_evaluate_start()
        
        try:
            # 获取时间和相机数据
            time = facts.get_global("time")
            if time is None:
                time = 0.0
            self._time = time
            
            view_matrix = facts.get_global("view_matrix")
            if view_matrix is None:
                view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = facts.get_global("projection_matrix")
            if proj_matrix is None:
                proj_matrix = np.eye(4, dtype=np.float32)
            
            camera_pos = facts.get_global("camera_position")
            if camera_pos is None:
                camera_pos = np.array([0.0, 10.0, 50.0], dtype=np.float32)
            
            # 获取纹理尺寸
            shared_color = self.get_shared_texture("color_buffer")
            if shared_color:
                width, height = shared_color.size
            else:
                width, height = 1920, 1080
            
            # 初始化纹理
            if self.texture_size != (width, height):
                self._init_textures(width, height)
                self.mark_all_textures_dirty()
            
            # 1. 计算Gerstner波浪
            self._compute_gerstner_waves(time)
            
            # 2. 计算SSR
            self._compute_ssr(view_matrix, proj_matrix, camera_pos)
            
            # 3. 计算焦散
            light_dir = facts.get_global("sun_direction")
            if light_dir is None:
                light_dir = np.array([0.0, -1.0, -0.5], dtype=np.float32)
            self._compute_caustics(time, light_dir)
            
            # 4. 计算涟漪
            self._compute_ripples(time)
            
            # 5. 计算水下效果
            self._compute_underwater(time, camera_pos)
            
            # 6. 合成最终图像
            self._compose_output(camera_pos)
            
            # 注册输出纹理
            self.register_shared_texture("water_output", self.texture_output)
            self.register_shared_texture("water_normal", self.texture_normal)
            self.register_shared_texture("water_caustics", self.texture_caustics)
            
            # 读回CPU（如果需要）
            if self.readback:
                output_data = np.frombuffer(self.texture_output.read(), dtype=np.float32)
                facts.set_global("water_render_result", output_data.reshape((height, width, 4)))
            
        except Exception as e:
            self.logger.error(f"Error in water evaluation: {e}")
    
    def _compute_gerstner_waves(self, time: float):
        """计算Gerstner波浪"""
        if not self.program_gerstner:
            return
        
        # 准备波浪参数缓冲区
        wave_data = np.zeros((self.num_waves, 4), dtype=np.float32)
        for i in range(self.num_waves):
            wave_data[i] = [
                self.wave_frequencies[i],
                self.wave_amplitudes[i],
                self.wave_phases[i],
                0.0
            ]
        
        # 创建/更新SSBO
        if not hasattr(self, '_wave_data_buffer') or self._wave_data_buffer is None:
            self._wave_data_buffer = self.ctx.buffer(wave_data.tobytes())
            self._wave_dir_buffer = self.ctx.buffer(self.wave_directions.tobytes())
        else:
            self._wave_data_buffer.write(wave_data.tobytes())
        
        # 绑定纹理和缓冲区
        self.texture_position.bind_to_image(0, read=False, write=True)
        self.texture_normal.bind_to_image(1, read=False, write=True)
        self._wave_data_buffer.bind_to_storage_buffer(2)
        self._wave_dir_buffer.bind_to_storage_buffer(3)
        
        # 设置uniform
        self.program_gerstner['time'].value = time
        self.program_gerstner['num_waves'].value = self.num_waves
        self.program_gerstner['wave_speed'].value = self.wave_speed
        
        # 执行
        nx = (self.texture_size[0] + 15) // 16
        ny = (self.texture_size[1] + 15) // 16
        self.program_gerstner.run(nx, ny, 1)
    
    def _compute_ssr(self, view_matrix: np.ndarray, proj_matrix: np.ndarray, 
                     camera_pos: np.ndarray):
        """计算屏幕空间反射"""
        if not self.program_ssr:
            return
        
        inv_view = np.linalg.inv(view_matrix)
        inv_proj = np.linalg.inv(proj_matrix)
        
        # 绑定共享纹理
        shared_color = self.get_shared_texture("color_buffer")
        shared_depth = self.get_shared_texture("depth_buffer")
        shared_roughness = self.get_shared_texture("roughness")
        
        if shared_color:
            shared_color.bind_to_image(0, read=True, write=False)
        else:
            self.texture_color.bind_to_image(0, read=True, write=False)
        
        if shared_depth:
            shared_depth.bind_to_image(1, read=True, write=False)
        else:
            self.texture_depth.bind_to_image(1, read=True, write=False)
        
        self.texture_normal.bind_to_image(2, read=True, write=False)
        
        if shared_roughness:
            shared_roughness.bind_to_image(3, read=True, write=False)
        else:
            self.texture_depth.bind_to_image(3, read=True, write=False)
        
        self.texture_reflection.bind_to_image(4, read=False, write=True)
        
        # 设置uniform
        self.program_ssr['view_matrix'].value = tuple(view_matrix.T.flatten())
        self.program_ssr['projection_matrix'].value = tuple(proj_matrix.T.flatten())
        self.program_ssr['inv_view_matrix'].value = tuple(inv_view.T.flatten())
        self.program_ssr['inv_projection_matrix'].value = tuple(inv_proj.T.flatten())
        self.program_ssr['camera_position'].value = tuple(camera_pos)
        self.program_ssr['max_steps'].value = self.ssr_max_steps
        self.program_ssr['binary_search_steps'].value = self.ssr_binary_search_steps
        self.program_ssr['intensity'].value = self.ssr_intensity
        self.program_ssr['max_distance'].value = self.ssr_max_distance
        
        # 执行（使用较小的workgroup以优化GTX 1650）
        nx = (self.texture_size[0] + 7) // 8
        ny = (self.texture_size[1] + 7) // 8
        self.program_ssr.run(nx, ny, 1)
    
    def _compute_caustics(self, time: float, light_dir: np.ndarray):
        """计算水下焦散效果"""
        if not self.program_caustics:
            return
        
        shared_depth = self.get_shared_texture("depth_buffer")
        
        if shared_depth:
            shared_depth.bind_to_image(0, read=True, write=False)
        else:
            self.texture_depth.bind_to_image(0, read=True, write=False)
        
        self.texture_normal.bind_to_image(1, read=True, write=False)
        self.texture_caustics.bind_to_image(2, read=False, write=True)
        
        self.program_caustics['time'].value = time
        self.program_caustics['intensity'].value = self.caustics_intensity
        self.program_caustics['scale'].value = self.caustics_scale
        self.program_caustics['light_direction'].value = tuple(light_dir)
        self.program_caustics['water_level'].value = 0.0
        
        nx = (self.texture_size[0] + 15) // 16
        ny = (self.texture_size[1] + 15) // 16
        self.program_caustics.run(nx, ny, 1)
    
    def _compute_ripples(self, time: float):
        """计算动态涟漪"""
        if not self.program_ripples:
            return
        
        # 更新涟漪缓冲区
        num_active = min(len(self.ripple_positions), self.max_ripples)
        
        for i in range(num_active):
            self.ripple_buffer[i] = [
                self.ripple_positions[i][0],
                self.ripple_positions[i][1],
                self.ripple_strengths[i],
                self.ripple_times[i]
            ]
        
        if not hasattr(self, '_ripple_ssbo') or self._ripple_ssbo is None:
            self._ripple_ssbo = self.ctx.buffer(self.ripple_buffer.tobytes())
        else:
            self._ripple_ssbo.write(self.ripple_buffer.tobytes())
        
        # 双缓冲涟漪纹理
        if not hasattr(self, '_ripple_pingpong'):
            self._ripple_pingpong = 0
            self.texture_ripple_2 = self.ctx.texture(self.texture_size, 1, dtype='f4')
        
        src_tex = self.texture_ripple if self._ripple_pingpong == 0 else self.texture_ripple_2
        dst_tex = self.texture_ripple_2 if self._ripple_pingpong == 0 else self.texture_ripple
        
        src_tex.bind_to_image(0, read=True, write=False)
        dst_tex.bind_to_image(1, read=False, write=True)
        self._ripple_ssbo.bind_to_storage_buffer(2)
        
        self.program_ripples['num_ripples'].value = num_active
        self.program_ripples['time'].value = time
        self.program_ripples['decay'].value = self.ripple_decay
        self.program_ripples['ripple_speed'].value = self.ripple_speed
        
        nx = (self.texture_size[0] + 15) // 16
        ny = (self.texture_size[1] + 15) // 16
        self.program_ripples.run(nx, ny, 1)
        
        self._ripple_pingpong = 1 - self._ripple_pingpong
    
    def _compute_underwater(self, time: float, camera_pos: np.ndarray):
        """计算水下效果"""
        if not self.program_underwater:
            return
        
        shared_color = self.get_shared_texture("color_buffer")
        shared_depth = self.get_shared_texture("depth_buffer")
        
        if shared_color:
            shared_color.bind_to_image(0, read=True, write=False)
        else:
            self.texture_color.bind_to_image(0, read=True, write=False)
        
        if shared_depth:
            shared_depth.bind_to_image(1, read=True, write=False)
        else:
            self.texture_depth.bind_to_image(1, read=True, write=False)
        
        self.texture_underwater.bind_to_image(2, read=False, write=True)
        
        self.program_underwater['time'].value = time
        self.program_underwater['blur_amount'].value = self.underwater_blur
        self.program_underwater['distortion_strength'].value = self.underwater_distortion
        self.program_underwater['fog_density'].value = self.underwater_fog_density
        self.program_underwater['fog_color'].value = (0.0, 0.3, 0.4)
        self.program_underwater['water_level'].value = 0.0
        self.program_underwater['camera_position'].value = tuple(camera_pos)
        
        nx = (self.texture_size[0] + 15) // 16
        ny = (self.texture_size[1] + 15) // 16
        self.program_underwater.run(nx, ny, 1)
    
    def _compose_output(self, camera_pos: np.ndarray):
        """合成最终输出"""
        if not self.program_compose:
            return
        
        shared_color = self.get_shared_texture("color_buffer")
        shared_depth = self.get_shared_texture("depth_buffer")
        
        if shared_color:
            shared_color.bind_to_image(0, read=True, write=False)
        else:
            self.texture_color.bind_to_image(0, read=True, write=False)
        
        self.texture_reflection.bind_to_image(1, read=True, write=False)
        self.texture_caustics.bind_to_image(2, read=True, write=False)
        self.texture_ripple.bind_to_image(3, read=True, write=False)
        self.texture_underwater.bind_to_image(4, read=True, write=False)
        
        if shared_depth:
            shared_depth.bind_to_image(5, read=True, write=False)
        else:
            self.texture_depth.bind_to_image(5, read=True, write=False)
        
        self.texture_output.bind_to_image(6, read=False, write=True)
        
        self.program_compose['water_level'].value = 0.0
        self.program_compose['camera_position'].value = tuple(camera_pos)
        self.program_compose['use_planar_reflection'].value = self.use_planar_reflection
        self.program_compose['planar_reflection_blend'].value = 0.5
        
        nx = (self.texture_size[0] + 15) // 16
        ny = (self.texture_size[1] + 15) // 16
        self.program_compose.run(nx, ny, 1)
    
    def _init_textures(self, width: int, height: int):
        """初始化所有纹理"""
        # 释放旧纹理
        if self.texture_position:
            self.texture_position.release()
            self.texture_normal.release()
            self.texture_depth.release()
            self.texture_color.release()
            self.texture_reflection.release()
            self.texture_caustics.release()
            self.texture_ripple.release()
            self.texture_underwater.release()
            self.texture_output.release()
        
        self.texture_size = (width, height)
        
        # 创建新纹理
        self.texture_position = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_normal = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_depth = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_color = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_reflection = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_caustics = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_ripple = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_underwater = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_output = self.ctx.texture((width, height), 4, dtype='f4')
        
        self.logger.debug(f"Initialized textures: {width}x{height}")
    
    def add_ripple(self, position: Tuple[float, float], strength: float = 1.0):
        """
        添加涟漪
        
        Args:
            position: 涟漪中心位置 (x, z)
            strength: 涟漪强度
        """
        if len(self.ripple_positions) >= self.max_ripples:
            # 移除最旧的涟漪
            self.ripple_positions.pop(0)
            self.ripple_strengths.pop(0)
            self.ripple_times.pop(0)
        
        self.ripple_positions.append(position)
        self.ripple_strengths.append(strength)
        self.ripple_times.append(self._time)
        
        self.logger.debug(f"Added ripple at {position} with strength {strength}")
    
    def set_planar_reflection(self, texture_data: Optional[np.ndarray] = None):
        """
        设置平面反射纹理
        
        Args:
            texture_data: 反射纹理数据，None表示禁用
        """
        if texture_data is None:
            self.use_planar_reflection = False
            if self.texture_planar_reflection:
                self.texture_planar_reflection.release()
                self.texture_planar_reflection = None
        else:
            self.use_planar_reflection = True
            if self.texture_planar_reflection is None:
                h, w = texture_data.shape[:2]
                self.texture_planar_reflection = self.ctx.texture((w, h), 4, dtype='f4')
            self.texture_planar_reflection.write(texture_data.tobytes())
    
    def set_parameters(self, 
                       wave_amplitude: Optional[float] = None,
                       wave_speed: Optional[float] = None,
                       ssr_intensity: Optional[float] = None,
                       caustics_intensity: Optional[float] = None,
                       underwater_blur: Optional[float] = None):
        """
        动态调整参数
        
        Args:
            wave_amplitude: 波浪振幅
            wave_speed: 波浪速度
            ssr_intensity: SSR强度
            caustics_intensity: 焦散强度
            underwater_blur: 水下模糊强度
        """
        if wave_amplitude is not None:
            self.wave_amplitude = wave_amplitude
            self._init_wave_params()
        
        if wave_speed is not None:
            self.wave_speed = wave_speed
        
        if ssr_intensity is not None:
            self.ssr_intensity = ssr_intensity
        
        if caustics_intensity is not None:
            self.caustics_intensity = caustics_intensity
        
        if underwater_blur is not None:
            self.underwater_blur = underwater_blur
    
    def cleanup(self):
        """清理资源"""
        textures = [
            self.texture_position, self.texture_normal, self.texture_depth,
            self.texture_color, self.texture_reflection, self.texture_caustics,
            self.texture_ripple, self.texture_underwater, self.texture_output,
            self.texture_planar_reflection
        ]
        
        for tex in textures:
            if tex:
                tex.release()
        
        buffers = [
            getattr(self, '_wave_data_buffer', None),
            getattr(self, '_wave_dir_buffer', None),
            getattr(self, '_ripple_ssbo', None)
        ]
        
        for buf in buffers:
            if buf:
                buf.release()
        
        self.logger.info("Cleaned up GpuAdvancedWaterRule resources")
