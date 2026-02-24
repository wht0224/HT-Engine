import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuPathTraceRule(Rule):
    """
    GPU 路径追踪模拟规则 (GPU Path Trace Rule)
    
    使用 ModernGL Compute Shader 实现高性能路径追踪模拟。
    针对 GTX 1650 Max-Q 优化，使用降采样和降噪。
    
    优先级: 50 (最后执行的后处理)
    """
    
    def __init__(self, sample_count=1, max_bounces=2, downsample_factor=4,
                 enable_denoising=True, context=None, manager=None,
                 table_name: str = "postprocess", use_shared_textures: bool = True):
        super().__init__("PostProcess.PathTrace", priority=50)
        self.sample_count = sample_count
        self.max_bounces = max_bounces
        self.downsample_factor = downsample_factor
        self.enable_denoising = enable_denoising
        self.manager = manager
        self.table_name = table_name
        self.use_shared_textures = use_shared_textures
        
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuPathTraceRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        layout(rgba16f, binding = 0) writeonly uniform image2D output_image;
        layout(rgba16f, binding = 1) readonly uniform image2D prev_frame;
        
        struct Ray {
            vec3 origin;
            vec3 direction;
        };
        
        struct HitInfo {
            bool hit;
            float distance;
            vec3 position;
            vec3 normal;
            vec2 uv;
            uint material_id;
        };
        
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform mat4 inv_view_matrix;
        uniform mat4 inv_projection_matrix;
        uniform vec3 camera_position;
        uniform vec3 sun_direction;
        uniform vec3 sun_color;
        uniform float sun_intensity;
        uniform uint frame_index;
        uniform uint sample_count;
        uniform uint max_bounces;
        uniform uint random_seed;
        
        float random(uint seed, uint index) {
            uint n = seed + index * 12345;
            n = (n << 13) ^ n;
            return float((n * (n * n * 15731 + 789221) + 1376312589) & 0x7fffffff) / 2147483647.0;
        }
        
        vec3 random_in_hemisphere(vec3 normal, uint seed, uint index) {
            float u = random(seed, index);
            float v = random(seed, index + 1);
            
            float theta = 2.0 * 3.14159 * u;
            float phi = acos(2.0 * v - 1.0);
            
            vec3 dir = vec3(
                sin(phi) * cos(theta),
                sin(phi) * sin(theta),
                cos(phi)
            );
            
            if (dot(dir, normal) < 0.0) {
                dir = -dir;
            }
            
            return dir;
        }
        
        HitInfo intersect_scene(Ray ray) {
            HitInfo info;
            info.hit = false;
            info.distance = 10000.0;
            
            // 地面
            float t = -ray.origin.y / ray.direction.y;
            if (t > 0.001) {
                vec3 hit_pos = ray.origin + ray.direction * t;
                if (abs(hit_pos.x) < 500.0 && abs(hit_pos.z) < 500.0) {
                    info.hit = true;
                    info.distance = t;
                    info.position = hit_pos;
                    info.normal = vec3(0.0, 1.0, 0.0);
                    info.material_id = 0;
                }
            }
            
            // 球体 - 模拟树木
            vec3 sphere_positions[10] = vec3[](
                vec3(0.0, 10.0, 0.0),
                vec3(-50.0, 5.0, 30.0),
                vec3(60.0, 8.0, -40.0),
                vec3(-30.0, 3.0, 80.0),
                vec3(90.0, 6.0, 50.0),
                vec3(-80.0, 4.0, -60.0),
                vec3(40.0, 7.0, 90.0),
                vec3(-70.0, 5.0, 20.0),
                vec3(20.0, 3.0, -80.0),
                vec3(-40.0, 6.0, -30.0)
            );
            
            float sphere_radii[10] = float[](
                5.0, 3.0, 4.0, 2.0, 3.5,
                2.5, 4.0, 3.0, 2.0, 3.5
            );
            
            for (int i = 0; i < 10; i++) {
                vec3 oc = ray.origin - sphere_positions[i];
                float a = dot(ray.direction, ray.direction);
                float b = 2.0 * dot(oc, ray.direction);
                float c = dot(oc, oc) - sphere_radii[i] * sphere_radii[i];
                float discriminant = b * b - 4.0 * a * c;
                
                if (discriminant > 0.0) {
                    float t1 = (-b - sqrt(discriminant)) / (2.0 * a);
                    if (t1 > 0.001 && t1 < info.distance) {
                        info.hit = true;
                        info.distance = t1;
                        info.position = ray.origin + ray.direction * t1;
                        info.normal = normalize(info.position - sphere_positions[i]);
                        info.material_id = 1 + (i % 2);
                    }
                }
            }
            
            return info;
        }
        
        vec3 pbr_shading(vec3 normal, vec3 view_dir, vec3 light_dir, 
                         vec3 albedo, float metallic, float roughness) {
            vec3 half_vec = normalize(view_dir + light_dir);
            
            float NdotL = max(dot(normal, light_dir), 0.0);
            float NdotV = max(dot(normal, view_dir), 0.0);
            float NdotH = max(dot(normal, half_vec), 0.0);
            
            vec3 F0 = mix(vec3(0.04), albedo, metallic);
            vec3 F = F0 + (1.0 - F0) * pow(1.0 - NdotV, 5.0);
            
            float alpha = roughness * roughness;
            float alpha_sq = alpha * alpha;
            
            float k = (alpha + 1.0) * (alpha + 1.0) / 8.0;
            float G1L = NdotL / (NdotL * (1.0 - k) + k);
            float G1V = NdotV / (NdotV * (1.0 - k) + k);
            float G = G1L * G1V;
            
            float denom = NdotH * NdotH * (alpha_sq - 1.0) + 1.0;
            float D = alpha_sq / (3.14159 * denom * denom);
            
            vec3 specular = F * G * D / (4.0 * NdotL * NdotV + 0.001);
            vec3 diffuse = albedo * (1.0 - metallic) / 3.14159;
            
            return (diffuse + specular) * NdotL;
        }
        
        // 迭代版路径追踪（避免递归，兼容Intel HD530）
        vec3 trace_ray_iterative(Ray ray, uint seed) {
            vec3 throughput = vec3(1.0);
            vec3 radiance = vec3(0.0);
            
            for (uint bounce = 0; bounce < max_bounces; bounce++) {
                HitInfo hit = intersect_scene(ray);
                
                if (!hit.hit) {
                    float sun_dot = max(dot(ray.direction, -sun_direction), 0.0);
                    vec3 sky_color = mix(vec3(0.5, 0.7, 1.0), vec3(0.1, 0.1, 0.3), ray.direction.y * 0.5 + 0.5);
                    float sun_contribution = pow(sun_dot, 256.0) * sun_intensity;
                    radiance += throughput * (sky_color + sun_color * sun_contribution);
                    break;
                }
                
                vec3 albedo;
                float metallic;
                float roughness;
                
                if (hit.material_id == 0) {
                    // 地面
                    albedo = vec3(0.4, 0.35, 0.3);
                    metallic = 0.0;
                    roughness = 0.9;
                } else if (hit.material_id == 1) {
                    // 树木
                    albedo = vec3(0.2, 0.5, 0.15);
                    metallic = 0.0;
                    roughness = 0.7;
                } else {
                    // 岩石
                    albedo = vec3(0.3, 0.3, 0.32);
                    metallic = 0.0;
                    roughness = 0.85;
                }
                
                vec3 view_dir = -ray.direction;
                vec3 light_dir = -sun_direction;
                
                vec3 direct_light = pbr_shading(hit.normal, view_dir, light_dir, 
                                                albedo, metallic, roughness) * sun_color * sun_intensity;
                
                float shadow = 1.0;
                Ray shadow_ray;
                shadow_ray.origin = hit.position + hit.normal * 0.001;
                shadow_ray.direction = -sun_direction;
                HitInfo shadow_hit = intersect_scene(shadow_ray);
                if (shadow_hit.hit) {
                    shadow = 0.2;
                }
                direct_light *= shadow;
                
                radiance += throughput * direct_light;
                
                vec3 bounce_dir = random_in_hemisphere(hit.normal, seed, bounce * 3);
                float NdotB = max(dot(hit.normal, bounce_dir), 0.0);
                throughput *= albedo * NdotB * 0.5;
                
                ray.origin = hit.position + hit.normal * 0.001;
                ray.direction = bounce_dir;
            }
            
            return radiance;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(output_image);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec2 uv = vec2(pos) / vec2(size);
            
            vec4 ndc = vec4(uv * 2.0 - 1.0, 0.0, 1.0);
            vec4 view_pos = inv_projection_matrix * ndc;
            view_pos /= view_pos.w;
            
            vec3 ray_dir = normalize(vec3(inv_view_matrix * vec4(view_pos.xyz, 0.0)));
            
            Ray ray;
            ray.origin = camera_position;
            ray.direction = ray_dir;
            
            uint seed = random_seed + pos.x + pos.y * uint(size.x) + frame_index * 12345;
            
            vec3 color = vec3(0.0);
            
            for (uint s = 0; s < sample_count; s++) {
                vec2 jitter = vec2(random(seed, s * 2), random(seed, s * 2 + 1)) - 0.5;
                jitter /= vec2(size);
                
                vec2 sample_uv = uv + jitter;
                vec4 sample_ndc = vec4(sample_uv * 2.0 - 1.0, 0.0, 1.0);
                vec4 sample_view = inv_projection_matrix * sample_ndc;
                sample_view /= sample_view.w;
                
                vec3 sample_dir = normalize(vec3(inv_view_matrix * vec4(sample_view.xyz, 0.0)));
                
                Ray sample_ray;
                sample_ray.origin = camera_position;
                sample_ray.direction = sample_dir;
                
                color += trace_ray_iterative(sample_ray, seed + s);
            }
            
            color /= float(sample_count);
            
            if (frame_index > 0) {
                vec4 prev_color = imageLoad(prev_frame, pos);
                float blend = 1.0 / float(frame_index + 1);
                color = mix(prev_color.rgb, color, blend);
            }
            
            color = clamp(color, 0.0, 10.0);
            
            imageStore(output_image, pos, vec4(color, 1.0));
        }
        """
        
        self.denoise_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D input_image;
        layout(rgba16f, binding = 1) writeonly uniform image2D output_image;
        
        uniform float filter_strength;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 center = imageLoad(input_image, pos);
            vec4 sum = center;
            float weight_sum = 1.0;
            
            for (int dy = -1; dy <= 1; dy++) {
                for (int dx = -1; dx <= 1; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    ivec2 sample_pos = pos + ivec2(dx, dy);
                    if (sample_pos.x < 0 || sample_pos.x >= size.x ||
                        sample_pos.y < 0 || sample_pos.y >= size.y) continue;
                    
                    vec4 sample_color = imageLoad(input_image, sample_pos);
                    
                    float color_diff = length(sample_color.rgb - center.rgb);
                    float weight = exp(-color_diff * filter_strength);
                    
                    sum += sample_color * weight;
                    weight_sum += weight;
                }
            }
            
            vec3 result = (sum / weight_sum).rgb;
            result = clamp(result, 0.0, 10.0);
            
            imageStore(output_image, pos, vec4(result, center.a));
        }
        """
        
        self.upsample_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D low_res;
        layout(rgba16f, binding = 1) readonly uniform image2D original;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_image;
        
        uniform float blend_factor;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(output_image);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec2 uv = vec2(pos) / vec2(size);
            ivec2 low_size = imageSize(low_res);
            ivec2 low_pos = ivec2(uv * vec2(low_size));
            low_pos = clamp(low_pos, ivec2(0), low_size - 1);
            
            vec4 low_color = imageLoad(low_res, low_pos);
            vec4 original_color = imageLoad(original, pos);
            
            vec3 result = mix(original_color.rgb, low_color.rgb, blend_factor);
            result = clamp(result, 0.0, 1.0);
            
            imageStore(output_image, pos, vec4(result, 1.0));
        }
        """
        
        self.program_pathtrace = None
        self.program_denoise = None
        self.program_upsample = None
        
        self.texture_output = None
        self.texture_prev_frame = None
        self.texture_denoised = None
        self.texture_original = None
        self.texture_final = None
        
        self.frame_index = 0
        self.texture_size = (0, 0)
        self._initialized = False
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            screen_size = facts.get_global("screen_size")
            if screen_size is None:
                screen_size = (1920, 1080)
            
            width, height = int(screen_size[0]), int(screen_size[1])
            
            if not self._initialized or self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program_pathtrace is None:
                    self.program_pathtrace = self.ctx.compute_shader(self.compute_shader_source)
                    self.program_denoise = self.ctx.compute_shader(self.denoise_shader)
                    self.program_upsample = self.ctx.compute_shader(self.upsample_shader)
                self._initialized = True
            
            view_matrix = facts.get_global("view_matrix")
            if view_matrix is None:
                view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = facts.get_global("projection_matrix")
            if proj_matrix is None:
                proj_matrix = np.eye(4, dtype=np.float32)
            
            inv_view = np.linalg.inv(view_matrix)
            inv_proj = np.linalg.inv(proj_matrix)
            
            camera_pos = facts.get_global("camera_position")
            if camera_pos is None:
                camera_pos = np.array([0.0, 5.0, 10.0], dtype=np.float32)
            
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-5)
            
            sun_color = facts.get_global("sun_color")
            if sun_color is None:
                sun_color = np.array([1.0, 0.95, 0.9], dtype=np.float32)
            
            sun_intensity = facts.get_global("sun_intensity")
            if sun_intensity is None:
                sun_intensity = 1.0
            
            self.program_pathtrace['view_matrix'].value = tuple(view_matrix.T.flatten())
            self.program_pathtrace['projection_matrix'].value = tuple(proj_matrix.T.flatten())
            self.program_pathtrace['inv_view_matrix'].value = tuple(inv_view.T.flatten())
            self.program_pathtrace['inv_projection_matrix'].value = tuple(inv_proj.T.flatten())
            self.program_pathtrace['camera_position'].value = tuple(camera_pos)
            self.program_pathtrace['sun_direction'].value = tuple(sun_dir)
            self.program_pathtrace['sun_color'].value = tuple(sun_color)
            self.program_pathtrace['sun_intensity'].value = float(sun_intensity)
            self.program_pathtrace['frame_index'].value = self.frame_index
            self.program_pathtrace['sample_count'].value = self.sample_count
            self.program_pathtrace['max_bounces'].value = self.max_bounces
            self.program_pathtrace['random_seed'].value = np.random.randint(0, 1000000)
            
            self.texture_output.bind_to_image(0, read=False, write=True)
            self.texture_prev_frame.bind_to_image(1, read=True, write=False)
            
            low_w = width // self.downsample_factor
            low_h = height // self.downsample_factor
            
            nx = (low_w + 7) // 8
            ny = (low_h + 7) // 8
            self.program_pathtrace.run(nx, ny, 1)
            
            if self.enable_denoising:
                self.program_denoise['filter_strength'].value = 2.0
                
                self.texture_output.bind_to_image(0, read=True, write=False)
                self.texture_denoised.bind_to_image(1, read=False, write=True)
                
                self.program_denoise.run(nx, ny, 1)
                
                temp = self.texture_output
                self.texture_output = self.texture_denoised
                self.texture_denoised = temp
            
            self.program_upsample['blend_factor'].value = 0.3
            
            self.texture_output.bind_to_image(0, read=True, write=False)
            self.texture_original.bind_to_image(1, read=True, write=False)
            self.texture_final.bind_to_image(2, read=False, write=True)
            
            nx = (width + 15) // 16
            ny = (height + 15) // 16
            self.program_upsample.run(nx, ny, 1)
            
            self.texture_prev_frame, self.texture_output = self.texture_output, self.texture_prev_frame
            
            self.frame_index += 1
            
            if self.manager:
                self.manager.register_texture("pathtrace_result", self.texture_final)
            
        except KeyError:
            pass
    
    def _init_textures(self, width, height):
        if self.texture_output:
            self.texture_output.release()
            self.texture_prev_frame.release()
            self.texture_denoised.release()
            self.texture_original.release()
            self.texture_final.release()
        
        self.texture_size = (width, height)
        
        low_w = width // self.downsample_factor
        low_h = height // self.downsample_factor
        
        self.texture_output = self.ctx.texture((low_w, low_h), 4, dtype='f4')
        self.texture_prev_frame = self.ctx.texture((low_w, low_h), 4, dtype='f4')
        self.texture_denoised = self.ctx.texture((low_w, low_h), 4, dtype='f4')
        self.texture_original = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_final = self.ctx.texture((width, height), 4, dtype='f4')
    
    def set_parameters(self, sample_count=None, max_bounces=None, enable_denoising=None):
        if sample_count is not None:
            self.sample_count = sample_count
        if max_bounces is not None:
            self.max_bounces = max_bounces
        if enable_denoising is not None:
            self.enable_denoising = enable_denoising
    
    def reset_accumulation(self):
        self.frame_index = 0
