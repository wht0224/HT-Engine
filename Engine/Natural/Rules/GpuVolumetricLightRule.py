import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuVolumetricLightRule(Rule):
    """
    GPU 体积光规则 (GPU Volumetric Light Rule)
    
    使用 ModernGL Compute Shader 实现高性能体积光效果。
    针对 GTX 1650 Max-Q 优化，使用降采样和光线步进。
    
    优先级: 65 (后处理阶段)
    """
    
    def __init__(self, step_count=16, intensity=0.5, scattering=0.3,
                 downsample_factor=4, context=None, manager=None,
                 table_name: str = "postprocess", use_shared_textures: bool = True):
        super().__init__("PostProcess.VolumetricLight", priority=65)
        self.step_count = step_count
        self.intensity = intensity
        self.scattering = scattering
        self.downsample_factor = downsample_factor
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
                print(f"[GpuVolumetricLightRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        layout(r32f, binding = 0) readonly uniform image2D depth_buffer;
        layout(rgba16f, binding = 1) writeonly uniform image2D volumetric_buffer;
        
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform mat4 inv_view_matrix;
        uniform mat4 inv_projection_matrix;
        uniform vec3 camera_position;
        uniform vec3 light_position;
        uniform vec3 light_direction;
        uniform vec3 light_color;
        uniform float light_intensity;
        uniform int step_count;
        uniform float max_distance;
        uniform float scattering;
        uniform float absorption;
        
        vec3 screen_to_world(vec2 uv, float depth) {
            vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
            vec4 view = inv_projection_matrix * ndc;
            vec4 world = inv_view_matrix * vec4(view.xyz / view.w, 1.0);
            return world.xyz;
        }
        
        float henyey_greenstein(float cos_theta, float g) {
            float g2 = g * g;
            return (1.0 - g2) / (4.0 * 3.14159 * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
        }
        
        float shadow_test(vec3 world_pos, vec3 light_pos) {
            vec3 to_light = light_pos - world_pos;
            float dist = length(to_light);
            vec3 light_dir = to_light / dist;
            
            return 1.0;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(depth_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pos).r;
            vec2 uv = vec2(pos) / vec2(size);
            
            vec3 world_pos = screen_to_world(uv, depth);
            
            vec3 ray_start = camera_position;
            vec3 ray_end = world_pos;
            vec3 ray_dir = normalize(ray_end - ray_start);
            float ray_length = length(ray_end - ray_start);
            
            float step_size = min(ray_length, max_distance) / float(step_count);
            
            vec3 accumulated_light = vec3(0.0);
            float transmittance = 1.0;
            
            vec3 to_light = normalize(light_position - camera_position);
            float cos_theta = dot(ray_dir, to_light);
            float phase = henyey_greenstein(cos_theta, scattering);
            
            for (int i = 0; i < step_count; i++) {
                float t = float(i) * step_size;
                vec3 sample_pos = ray_start + ray_dir * t;
                
                vec3 to_sample_light = light_position - sample_pos;
                float light_dist = length(to_sample_light);
                vec3 sample_light_dir = to_sample_light / light_dist;
                
                float shadow = shadow_test(sample_pos, light_position);
                
                float attenuation = 1.0 / (1.0 + 0.1 * light_dist + 0.01 * light_dist * light_dist);
                
                float cos_sample = dot(-light_direction, sample_light_dir);
                float cone_attenuation = smoothstep(0.9, 1.0, cos_sample);
                
                vec3 sample_light = light_color * light_intensity * attenuation * cone_attenuation * shadow;
                
                float density = 0.01;
                vec3 scatter = sample_light * phase * density * step_size;
                
                accumulated_light += scatter * transmittance;
                transmittance *= exp(-absorption * density * step_size);
                
                if (transmittance < 0.01) break;
            }
            
            accumulated_light = clamp(accumulated_light, 0.0, 1.0);
            
            imageStore(volumetric_buffer, pos, vec4(accumulated_light * intensity, 1.0));
        }
        """
        
        self.blend_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
        layout(rgba16f, binding = 1) readonly uniform image2D volumetric_buffer;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_buffer;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 color = imageLoad(color_buffer, pos);
            
            ivec2 vol_size = imageSize(volumetric_buffer);
            vec2 vol_uv = vec2(pos) / vec2(size);
            ivec2 vol_pos = ivec2(vol_uv * vec2(vol_size));
            vol_pos = clamp(vol_pos, ivec2(0), vol_size - 1);
            
            vec4 volumetric = imageLoad(volumetric_buffer, vol_pos);
            
            vec3 result = color.rgb + volumetric.rgb;
            result = clamp(result, 0.0, 1.0);
            
            imageStore(output_buffer, pos, vec4(result, color.a));
        }
        """
        
        self.program_volumetric = None
        self.program_blend = None
        
        self.texture_depth = None
        self.texture_volumetric = None
        self.texture_color = None
        self.texture_output = None
        
        self.texture_size = (0, 0)
        self._initialized = False
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            shared_depth = self.manager.get_texture("depth_buffer") if (self.use_shared_textures and self.manager) else None
            
            if not shared_depth:
                depth_data = facts.get_column(self.table_name, "depth_buffer")
                if depth_data is None:
                    return
                height, width = depth_data.shape[:2] if depth_data.ndim > 1 else (int(np.sqrt(len(depth_data))), int(np.sqrt(len(depth_data))))
            else:
                width, height = shared_depth.size
            
            if not self._initialized or self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program_volumetric is None:
                    self.program_volumetric = self.ctx.compute_shader(self.compute_shader_source)
                    self.program_blend = self.ctx.compute_shader(self.blend_shader)
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
                camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            light_pos = facts.get_global("light_position")
            if light_pos is None:
                light_pos = np.array([0.0, 100.0, 0.0], dtype=np.float32)
            
            light_dir = facts.get_global("light_direction")
            if light_dir is None:
                light_dir = np.array([0.0, -1.0, 0.0], dtype=np.float32)
            
            light_color = facts.get_global("light_color")
            if light_color is None:
                light_color = np.array([1.0, 0.9, 0.7], dtype=np.float32)
            
            light_intensity = facts.get_global("light_intensity")
            if light_intensity is None:
                light_intensity = 1.0
            
            self.program_volumetric['view_matrix'].value = tuple(view_matrix.T.flatten())
            self.program_volumetric['projection_matrix'].value = tuple(proj_matrix.T.flatten())
            self.program_volumetric['inv_view_matrix'].value = tuple(inv_view.T.flatten())
            self.program_volumetric['inv_projection_matrix'].value = tuple(inv_proj.T.flatten())
            self.program_volumetric['camera_position'].value = tuple(camera_pos)
            self.program_volumetric['light_position'].value = tuple(light_pos)
            self.program_volumetric['light_direction'].value = tuple(light_dir)
            self.program_volumetric['light_color'].value = tuple(light_color)
            self.program_volumetric['light_intensity'].value = float(light_intensity)
            self.program_volumetric['step_count'].value = self.step_count
            self.program_volumetric['max_distance'].value = 100.0
            self.program_volumetric['scattering'].value = self.scattering
            self.program_volumetric['absorption'].value = 0.1
            
            if shared_depth:
                shared_depth.bind_to_image(0, read=True, write=False)
            else:
                self.texture_depth.bind_to_image(0, read=True, write=False)
            
            self.texture_volumetric.bind_to_image(1, read=False, write=True)
            
            low_w = width // self.downsample_factor
            low_h = height // self.downsample_factor
            
            nx = (low_w + 7) // 8
            ny = (low_h + 7) // 8
            self.program_volumetric.run(nx, ny, 1)
            
            self.texture_color.bind_to_image(0, read=True, write=False)
            self.texture_volumetric.bind_to_image(1, read=True, write=False)
            self.texture_output.bind_to_image(2, read=False, write=True)
            
            nx = (width + 15) // 16
            ny = (height + 15) // 16
            self.program_blend.run(nx, ny, 1)
            
            if self.manager:
                self.manager.register_texture("volumetric_result", self.texture_output)
            
        except KeyError:
            pass
    
    def _init_textures(self, width, height):
        if self.texture_depth:
            self.texture_depth.release()
            self.texture_volumetric.release()
            self.texture_color.release()
            self.texture_output.release()
        
        self.texture_size = (width, height)
        
        self.texture_depth = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_color = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_output = self.ctx.texture((width, height), 4, dtype='f4')
        
        low_w = width // self.downsample_factor
        low_h = height // self.downsample_factor
        self.texture_volumetric = self.ctx.texture((low_w, low_h), 4, dtype='f4')
    
    def set_parameters(self, step_count=None, intensity=None, scattering=None):
        if step_count is not None:
            self.step_count = step_count
        if intensity is not None:
            self.intensity = intensity
        if scattering is not None:
            self.scattering = scattering
