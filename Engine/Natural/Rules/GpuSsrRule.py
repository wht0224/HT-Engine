import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuSsrRule(Rule):
    """
    GPU 屏幕空间反射规则 (GPU SSR Rule)
    
    使用 ModernGL Compute Shader 实现高性能屏幕空间反射。
    针对 GTX 1650 Max-Q 优化，使用降采样和自适应步进。
    
    优先级: 75 (后处理阶段，在泛光之后)
    """
    
    def __init__(self, max_steps=16, binary_search_steps=4, intensity=0.5,
                 downsample_factor=2, context=None, manager=None,
                 table_name: str = "postprocess", use_shared_textures: bool = True):
        super().__init__("PostProcess.SSR", priority=75)
        self.max_steps = max_steps
        self.binary_search_steps = binary_search_steps
        self.intensity = intensity
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
                print(f"[GpuSsrRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.compute_shader_source = """
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
        uniform float pixel_stride;
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
        
        bool ray_march(vec3 origin, vec3 direction, out vec2 hit_uv, out float hit_depth) {
            vec3 current_pos = origin;
            float step_size = max_distance / float(max_steps);
            
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
                
                float depth_diff = ray_depth - linear_depth(scene_depth);
                
                if (depth_diff > 0.0 && depth_diff < step_size * 2.0) {
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
                
                step_size *= 1.1;
            }
            
            return false;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pos).r;
            if (depth >= 1.0) {
                imageStore(reflection_buffer, pos, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            vec4 normal_roughness = imageLoad(normal_buffer, pos);
            vec3 normal = normalize(normal_roughness.xyz * 2.0 - 1.0);
            float roughness = imageLoad(roughness_buffer, pos).r;
            
            if (roughness > 0.8) {
                imageStore(reflection_buffer, pos, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            vec2 uv = vec2(pos) / vec2(size);
            vec3 view_pos = screen_to_view(uv, depth);
            vec3 world_pos = view_to_world(view_pos);
            
            vec3 view_dir = normalize(camera_position - world_pos);
            vec3 reflect_dir = reflect(-view_dir, normal);
            
            if (dot(reflect_dir, normal) < 0.0) {
                imageStore(reflection_buffer, pos, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            vec2 hit_uv;
            float hit_depth;
            
            bool hit = ray_march(world_pos + normal * 0.1, reflect_dir, hit_uv, hit_depth);
            
            if (hit) {
                ivec2 hit_pos = ivec2(hit_uv * vec2(size));
                hit_pos = clamp(hit_pos, ivec2(0), size - 1);
                
                vec4 reflection_color = imageLoad(color_buffer, hit_pos);
                
                float fade = 1.0;
                float edge_fade = 1.0 - smoothstep(0.4, 0.6, abs(hit_uv.x - 0.5) + abs(hit_uv.y - 0.5));
                fade *= edge_fade;
                
                float roughness_factor = 1.0 - roughness;
                float reflection_strength = intensity * fade * roughness_factor;
                
                vec3 result = reflection_color.rgb * reflection_strength;
                imageStore(reflection_buffer, pos, vec4(result, reflection_strength));
            } else {
                imageStore(reflection_buffer, pos, vec4(0.0, 0.0, 0.0, 0.0));
            }
        }
        """
        
        self.blend_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
        layout(rgba16f, binding = 1) readonly uniform image2D reflection_buffer;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_buffer;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 color = imageLoad(color_buffer, pos);
            vec4 reflection = imageLoad(reflection_buffer, pos);
            
            vec3 result = color.rgb + reflection.rgb;
            result = clamp(result, 0.0, 1.0);
            
            imageStore(output_buffer, pos, vec4(result, color.a));
        }
        """
        
        self.program_ssr = None
        self.program_blend = None
        
        self.texture_color = None
        self.texture_depth = None
        self.texture_normal = None
        self.texture_roughness = None
        self.texture_reflection = None
        self.texture_output = None
        
        self.texture_size = (0, 0)
        self._initialized = False
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            shared_color = self.manager.get_texture("color_buffer") if (self.use_shared_textures and self.manager) else None
            shared_depth = self.manager.get_texture("depth_buffer") if (self.use_shared_textures and self.manager) else None
            
            if not shared_color:
                color_data = facts.get_column(self.table_name, "color_buffer")
                if color_data is None:
                    return
                height, width = color_data.shape[:2] if color_data.ndim == 3 else (int(np.sqrt(len(color_data)//4)), int(np.sqrt(len(color_data)//4)))
            else:
                width, height = shared_color.size
            
            if not self._initialized or self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program_ssr is None:
                    self.program_ssr = self.ctx.compute_shader(self.compute_shader_source)
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
            
            self.program_ssr['view_matrix'].value = tuple(view_matrix.T.flatten())
            self.program_ssr['projection_matrix'].value = tuple(proj_matrix.T.flatten())
            self.program_ssr['inv_view_matrix'].value = tuple(inv_view.T.flatten())
            self.program_ssr['inv_projection_matrix'].value = tuple(inv_proj.T.flatten())
            self.program_ssr['camera_position'].value = tuple(camera_pos)
            self.program_ssr['max_steps'].value = self.max_steps
            self.program_ssr['binary_search_steps'].value = self.binary_search_steps
            self.program_ssr['intensity'].value = self.intensity
            self.program_ssr['pixel_stride'].value = 2.0
            self.program_ssr['max_distance'].value = 100.0
            
            if shared_color:
                shared_color.bind_to_image(0, read=True, write=False)
            else:
                self.texture_color.bind_to_image(0, read=True, write=False)
            
            if shared_depth:
                shared_depth.bind_to_image(1, read=True, write=False)
            else:
                self.texture_depth.bind_to_image(1, read=True, write=False)
            
            self.texture_normal.bind_to_image(2, read=True, write=False)
            self.texture_roughness.bind_to_image(3, read=True, write=False)
            self.texture_reflection.bind_to_image(4, read=False, write=True)
            
            low_w = width // self.downsample_factor
            low_h = height // self.downsample_factor
            
            nx = (low_w + 7) // 8
            ny = (low_h + 7) // 8
            self.program_ssr.run(nx, ny, 1)
            
            self.texture_color.bind_to_image(0, read=True, write=False)
            self.texture_reflection.bind_to_image(1, read=True, write=False)
            self.texture_output.bind_to_image(2, read=False, write=True)
            
            nx = (width + 15) // 16
            ny = (height + 15) // 16
            self.program_blend.run(nx, ny, 1)
            
            if self.manager:
                self.manager.register_texture("ssr_result", self.texture_output)
            
        except KeyError:
            pass
    
    def _init_textures(self, width, height):
        if self.texture_color:
            self.texture_color.release()
            self.texture_depth.release()
            self.texture_normal.release()
            self.texture_roughness.release()
            self.texture_reflection.release()
            self.texture_output.release()
        
        self.texture_size = (width, height)
        
        self.texture_color = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_depth = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_normal = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_roughness = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_output = self.ctx.texture((width, height), 4, dtype='f4')
        
        low_w = width // self.downsample_factor
        low_h = height // self.downsample_factor
        self.texture_reflection = self.ctx.texture((low_w, low_h), 4, dtype='f4')
    
    def set_parameters(self, max_steps=None, intensity=None):
        if max_steps is not None:
            self.max_steps = max_steps
        if intensity is not None:
            self.intensity = intensity
