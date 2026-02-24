import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuMotionBlurRule(Rule):
    """
    GPU 运动模糊规则 (GPU Motion Blur Rule)
    
    使用 ModernGL Compute Shader 实现高性能运动模糊效果。
    针对 GTX 1650 Max-Q 优化，使用速度缓冲和历史帧混合。
    
    优先级: 60 (后处理阶段)
    """
    
    def __init__(self, intensity=0.5, sample_count=8, max_velocity=32,
                 context=None, manager=None, table_name: str = "postprocess",
                 use_shared_textures: bool = True):
        super().__init__("PostProcess.MotionBlur", priority=60)
        self.intensity = intensity
        self.sample_count = sample_count
        self.max_velocity = max_velocity
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
                print(f"[GpuMotionBlurRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.velocity_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D depth_buffer;
        layout(rg16f, binding = 1) writeonly uniform image2D velocity_buffer;
        
        uniform mat4 current_view_proj;
        uniform mat4 previous_view_proj;
        uniform mat4 inv_view_proj;
        uniform vec2 screen_size;
        
        vec3 screen_to_world(vec2 uv, float depth) {
            vec4 ndc = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
            vec4 world = inv_view_proj * ndc;
            return world.xyz / world.w;
        }
        
        vec2 world_to_screen(vec3 world_pos, mat4 vp) {
            vec4 ndc = vp * vec4(world_pos, 1.0);
            vec2 screen = ndc.xy / ndc.w;
            return screen * 0.5 + 0.5;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(depth_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pos).r;
            vec2 uv = vec2(pos) / vec2(size);
            
            vec3 world_pos = screen_to_world(uv, depth);
            
            vec2 current_screen = world_to_screen(world_pos, current_view_proj);
            vec2 previous_screen = world_to_screen(world_pos, previous_view_proj);
            
            vec2 velocity = (current_screen - previous_screen) * screen_size;
            
            float velocity_length = length(velocity);
            if (velocity_length > float(max_velocity)) {
                velocity = normalize(velocity) * float(max_velocity);
            }
            
            imageStore(velocity_buffer, pos, vec4(velocity, 0.0, 1.0));
        }
        """
        
        self.blur_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D color_buffer;
        layout(rg16f, binding = 1) readonly uniform image2D velocity_buffer;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_buffer;
        
        uniform float intensity;
        uniform int sample_count;
        uniform float max_velocity;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(color_buffer);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 center_color = imageLoad(color_buffer, pos);
            vec2 velocity = imageLoad(velocity_buffer, pos).xy;
            
            float velocity_length = length(velocity);
            
            if (velocity_length < 0.5 || intensity < 0.01) {
                imageStore(output_buffer, pos, center_color);
                return;
            }
            
            vec2 velocity_norm = normalize(velocity);
            float step_size = velocity_length / float(sample_count);
            
            vec4 accumulated_color = center_color;
            float total_weight = 1.0;
            
            for (int i = 1; i <= sample_count; i++) {
                float offset = float(i) * step_size * intensity;
                
                vec2 sample_pos = vec2(pos) - velocity_norm * offset;
                ivec2 sample_texel = ivec2(sample_pos);
                
                if (sample_texel.x >= 0 && sample_texel.x < size.x &&
                    sample_texel.y >= 0 && sample_texel.y < size.y) {
                    
                    vec4 sample_color = imageLoad(color_buffer, sample_texel);
                    vec2 sample_velocity = imageLoad(velocity_buffer, sample_texel).xy;
                    
                    float weight = 1.0 - float(i) / float(sample_count);
                    accumulated_color += sample_color * weight;
                    total_weight += weight;
                }
                
                sample_pos = vec2(pos) + velocity_norm * offset;
                sample_texel = ivec2(sample_pos);
                
                if (sample_texel.x >= 0 && sample_texel.x < size.x &&
                    sample_texel.y >= 0 && sample_texel.y < size.y) {
                    
                    vec4 sample_color = imageLoad(color_buffer, sample_texel);
                    
                    float weight = 1.0 - float(i) / float(sample_count);
                    accumulated_color += sample_color * weight;
                    total_weight += weight;
                }
            }
            
            vec3 result = accumulated_color.rgb / total_weight;
            result = clamp(result, 0.0, 1.0);
            
            imageStore(output_buffer, pos, vec4(result, center_color.a));
        }
        """
        
        self.program_velocity = None
        self.program_blur = None
        
        self.texture_depth = None
        self.texture_velocity = None
        self.texture_color = None
        self.texture_output = None
        
        self.previous_view_proj = None
        
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
                if self.program_velocity is None:
                    self.program_velocity = self.ctx.compute_shader(self.velocity_shader)
                    self.program_blur = self.ctx.compute_shader(self.blur_shader)
                self._initialized = True
            
            view_matrix = facts.get_global("view_matrix")
            if view_matrix is None:
                view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = facts.get_global("projection_matrix")
            if proj_matrix is None:
                proj_matrix = np.eye(4, dtype=np.float32)
            
            current_view_proj = proj_matrix @ view_matrix
            
            if self.previous_view_proj is None:
                self.previous_view_proj = current_view_proj.copy()
            
            inv_view_proj = np.linalg.inv(current_view_proj)
            
            self.program_velocity['current_view_proj'].value = tuple(current_view_proj.T.flatten())
            self.program_velocity['previous_view_proj'].value = tuple(self.previous_view_proj.T.flatten())
            self.program_velocity['inv_view_proj'].value = tuple(inv_view_proj.T.flatten())
            self.program_velocity['screen_size'].value = (float(width), float(height))
            
            if shared_depth:
                shared_depth.bind_to_image(0, read=True, write=False)
            else:
                self.texture_depth.bind_to_image(0, read=True, write=False)
            
            self.texture_velocity.bind_to_image(1, read=False, write=True)
            
            nx = (width + 15) // 16
            ny = (height + 15) // 16
            self.program_velocity.run(nx, ny, 1)
            
            self.program_blur['intensity'].value = self.intensity
            self.program_blur['sample_count'].value = self.sample_count
            self.program_blur['max_velocity'].value = float(self.max_velocity)
            
            self.texture_color.bind_to_image(0, read=True, write=False)
            self.texture_velocity.bind_to_image(1, read=True, write=False)
            self.texture_output.bind_to_image(2, read=False, write=True)
            
            self.program_blur.run(nx, ny, 1)
            
            self.previous_view_proj = current_view_proj.copy()
            
            if self.manager:
                self.manager.register_texture("motion_blur_result", self.texture_output)
            
        except KeyError:
            pass
    
    def _init_textures(self, width, height):
        if self.texture_depth:
            self.texture_depth.release()
            self.texture_velocity.release()
            self.texture_color.release()
            self.texture_output.release()
        
        self.texture_size = (width, height)
        
        self.texture_depth = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_velocity = self.ctx.texture((width, height), 2, dtype='f4')
        self.texture_color = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_output = self.ctx.texture((width, height), 4, dtype='f4')
    
    def set_parameters(self, intensity=None, sample_count=None):
        if intensity is not None:
            self.intensity = intensity
        if sample_count is not None:
            self.sample_count = sample_count
    
    def reset_history(self):
        self.previous_view_proj = None
