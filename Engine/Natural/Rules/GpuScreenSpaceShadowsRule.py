import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuScreenSpaceShadowsRule(Rule):
    """
    GPU 屏幕空间阴影规则 (GPU Screen Space Shadows Rule)
    
    实现屏幕空间接触阴影:
    - 基于深度的光线步进
    - 软阴影边缘
    - 自适应步进
    - 性能目标: 0.3-0.5ms
    """
    
    def __init__(self, context=None, manager=None, readback=False, quality="medium"):
        super().__init__("Lighting.ScreenSpaceShadows", priority=72)
        self.manager = manager
        self.readback = readback
        self.quality = quality
        
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuScreenSpaceShadowsRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.quality_params = {
            "low": {"max_steps": 8, "step_size": 4.0},
            "medium": {"max_steps": 16, "step_size": 2.0},
            "high": {"max_steps": 32, "step_size": 1.0}
        }
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        layout(r32f, binding = 0) readonly uniform image2D depth_buffer;
        layout(r32f, binding = 1) readonly uniform image2D normal_buffer;
        layout(r32f, binding = 2) writeonly uniform image2D shadow_output;
        
        uniform mat4 inv_view_proj;
        uniform vec3 light_direction;
        uniform vec3 camera_position;
        uniform float max_distance;
        uniform float thickness;
        uniform int max_steps;
        uniform float step_size;
        uniform vec2 resolution;
        
        vec3 world_pos_from_depth(float depth, vec2 uv) {
            vec4 clip_pos = vec4(uv * 2.0 - 1.0, depth * 2.0 - 1.0, 1.0);
            vec4 world_pos = inv_view_proj * clip_pos;
            return world_pos.xyz / world_pos.w;
        }
        
        vec2 uv_from_world_pos(vec3 world_pos, out float depth) {
            vec4 clip_pos = inv_view_proj * vec4(world_pos, 1.0);
            vec3 ndc = clip_pos.xyz / clip_pos.w;
            depth = ndc.z * 0.5 + 0.5;
            return ndc.xy * 0.5 + 0.5;
        }
        
        float linearize_depth(float depth, float near, float far) {
            return (2.0 * near) / (far + near - depth * (far - near));
        }
        
        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(depth_buffer);
            
            if (pixel.x >= size.x || pixel.y >= size.y) return;
            
            vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
            
            float depth = imageLoad(depth_buffer, pixel).r;
            
            if (depth >= 1.0) {
                imageStore(shadow_output, pixel, vec4(1.0, 0.0, 0.0, 0.0));
                return;
            }
            
            vec3 world_pos = world_pos_from_depth(depth, uv);
            
            vec3 normal;
            normal.x = imageLoad(normal_buffer, pixel).r;
            normal.y = imageLoad(normal_buffer, ivec2(pixel.x + 1, pixel.y)).r;
            normal.z = imageLoad(normal_buffer, ivec2(pixel.x, pixel.y + 1)).r;
            normal = normalize(normal * 2.0 - 1.0);
            
            float NdotL = dot(normal, light_direction);
            if (NdotL <= 0.0) {
                imageStore(shadow_output, pixel, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            vec3 ray_origin = world_pos + normal * 0.01;
            vec3 ray_dir = light_direction;
            
            float shadow = 1.0;
            float current_distance = 0.0;
            float step = step_size;
            
            for (int i = 0; i < max_steps; i++) {
                current_distance += step;
                
                if (current_distance > max_distance) break;
                
                vec3 sample_pos = ray_origin + ray_dir * current_distance;
                
                float sample_depth;
                vec2 sample_uv = uv_from_world_pos(sample_pos, sample_depth);
                
                if (sample_uv.x < 0.0 || sample_uv.x > 1.0 || 
                    sample_uv.y < 0.0 || sample_uv.y > 1.0) {
                    break;
                }
                
                ivec2 sample_pixel = ivec2(sample_uv * resolution);
                float scene_depth = imageLoad(depth_buffer, sample_pixel).r;
                
                float depth_diff = sample_depth - scene_depth;
                
                // 改进的软阴影计算
                if (depth_diff > 0.0 && depth_diff < thickness) {
                    float blocker_distance = current_distance;
                    float penumbra = blocker_distance / max_distance;
                    // 更柔和的阴影过渡
                    float shadow_fade = smoothstep(0.0, thickness, depth_diff);
                    shadow *= (1.0 - (1.0 - shadow_fade) * (1.0 - penumbra * 0.7));
                    
                    if (shadow < 0.05) {
                        shadow = 0.0;
                        break;
                    }
                }
                
                step *= 1.1;
            }
            
            // 应用平滑处理
            shadow = smoothstep(0.0, 1.0, shadow);
            
            imageStore(shadow_output, pixel, vec4(shadow, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_shadow = None

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            width = facts.get_global("viewport_width") or 1920
            height = facts.get_global("viewport_height") or 1080
            
            width = max(1, int(width))
            height = max(1, int(height))
            
            shared_depth = self.manager.get_texture("depth_buffer") if self.manager else None
            shared_normal = self.manager.get_texture("normal_buffer") if self.manager else None
            
            if self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            light_dir = facts.get_global("sun_direction")
            if light_dir is None:
                light_dir = np.array([0.5, -0.8, 0.3], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            cam_pos = facts.get_global("camera_position")
            if cam_pos is None:
                cam_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            view_proj = facts.get_global("view_projection_matrix")
            if view_proj is None:
                view_proj = np.eye(4, dtype=np.float32)
            
            try:
                inv_view_proj = np.linalg.inv(view_proj)
            except:
                inv_view_proj = np.eye(4, dtype=np.float32)
            
            params = self.quality_params[self.quality]
            
            self.program['light_direction'].value = tuple(light_dir)
            self.program['camera_position'].value = tuple(cam_pos)
            self.program['inv_view_proj'].value = tuple(inv_view_proj.T.flatten())
            self.program['max_distance'].value = 50.0
            self.program['thickness'].value = 0.5
            self.program['max_steps'].value = params["max_steps"]
            self.program['step_size'].value = params["step_size"]
            self.program['resolution'].value = (float(width), float(height))
            
            if shared_depth:
                shared_depth.bind_to_image(0, read=True, write=False)
            if shared_normal:
                shared_normal.bind_to_image(1, read=True, write=False)
            
            self.texture_shadow.bind_to_image(2, read=False, write=True)
            
            nx = (width + 7) // 8
            ny = (height + 7) // 8
            self.program.run(nx, ny, 1)
            
            if self.manager:
                self.manager.register_texture("screen_space_shadows", self.texture_shadow)
            
            if self.readback:
                shadow_data = np.frombuffer(self.texture_shadow.read(), dtype=np.float32)
                facts.set_global("screen_space_shadow_buffer", shadow_data)
                
        except Exception as e:
            pass

    def _init_textures(self, width, height):
        if self.texture_shadow:
            self.texture_shadow.release()
            
        self.texture_size = (width, height)
        self.texture_shadow = self.ctx.texture((width, height), 1, dtype='f4')
