import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuOcclusionRule(Rule):
    """
    GPU 软阴影规则 (GPU Occlusion Rule)
    
    使用 ModernGL Compute Shader 实现并行软阴影计算。
    基于遮挡距离的软阴影因果推导。
    """
    
    def __init__(self, hard_shadow_threshold=2.0, soft_shadow_range=15.0, 
                 penumbra_scale=1.0, context=None, manager=None, readback=True,
                 table_name: str = "terrain_main", use_shared_textures: bool = True):
        super().__init__("Lighting.OcclusionGPU", priority=100)
        self.hard_shadow_threshold = hard_shadow_threshold
        self.soft_shadow_range = soft_shadow_range
        self.penumbra_scale = penumbra_scale
        self.manager = manager
        self.readback = readback
        self.table_name = table_name
        self.use_shared_textures = use_shared_textures
        
        # 初始化 OpenGL Context
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuOcclusionRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        # Compute Shader: 软阴影计算
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D height_map;
        layout(r32f, binding = 1) writeonly uniform image2D shadow_soft;
        layout(r32f, binding = 2) writeonly uniform image2D shadow_softness;
        
        uniform vec3 sun_dir;
        uniform float hard_threshold;
        uniform float soft_range;
        uniform float penumbra_scale;
        
        // 双线性插值采样高度
        float sampleHeightBilinear(ivec2 size, vec2 pos) {
            int x0 = int(floor(pos.x));
            int y0 = int(floor(pos.y));
            int x1 = min(x0 + 1, size.x - 1);
            int y1 = min(y0 + 1, size.y - 1);
            
            float wx = pos.x - float(x0);
            float wy = pos.y - float(y0);
            
            float h00 = imageLoad(height_map, ivec2(x0, y0)).r;
            float h10 = imageLoad(height_map, ivec2(x1, y0)).r;
            float h01 = imageLoad(height_map, ivec2(x0, y1)).r;
            float h11 = imageLoad(height_map, ivec2(x1, y1)).r;
            
            return (1.0 - wx) * (1.0 - wy) * h00 +
                   wx * (1.0 - wy) * h10 +
                   (1.0 - wx) * wy * h01 +
                   wx * wy * h11;
        }
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_map);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float base_height = imageLoad(height_map, pos).r;
            
            // 太阳方向
            vec2 sun_xz = normalize(vec2(sun_dir.x, sun_dir.z));
            float sun_height_factor = -sun_dir.y;
            
            // 初始化
            float shadow_value = 1.0;
            float softness = 0.0;
            
            // 如果太阳在下方，完全阴影
            if (sun_dir.y >= 0.0) {
                shadow_value = 0.0;
                softness = 0.0;
            } else {
                // Ray Marching 搜索遮挡物
                float step_size = 1.0;
                int max_steps = int(soft_range * 2.0 / step_size);
                
                bool found_occluder = false;
                float nearest_distance = 0.0;
                float max_height_diff = 0.0;
                
                for (int step = 1; step <= max_steps; step++) {
                    float distance = float(step) * step_size;
                    
                    // 计算采样位置
                    vec2 sample_pos = vec2(pos) + sun_xz * distance;
                    
                    // 边界检查
                    if (sample_pos.x < 0.0 || sample_pos.x >= float(size.x - 1) ||
                        sample_pos.y < 0.0 || sample_pos.y >= float(size.y - 1)) {
                        break;
                    }
                    
                    // 计算射线高度
                    float ray_height = base_height + sun_height_factor * distance;
                    
                    // 双线性插值采样高度
                    float sample_height = sampleHeightBilinear(size, sample_pos);
                    
                    // 检查遮挡
                    if (sample_height > ray_height) {
                        found_occluder = true;
                        nearest_distance = distance;
                        max_height_diff = sample_height - ray_height;
                        break;
                    }
                }
                
                // 计算软阴影
                if (found_occluder) {
                    // 基础阴影值
                    float base_shadow = 1.0 - min(1.0, max_height_diff * 0.1);
                    
                    // 根据距离计算软度
                    if (nearest_distance < hard_threshold) {
                        // 硬阴影
                        shadow_value = base_shadow;
                        softness = 0.0;
                    } else if (nearest_distance < soft_range) {
                        // 软阴影
                        softness = (nearest_distance - hard_threshold) / 
                                   (soft_range - hard_threshold);
                        softness = min(1.0, softness * penumbra_scale);
                        shadow_value = base_shadow * (0.3 + 0.7 * (1.0 - softness));
                    } else {
                        // 极软阴影
                        softness = 1.0;
                        shadow_value = base_shadow * 0.3;
                    }
                }
            }
            
            // 写入结果
            imageStore(shadow_soft, pos, vec4(shadow_value, 0.0, 0.0, 0.0));
            imageStore(shadow_softness, pos, vec4(softness, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_height = None
        self.texture_shadow_soft = None
        self.texture_shadow_softness = None
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        table_name = self.table_name
        try:
            shared_height = self.manager.get_texture("height") if (self.use_shared_textures and self.manager) else None
            
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
            
            # 绑定输入
            if shared_height:
                shared_height.bind_to_image(0, read=True, write=False)
            else:
                self.texture_height.write(flat_height.astype(np.float32).tobytes())
                self.texture_height.bind_to_image(0, read=True, write=False)
            
            # Uniforms
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-5)
            
            self.program['sun_dir'].value = tuple(sun_dir)
            self.program['hard_threshold'].value = self.hard_shadow_threshold
            self.program['soft_range'].value = self.soft_shadow_range
            self.program['penumbra_scale'].value = self.penumbra_scale
            
            # 绑定输出
            self.texture_shadow_soft.bind_to_image(1, read=False, write=True)
            self.texture_shadow_softness.bind_to_image(2, read=False, write=True)
            
            # 运行
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            # 注册输出
            if self.manager:
                self.manager.register_texture("shadow_soft", self.texture_shadow_soft)
                self.manager.register_texture("shadow_softness", self.texture_shadow_softness)
            
            # 读回CPU
            if self.readback:
                shadow_soft_data = np.frombuffer(self.texture_shadow_soft.read(), dtype=np.float32)
                shadow_softness_data = np.frombuffer(self.texture_shadow_softness.read(), dtype=np.float32)
                facts.set_column(table_name, "shadow_soft", shadow_soft_data)
                facts.set_column(table_name, "shadow_softness", shadow_softness_data)
        
        except KeyError:
            pass
    
    def _init_textures(self, size):
        if self.texture_height:
            self.texture_height.release()
            self.texture_shadow_soft.release()
            self.texture_shadow_softness.release()
        
        self.texture_size = (size, size)
        self.texture_height = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_shadow_soft = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_shadow_softness = self.ctx.texture((size, size), 1, dtype='f4')
