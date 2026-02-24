import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuLightPropagationRule(Rule):
    """
    GPU 光传播规则 (GPU Light Propagation Rule)
    
    使用 ModernGL Compute Shader 实现并行光传播计算。
    光像水一样从亮处流向暗处。
    """
    
    def __init__(self, propagation_range=10.0, propagation_strength=0.5, iterations=3,
                 context=None, manager=None, readback=True, table_name: str = "terrain_main"):
        super().__init__("Lighting.PropagationGPU", priority=80)
        self.propagation_range = propagation_range
        self.propagation_strength = propagation_strength
        self.iterations = iterations
        self.manager = manager
        self.readback = readback
        self.table_name = table_name
        
        # 初始化 OpenGL Context
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuLightPropagationRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        # Compute Shader: 光传播计算（在GPU内做迭代累加，避免每轮读回CPU）
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D sources;
        layout(r32f, binding = 1) readonly uniform image2D current_indirect;
        layout(r32f, binding = 2) writeonly uniform image2D new_indirect;
        
        uniform float propagation_range;
        uniform float propagation_strength;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(sources);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            // 如果当前位置已经有直接光，跳过
            float source_val = imageLoad(sources, pos).r;
            if (source_val > 0.5) {
                imageStore(new_indirect, pos, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            float received_light = 0.0;
            int radius = int(propagation_range);
            
            // 在传播范围内搜索光源
            for (int dy = -radius; dy <= radius; dy++) {
                for (int dx = -radius; dx <= radius; dx++) {
                    if (dx == 0 && dy == 0) continue;
                    
                    ivec2 sample_pos = pos + ivec2(dx, dy);
                    
                    // 边界检查
                    if (sample_pos.x < 0 || sample_pos.x >= size.x ||
                        sample_pos.y < 0 || sample_pos.y >= size.y) {
                        continue;
                    }
                    
                    // 计算距离
                    float distance = sqrt(float(dx * dx + dy * dy));
                    if (distance > propagation_range) continue;
                    
                    // 采样光源亮度
                    float source_brightness = imageLoad(sources, sample_pos).r;
                    float indirect_brightness = imageLoad(current_indirect, sample_pos).r;
                    float total_brightness = max(source_brightness, indirect_brightness);
                    
                    if (total_brightness < 0.1) continue;
                    
                    // 距离衰减（平方反比）
                    float distance_factor = 1.0 / (1.0 + distance * distance * 0.1);
                    
                    // 计算贡献
                    float contribution = total_brightness * distance_factor;
                    received_light += contribution;
                }
            }
            
            // 限制最大值
            received_light = min(received_light, 1.0);
            
            // 写入结果（累加到当前间接光，避免CPU端循环读回）
            float current_val = imageLoad(current_indirect, pos).r;
            float next_val = clamp(current_val + received_light * propagation_strength, 0.0, 1.0);
            imageStore(new_indirect, pos, vec4(next_val, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_sources = None
        self.texture_current = None
        self.texture_new = None
        self._zero_bytes_size = 0
        self._zero_bytes = b""
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        table_name = self.table_name
        try:
            # 获取输入
            try:
                direct = facts.get_column(table_name, "shadow_mask")
            except KeyError:
                direct = np.ones(facts.get_column(table_name, "height").shape)
            
            grid_len = len(direct)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            if self.texture_size != (size, size):
                self._init_textures(size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            self.texture_sources.write(direct.astype(np.float32).tobytes())
            if self._zero_bytes_size != size:
                self._zero_bytes_size = size
                self._zero_bytes = (np.zeros(size * size, dtype=np.float32)).tobytes()
            self.texture_current.write(self._zero_bytes)
            self.texture_new.write(self._zero_bytes)
            
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            
            for iteration in range(self.iterations):
                self.texture_sources.bind_to_image(0, read=True, write=False)
                self.texture_current.bind_to_image(1, read=True, write=False)
                self.texture_new.bind_to_image(2, read=False, write=True)
                
                self.program["propagation_range"].value = self.propagation_range
                self.program["propagation_strength"].value = self.propagation_strength ** (iteration + 1)
                
                self.program.run(nx, ny, 1)
                
                self.texture_current, self.texture_new = self.texture_new, self.texture_current
            
            # 注册输出
            if self.manager:
                self.manager.register_texture("indirect_light", self.texture_current)
            
            # 读回CPU
            if self.readback:
                indirect = np.frombuffer(self.texture_current.read(), dtype=np.float32)
                facts.set_column(table_name, "indirect_light", indirect)
                facts.set_global("lighting_indirect", indirect.reshape((size, size)))
        
        except KeyError:
            pass
    
    def _init_textures(self, size):
        if self.texture_sources:
            self.texture_sources.release()
            self.texture_current.release()
            self.texture_new.release()
        
        self.texture_size = (size, size)
        self.texture_sources = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_current = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_new = self.ctx.texture((size, size), 1, dtype='f4')
