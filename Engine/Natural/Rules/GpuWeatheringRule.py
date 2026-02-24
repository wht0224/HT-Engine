import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuWeatheringRule(Rule):
    """
    GPU 热力风化规则 (GPU Thermal Weathering Rule)
    
    使用 ModernGL Compute Shader 实现地形热力风化 (Talus Formation)。
    替代 ThermalWeatheringRule 的 CPU 实现，消除 500k 面下的最大性能瓶颈。
    """
    
    def __init__(self, critical_angle=0.8, weathering_rate=0.1, iterations=5, context=None, manager=None, readback=True):
        super().__init__("Terrain.ThermalWeatheringGPU", priority=25)
        self.critical_angle = critical_angle
        self.weathering_rate = weathering_rate
        self.iterations = iterations
        self.manager = manager
        self.readback = readback
        
        # 初始化 OpenGL Context
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuWeatheringRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # Compute Shader: Flux Calculation & Height Update
        # 我们使用 "Gather" 方法或者 Ping-Pong 技术。
        # 为了简单且正确，我们使用两个 Pass (或者在同一个 Shader 里仔细处理，但这里为了逻辑清晰，使用 Ping-Pong 纹理)
        # 实际上，风化是一个局部操作，我们可以简单地在一个 Pass 里计算 "我要给出多少" 和 "我要收回多少" 吗？
        # 不行，因为 "我要收回多少" 依赖于邻居的 "我要给出多少"。
        # 所以标准做法是：
        # Pass 1: 计算每个像素向四周的流出量 (Flux)
        # Pass 2: 根据流出量和流入量更新高度
        # 
        # 为了极致性能，我们可以简化模型：
        # 既然是 "迭代" 的，我们可以直接计算 "Height Delta"。
        # Delta = Sum(In from neighbors) - Sum(Out to neighbors)
        # 这需要读取邻居的高度来计算邻居流向我的量。
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D height_in;
        layout(r32f, binding = 1) writeonly uniform image2D height_out;
        
        uniform float critical_angle;
        uniform float weathering_rate;
        
        // 计算流出量 (Flux Out)
        float get_flux_out(float h_center, float h_neighbor) {
            float diff = h_center - h_neighbor;
            // 简单的距离假设为 1.0
            float slope = diff; 
            
            if (slope > critical_angle) {
                return (slope - critical_angle) * weathering_rate * 0.25; // 分给4个邻居，简单均分权重的近似
            }
            return 0.0;
        }

        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_in);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float h = imageLoad(height_in, pos).r;
            
            // 读取邻居高度
            // 边界处理：如果越界，假设高度无穷大(无法流出)或者和中心一样
            // 这里简单处理：越界取中心值 (无流动)
            
            float h_l = (pos.x > 0) ? imageLoad(height_in, pos + ivec2(-1, 0)).r : h;
            float h_r = (pos.x < size.x - 1) ? imageLoad(height_in, pos + ivec2(1, 0)).r : h;
            float h_u = (pos.y < size.y - 1) ? imageLoad(height_in, pos + ivec2(0, 1)).r : h;
            float h_d = (pos.y > 0) ? imageLoad(height_in, pos + ivec2(0, -1)).r : h;
            
            // 1. 计算我不稳定的物质流出 (Out Flux)
            // 只有当我比邻居高，且坡度 > 临界值时，才流出
            float out_l = get_flux_out(h, h_l);
            float out_r = get_flux_out(h, h_r);
            float out_u = get_flux_out(h, h_u);
            float out_d = get_flux_out(h, h_d);
            
            float total_out = out_l + out_r + out_u + out_d;
            
            // 2. 计算邻居流向我的物质 (In Flux)
            // 对称的：邻居流向我 = get_flux_out(neighbor, me)
            float in_l = get_flux_out(h_l, h);
            float in_r = get_flux_out(h_r, h);
            float in_u = get_flux_out(h_u, h);
            float in_d = get_flux_out(h_d, h);
            
            float total_in = in_l + in_r + in_u + in_d;
            
            // 更新高度
            float h_new = h + total_in - total_out;
            
            imageStore(height_out, pos, vec4(h_new, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        # Ping-Pong Textures
        self.texture_a = None
        self.texture_b = None

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return

        table_name = "terrain_main"
        try:
            # Check shared textures
            shared_height = self.manager.get_texture("height") if self.manager else None
            
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

            # Upload or Bind Inputs
            if shared_height:
                shared_height.bind_to_image(0, read=True, write=False)
            else:
                self.texture_a.write(flat_height.astype(np.float32).tobytes())
                self.texture_a.bind_to_image(0, read=True, write=False)

            self.texture_b.bind_to_image(1, read=False, write=True)

            # Uniforms
            self.program['critical_angle'].value = self.critical_angle
            self.program['weathering_rate'].value = self.weathering_rate
            
            # Run (Ping-Pong logic could be better, but for now simple pass)
            # Actually, weathering modifies height.
            # If we are in "readback=False" mode, we should update the shared texture "height" for next rules!
            # But here we write to 'height_out'.
            # We need to swap or copy back.
            
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            # Register Output
            if self.manager:
                # Update the 'height' texture in manager to point to our output?
                # This is tricky because others might hold reference to the old one.
                # Better: Copy out to in for next frame, or register "height" as out.
                self.manager.register_texture("height", self.texture_b)
            
            # Read Back
            if self.readback:
                new_height_data = np.frombuffer(self.texture_b.read(), dtype=np.float32)
                facts.set_column(table_name, "height", new_height_data)
                
        except KeyError:
            pass
            
    def _init_textures(self, size):
        if self.texture_a:
            self.texture_a.release()
            self.texture_b.release()
            
        self.texture_size = (size, size)
        self.texture_a = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_b = self.ctx.texture((size, size), 1, dtype='f4')
