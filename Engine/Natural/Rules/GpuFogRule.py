import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuFogRule(Rule):
    """
    GPU 雾效规则 (GPU Fog Rule)
    
    使用 ModernGL Compute Shader 计算雾的密度分布。
    替代 FogRule 的 CPU 实现，消除 2M 面下的性能瓶颈。
    """
    
    def __init__(self, base_density=0.1, humidity_factor=0.5, 
                 context=None, manager=None, readback=True,
                 table_name: str = "terrain_main", use_shared_textures: bool = True):
        super().__init__("Atmosphere.FogGPU", priority=60)
        self.base_density = base_density
        self.humidity_factor = humidity_factor
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
                print(f"[GpuFogRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # Compute Shader: Fog Density Calculation
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D height_map;
        layout(r32f, binding = 1) readonly uniform image2D water_map;
        layout(r32f, binding = 2) writeonly uniform image2D fog_map;
        
        uniform float base_density;
        uniform float humidity_factor;
        uniform float temperature;
        uniform bool has_water;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_map);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            // 读取高度
            float h = imageLoad(height_map, pos).r;
            
            // 读取水深 (如果有)
            float water = 0.0;
            if (has_water) {
                water = imageLoad(water_map, pos).r;
            }
            
            // 计算湿度贡献
            // humidity = clamp(water * humidity_factor, 0, 1)
            float humidity = clamp(water * humidity_factor, 0.0, 1.0);
            
            // 基础雾密度
            // fog_density = base_density + humidity * 0.5
            float density = base_density + humidity * 0.5;
            
            // 高度衰减
            // height_factor = exp(-H / (100.0 + temperature * 5.0))
            float height_denom = 100.0 + temperature * 5.0;
            float height_factor = exp(-h / height_denom);
            
            density = density * height_factor;
            
            imageStore(fog_map, pos, vec4(density, 0.0, 0.0, 0.0));
        }
        """
        
        self.compute_shader = None
        self.texture_fog_out = None
        
    def compile_shader(self):
        if self.compute_shader:
            return
            
        try:
            self.compute_shader = self.ctx.compute_shader(self.compute_shader_source)
        except Exception as e:
            print(f"[GpuFogRule] Shader compilation failed: {e}")

    def ensure_textures(self, size):
        """确保输出纹理存在且尺寸正确"""
        if self.texture_fog_out is None or self.texture_fog_out.width != size:
            self.texture_fog_out = self.ctx.texture((size, size), 1, dtype='f4')
            if self.manager:
                self.manager.register_texture("fog_density_map", self.texture_fog_out)

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return

        self.compile_shader()
        if not self.compute_shader:
            return
            
        table_name = self.table_name
        
        try:
            # 1. 获取输入纹理 (Height & Water)
            # 优先从 Manager 获取共享纹理
            shared_height = self.manager.get_texture("height") if (self.use_shared_textures and self.manager) else None
            shared_water = self.manager.get_texture("water") if (self.use_shared_textures and self.manager) else None
            
            height_tex = None
            water_tex = None
            size = 0
            
            # 准备 Height Texture
            if shared_height:
                height_tex = shared_height
                size = height_tex.width
            else:
                # Fallback: 从 CPU 创建纹理 (性能较差)
                height_data = facts.get_column(table_name, "height")
                if height_data is None: return
                count = len(height_data)
                size = int(np.sqrt(count))
                height_tex = self.ctx.texture((size, size), 1, data=height_data.tobytes(), dtype='f4')
            
            # 准备 Water Texture
            has_water = False
            if shared_water:
                water_tex = shared_water
                has_water = True
            else:
                try:
                    water_data = facts.get_column(table_name, "water")
                    if water_data is not None:
                        water_tex = self.ctx.texture((size, size), 1, data=water_data.tobytes(), dtype='f4')
                        has_water = True
                except:
                    pass
            
            # 2. 准备输出纹理
            self.ensure_textures(size)
            
            # 3. 设置 Uniforms
            temperature = facts.get_global("temperature")
            if temperature is None: temperature = 20.0
            
            self.compute_shader['base_density'] = self.base_density
            self.compute_shader['humidity_factor'] = self.humidity_factor
            self.compute_shader['temperature'] = float(temperature)
            self.compute_shader['has_water'] = has_water
            
            # 4. 绑定纹理并运行 Shader
            height_tex.bind_to_image(0, read=True, write=False)
            if has_water:
                water_tex.bind_to_image(1, read=True, write=False)
            self.texture_fog_out.bind_to_image(2, read=False, write=True)
            
            # Dispatch
            group_size = 16
            groups_x = (size + group_size - 1) // group_size
            groups_y = (size + group_size - 1) // group_size
            self.compute_shader.run(groups_x, groups_y)
            
            # 5. Readback (如果需要)
            if self.readback:
                # 读取完整数据回 FactBase
                fog_bytes = self.texture_fog_out.read()
                fog_data = np.frombuffer(fog_bytes, dtype=np.float32)
                
                # 计算平均值用于 Global (这是 AtmosphereRule 需要的)
                avg_fog = np.mean(fog_data)
                facts.set_global("fog_density", avg_fog)
                
                # 如果不需要在 CPU 端使用 fog map，可以不存这个 global map
                # 但为了兼容性，还是存一下
                facts.set_global("fog_density_map", fog_data.reshape((size, size)))
                
            else:
                # Resident Mode: 
                # 我们仍然需要更新 fog_density 这个标量，否则 God Rays 可能会罢工
                # 简单估算：直接用 base_density 代替，或者不更新
                # 为了保持 God Rays 工作，设置一个默认值
                facts.set_global("fog_density", self.base_density)
                
                # 确保 fog_density_map 在 GPU 上可用 (已经通过 register_texture 完成)
                pass

        except Exception as e:
            # print(f"[GpuFogRule] Error: {e}")
            pass
