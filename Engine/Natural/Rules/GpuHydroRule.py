import numpy as np
import moderngl
from ..Core.GpuRuleBase import GpuRuleBase
from ..Core.FactBase import FactBase

class GpuHydroRule(GpuRuleBase):
    """
    GPU 水文视觉规则 (GPU Hydro Rule)
    
    使用 ModernGL Compute Shader 实现水文视觉效果 (Wetness, Reflection, Caustics)。
    替代 HydroVisualRule 和 WaterCausticsRule 的 CPU 实现。
    
    优化:
    - 纹理脏标记: 仅在数据变化时上传
    - readback默认关闭: 避免GPU-CPU回读
    - 共享纹理: 注册height/water供其他规则使用
    """
    
    def __init__(self, wetness_smoothness=0.8, reflection_threshold=0.5,
                 caustics_speed=1.0, caustics_intensity=0.5, context=None, manager=None, readback=False,
                 table_name: str = "terrain_main", use_shared_textures: bool = True):
        super().__init__(
            name="Hydro.VisualGPU", 
            priority=55,
            manager=manager,
            readback=readback,
            use_shared_textures=use_shared_textures
        )
        self.wetness_smoothness = wetness_smoothness
        self.reflection_threshold = reflection_threshold
        self.caustics_speed = caustics_speed
        self.caustics_intensity = caustics_intensity
        self.table_name = table_name
        
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuHydroRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # Compute Shader: Wetness + Caustics + Flow
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D height_map;
        layout(r32f, binding = 1) readonly uniform image2D water_map;
        
        layout(r32f, binding = 2) writeonly uniform image2D wetness_map;
        layout(r32f, binding = 3) writeonly uniform image2D roughness_map;
        layout(r32f, binding = 4) writeonly uniform image2D caustics_map;
        layout(r32f, binding = 5) writeonly uniform image2D flow_x_map;
        layout(r32f, binding = 6) writeonly uniform image2D flow_y_map;
        
        uniform float rain_intensity;
        uniform float time;
        uniform float wetness_smoothness;
        uniform float caustics_speed;
        uniform float caustics_intensity;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_map);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float h = imageLoad(height_map, pos).r;
            float w = imageLoad(water_map, pos).r;
            
            // --- 1. Wetness & Roughness ---
            float wetness = clamp(w + rain_intensity * 0.5, 0.0, 1.0);
            float roughness = 0.8 * (1.0 - wetness * wetness_smoothness);
            roughness = clamp(roughness, 0.1, 1.0);
            
            imageStore(wetness_map, pos, vec4(wetness, 0.0, 0.0, 0.0));
            imageStore(roughness_map, pos, vec4(roughness, 0.0, 0.0, 0.0));
            
            // --- 2. Caustics (Procedural) ---
            vec2 uv = vec2(pos) / vec2(size) * 12.0; // Scale
            float t = time * caustics_speed;
            
            float c = 0.0;
            c += sin(uv.x * 2.0 + t) * cos(uv.y * 2.0 + t * 0.7);
            c += sin(uv.x * 5.0 - t * 1.2) * sin(uv.y * 5.0 + t * 0.5) * 0.5;
            
            // Normalize and intensity
            c = (c + 1.5) / 3.0;
            c = pow(c, 3.0) * caustics_intensity;
            c = clamp(c, 0.0, 1.0);
            
            // Mask by water presence (simple check)
            if (w < 0.01) c = 0.0;
            
            imageStore(caustics_map, pos, vec4(c, 0.0, 0.0, 0.0));
            
            // --- 3. Flow Direction (Gradient) ---
            float h_l = imageLoad(height_map, pos + ivec2(-1, 0)).r;
            float h_r = imageLoad(height_map, pos + ivec2(1, 0)).r;
            float h_u = imageLoad(height_map, pos + ivec2(0, 1)).r;
            float h_d = imageLoad(height_map, pos + ivec2(0, -1)).r;
            
            // Gradient (-grad)
            float flow_x = -(h_r - h_l) * 0.5;
            float flow_y = -(h_u - h_d) * 0.5;
            
            // Normalize
            float len = sqrt(flow_x*flow_x + flow_y*flow_y) + 1e-5;
            flow_x /= len;
            flow_y /= len;
            
            if (w < 0.1) {
                flow_x = 0.0;
                flow_y = 0.0;
            }
            
            imageStore(flow_x_map, pos, vec4(flow_x, 0.0, 0.0, 0.0));
            imageStore(flow_y_map, pos, vec4(flow_y, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_height = None
        self.texture_water = None
        
        self.texture_wetness = None
        self.texture_roughness = None
        self.texture_caustics = None
        self.texture_flow_x = None
        self.texture_flow_y = None

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return

        table_name = self.table_name
        try:
            # Check shared textures
            shared_height = self.manager.get_texture("height") if (self.use_shared_textures and self.manager) else None
            shared_water = self.manager.get_texture("water") if (self.use_shared_textures and self.manager) else None

            # Fallback or Setup
            if not shared_height:
                height_data = facts.get_column(table_name, "height")
                grid_len = len(height_data)
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
            
            # Upload or Bind
            if shared_height:
                shared_height.bind_to_image(0, read=True, write=False)
            else:
                self.texture_height.write(height_data.astype(np.float32).tobytes())
                self.texture_height.bind_to_image(0, read=True, write=False)
                # 注册为共享纹理供其他规则使用
                if self.manager and self.use_shared_textures:
                    self.manager.register_texture("height", self.texture_height)

            if shared_water:
                shared_water.bind_to_image(1, read=True, write=False)
            else:
                try:
                    water_data = facts.get_column(table_name, "water")
                except KeyError:
                    water_data = np.zeros(grid_len, dtype=np.float32)
                self.texture_water.write(water_data.astype(np.float32).tobytes())
                self.texture_water.bind_to_image(1, read=True, write=False)
                # 注册为共享纹理供其他规则使用
                if self.manager and self.use_shared_textures:
                    self.manager.register_texture("water", self.texture_water)
            
            # Uniforms
            rain_intensity = facts.get_global("rain_intensity") or 0.0
            time = facts.get_global("time") or 0.0
            
            self.program['rain_intensity'].value = rain_intensity
            self.program['time'].value = time
            self.program['wetness_smoothness'].value = self.wetness_smoothness
            self.program['caustics_speed'].value = self.caustics_speed
            self.program['caustics_intensity'].value = self.caustics_intensity
            
            # Bind Outputs
            self.texture_wetness.bind_to_image(2, read=False, write=True)
            self.texture_roughness.bind_to_image(3, read=False, write=True)
            self.texture_caustics.bind_to_image(4, read=False, write=True)
            self.texture_flow_x.bind_to_image(5, read=False, write=True)
            self.texture_flow_y.bind_to_image(6, read=False, write=True)
            
            # Run
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            # Register outputs
            if self.manager:
                self.manager.register_texture("hydro_wetness", self.texture_wetness)
                self.manager.register_texture("hydro_caustics", self.texture_caustics)
                self.manager.register_texture("wetness", self.texture_wetness)
                self.manager.register_texture("roughness", self.texture_roughness)

            # Read Back
            if self.readback:
                wetness_data = np.frombuffer(self.texture_wetness.read(), dtype=np.float32)
                roughness_data = np.frombuffer(self.texture_roughness.read(), dtype=np.float32)
                caustics_data = np.frombuffer(self.texture_caustics.read(), dtype=np.float32)
                flow_x_data = np.frombuffer(self.texture_flow_x.read(), dtype=np.float32)
                flow_y_data = np.frombuffer(self.texture_flow_y.read(), dtype=np.float32)
                
                facts.set_column(table_name, "wetness", wetness_data)
                facts.set_column(table_name, "roughness", roughness_data)
                facts.set_column(table_name, "caustics", caustics_data)
                facts.set_column(table_name, "flow_direction_x", flow_x_data)
                facts.set_column(table_name, "flow_direction_y", flow_y_data)
                
                facts.set_global("hydro_wetness_map", wetness_data.reshape(size, size))
                facts.set_global("hydro_roughness_map", roughness_data.reshape(size, size))
                facts.set_global("hydro_caustics", caustics_data.reshape(size, size))
            
        except KeyError:
            pass
            
    def _init_textures(self, size):
        if self.texture_height:
            self.texture_height.release()
            self.texture_water.release()
            self.texture_wetness.release()
            self.texture_roughness.release()
            self.texture_caustics.release()
            self.texture_flow_x.release()
            self.texture_flow_y.release()
            
        self.texture_size = (size, size)
        self.texture_height = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_water = self.ctx.texture((size, size), 1, dtype='f4')
        
        self.texture_wetness = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_roughness = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_caustics = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_flow_x = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_flow_y = self.ctx.texture((size, size), 1, dtype='f4')
