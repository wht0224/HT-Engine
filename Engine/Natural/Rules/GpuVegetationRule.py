import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuVegetationRule(Rule):
    """
    GPU 植被生长规则 (GPU Vegetation Growth Rule)
    
    使用 ModernGL Compute Shader 实现植被动态演替。
    替代 VegetationGrowthRule 的 CPU 实现，解决 500k 面下的性能瓶颈。
    """
    
    def __init__(self, growth_rate=0.1, death_rate=0.05, optimum_water=1.0, max_slope=1.5, context=None, manager=None, readback=True):
        super().__init__("Evolution.VegetationGPU", priority=30)
        self.growth_rate = growth_rate
        self.death_rate = death_rate
        self.optimum_water = optimum_water
        self.max_slope = max_slope
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
                print(f"[GpuVegetationRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # Compute Shader
        self.compute_shader_source = """
        #version 430
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D height_map;
        layout(r32f, binding = 1) readonly uniform image2D water_map;
        layout(r32f, binding = 2) readonly uniform image2D density_in_map;
        layout(r32f, binding = 3) writeonly uniform image2D density_out_map;
        
        uniform float growth_rate;
        uniform float death_rate;
        uniform float optimum_water;
        uniform float max_slope;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(height_map);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float h = imageLoad(height_map, pos).r;
            float w = imageLoad(water_map, pos).r;
            float density = imageLoad(density_in_map, pos).r;
            
            // Calculate Slope (Gradient magnitude)
            // Central difference
            float h_l = imageLoad(height_map, clamp(pos + ivec2(-1, 0), ivec2(0), size-1)).r;
            float h_r = imageLoad(height_map, clamp(pos + ivec2(1, 0), ivec2(0), size-1)).r;
            float h_u = imageLoad(height_map, clamp(pos + ivec2(0, 1), ivec2(0), size-1)).r;
            float h_d = imageLoad(height_map, clamp(pos + ivec2(0, -1), ivec2(0), size-1)).r;
            
            float dz_dx = (h_r - h_l) * 0.5;
            float dz_dy = (h_u - h_d) * 0.5;
            float slope = sqrt(dz_dx*dz_dx + dz_dy*dz_dy);
            
            // Logic: Growth & Death
            float water_factor = clamp(w / optimum_water, 0.0, 1.0);
            float slope_factor = clamp(1.0 - (slope / max_slope), 0.0, 1.0);
            
            float growth_potential = water_factor * slope_factor;
            float new_growth = growth_rate * growth_potential * (1.0 - density);
            
            float drought_factor = 1.0 - water_factor;
            float death = death_rate * density * (1.0 + drought_factor * 5.0);
            
            float new_density = clamp(density + new_growth - death, 0.0, 1.0);
            
            imageStore(density_out_map, pos, vec4(new_density, 0.0, 0.0, 0.0));
        }
        """
        self.program = None
        self.texture_size = (0, 0)
        self.texture_height = None
        self.texture_water = None
        self.texture_density_in = None
        self.texture_density_out = None
        
    def evaluate(self, facts: FactBase):
        # Lazy Init Context from Manager
        if not self.ctx and self.manager and self.manager.context:
            self.ctx = self.manager.context
            
        if not self.ctx:
            return

        table_name = "terrain_main"
        try:
            # Check for shared textures first (GPU Residency)
            shared_height = self.manager.get_texture("height") if self.manager else None
            shared_water = self.manager.get_texture("water") if self.manager else None
            
            # Fallback to CPU data
            if not shared_height:
                height_data = facts.get_column(table_name, "height")
                count = len(height_data)
                size = int(np.sqrt(count))
            else:
                size = shared_height.width
                count = size * size

            # Ensure density data exists (CPU side is still master for now)
            try:
                density_data = facts.get_column(table_name, "vegetation_density")
            except KeyError:
                density_data = np.zeros(count, dtype=np.float32)
                facts.add_column(table_name, "vegetation_density", density_data)

            if size * size != count:
                return

            if self.texture_size != (size, size):
                self._init_textures(size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)

            # Upload or Bind Inputs
            if shared_height:
                shared_height.bind_to_image(0, read=True, write=False)
            else:
                self.texture_height.write(height_data.astype('f4').tobytes())
                self.texture_height.bind_to_image(0, read=True, write=False)
                
            if shared_water:
                shared_water.bind_to_image(1, read=True, write=False)
            else:
                try:
                    water_data = facts.get_column(table_name, "water")
                except KeyError:
                    water_data = np.zeros(count, dtype=np.float32)
                self.texture_water.write(water_data.astype('f4').tobytes())
                self.texture_water.bind_to_image(1, read=True, write=False)

            self.texture_density_in.write(density_data.astype('f4').tobytes())
            
            # Uniforms
            self.program['growth_rate'].value = self.growth_rate
            self.program['death_rate'].value = self.death_rate
            self.program['optimum_water'].value = self.optimum_water
            self.program['max_slope'].value = self.max_slope

            self.texture_density_in.bind_to_image(2, read=True, write=False)
            self.texture_density_out.bind_to_image(3, read=False, write=True)
            
            # Dispatch
            group_x = (size + 15) // 16
            group_y = (size + 15) // 16
            self.program.run(group_x, group_y)
            
            # Read back result (unless disabled)
            if self.manager:
                # Register output texture for others to use
                self.manager.register_texture("vegetation_density", self.texture_density_out)
                
            if self.readback:
                new_density_bytes = self.texture_density_out.read()
                new_density_data = np.frombuffer(new_density_bytes, dtype=np.float32)
                facts.set_column(table_name, "vegetation_density", new_density_data)
        except KeyError:
            pass
            
    def _init_textures(self, size):
        if self.texture_height:
            self.texture_height.release()
            self.texture_water.release()
            self.texture_density_in.release()
            self.texture_density_out.release()
            
        self.texture_size = (size, size)
        self.texture_height = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_water = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_density_in = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_density_out = self.ctx.texture((size, size), 1, dtype='f4')
