import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuAtmosphereRule(Rule):
    """
    GPU 大气光学规则 (GPU Atmosphere Rule)
    
    使用 ModernGL Compute Shader 实现丁达尔效应 (God Rays) 和云影。
    替代 AtmosphereRule 的 CPU 实现，解决 500k 面下的性能瓶颈。
    """
    
    def __init__(self, god_ray_samples=32, god_ray_density=0.5,
                 cloud_speed=1.0, cloud_scale=0.01, context=None, manager=None, readback=True):
        super().__init__("Atmosphere.OpticalGPU", priority=70)
        self.god_ray_samples = god_ray_samples
        self.god_ray_density = god_ray_density
        self.cloud_speed = cloud_speed
        self.cloud_scale = cloud_scale
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
                print(f"[GpuAtmosphereRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return

        # Compute Shader: God Rays + Cloud Shadows
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D shadow_mask;
        layout(r32f, binding = 1) readonly uniform image2D cloud_noise;
        layout(r32f, binding = 2) writeonly uniform image2D god_ray_map;
        layout(r32f, binding = 3) writeonly uniform image2D cloud_shadow_map;
        
        uniform vec2 sun_pos_screen; // 0-1 range
        uniform int samples;
        uniform float density;
        uniform float weight;
        uniform float decay;
        uniform float exposure;
        
        uniform vec2 cloud_offset;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(shadow_mask);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            // --- 1. God Rays (Radial Blur) ---
            vec2 uv = (vec2(pos) + 0.5) / vec2(size);
            vec2 deltaTextCoord = vec2(uv - sun_pos_screen);
            
            // 每次步进的距离
            deltaTextCoord *= 1.0 / float(samples) * density;
            
            vec2 coord = uv;
            float illuminationDecay = 1.0;
            float god_ray = 0.0;
            
            for(int i=0; i < samples; i++) {
                coord -= deltaTextCoord;
                
                // 边界检查
                if(coord.x < 0.0 || coord.x > 1.0 || coord.y < 0.0 || coord.y > 1.0) break;
                
                // 采样 Shadow Mask (1=Lit, 0=Shadow)
                // God Rays 产生于光线通过的地方 (1)
                float s = imageLoad(shadow_mask, ivec2(coord * size)).r;
                
                s *= illuminationDecay * weight;
                god_ray += s;
                illuminationDecay *= decay;
            }
            
            god_ray *= exposure;
            imageStore(god_ray_map, pos, vec4(god_ray, 0.0, 0.0, 0.0));
            
            // --- 2. Cloud Shadows (Texture Scroll) ---
            // 简单的 UV 偏移采样
            ivec2 cloud_size = imageSize(cloud_noise);
            
            // Wrap UV
            vec2 cloud_uv = uv + cloud_offset;
            cloud_uv = fract(cloud_uv); // 0-1
            
            ivec2 cloud_pos = ivec2(cloud_uv * cloud_size);
            float cloud_val = imageLoad(cloud_noise, cloud_pos).r;
            
            // 简单的阈值处理模拟云影
            float shadow = smoothstep(0.4, 0.6, cloud_val);
            
            imageStore(cloud_shadow_map, pos, vec4(shadow, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_shadow = None
        self.texture_cloud = None
        self.texture_god_ray = None
        self.texture_cloud_shadow = None
        
        self._cloud_noise_data = None

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return

        table_name = "terrain_main"
        try:
            # Check shared textures
            # We need 'shadow_mask' from LightingRule
            shared_shadow = self.manager.get_texture("shadow_mask") if self.manager else None
            
            # Setup Size
            if shared_shadow:
                size = shared_shadow.width
                grid_len = size * size
            else:
                try:
                    flat_shadow = facts.get_column(table_name, "shadow_mask")
                    grid_len = len(flat_shadow)
                except KeyError:
                    # Fallback to height for size
                    flat_height = facts.get_column(table_name, "height")
                    grid_len = len(flat_height)
                    flat_shadow = np.ones(grid_len, dtype=np.float32)
                size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return

            if self.texture_size != (size, size):
                self._init_textures(size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            # Upload or Bind
            if shared_shadow:
                shared_shadow.bind_to_image(0, read=True, write=False)
            else:
                self.texture_shadow.write(flat_shadow.astype(np.float32).tobytes())
                self.texture_shadow.bind_to_image(0, read=True, write=False)
            
            # Uniforms
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            
            sun_screen_x = 0.5 + sun_dir[0] * 0.5
            sun_screen_y = 0.5 + sun_dir[2] * 0.5
            
            self.program['sun_pos_screen'].value = (sun_screen_x, sun_screen_y)
            self.program['samples'].value = self.god_ray_samples
            self.program['density'].value = 1.0 
            self.program['weight'].value = 0.05 
            self.program['decay'].value = 0.95 
            self.program['exposure'].value = self.god_ray_density * 2.0
            
            time = facts.get_global("time") or 0.0
            wind_speed = facts.get_global("wind_speed") or 1.0
            wind_dir = facts.get_global("wind_direction")
            if wind_dir is None: wind_dir = np.array([1.0, 0.0, 0.0])
            
            offset_x = wind_dir[0] * wind_speed * self.cloud_speed * time * 0.01
            offset_y = wind_dir[2] * wind_speed * self.cloud_speed * time * 0.01
            self.program['cloud_offset'].value = (offset_x, offset_y)
            
            # Bind Others
            self.texture_cloud.bind_to_image(1, read=True, write=False)
            self.texture_god_ray.bind_to_image(2, read=False, write=True)
            self.texture_cloud_shadow.bind_to_image(3, read=False, write=True)
            
            # Run
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            # Register Outputs
            if self.manager:
                self.manager.register_texture("god_ray", self.texture_god_ray)
                self.manager.register_texture("cloud_shadow", self.texture_cloud_shadow)
            
            # Read Back
            if self.readback:
                god_ray_data = np.frombuffer(self.texture_god_ray.read(), dtype=np.float32)
                cloud_shadow_data = np.frombuffer(self.texture_cloud_shadow.read(), dtype=np.float32)
                facts.set_column(table_name, "god_ray", god_ray_data)
                facts.set_column(table_name, "cloud_shadow", cloud_shadow_data)
                
                # Merge on CPU (or move to shader later)
                atmosphere_intensity = god_ray_data * (1.0 - cloud_shadow_data * 0.5)
                facts.set_column(table_name, "atmosphere_intensity", atmosphere_intensity)
                
        except KeyError:
            pass
            
    def _init_textures(self, size):
        if self.texture_shadow:
            self.texture_shadow.release()
            self.texture_cloud.release()
            self.texture_god_ray.release()
            self.texture_cloud_shadow.release()
            
        self.texture_size = (size, size)
        self.texture_shadow = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_god_ray = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_cloud_shadow = self.ctx.texture((size, size), 1, dtype='f4')
        
        # Cloud Noise (生成一次)
        if self._cloud_noise_data is None or self._cloud_noise_data.shape != (size, size):
            self._cloud_noise_data = np.random.rand(size, size).astype(np.float32)
            # 简单模糊一下作为噪声
            from scipy.ndimage import gaussian_filter
            self._cloud_noise_data = gaussian_filter(self._cloud_noise_data, sigma=4.0)
            
        self.texture_cloud = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_cloud.write(self._cloud_noise_data.tobytes())