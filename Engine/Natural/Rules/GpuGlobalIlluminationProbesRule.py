import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuGlobalIlluminationProbesRule(Rule):
    """
    GPU 全局光照探针规则 (GPU Global Illumination Probes Rule)
    
    实现简化的全局光照:
    - 3D探针网格
    - 辐射度缓存
    - 插值采样
    - 性能目标: 0.5-1.0ms
    """
    
    def __init__(self, context=None, manager=None, readback=False, quality="medium"):
        super().__init__("Lighting.GlobalIlluminationProbes", priority=73)
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
                print(f"[GpuGlobalIlluminationProbesRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.quality_params = {
            "low": {"probe_grid": 8, "bounce_count": 1},
            "medium": {"probe_grid": 16, "bounce_count": 2},
            "high": {"probe_grid": 32, "bounce_count": 3}
        }
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 4, local_size_y = 4, local_size_z = 4) in;
        
        layout(rgba32f, binding = 0) writeonly uniform image3D probe_grid;
        layout(rgba32f, binding = 1) readonly uniform image3D probe_grid_prev;
        layout(r32f, binding = 2) readonly uniform image2D depth_buffer;
        layout(rgba32f, binding = 3) readonly uniform image2D normal_buffer;
        
        uniform vec3 scene_min;
        uniform vec3 scene_max;
        uniform vec3 light_direction;
        uniform vec3 light_color;
        uniform float light_intensity;
        uniform vec3 ambient_color;
        uniform int bounce_count;
        uniform vec3 camera_position;
        uniform float probe_spacing;
        
        #define PI 3.14159265359
        #define SAMPLE_COUNT 16
        
        vec3 hemisphere_sample(uint index, uint sample_count) {
            float phi = float(index) * 2.4;
            float theta = float(index) * PI / float(sample_count);
            return vec3(
                sin(theta) * cos(phi),
                cos(theta),
                sin(theta) * sin(phi)
            );
        }
        
        float lambert(vec3 normal, vec3 light_dir) {
            return max(0.0, dot(normal, light_dir));
        }
        
        void main() {
            ivec3 probe_idx = ivec3(gl_GlobalInvocationID.xyz);
            ivec3 grid_size = imageSize(probe_grid);
            
            if (probe_idx.x >= grid_size.x || probe_idx.y >= grid_size.y || probe_idx.z >= grid_size.z) return;
            
            vec3 probe_pos = scene_min + (vec3(probe_idx) + 0.5) * probe_spacing;
            
            vec3 direct_light = vec3(0.0);
            vec3 indirect_light = vec3(0.0);
            
            float direct_visibility = 1.0;
            float NdotL = max(0.0, -light_direction.y);
            direct_light = light_color * light_intensity * NdotL * direct_visibility;
            
            vec3 indirect_accum = vec3(0.0);
            
            for (int bounce = 0; bounce < bounce_count; bounce++) {
                vec3 bounce_light = vec3(0.0);
                
                for (int s = 0; s < SAMPLE_COUNT; s++) {
                    vec3 sample_dir = hemisphere_sample(uint(s), SAMPLE_COUNT);
                    
                    ivec3 neighbor_idx = probe_idx + ivec3(sample_dir * 2.0);
                    
                    if (neighbor_idx.x >= 0 && neighbor_idx.x < grid_size.x &&
                        neighbor_idx.y >= 0 && neighbor_idx.y < grid_size.y &&
                        neighbor_idx.z >= 0 && neighbor_idx.z < grid_size.z) {
                        
                        vec4 neighbor_irradiance = imageLoad(probe_grid_prev, neighbor_idx);
                        bounce_light += neighbor_irradiance.rgb * lambert(vec3(0.0, 1.0, 0.0), sample_dir);
                    }
                }
                
                bounce_light /= float(SAMPLE_COUNT);
                indirect_accum += bounce_light * 0.5;
            }
            
            indirect_light = indirect_accum;
            
            vec3 total_light = direct_light + indirect_light + ambient_color;
            
            total_light = min(total_light, vec3(4.0));
            
            imageStore(probe_grid, probe_idx, vec4(total_light, 1.0));
        }
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        void main_interpolate() {
            ivec2 pixel = ivec3(gl_GlobalInvocationID.xyz).xy;
            ivec2 size = imageSize(depth_buffer);
            
            if (pixel.x >= size.x || pixel.y >= size.y) return;
            
            float depth = imageLoad(depth_buffer, pixel).r;
            if (depth >= 1.0) return;
            
            vec3 world_pos = vec3(0.0);
            
            vec3 probe_uv = (world_pos - scene_min) / (scene_max - scene_min);
            vec3 probe_coord = probe_uv * vec3(grid_size);
            
            ivec3 probe_min = ivec3(floor(probe_coord));
            ivec3 probe_max = ivec3(ceil(probe_coord));
            
            vec3 weights = fract(probe_coord);
            
            vec3 result = vec3(0.0);
            float total_weight = 0.0;
            
            for (int x = 0; x <= 1; x++) {
                for (int y = 0; y <= 1; y++) {
                    for (int z = 0; z <= 1; z++) {
                        ivec3 idx = ivec3(
                            clamp(probe_min.x + x, 0, grid_size.x - 1),
                            clamp(probe_min.y + y, 0, grid_size.y - 1),
                            clamp(probe_min.z + z, 0, grid_size.z - 1)
                        );
                        
                        vec3 w = vec3(
                            x == 0 ? 1.0 - weights.x : weights.x,
                            y == 0 ? 1.0 - weights.y : weights.y,
                            z == 0 ? 1.0 - weights.z : weights.z
                        );
                        
                        float weight = w.x * w.y * w.z;
                        vec4 probe_val = imageLoad(probe_grid, idx);
                        result += probe_val.rgb * weight;
                        total_weight += weight;
                    }
                }
            }
            
            if (total_weight > 0.0) {
                result /= total_weight;
            }
        }
        """
        
        self.program = None
        self.program_interpolate = None
        self.probe_grid_size = (0, 0, 0)
        self.texture_probe_grid = None
        self.texture_probe_grid_prev = None

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            params = self.quality_params[self.quality]
            grid_size = params["probe_grid"]
            
            scene_min = facts.get_global("scene_min")
            if scene_min is None:
                scene_min = np.array([-100.0, -10.0, -100.0], dtype=np.float32)
            
            scene_max = facts.get_global("scene_max")
            if scene_max is None:
                scene_max = np.array([100.0, 50.0, 100.0], dtype=np.float32)
            
            light_dir = facts.get_global("sun_direction")
            if light_dir is None:
                light_dir = np.array([0.5, -0.8, 0.3], dtype=np.float32)
            light_dir = light_dir / np.linalg.norm(light_dir)
            
            light_color = facts.get_global("sun_color")
            if light_color is None:
                light_color = np.array([1.0, 0.95, 0.9], dtype=np.float32)
            
            light_intensity = facts.get_global("sun_intensity") or 1.0
            
            ambient_color = facts.get_global("ambient_color")
            if ambient_color is None:
                ambient_color = np.array([0.1, 0.12, 0.15], dtype=np.float32)
            
            probe_spacing = np.max(scene_max - scene_min) / grid_size
            
            if self.probe_grid_size != (grid_size, grid_size, grid_size):
                self._init_textures(grid_size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            self.program['scene_min'].value = tuple(scene_min)
            self.program['scene_max'].value = tuple(scene_max)
            self.program['light_direction'].value = tuple(light_dir)
            self.program['light_color'].value = tuple(light_color)
            self.program['light_intensity'].value = light_intensity
            self.program['ambient_color'].value = tuple(ambient_color)
            self.program['bounce_count'].value = params["bounce_count"]
            self.program['probe_spacing'].value = probe_spacing
            
            self.texture_probe_grid.bind_to_image(0, read=False, write=True)
            self.texture_probe_grid_prev.bind_to_image(1, read=True, write=False)
            
            ng = (grid_size + 3) // 4
            self.program.run(ng, ng, ng)
            
            self.texture_probe_grid, self.texture_probe_grid_prev = \
                self.texture_probe_grid_prev, self.texture_probe_grid
            
            if self.manager:
                self.manager.register_texture("gi_probe_grid", self.texture_probe_grid)
            
            if self.readback:
                probe_data = np.frombuffer(self.texture_probe_grid.read(), dtype=np.float32)
                probe_data = probe_data.reshape((grid_size, grid_size, grid_size, 4))
                facts.set_global("gi_probe_data", probe_data)
                
        except Exception as e:
            pass

    def _init_textures(self, grid_size):
        if self.texture_probe_grid:
            self.texture_probe_grid.release()
            self.texture_probe_grid_prev.release()
            
        self.probe_grid_size = (grid_size, grid_size, grid_size)
        self.texture_probe_grid = self.ctx.texture3d((grid_size, grid_size, grid_size), 4, dtype='f4')
        self.texture_probe_grid_prev = self.ctx.texture3d((grid_size, grid_size, grid_size), 4, dtype='f4')
        
        initial_data = np.zeros((grid_size, grid_size, grid_size, 4), dtype=np.float32)
        initial_data[:, :, :, 3] = 1.0
        self.texture_probe_grid.write(initial_data.tobytes())
        self.texture_probe_grid_prev.write(initial_data.tobytes())
