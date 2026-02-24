import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuAtmosphereEnhancedRule(Rule):
    """
    GPU 大气散射增强规则 (GPU Atmosphere Enhanced Rule)
    
    实现真实的大气散射效果:
    - Rayleigh散射 (蓝色天空)
    - Mie散射 (太阳光晕)
    - 高度衰减
    - 日出/日落颜色变化
    
    性能目标: 0.3-0.5ms
    """
    
    def __init__(self, context=None, manager=None, readback=False, quality="medium"):
        super().__init__("Atmosphere.ScatteringEnhanced", priority=68)
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
                print(f"[GpuAtmosphereEnhancedRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.quality_params = {
            "low": {"samples": 8, "transmittance_samples": 4},
            "medium": {"samples": 16, "transmittance_samples": 8},
            "high": {"samples": 32, "transmittance_samples": 16}
        }
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 8, local_size_y = 8) in;
        
        layout(rgba32f, binding = 0) writeonly uniform image2D sky_output;
        layout(r32f, binding = 1) readonly uniform image2D depth_buffer;
        
        uniform vec3 sun_direction;
        uniform vec3 camera_position;
        uniform vec3 camera_forward;
        uniform vec3 camera_right;
        uniform vec3 camera_up;
        uniform float planet_radius;
        uniform float atmosphere_height;
        uniform float rayleigh_height;
        uniform float mie_height;
        uniform vec3 rayleigh_beta;
        uniform vec3 mie_beta;
        uniform float mie_g;
        uniform int samples;
        uniform int transmittance_samples;
        uniform float time_of_day;
        
        const float PI = 3.14159265359;
        
        vec3 rayleigh_phase(float cos_theta) {
            return vec3(3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta));
        }
        
        float mie_phase(float cos_theta, float g) {
            float g2 = g * g;
            float num = (1.0 - g2);
            float denom = 4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5);
            return num / denom;
        }
        
        float rayleigh_density(float h) {
            return exp(-h / rayleigh_height);
        }
        
        float mie_density(float h) {
            return exp(-h / mie_height);
        }
        
        bool intersect_sphere(vec3 ro, vec3 rd, float radius, out float t_near, out float t_far) {
            vec3 center = vec3(0.0, -planet_radius, 0.0);
            vec3 oc = ro - center;
            float b = dot(oc, rd);
            float c = dot(oc, oc) - radius * radius;
            float discriminant = b * b - c;
            
            if (discriminant < 0.0) return false;
            
            float sqrt_d = sqrt(discriminant);
            t_near = -b - sqrt_d;
            t_far = -b + sqrt_d;
            return t_far > 0.0;
        }
        
        vec3 compute_transmittance(vec3 pos, vec3 dir, float t_max) {
            float segment = t_max / float(transmittance_samples);
            vec3 optical_depth = vec3(0.0);
            
            for (int i = 0; i < transmittance_samples; i++) {
                float t = (float(i) + 0.5) * segment;
                vec3 p = pos + dir * t;
                float h = length(p) - planet_radius;
                
                if (h < 0.0) break;
                
                optical_depth += rayleigh_beta * rayleigh_density(h) * segment;
                optical_depth += mie_beta * mie_density(h) * segment;
            }
            
            return exp(-optical_depth);
        }
        
        vec3 compute_scattering(vec3 ray_origin, vec3 ray_dir, float t_max) {
            float segment = t_max / float(samples);
            vec3 total_rayleigh = vec3(0.0);
            vec3 total_mie = vec3(0.0);
            vec3 optical_depth = vec3(0.0);
            
            float cos_theta = dot(ray_dir, sun_direction);
            vec3 rayleigh_phase_val = rayleigh_phase(cos_theta);
            float mie_phase_val = mie_phase(cos_theta, mie_g);
            
            for (int i = 0; i < samples; i++) {
                float t = (float(i) + 0.5) * segment;
                vec3 p = ray_origin + ray_dir * t;
                float h = length(p) - planet_radius;
                
                if (h < 0.0) break;
                
                float rayleigh_d = rayleigh_density(h);
                float mie_d = mie_density(h);
                
                optical_depth += rayleigh_beta * rayleigh_d * segment;
                optical_depth += mie_beta * mie_d * segment;
                
                float t_near, t_far;
                if (intersect_sphere(p, sun_direction, planet_radius + atmosphere_height, t_near, t_far)) {
                    float t_sun = t_far;
                    vec3 transmittance_sun = compute_transmittance(p, sun_direction, t_sun);
                    vec3 transmittance_view = exp(-optical_depth);
                    
                    total_rayleigh += rayleigh_beta * rayleigh_d * transmittance_view * transmittance_sun * segment;
                    total_mie += mie_beta * mie_d * transmittance_view * transmittance_sun * segment;
                }
            }
            
            vec3 result = total_rayleigh * rayleigh_phase_val + total_mie * mie_phase_val;
            
            // 改进的太阳光晕效果
            float sun_disk = smoothstep(0.9995, 0.9999, cos_theta);
            float sun_glow = pow(max(0.0, cos_theta), 64.0);
            float sun_halo = pow(max(0.0, cos_theta), 8.0) * 0.3;
            
            vec3 sun_color = vec3(1.0, 0.95, 0.8);
            vec3 sun_contribution = sun_color * (sun_disk * 15.0 + sun_glow * 2.0 + sun_halo);
            result += sun_contribution * exp(-optical_depth.r * 5.0);
            
            return result;
        }
        
        void main() {
            ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(sky_output);
            
            if (pixel.x >= size.x || pixel.y >= size.y) return;
            
            vec2 uv = (vec2(pixel) + 0.5) / vec2(size);
            uv = uv * 2.0 - 1.0;
            
            vec3 ray_dir = normalize(camera_forward + uv.x * camera_right + uv.y * camera_up);
            
            float t_near, t_far;
            float t_max = 100000.0;
            
            if (intersect_sphere(camera_position, ray_dir, planet_radius + atmosphere_height, t_near, t_far)) {
                t_max = t_far;
            }
            
            float ground_t;
            if (intersect_sphere(camera_position, ray_dir, planet_radius, t_near, ground_t)) {
                if (ground_t > 0.0) {
                    t_max = min(t_max, ground_t);
                }
            }
            
            vec3 sky_color = compute_scattering(camera_position, ray_dir, t_max);
            
            float horizon_factor = 1.0 - abs(ray_dir.y);
            horizon_factor = pow(horizon_factor, 3.0);
            sky_color += vec3(0.1, 0.05, 0.02) * horizon_factor;
            
            float sunset_factor = 1.0 - abs(time_of_day - 0.5) * 2.0;
            sunset_factor = max(0.0, sunset_factor);
            sunset_factor *= horizon_factor;
            sky_color += vec3(0.3, 0.1, 0.0) * sunset_factor;
            
            sky_color = pow(sky_color, vec3(1.0 / 2.2));
            
            imageStore(sky_output, pixel, vec4(sky_color, 1.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_sky = None
        self.texture_depth = None
        
        self.earth_params = {
            "planet_radius": 6371000.0,
            "atmosphere_height": 100000.0,
            "rayleigh_height": 8000.0,
            "mie_height": 1200.0,
            "rayleigh_beta": np.array([5.8e-6, 13.5e-6, 33.1e-6], dtype=np.float32),
            "mie_beta": np.array([21e-6, 21e-6, 21e-6], dtype=np.float32),
            "mie_g": 0.76
        }

    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            width = facts.get_global("viewport_width") or 1920
            height = facts.get_global("viewport_height") or 1080
            
            width = max(1, int(width))
            height = max(1, int(height))
            
            if self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
            
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -0.8, 0.3], dtype=np.float32)
            sun_dir = sun_dir / np.linalg.norm(sun_dir)
            
            cam_pos = facts.get_global("camera_position")
            if cam_pos is None:
                cam_pos = np.array([0.0, 100.0, 0.0], dtype=np.float32)
            
            cam_forward = facts.get_global("camera_forward")
            if cam_forward is None:
                cam_forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
            
            cam_right = facts.get_global("camera_right")
            if cam_right is None:
                cam_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            
            cam_up = facts.get_global("camera_up")
            if cam_up is None:
                cam_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
            time_of_day = facts.get_global("time_of_day") or 0.5
            
            params = self.quality_params[self.quality]
            
            self.program['sun_direction'].value = tuple(sun_dir)
            self.program['camera_position'].value = (0.0, self.earth_params["planet_radius"] + cam_pos[1], 0.0)
            self.program['camera_forward'].value = tuple(cam_forward)
            self.program['camera_right'].value = tuple(cam_right)
            self.program['camera_up'].value = tuple(cam_up)
            self.program['planet_radius'].value = self.earth_params["planet_radius"]
            self.program['atmosphere_height'].value = self.earth_params["atmosphere_height"]
            self.program['rayleigh_height'].value = self.earth_params["rayleigh_height"]
            self.program['mie_height'].value = self.earth_params["mie_height"]
            self.program['rayleigh_beta'].value = tuple(self.earth_params["rayleigh_beta"])
            self.program['mie_beta'].value = tuple(self.earth_params["mie_beta"])
            self.program['mie_g'].value = self.earth_params["mie_g"]
            self.program['samples'].value = params["samples"]
            self.program['transmittance_samples'].value = params["transmittance_samples"]
            self.program['time_of_day'].value = time_of_day
            
            self.texture_sky.bind_to_image(0, read=False, write=True)
            
            nx = (width + 7) // 8
            ny = (height + 7) // 8
            self.program.run(nx, ny, 1)
            
            if self.manager:
                self.manager.register_texture("sky_scattering", self.texture_sky)
            
            if self.readback:
                sky_data = np.frombuffer(self.texture_sky.read(), dtype=np.float32)
                sky_data = sky_data.reshape((height, width, 4))
                facts.set_global("sky_color_buffer", sky_data)
                
        except Exception as e:
            pass

    def _init_textures(self, width, height):
        if self.texture_sky:
            self.texture_sky.release()
            
        self.texture_size = (width, height)
        self.texture_sky = self.ctx.texture((width, height), 4, dtype='f4')
