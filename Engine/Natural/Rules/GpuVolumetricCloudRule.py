import numpy as np
import moderngl
from ..Core.GpuRuleBase import GpuRuleBase
from ..Core.FactBase import FactBase


class GpuVolumetricCloudRule(GpuRuleBase):
    """
    GPU 体积云规则 (GPU Volumetric Cloud Rule)
    
    使用 ModernGL Compute Shader 实现3D体积云生成。
    基于符号主义AI架构，用规则推理云的形态、光照和动画。
    
    核心特性:
    - 3D噪声生成云密度场
    - 风场驱动的云动画
    - 简化光照模型（亮面/暗面）
    - 与Natural系统集成
    """
    
    def __init__(self, 
                 cloud_height=300.0,           # 云底高度
                 cloud_thickness=150.0,        # 云厚度
                 cloud_coverage=0.6,           # 覆盖率 (0-1)
                 wind_speed=15.0,              # 风速
                 wind_direction=None,          # 风向 (默认X轴)
                 context=None, 
                 manager=None, 
                 readback=False,
                 table_name: str = "terrain_main",
                 use_shared_textures: bool = True):
        super().__init__(
            name="Atmosphere.VolumetricCloudGPU",
            priority=75,  # 在大气效果之后
            manager=manager,
            readback=readback,
            use_shared_textures=use_shared_textures
        )
        
        self.cloud_height = cloud_height
        self.cloud_thickness = cloud_thickness
        self.cloud_coverage = cloud_coverage
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction or np.array([1.0, 0.0, 0.5], dtype=np.float32)
        self.wind_direction = self.wind_direction / (np.linalg.norm(self.wind_direction) + 1e-5)
        self.table_name = table_name
        
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuVolumetricCloudRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        # 3D云纹理尺寸
        self.cloud_resolution = (128, 64, 128)  # X, Y(高度), Z
        
        # Compute Shader: 3D云生成
        self.compute_shader_source = """#version 430
        layout(local_size_x = 8, local_size_y = 4, local_size_z = 8) in;
        
        layout(r8, binding = 0) writeonly uniform image3D cloud_density;
        layout(rgba8, binding = 1) writeonly uniform image3D cloud_color;
        
        uniform float time;
        uniform vec3 wind_direction;
        uniform float coverage;
        uniform vec3 sun_dir;
        uniform float cloud_height;
        uniform float cloud_thickness;
        uniform float wind_speed;
        
        // 3D哈希噪声
        float hash3D(vec3 p) {
            p = vec3(dot(p, vec3(127.1, 311.7, 74.7)),
                     dot(p, vec3(269.5, 183.3, 246.1)),
                     dot(p, vec3(113.5, 271.9, 124.6)));
            return fract(sin(dot(p, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
        }
        
        // 3D噪声（三线性插值）
        float noise3D(vec3 p) {
            vec3 i = floor(p);
            vec3 f = fract(p);
            f = f * f * (3.0 - 2.0 * f);  // smoothstep
            
            float n = mix(
                mix(
                    mix(hash3D(i + vec3(0,0,0)), hash3D(i + vec3(1,0,0)), f.x),
                    mix(hash3D(i + vec3(0,1,0)), hash3D(i + vec3(1,1,0)), f.x),
                    f.y
                ),
                mix(
                    mix(hash3D(i + vec3(0,0,1)), hash3D(i + vec3(1,0,1)), f.x),
                    mix(hash3D(i + vec3(0,1,1)), hash3D(i + vec3(1,1,1)), f.x),
                    f.y
                ),
                f.z
            );
            return n;
        }
        
        // FBM（分形布朗运动）
        float fbm3D(vec3 p, int octaves) {
            float value = 0.0;
            float amplitude = 0.5;
            float frequency = 1.0;
            
            for (int i = 0; i < octaves; i++) {
                value += amplitude * noise3D(p * frequency);
                amplitude *= 0.5;
                frequency *= 2.0;
            }
            
            return value;
        }
        
        void main() {
            ivec3 pos = ivec3(gl_GlobalInvocationID.xyz);
            ivec3 size = imageSize(cloud_density);
            
            if (pos.x >= size.x || pos.y >= size.y || pos.z >= size.z) return;
            
            // 归一化坐标 (0-1)
            vec3 uvw = vec3(pos) / vec3(size);
            
            // 世界坐标（带风移动）
            vec3 worldPos = uvw * vec3(2000.0, cloud_thickness, 2000.0);
            worldPos.xz += wind_direction.xz * time * wind_speed;
            worldPos.y += cloud_height;
            
            // 云密度（FBM噪声）
            float density = fbm3D(worldPos * 0.005, 4);
            
            // 高度塑形（云底和云顶稀疏，中间浓密）
            float heightInCloud = uvw.y;
            float heightShape = sin(heightInCloud * 3.14159);  // 中间高两边低
            heightShape = pow(heightShape, 0.5);  // 锐化
            density *= heightShape;
            
            // 覆盖率阈值
            float threshold = 1.0 - coverage;
            density = density > threshold ? (density - threshold) / (1.0 - threshold) : 0.0;
            
            // 写入密度
            imageStore(cloud_density, pos, vec4(density, 0.0, 0.0, 0.0));
            
            // 预计算云颜色（简化光照）
            vec3 cloudBaseColor = vec3(0.95, 0.95, 0.98);  // 白云基础色
            
            // 简单的光照（假设云法线朝上）
            vec3 normal = vec3(0.0, 1.0, 0.0);
            float NdotL = max(dot(normal, -sun_dir), 0.0);
            
            // 亮面/暗面
            vec3 litColor = cloudBaseColor * (0.3 + NdotL * 0.7);  // 亮面
            vec3 shadowColor = cloudBaseColor * 0.4;  // 暗面
            
            // 根据密度混合（浓密处更亮，稀疏处更暗）
            vec3 finalColor = mix(shadowColor, litColor, density);
            
            imageStore(cloud_color, pos, vec4(finalColor, density));
        }
        """
        
        self.program = None
        self.texture_cloud_density = None
        self.texture_cloud_color = None
        self.time = 0.0
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            # 初始化纹理和程序
            if self.program is None:
                self._init_resources()
            
            # 更新时间
            dt = facts.get_global("delta_time") or 0.016
            self.time += dt
            
            # 获取太阳方向
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-5)
            
            # 绑定输出
            self.texture_cloud_density.bind_to_image(0, read=False, write=True)
            self.texture_cloud_color.bind_to_image(1, read=False, write=True)
            
            # 设置Uniform
            self.program['time'].value = self.time
            self.program['wind_direction'].value = tuple(self.wind_direction)
            self.program['coverage'].value = self.cloud_coverage
            self.program['sun_dir'].value = tuple(sun_dir)
            self.program['cloud_height'].value = self.cloud_height
            self.program['cloud_thickness'].value = self.cloud_thickness
            self.program['wind_speed'].value = self.wind_speed
            
            # 运行Compute Shader
            nx = (self.cloud_resolution[0] + 7) // 8
            ny = (self.cloud_resolution[1] + 3) // 4
            nz = (self.cloud_resolution[2] + 7) // 8
            self.program.run(nx, ny, nz)
            
            # 注册到共享纹理
            if self.manager:
                self.manager.register_texture("cloud_density", self.texture_cloud_density)
                self.manager.register_texture("cloud_color", self.texture_cloud_color)
            
            # 可选：读回CPU（用于调试）
            if self.readback:
                density_data = np.frombuffer(self.texture_cloud_density.read(), dtype=np.float32)
                facts.set_global("cloud_density_field", density_data.reshape(self.cloud_resolution))
        
        except Exception as e:
            print(f"[GpuVolumetricCloudRule] Error: {e}")
    
    def _init_resources(self):
        """初始化3D纹理和Compute Shader"""
        # 创建3D纹理
        self.texture_cloud_density = self.ctx.texture3d(
            self.cloud_resolution, 
            1,  # R8格式
            dtype='f4'
        )
        
        self.texture_cloud_color = self.ctx.texture3d(
            self.cloud_resolution,
            4,  # RGBA8格式
            dtype='f4'
        )
        
        # 创建Compute Shader
        self.program = self.ctx.compute_shader(self.compute_shader_source)
        
        print(f"[GpuVolumetricCloudRule] 3D云纹理初始化: {self.cloud_resolution}")
