import numpy as np
import moderngl
from ..Core.GpuRuleBase import GpuRuleBase
from ..Core.FactBase import FactBase


class GpuReflectionRule(GpuRuleBase):
    """
    GPU 反射规则 (GPU Reflection Rule)
    
    使用 ModernGL Compute Shader 实现并行反射计算。
    表面"性格"决定反射类型（镜面/漫反射）。
    
    优化:
    - 纹理脏标记: 仅在数据变化时上传
    - readback默认关闭: 避免GPU-CPU回读
    """
    
    def __init__(self, smoothness_threshold=0.7, reflection_range=20.0, max_bounces=2,
                 context=None, manager=None, readback=False, table_name: str = "terrain_main",
                 use_shared_textures: bool = True):
        super().__init__(
            name="Lighting.ReflectionGPU", 
            priority=90,
            manager=manager,
            readback=readback,
            use_shared_textures=use_shared_textures
        )
        self.smoothness_threshold = smoothness_threshold
        self.reflection_range = reflection_range
        self.max_bounces = max_bounces
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
                print(f"[GpuReflectionRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        # Compute Shader: 反射计算
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r32f, binding = 0) readonly uniform image2D roughness;
        layout(r32f, binding = 1) readonly uniform image2D wetness;
        layout(r32f, binding = 2) readonly uniform image2D direct_light;
        layout(r32f, binding = 3) readonly uniform image2D indirect_light;
        layout(r32f, binding = 4) writeonly uniform image2D reflection_map;
        
        uniform float smoothness_threshold;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(roughness);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            // 读取属性
            float r = imageLoad(roughness, pos).r;
            float w = imageLoad(wetness, pos).r;
            float light = clamp(imageLoad(direct_light, pos).r + imageLoad(indirect_light, pos).r, 0.0, 1.0);
            
            // 如果没有光照，就没有反射
            if (light < 0.1) {
                imageStore(reflection_map, pos, vec4(0.0, 0.0, 0.0, 0.0));
                return;
            }
            
            // 湿润度降低粗糙度
            float effective_roughness = r * (1.0 - w * 0.5);
            
            float reflection_intensity = 0.0;
            
            // 判断反射类型
            if (effective_roughness < smoothness_threshold) {
                // 镜面反射
                reflection_intensity = (1.0 - effective_roughness) * light * (1.0 + w * 0.5);
            } else {
                // 漫反射
                reflection_intensity = (1.0 - effective_roughness) * light * 0.5;
            }
            
            // 限制最大值
            reflection_intensity = clamp(reflection_intensity, 0.0, 1.0);
            
            // 写入结果
            imageStore(reflection_map, pos, vec4(reflection_intensity, 0.0, 0.0, 0.0));
        }
        """
        
        self.program = None
        self.texture_size = (0, 0)
        self.texture_roughness = None
        self.texture_wetness = None
        self.texture_direct = None
        self.texture_indirect = None
        self.texture_reflection = None
        
        # 初始化脏标记
        self._texture_dirty = {
            'roughness': True,
            'wetness': True,
            'direct': True,
            'indirect': True
        }
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        self._on_evaluate_start()
        
        table_name = self.table_name
        try:
            shared_roughness = self.get_shared_texture("roughness")
            shared_wetness = self.get_shared_texture("wetness")
            shared_direct = self.get_shared_texture("shadow_mask")
            shared_indirect = self.get_shared_texture("indirect_light")
            
            size = 0
            if shared_direct:
                size = shared_direct.width
            elif shared_roughness:
                size = shared_roughness.width
            elif shared_wetness:
                size = shared_wetness.width
            elif shared_indirect:
                size = shared_indirect.width
            else:
                flat_roughness = facts.get_column(table_name, "roughness")
                grid_len = len(flat_roughness)
                size = int(np.sqrt(grid_len))
                if size * size != grid_len:
                    return
                try:
                    flat_wetness = facts.get_column(table_name, "wetness")
                except KeyError:
                    flat_wetness = np.zeros(grid_len, dtype=np.float32)
                try:
                    flat_direct = facts.get_column(table_name, "shadow_mask")
                except KeyError:
                    flat_direct = np.ones(grid_len, dtype=np.float32)
                try:
                    flat_indirect = facts.get_column(table_name, "indirect_light")
                except KeyError:
                    flat_indirect = np.zeros(grid_len, dtype=np.float32)
            
            grid_len = int(size * size)
            if not shared_roughness:
                try:
                    flat_roughness = facts.get_column(table_name, "roughness")
                except KeyError:
                    flat_roughness = np.zeros(grid_len, dtype=np.float32)
            if not shared_wetness:
                try:
                    flat_wetness = facts.get_column(table_name, "wetness")
                except KeyError:
                    flat_wetness = np.zeros(grid_len, dtype=np.float32)
            if not shared_direct:
                try:
                    flat_direct = facts.get_column(table_name, "shadow_mask")
                except KeyError:
                    flat_direct = np.ones(grid_len, dtype=np.float32)
            if not shared_indirect:
                try:
                    flat_indirect = facts.get_column(table_name, "indirect_light")
                except KeyError:
                    flat_indirect = np.zeros(grid_len, dtype=np.float32)
            
            if self.texture_size != (size, size):
                self._init_textures(size)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
                self.mark_all_textures_dirty()
            
            if shared_roughness:
                shared_roughness.bind_to_image(0, read=True, write=False)
            else:
                if self.should_upload_texture('roughness', flat_roughness):
                    self.texture_roughness.write(flat_roughness.astype(np.float32).tobytes())
                    self.mark_texture_clean('roughness')
                self.texture_roughness.bind_to_image(0, read=True, write=False)
            
            if shared_wetness:
                shared_wetness.bind_to_image(1, read=True, write=False)
            else:
                if self.should_upload_texture('wetness', flat_wetness):
                    self.texture_wetness.write(flat_wetness.astype(np.float32).tobytes())
                    self.mark_texture_clean('wetness')
                self.texture_wetness.bind_to_image(1, read=True, write=False)
            
            if shared_direct:
                shared_direct.bind_to_image(2, read=True, write=False)
            else:
                if self.should_upload_texture('direct', flat_direct):
                    self.texture_direct.write(flat_direct.astype(np.float32).tobytes())
                    self.mark_texture_clean('direct')
                self.texture_direct.bind_to_image(2, read=True, write=False)
            
            if shared_indirect:
                shared_indirect.bind_to_image(3, read=True, write=False)
            else:
                if self.should_upload_texture('indirect', flat_indirect):
                    self.texture_indirect.write(flat_indirect.astype(np.float32).tobytes())
                    self.mark_texture_clean('indirect')
                self.texture_indirect.bind_to_image(3, read=True, write=False)
            
            self.texture_reflection.bind_to_image(4, read=False, write=True)
            
            self.program['smoothness_threshold'].value = self.smoothness_threshold
            
            nx = (size + 15) // 16
            ny = (size + 15) // 16
            self.program.run(nx, ny, 1)
            
            self.register_shared_texture("reflection", self.texture_reflection)
            
            # 读回CPU
            if self.readback:
                reflection_data = np.frombuffer(self.texture_reflection.read(), dtype=np.float32)
                facts.set_column(table_name, "reflection", reflection_data)
                facts.set_global("lighting_reflection", reflection_data.reshape((size, size)))
        
        except KeyError:
            pass
    
    def _init_textures(self, size):
        if self.texture_roughness:
            self.texture_roughness.release()
            self.texture_wetness.release()
            self.texture_direct.release()
            self.texture_indirect.release()
            self.texture_reflection.release()
        
        self.texture_size = (size, size)
        self.texture_roughness = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_wetness = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_direct = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_indirect = self.ctx.texture((size, size), 1, dtype='f4')
        self.texture_reflection = self.ctx.texture((size, size), 1, dtype='f4')
