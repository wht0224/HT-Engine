import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuBloomRule(Rule):
    """
    GPU 泛光效果规则 (GPU Bloom Rule)
    
    使用 ModernGL Compute Shader 实现高性能泛光后处理。
    针对 GTX 1650 Max-Q 优化，支持降采样和模糊分离。
    
    优先级: 70 (后处理阶段)
    """
    
    def __init__(self, threshold=0.8, intensity=0.5, blur_radius=4, 
                 downsample_factor=4, context=None, manager=None,
                 table_name: str = "postprocess", use_shared_textures: bool = True):
        super().__init__("PostProcess.Bloom", priority=70)
        self.threshold = threshold
        self.intensity = intensity
        self.blur_radius = blur_radius
        self.downsample_factor = downsample_factor
        self.manager = manager
        self.table_name = table_name
        self.use_shared_textures = use_shared_textures
        
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
        else:
            try:
                self.ctx = moderngl.create_context(standalone=True)
            except Exception as e:
                print(f"[GpuBloomRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.brightness_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(rgba16f, binding = 0) readonly uniform image2D input_image;
        layout(r16f, binding = 1) writeonly uniform image2D bright_image;
        
        uniform float threshold;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            vec4 color = imageLoad(input_image, pos);
            
            float brightness = dot(color.rgb, vec3(0.299, 0.587, 0.114));
            
            if (brightness > threshold) {
                float contribution = (brightness - threshold) / (1.0 - threshold);
                vec3 bright_color = color.rgb * contribution;
                imageStore(bright_image, pos, vec4(bright_color, 1.0));
            } else {
                imageStore(bright_image, pos, vec4(0.0, 0.0, 0.0, 1.0));
            }
        }
        """
        
        self.blur_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r16f, binding = 0) readonly uniform image2D input_image;
        layout(r16f, binding = 1) writeonly uniform image2D output_image;
        
        uniform vec2 direction;
        uniform int radius;
        
        float weights[9] = float[](
            0.0162162162, 0.0540540541, 0.1216216216,
            0.1945945946, 0.2270270270, 0.1945945946,
            0.1216216216, 0.0540540541, 0.0162162162
        );
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(input_image);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            float sum = 0.0;
            float total_weight = 0.0;
            
            for (int i = -4; i <= 4; i++) {
                ivec2 sample_pos = pos + ivec2(direction * float(i));
                
                sample_pos.x = clamp(sample_pos.x, 0, size.x - 1);
                sample_pos.y = clamp(sample_pos.y, 0, size.y - 1);
                
                float sample_val = imageLoad(input_image, sample_pos).r;
                sum += sample_val * weights[i + 4];
                total_weight += weights[i + 4];
            }
            
            imageStore(output_image, pos, vec4(sum / total_weight, 0.0, 0.0, 1.0));
        }
        """
        
        self.upsample_shader = """
        #version 430
        
        layout(local_size_x = 16, local_size_y = 16) in;
        
        layout(r16f, binding = 0) readonly uniform image2D bloom_low;
        layout(rgba16f, binding = 1) readonly uniform image2D original;
        layout(rgba16f, binding = 2) writeonly uniform image2D output_image;
        
        uniform float intensity;
        uniform float factor;
        
        void main() {
            ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
            ivec2 size = imageSize(original);
            
            if (pos.x >= size.x || pos.y >= size.y) return;
            
            ivec2 low_size = imageSize(bloom_low);
            vec2 uv = vec2(pos) / vec2(size);
            ivec2 low_pos = ivec2(uv * vec2(low_size));
            
            low_pos = clamp(low_pos, ivec2(0), low_size - 1);
            
            float bloom_val = imageLoad(bloom_low, low_pos).r;
            vec4 original_color = imageLoad(original, pos);
            
            vec3 bloom_color = vec3(bloom_val) * intensity * factor;
            vec3 final_color = original_color.rgb + bloom_color;
            
            final_color = clamp(final_color, 0.0, 1.0);
            
            imageStore(output_image, pos, vec4(final_color, 1.0));
        }
        """
        
        self.program_brightness = None
        self.program_blur = None
        self.program_upsample = None
        
        self.texture_input = None
        self.texture_bright = None
        self.texture_blur_h = None
        self.texture_blur_v = None
        self.texture_output = None
        
        self.texture_size = (0, 0)
        self._initialized = False
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            input_data = facts.get_column(self.table_name, "color_buffer")
            if input_data is None:
                return
            
            height, width = input_data.shape[:2] if input_data.ndim == 3 else (int(np.sqrt(len(input_data)//4)), int(np.sqrt(len(input_data)//4)))
            
            if not self._initialized or self.texture_size != (width, height):
                self._init_textures(width, height)
                if self.program_brightness is None:
                    self.program_brightness = self.ctx.compute_shader(self.brightness_shader)
                    self.program_blur = self.ctx.compute_shader(self.blur_shader)
                    self.program_upsample = self.ctx.compute_shader(self.upsample_shader)
                self._initialized = True
            
            if input_data.ndim == 1:
                input_data = input_data.reshape((height, width, 4))
            
            self.texture_input.write(input_data.astype(np.float32).tobytes())
            
            self._extract_brightness(width, height)
            
            low_w = width // self.downsample_factor
            low_h = height // self.downsample_factor
            
            self._apply_blur(low_w, low_h)
            
            self._upsample_and_blend(width, height)
            
            if self.manager:
                self.manager.register_texture("bloom_result", self.texture_output)
            
            if facts.get_global("enable_bloom_readback"):
                result = np.frombuffer(self.texture_output.read(), dtype=np.float32).reshape((height, width, 4))
                facts.set_column(self.table_name, "bloom_result", result)
            
        except KeyError:
            pass
    
    def _extract_brightness(self, width, height):
        self.texture_input.bind_to_image(0, read=True, write=False)
        self.texture_bright.bind_to_image(1, read=False, write=True)
        
        self.program_brightness['threshold'].value = self.threshold
        
        nx = (width + 15) // 16
        ny = (height + 15) // 16
        self.program_brightness.run(nx, ny, 1)
    
    def _apply_blur(self, width, height):
        self.texture_bright.bind_to_image(0, read=True, write=False)
        self.texture_blur_h.bind_to_image(1, read=False, write=True)
        
        self.program_blur['direction'].value = (1.0, 0.0)
        self.program_blur['radius'].value = self.blur_radius
        
        nx = (width + 15) // 16
        ny = (height + 15) // 16
        self.program_blur.run(nx, ny, 1)
        
        self.texture_blur_h.bind_to_image(0, read=True, write=False)
        self.texture_blur_v.bind_to_image(1, read=False, write=True)
        
        self.program_blur['direction'].value = (0.0, 1.0)
        self.program_blur.run(nx, ny, 1)
    
    def _upsample_and_blend(self, width, height):
        self.texture_blur_v.bind_to_image(0, read=True, write=False)
        self.texture_input.bind_to_image(1, read=True, write=False)
        self.texture_output.bind_to_image(2, read=False, write=True)
        
        self.program_upsample['intensity'].value = self.intensity
        self.program_upsample['factor'].value = 1.0
        
        nx = (width + 15) // 16
        ny = (height + 15) // 16
        self.program_upsample.run(nx, ny, 1)
    
    def _init_textures(self, width, height):
        if self.texture_input:
            self.texture_input.release()
            self.texture_bright.release()
            self.texture_blur_h.release()
            self.texture_blur_v.release()
            self.texture_output.release()
        
        self.texture_size = (width, height)
        
        self.texture_input = self.ctx.texture((width, height), 4, dtype='f4')
        self.texture_bright = self.ctx.texture((width, height), 1, dtype='f4')
        self.texture_output = self.ctx.texture((width, height), 4, dtype='f4')
        
        low_w = width // self.downsample_factor
        low_h = height // self.downsample_factor
        self.texture_blur_h = self.ctx.texture((low_w, low_h), 1, dtype='f4')
        self.texture_blur_v = self.ctx.texture((low_w, low_h), 1, dtype='f4')
    
    def set_parameters(self, threshold=None, intensity=None, blur_radius=None):
        if threshold is not None:
            self.threshold = threshold
        if intensity is not None:
            self.intensity = intensity
        if blur_radius is not None:
            self.blur_radius = blur_radius
