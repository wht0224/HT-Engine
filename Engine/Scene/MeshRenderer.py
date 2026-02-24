import moderngl
import numpy as np
from Engine.Scene.SceneNode import SceneNode
from Engine.Math import Vector3, Matrix4x4
from Engine.Scene.OBJLoader import OBJLoader

class MeshRenderer(SceneNode):
    """
    Renders a 3D mesh (loaded from OBJ) using ModernGL.
    Supports basic diffuse lighting and texture mapping.
    """
    
    def __init__(self, name: str, model_path: str, texture_path: str = None):
        super().__init__(name)
        self.model_path = model_path
        self.texture_path = texture_path
        self.ctx = None
        self.program = None
        self.vbo = None
        self.vao = None
        self.texture = None
        self.num_vertices = 0
        
        # Shader
        self.vertex_shader = """
        #version 330
        
        in vec3 in_pos;
        in vec3 in_norm;
        in vec2 in_uv;
        
        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform mat4 m_model;
        
        out vec3 v_norm;
        out vec2 v_uv;
        out vec3 v_pos;
        
        void main() {
            vec4 pos_world = m_model * vec4(in_pos, 1.0);
            v_pos = pos_world.xyz;
            v_uv = in_uv;
            v_norm = mat3(transpose(inverse(m_model))) * in_norm;
            
            gl_Position = m_proj * m_view * pos_world;
        }
        """
        
        self.fragment_shader = """
        #version 330
        
        in vec3 v_norm;
        in vec2 v_uv;
        in vec3 v_pos;
        
        uniform vec3 color;
        uniform vec3 light_dir;
        uniform vec3 emissive_color;
        uniform float emissive_strength;
        uniform float emissive_enabled;
        
        out vec4 f_color;
        
        void main() {
            vec3 norm = normalize(v_norm);
            vec3 light = normalize(light_dir);
            
            // Diffuse
            float diff = max(dot(norm, light), 0.0);
            
            // Ambient
            float ambient = 0.4;
            
            // Material color (from Material.set_color())
            vec3 material_color = color;
            
            // Emissive glow
            vec3 final_color = material_color * (ambient + diff);
            if (emissive_enabled > 0.5) {
                final_color = mix(final_color, emissive_color, emissive_strength);
            }
            
            // Fog match
            float dist = gl_FragCoord.z / gl_FragCoord.w;
            float fog_factor = exp(-dist * 0.002);
            vec3 fog_color = vec3(0.6, 0.7, 0.8);
            final_color = mix(fog_color, final_color, clamp(fog_factor, 0.0, 1.0));
            
            // Gamma
            final_color = pow(final_color, vec3(1.0/2.2));
            
            f_color = vec4(final_color, 1.0);
        }
        """
        
    def initialize(self, context):
        self.ctx = context
        try:
            self.program = self.ctx.program(
                vertex_shader=self.vertex_shader,
                fragment_shader=self.fragment_shader
            )
        except Exception as e:
            print(f"MeshRenderer Shader Error: {e}")
            print("Using Fallback Shader (Magenta)")
            self.program = self.ctx.program(
                vertex_shader="""
                #version 330
                in vec3 in_pos;
                uniform mat4 m_proj;
                uniform mat4 m_view;
                uniform mat4 m_model;
                void main() {
                    gl_Position = m_proj * m_view * m_model * vec4(in_pos, 1.0);
                }
                """,
                fragment_shader="""
                #version 330
                out vec4 f_color;
                void main() {
                    f_color = vec4(1.0, 0.0, 1.0, 1.0); // MAGENTA ERROR
                }
                """
            )
        
        # Load Model
        data = OBJLoader.load(self.model_path)
        if not data:
            print(f"Failed to load model: {self.model_path}")
            return
            
        vertices = data['vertices']
        normals = data['normals']
        uvs = data['uvs']
        
        self.num_vertices = len(vertices)
        
        # Interleave
        # Format: 3f 3f 2f (pos, norm, uv)
        buffer_data = np.column_stack((vertices, normals, uvs)).astype(np.float32)
        
        self.vbo = self.ctx.buffer(buffer_data.tobytes())
        
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '3f 3f 2f', 'in_pos', 'in_norm', 'in_uv')
            ]
        )
        
    def render(self, camera, light_manager=None):
        if not self.ctx:
            # Try lazy init
             if hasattr(camera, 'ctx') and camera.ctx:
                 self.initialize(camera.ctx)
             # Fallback to finding context from somewhere else or just returning
             # Usually SceneManager should have initialized us or passed context
             if not self.ctx:
                 return
             
        self.ctx.enable(moderngl.DEPTH_TEST)
             
        self.program['m_proj'].write(camera.get_projection_matrix().tobytes())
        self.program['m_view'].write(camera.get_view_matrix().tobytes())
        self.program['m_model'].write(self.get_world_transform().tobytes())
        
        # Lighting
        if 'light_dir' in self.program:
            self.program['light_dir'].value = (0.5, 1.0, 0.3)
        if 'color' in self.program:
            self.program['color'].value = (0.8, 0.8, 0.8) # Grey default
        
        # Emissive glow support
        if 'emissive_enabled' in self.program:
            self.program['emissive_enabled'].value = 1.0
        if 'emissive_color' in self.program:
            self.program['emissive_color'].value = (1.0, 0.9, 0.6)
        if 'emissive_strength' in self.program:
            self.program['emissive_strength'].value = 20.0
        
        self.vao.render(mode=moderngl.TRIANGLES)
