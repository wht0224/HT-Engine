import moderngl
import numpy as np
import ctypes
from Engine.Scene.SceneNode import SceneNode

class SimpleEntityRenderer(SceneNode):
    """
    简单的实体渲染器 (Simple Entity Renderer)
    
    使用 GL_POINTS 渲染实体位置。
    直接从 NaturalSystem 的 FactBase 中读取数据。
    """
    
    def __init__(self, name: str, natural_system, entity_type: str = "herbivore", color=(1.0, 1.0, 1.0), map_size=(512.0, 512.0)):
        super().__init__(name)
        self.system = natural_system
        self.entity_type = entity_type
        self.color = color
        self.map_size = map_size
        self.ctx = None # Will be set during render or init
        
        # OpenGL objects
        self.vbo = None
        self.vao = None
        self.program = None
        self.point_size = 5.0
        
        # Shader
        self.vertex_shader = """
        #version 330
        
        in vec2 in_pos;
        // in float in_heading; // If available
        
        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform vec2 map_size; // Terrain size in world units
        
        uniform sampler2D heightmap;
        
        void main() {
            // Map world pos to UV (0..1)
            // Assuming map_size is the extent of the terrain in world units
            vec2 uv = in_pos / map_size;
            
            // Sample height
            // Note: texture lookup returns 0..1 usually, need real height
            // If texture stores real height (float32), then .r is the height.
            float h = texture(heightmap, uv).r;
            
            // Entities are slightly above ground
            vec3 pos_3d = vec3(in_pos.x, h + 0.5, in_pos.y);
            
            gl_Position = m_proj * m_view * vec4(pos_3d, 1.0);
            gl_PointSize = 5.0;
        }
        """
        
        self.fragment_shader = """
        #version 330
        
        uniform vec3 color;
        out vec4 f_color;
        
        void main() {
            f_color = vec4(color, 1.0);
        }
        """

    def initialize(self, context):
        self.ctx = context
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
    def render(self, camera, light_manager=None):
        if not self.ctx:
            # Try to grab context from system if possible, or wait for explicit init
            if hasattr(self.system, 'gpu_manager') and self.system.gpu_manager.context:
                 self.initialize(self.system.gpu_manager.context)
            else:
                return

        try:
            # 1. Get Data from NaturalSystem
            facts = self.system.engine.facts
            px = facts.get_column(self.entity_type, "pos_x")
            pz = facts.get_column(self.entity_type, "pos_z")
            
            count = len(px)
            if count == 0:
                return
                
            # Interleave
            data = np.column_stack((px, pz)).astype(np.float32)
            
            # 2. Update Buffer
            if not self.vbo or self.vbo.size < data.nbytes:
                if self.vbo: self.vbo.release()
                if self.vao: self.vao.release()
                self.vbo = self.ctx.buffer(data.tobytes())
                self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, 'in_pos')
            else:
                self.vbo.write(data.tobytes())
                
            # 3. Render
            m_proj = camera.get_projection_matrix()
            m_view = camera.get_view_matrix()
            proj_bytes = np.array(getattr(m_proj, "data", m_proj), dtype=np.float32).reshape(4, 4).T.tobytes()
            view_bytes = np.array(getattr(m_view, "data", m_view), dtype=np.float32).reshape(4, 4).T.tobytes()
            self.program['m_proj'].write(proj_bytes)
            self.program['m_view'].write(view_bytes)
            self.program['color'].value = self.color
            self.program['map_size'].value = (float(self.map_size[0]), float(self.map_size[1]))
            
            # Bind Heightmap
            if hasattr(self.system, 'gpu_manager') and self.system.gpu_manager:
                h_tex = self.system.gpu_manager.get_texture("height")
                if h_tex:
                    h_tex.use(location=0)
                    self.program['heightmap'].value = 0
            
            # Disable depth test? No, points should be occluded.
            # But enable depth test.
            
            self.vao.render(mode=moderngl.POINTS)
            
        except Exception as e:
            # print(f"Render Error: {e}")
            pass
