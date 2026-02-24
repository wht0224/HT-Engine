import moderngl
import numpy as np
from Engine.Scene.SceneNode import SceneNode

class SimpleTerrainRenderer(SceneNode):
    """
    Simple Terrain Renderer using ModernGL.
    Visualizes height and vegetation density from NaturalSystem.
    """
    
    def __init__(self, name: str, natural_system, size=512, grid_size=512):
        super().__init__(name)
        self.system = natural_system
        self.size = size # World size
        self.grid_size = grid_size # Texture/Grid resolution
        self.ctx = None
        self.program = None
        self.vbo = None
        self.ibo = None
        self.vao = None
        self.index_count = 0
        
        # Shader
        self.vertex_shader = """
        #version 330
        
        in vec2 in_pos;
        in vec2 in_uv;
        
        uniform mat4 m_proj;
        uniform mat4 m_view;
        uniform sampler2D heightmap;
        uniform float height_scale;
        
        out vec2 v_uv;
        out float v_height;
        
        void main() {
            v_uv = in_uv;
            
            // Sample height
            float h = texture(heightmap, in_uv).r * height_scale;
            v_height = h;
            
            // Position
            // in_pos is 0..size
            vec3 pos = vec3(in_pos.x, h, in_pos.y);
            
            gl_Position = m_proj * m_view * vec4(pos, 1.0);
        }
        """
        
        self.fragment_shader = """
        #version 330
        
        in vec2 v_uv;
        in float v_height;
        
        uniform sampler2D densitymap;
        
        out vec4 f_color;
        
        void main() {
            float density = texture(densitymap, v_uv).r;
            
            // Photorealistic Earth Tones
            vec3 dirt_color = vec3(0.35, 0.28, 0.22); // Rich Soil
            vec3 sand_color = vec3(0.76, 0.70, 0.50); // Beach Sand
            vec3 rock_color = vec3(0.45, 0.45, 0.48); // Slate Grey
            vec3 grass_color = vec3(0.18, 0.38, 0.08); // Lush Green (Darker)
            vec3 dry_grass_color = vec3(0.60, 0.55, 0.30); // Savannah Yellow
            
            // Height-based blending
            vec3 ground_color;
            if (v_height < 5.0) {
                ground_color = sand_color;
            } else if (v_height > 100.0) {
                ground_color = rock_color;
            } else {
                ground_color = dirt_color;
            }
            
            // Vegetation Mixing
            // If density is high -> lush green
            // If density is low -> dry grass or dirt
            vec3 veg_color = mix(dry_grass_color, grass_color, density);
            
            // Final Mix: Ground vs Vegetation
            // We assume density > 0.1 means some vegetation exists
            float veg_mask = smoothstep(0.0, 0.4, density);
            
            // If high altitude, less vegetation (rocky)
            float rock_mask = smoothstep(80.0, 120.0, v_height);
            veg_mask *= (1.0 - rock_mask);
            
            vec3 final_color = mix(ground_color, veg_color, veg_mask);
            
            // Simple Lighting (Fake Sun)
            // Need normal for real lighting, but we can fake it with height derivative (ddx, ddy)
            vec3 dx = dFdx(vec3(v_uv, v_height));
            vec3 dy = dFdy(vec3(v_uv, v_height));
            vec3 normal = normalize(cross(dx, dy));
            
            vec3 sun_dir = normalize(vec3(0.5, 1.0, 0.3));
            float diff = max(dot(normal, sun_dir), 0.0);
            
            // Ambient + Diffuse
            vec3 ambient = vec3(0.2, 0.2, 0.3); // Blueish sky ambient
            vec3 light = ambient + vec3(1.0, 0.95, 0.8) * diff; // Warm sun
            
            final_color *= light;
            
            // Fog (Aerial Perspective)
            // Simple height fog
            float dist = gl_FragCoord.z / gl_FragCoord.w;
            float fog_factor = exp(-dist * 0.002);
            vec3 fog_color = vec3(0.6, 0.7, 0.8); // Sky blue
            
            final_color = mix(fog_color, final_color, clamp(fog_factor, 0.0, 1.0));
            
            // Gamma Correction
            final_color = pow(final_color, vec3(1.0/2.2));
            
            f_color = vec4(final_color, 1.0);
        }
        """

    def initialize(self, context):
        self.ctx = context
        self.program = self.ctx.program(
            vertex_shader=self.vertex_shader,
            fragment_shader=self.fragment_shader
        )
        
        # Create Grid Mesh
        self._create_mesh()
        
    def _create_mesh(self):
        # Generate grid vertices
        # size x size grid
        # vertices: x, z, u, v
        
        # For a 512x512 grid, we have 512*512 vertices? 
        # Actually 512 cells means 513 vertices.
        # But let's simplify: 1 vertex per pixel for now, or a slightly lower res mesh if needed.
        # If we use 1 vertex per pixel (512x512), that's 262k vertices. Totally fine.
        
        N = self.grid_size
        x = np.linspace(0, self.size, N)
        z = np.linspace(0, self.size, N)
        u = np.linspace(0, 1, N)
        v = np.linspace(0, 1, N)
        
        # Meshgrid
        X, Z = np.meshgrid(x, z)
        U, V = np.meshgrid(u, v)
        
        # Flatten
        X = X.flatten().astype(np.float32)
        Z = Z.flatten().astype(np.float32)
        U = U.flatten().astype(np.float32)
        V = V.flatten().astype(np.float32)
        
        # Interleave: x, z, u, v
        data = np.column_stack((X, Z, U, V))
        
        self.vbo = self.ctx.buffer(data.tobytes())
        
        # Indices (Triangle Strip or Triangles)
        # Using Triangles for simplicity
        indices = []
        # This is slow in Python, but done once.
        # Optimization: Use numpy for indices generation
        
        # Grid indices
        # (i, j) -> i * N + j
        # Quad: (i, j), (i+1, j), (i, j+1), (i+1, j+1)
        
        i = np.arange(N - 1)
        j = np.arange(N - 1)
        I, J = np.meshgrid(i, j)
        
        # Top-left of quad
        p0 = I * N + J
        p1 = p0 + 1
        p2 = p0 + N
        p3 = p2 + 1
        
        # Triangle 1: p0, p1, p2
        # Triangle 2: p1, p3, p2
        
        t1 = np.column_stack((p0.flatten(), p1.flatten(), p2.flatten()))
        t2 = np.column_stack((p1.flatten(), p3.flatten(), p2.flatten()))
        
        indices = np.vstack((t1, t2)).flatten().astype(np.uint32)
        
        self.ibo = self.ctx.buffer(indices.tobytes())
        self.index_count = len(indices)
        
        self.vao = self.ctx.vertex_array(
            self.program,
            [
                (self.vbo, '2f 2f', 'in_pos', 'in_uv')
            ],
            self.ibo
        )
        
    def render(self, camera, light_manager=None):
        if not self.ctx:
            # Try lazy init
             if hasattr(self.system, 'gpu_manager') and self.system.gpu_manager.context:
                 self.initialize(self.system.gpu_manager.context)
             else:
                 return
                 
        # Bind textures
        if hasattr(self.system, 'gpu_manager') and self.system.gpu_manager:
            h_tex = self.system.gpu_manager.get_texture("height")
            d_tex = self.system.gpu_manager.get_texture("vegetation_density")
            
            if h_tex:
                h_tex.use(location=0)
                self.program['heightmap'].value = 0
            
            if d_tex:
                d_tex.use(location=1)
                self.program['densitymap'].value = 1
                
        m_proj = camera.get_projection_matrix()
        m_view = camera.get_view_matrix()
        proj_bytes = np.array(getattr(m_proj, "data", m_proj), dtype=np.float32).reshape(4, 4).T.tobytes()
        view_bytes = np.array(getattr(m_view, "data", m_view), dtype=np.float32).reshape(4, 4).T.tobytes()
        self.program['m_proj'].write(proj_bytes)
        self.program['m_view'].write(view_bytes)
        self.program['height_scale'].value = 1.0 # Already baked in height texture?
        
        self.vao.render(mode=moderngl.TRIANGLES)
