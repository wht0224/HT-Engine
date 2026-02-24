import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuLodRule(Rule):
    """
    GPU LOD计算规则 (GPU LOD Rule)
    
    使用 ModernGL Compute Shader 实现基于屏幕空间误差的LOD选择。
    针对30M三角形场景优化，支持动态LOD切换。
    
    优先级: 105 (在剔除之后，渲染之前执行)
    """
    
    def __init__(self, max_objects=100000, lod_levels=5, context=None, manager=None,
                 table_name: str = "scene_objects", use_shared_textures: bool = True):
        super().__init__("Render.LOD", priority=105)
        self.max_objects = max_objects
        self.lod_levels = lod_levels
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
                print(f"[GpuLodRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 64) in;
        
        struct LODInfo {
            uint lod_level;
            float screen_error;
            float triangle_count;
        };
        
        layout(std430, binding = 0) readonly buffer PositionsBuffer {
            vec4 positions[];
        };
        
        layout(std430, binding = 4) readonly buffer LodParamsBuffer {
            vec4 lod_params[];
        };
        
        layout(std430, binding = 1) readonly buffer VisibleIndices {
            uint visible_indices[];
        };
        
        layout(std430, binding = 2) writeonly buffer LODResults {
            LODInfo lod_results[];
        };
        
        layout(std430, binding = 3) buffer LODStats {
            uint total_triangles;
            uint lod_counts[8];
        };
        
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform vec3 camera_position;
        uniform vec2 screen_size;
        uniform float pixel_error_threshold;
        uniform uint visible_count;
        uniform float lod_distances[8];
        
        float calculate_screen_error(vec3 center, float radius, float base_error) {
            vec4 view_pos = view_matrix * vec4(center, 1.0);
            float distance = length(view_pos.xyz);
            
            if (distance < 0.1) distance = 0.1;
            
            float projected_radius = radius / distance;
            
            float fov = 1.0 / projection_matrix[1][1];
            float screen_height = screen_size.y;
            float pixels_per_radian = screen_height * 0.5;
            
            float screen_pixels = projected_radius * pixels_per_radian * fov;
            
            float screen_error = (base_error / radius) * screen_pixels;
            
            return screen_error;
        }
        
        uint select_lod_level(float distance, float screen_error, vec4 lod_param) {
            uint min_lod = uint(lod_param.z);
            uint max_lod = uint(lod_param.w);
            
            uint lod = 0;
            
            for (int i = 0; i < 8; i++) {
                if (distance < lod_distances[i]) {
                    lod = uint(i);
                    break;
                }
                lod = uint(i);
            }
            
            if (screen_error < pixel_error_threshold * 0.5) {
                lod = min(lod + 1, max_lod);
            } else if (screen_error > pixel_error_threshold * 2.0) {
                lod = max(lod - 1, min_lod);
            }
            
            lod = clamp(lod, min_lod, max_lod);
            
            return lod;
        }
        
        float get_triangle_count_for_lod(uint lod, float base_triangles) {
            float factor = 1.0 / float(1 << (lod * 2));
            return base_triangles * factor;
        }
        
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= visible_count) return;
            
            uint obj_idx = visible_indices[idx];
            
            int pos_idx = int(obj_idx);
            vec4 pos_radius = positions[pos_idx];
            vec4 lod_param = lod_params[pos_idx];
            
            vec3 center = pos_radius.xyz;
            float radius = pos_radius.w;
            float base_error = lod_param.x;
            float base_triangles = lod_param.y;
            
            vec4 view_pos = view_matrix * vec4(center, 1.0);
            float distance = length(view_pos.xyz);
            
            float screen_error = calculate_screen_error(center, radius, base_error);
            
            uint lod_level = select_lod_level(distance, screen_error, lod_param);
            
            float tri_count = get_triangle_count_for_lod(lod_level, base_triangles);
            
            lod_results[idx].lod_level = lod_level;
            lod_results[idx].screen_error = screen_error;
            lod_results[idx].triangle_count = tri_count;
            
            atomicAdd(total_triangles, uint(tri_count));
            atomicAdd(lod_counts[lod_level], 1);
        }
        """
        
        self.program = None
        self.buffer_positions = None
        self.buffer_lod_params = None
        self.buffer_visible = None
        self.buffer_lod_results = None
        self.buffer_lod_stats = None
        self._initialized = False
        
        self.default_lod_distances = np.array([
            50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0
        ], dtype=np.float32)
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            visible_count = facts.get_global("visible_object_count")
            if visible_count is None or visible_count == 0:
                return
            
            if not self._initialized:
                self._init_buffers()
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
                self._initialized = True
            
            view_matrix = facts.get_global("view_matrix")
            if view_matrix is None:
                view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = facts.get_global("projection_matrix")
            if proj_matrix is None:
                proj_matrix = np.eye(4, dtype=np.float32)
            
            camera_pos = facts.get_global("camera_position")
            if camera_pos is None:
                camera_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
            screen_size = facts.get_global("screen_size")
            if screen_size is None:
                screen_size = (1920.0, 1080.0)
            
            pixel_threshold = facts.get_global("pixel_error_threshold")
            if pixel_threshold is None:
                pixel_threshold = 4.0
            
            lod_distances = facts.get_global("lod_distances")
            if lod_distances is None:
                lod_distances = self.default_lod_distances
            
            self.program['view_matrix'].value = tuple(view_matrix.T.flatten())
            self.program['projection_matrix'].value = tuple(proj_matrix.T.flatten())
            self.program['camera_position'].value = tuple(camera_pos)
            self.program['screen_size'].value = screen_size
            self.program['pixel_error_threshold'].value = float(pixel_threshold)
            self.program['visible_count'].value = int(visible_count)
            
            for i, dist in enumerate(lod_distances[:8]):
                self.program[f'lod_distances[{i}]'].value = float(dist)
            
            self.buffer_positions.bind_to_storage_buffer(0)
            self.buffer_visible.bind_to_storage_buffer(1)
            self.buffer_lod_results.bind_to_storage_buffer(2)
            self.buffer_lod_stats.bind_to_storage_buffer(3)
            self.buffer_lod_params.bind_to_storage_buffer(4)
            
            stats_data = np.zeros(9, dtype=np.uint32)
            self.buffer_lod_stats.write(stats_data.tobytes())
            
            groups = (visible_count + 63) // 64
            self.program.run(groups, 1, 1)
            
            stats = np.frombuffer(self.buffer_lod_stats.read(36), dtype=np.uint32)
            total_triangles = int(stats[0])
            lod_counts = stats[1:9]
            
            facts.set_global("lod_total_triangles", total_triangles)
            facts.set_global("lod_counts", lod_counts.tolist())
            
            if self.manager:
                self.manager.register_buffer("lod_results", self.buffer_lod_results)
            
        except KeyError:
            pass
    
    def _init_buffers(self):
        if self.buffer_positions:
            self.buffer_positions.release()
            self.buffer_lod_params.release()
            self.buffer_visible.release()
            self.buffer_lod_results.release()
            self.buffer_lod_stats.release()
        
        max_obj = self.max_objects
        self.buffer_positions = self.ctx.buffer(reserve=max_obj * 32)
        self.buffer_lod_params = self.ctx.buffer(reserve=max_obj * 16)
        self.buffer_visible = self.ctx.buffer(reserve=max_obj * 4)
        self.buffer_lod_results = self.ctx.buffer(reserve=max_obj * 12)
        self.buffer_lod_stats = self.ctx.buffer(reserve=36)
    
    def set_object_data(self, positions, lod_params):
        if self.buffer_positions:
            self.buffer_positions.write(positions.astype(np.float32).tobytes())
            self.buffer_lod_params.write(lod_params.astype(np.float32).tobytes())
    
    def set_visible_indices(self, visible_buffer):
        self.buffer_visible = visible_buffer
