import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class GpuCullingRule(Rule):
    """
    GPU 遮挡剔除规则 (GPU Culling Rule)
    
    使用 ModernGL Compute Shader 实现Hi-Z遮挡剔除和视锥体裁剪。
    针对 GTX 1650 Max-Q 优化，支持30M三角形场景。
    
    优先级: 110 (最高优先级，在其他渲染规则之前执行)
    """
    
    def __init__(self, max_objects=100000, context=None, manager=None, 
                 table_name: str = "scene_objects", use_shared_textures: bool = True):
        super().__init__("Render.Culling", priority=110)
        self.max_objects = max_objects
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
                print(f"[GpuCullingRule] Failed to create OpenGL context: {e}")
                self.ctx = None
                return
        
        self.compute_shader_source = """
        #version 430
        
        layout(local_size_x = 64) in;
        
        struct AABB {
            vec3 min_pos;
            vec3 max_pos;
        };
        
        struct DrawCommand {
            uint indexCount;
            uint instanceCount;
            uint firstIndex;
            uint baseVertex;
            uint baseInstance;
        };
        
        layout(std430, binding = 0) readonly buffer ObjectData {
            vec4 positions[];  // xyz = center, w = radius
        };
        
        layout(std430, binding = 1) readonly buffer AABBData {
            vec4 aabbs[];  // min.xyz, max.xyz packed
        };
        
        layout(std430, binding = 2) writeonly buffer VisibleObjects {
            uint visible_indices[];
        };
        
        layout(std430, binding = 3) buffer DrawCommands {
            DrawCommand commands[];
        };
        
        layout(r32f, binding = 4) readonly uniform image2D depth_pyramid;
        
        uniform mat4 view_matrix;
        uniform mat4 projection_matrix;
        uniform vec4 frustum_planes[6];
        uniform uint object_count;
        uniform uint depth_pyramid_levels;
        uniform vec2 screen_size;
        
        bool frustum_cull(vec3 center, float radius) {
            for (int i = 0; i < 6; i++) {
                float dist = dot(frustum_planes[i].xyz, center) + frustum_planes[i].w;
                if (dist < -radius) return false;
            }
            return true;
        }
        
        bool occlusion_cull(vec3 center, float radius, vec3 aabb_min, vec3 aabb_max) {
            vec4 clip_pos = projection_matrix * view_matrix * vec4(center, 1.0);
            if (clip_pos.w <= 0.0) return false;
            
            vec3 ndc = clip_pos.xyz / clip_pos.w;
            vec2 screen_pos = ndc.xy * 0.5 + 0.5;
            
            if (screen_pos.x < 0.0 || screen_pos.x > 1.0 ||
                screen_pos.y < 0.0 || screen_pos.y > 1.0) {
                return true;
            }
            
            vec4 clip_min = projection_matrix * view_matrix * vec4(aabb_min, 1.0);
            vec4 clip_max = projection_matrix * view_matrix * vec4(aabb_max, 1.0);
            
            if (clip_min.w <= 0.0 && clip_max.w <= 0.0) return false;
            
            float min_depth = min(clip_min.z / clip_min.w, clip_max.z / clip_max.w);
            min_depth = min_depth * 0.5 + 0.5;
            
            ivec2 pyramid_size = imageSize(depth_pyramid);
            float depth_sample = imageLoad(depth_pyramid, ivec2(screen_pos * vec2(pyramid_size))).r;
            
            return min_depth <= depth_sample + 0.01;
        }
        
        void main() {
            uint idx = gl_GlobalInvocationID.x;
            if (idx >= object_count) return;
            
            vec4 pos_radius = positions[idx];
            vec3 center = pos_radius.xyz;
            float radius = pos_radius.w;
            
            bool visible = true;
            
            if (visible) {
                visible = frustum_cull(center, radius);
            }
            
            if (visible && depth_pyramid_levels > 0) {
                vec4 aabb_data = aabbs[idx];
                vec3 aabb_min = aabb_data.xyz;
                vec3 aabb_max = vec3(aabb_data.w, aabbs[idx + 1].xy);
                visible = occlusion_cull(center, radius, aabb_min, aabb_max);
            }
            
            if (visible) {
                uint out_idx = atomicAdd(commands[0].instanceCount, 1);
                if (out_idx < object_count) {
                    visible_indices[out_idx] = idx;
                }
            }
        }
        """
        
        self.program = None
        self.buffer_positions = None
        self.buffer_aabbs = None
        self.buffer_visible = None
        self.buffer_commands = None
        self.texture_depth_pyramid = None
        self.depth_pyramid_levels = 0
        self._initialized = False
    
    def evaluate(self, facts: FactBase):
        if not self.ctx:
            return
        
        try:
            positions = facts.get_column(self.table_name, "positions")
            object_count = len(positions) // 4
            
            if object_count == 0:
                return
            
            if not self._initialized:
                self._init_buffers(object_count)
                if self.program is None:
                    self.program = self.ctx.compute_shader(self.compute_shader_source)
                self._initialized = True
            
            self.buffer_positions.write(positions.astype(np.float32).tobytes())
            
            try:
                aabbs = facts.get_column(self.table_name, "aabbs")
                self.buffer_aabbs.write(aabbs.astype(np.float32).tobytes())
            except KeyError:
                pass
            
            view_matrix = facts.get_global("view_matrix")
            if view_matrix is None:
                view_matrix = np.eye(4, dtype=np.float32)
            proj_matrix = facts.get_global("projection_matrix")
            if proj_matrix is None:
                proj_matrix = np.eye(4, dtype=np.float32)
            
            frustum_planes = self._extract_frustum_planes(view_matrix, proj_matrix)
            
            self.program['view_matrix'].value = tuple(view_matrix.T.flatten())
            self.program['projection_matrix'].value = tuple(proj_matrix.T.flatten())
            self.program['object_count'].value = object_count
            self.program['depth_pyramid_levels'].value = self.depth_pyramid_levels
            self.program['screen_size'].value = (1920.0, 1080.0)
            
            for i, plane in enumerate(frustum_planes):
                self.program[f'frustum_planes[{i}]'].value = tuple(plane)
            
            self.buffer_positions.bind_to_storage_buffer(0)
            self.buffer_aabbs.bind_to_storage_buffer(1)
            self.buffer_visible.bind_to_storage_buffer(2)
            self.buffer_commands.bind_to_storage_buffer(3)
            
            if self.texture_depth_pyramid:
                self.texture_depth_pyramid.bind_to_image(4, read=True, write=False)
            
            cmd_data = np.array([0, 0, 0, 0, 0], dtype=np.uint32)
            self.buffer_commands.write(cmd_data.tobytes())
            
            groups = (object_count + 63) // 64
            self.program.run(groups, 1, 1)
            
            visible_count = np.frombuffer(self.buffer_commands.read(4), dtype=np.uint32)[0]
            facts.set_global("visible_object_count", int(visible_count))
            
            if self.manager:
                self.manager.register_buffer("visible_indices", self.buffer_visible)
                self.manager.register_buffer("draw_commands", self.buffer_commands)
            
        except KeyError:
            pass
    
    def _extract_frustum_planes(self, view, proj):
        vp = proj @ view
        planes = []
        
        planes.append(vp[3, :] + vp[0, :])
        planes.append(vp[3, :] - vp[0, :])
        planes.append(vp[3, :] + vp[1, :])
        planes.append(vp[3, :] - vp[1, :])
        planes.append(vp[3, :] + vp[2, :])
        planes.append(vp[3, :] - vp[2, :])
        
        normalized_planes = []
        for plane in planes:
            length = np.sqrt(plane[0]**2 + plane[1]**2 + plane[2]**2)
            if length > 0:
                normalized_planes.append(plane / length)
            else:
                normalized_planes.append(plane)
        
        return normalized_planes
    
    def _init_buffers(self, max_objects):
        if self.buffer_positions:
            self.buffer_positions.release()
            self.buffer_aabbs.release()
            self.buffer_visible.release()
            self.buffer_commands.release()
        
        self.buffer_positions = self.ctx.buffer(reserve=max_objects * 16)
        self.buffer_aabbs = self.ctx.buffer(reserve=max_objects * 32)
        self.buffer_visible = self.ctx.buffer(reserve=max_objects * 4)
        self.buffer_commands = self.ctx.buffer(reserve=20)
    
    def set_depth_pyramid(self, depth_texture, levels):
        self.texture_depth_pyramid = depth_texture
        self.depth_pyramid_levels = levels
