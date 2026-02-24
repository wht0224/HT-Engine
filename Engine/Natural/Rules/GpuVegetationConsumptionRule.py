import numpy as np
import moderngl
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GpuVegetationConsumptionRule(Rule):
    """
    GPU 植被消耗规则 (GPU Vegetation Consumption Rule)
    
    处理实体对植被的消耗（吃草）。
    这是一个 Compute Shader，接收实体位置列表，并在 GPU 上直接修改植被纹理。
    """
    
    def __init__(self, context=None, manager=None):
        super().__init__("Interaction.VegetationConsumption", priority=35) # Run after Growth
        self.manager = manager
        self.ctx = None
        
        # Initialize Context (Lazy if needed)
        if manager:
            self.ctx = manager.context
        elif context:
            self.ctx = context
            
        # Compute Shader for "Stamping"
        self.compute_shader_source = """
        #version 430
        layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;
        
        layout(r32f, binding = 0) uniform image2D density_map;
        
        // Entity data: x, z, is_eating, padding
        struct Entity {
            float x;
            float z;
            float is_eating;
            float padding;
        };
        
        layout(std430, binding = 1) buffer EntityBuffer {
            Entity entities[];
        };
        
        uniform int entity_count;
        uniform float eat_amount;
        
        void main() {
            uint id = gl_GlobalInvocationID.x;
            if (id >= entity_count) return;
            
            Entity e = entities[id];
            
            // Filter: Only process if eating
            if (e.is_eating < 0.5) return;
            
            vec2 pos_world = vec2(e.x, e.z);
            ivec2 center_uv = ivec2(pos_world);
            
            // Simple 1-pixel brush
            // Check bounds
            ivec2 size = imageSize(density_map);
            if (center_uv.x < 0 || center_uv.x >= size.x || center_uv.y < 0 || center_uv.y >= size.y) return;
            
            float current = imageLoad(density_map, center_uv).r;
            float new_val = max(0.0, current - eat_amount);
            imageStore(density_map, center_uv, vec4(new_val, 0.0, 0.0, 0.0));
        }
        """
        self.program = None
        self.entity_buffer = None
        self.max_entities = 10000 # Buffer capacity
        self.eat_amount = 0.5 # Consumption rate per frame
        
    def _init_gl(self):
        """Compile shader and create buffers"""
        if not self.ctx:
            return False
            
        try:
            self.program = self.ctx.compute_shader(self.compute_shader_source)
            # Create buffer (10000 * 16 bytes for vec4)
            self.entity_buffer = self.ctx.buffer(reserve=self.max_entities * 16) 
            return True
        except Exception as e:
            print(f"Failed to init GpuVegetationConsumptionRule: {e}")
            return False
            
    def evaluate(self, facts: FactBase):
        # Lazy Init Context
        if not self.ctx and self.manager and self.manager.context:
            self.ctx = self.manager.context
            
        if not self.ctx:
            return
            
        if not self.program:
            if not self._init_gl():
                return
                
        # Get texture
        density_tex = self.manager.get_texture("vegetation_density")
        if not density_tex:
            return

        # 1. Check if we have entities to process
        if "herbivore" not in facts.tables:
            return
            
        count = facts.get_count("herbivore")
        if count == 0:
            return
            
        # Extract columns
        # We need x, z, is_eating
        # FactBase stores SoA as numpy arrays
        try:
            # We can't easily interleave them efficiently in Python loop
            # But numpy stack is fast enough for <100k entities
            
            # Access raw arrays if possible for speed, or use get_column
            # get_column returns a copy usually, but check implementation
            # For performance, direct access to storage is better if permitted.
            # facts.tables[name][col]
            
            # Use safe access first
            pos_x = facts.get_column("herbivore", "pos_x")
            pos_z = facts.get_column("herbivore", "pos_z")
            try:
                is_eating = facts.get_column("herbivore", "is_eating")
            except Exception:
                is_eating = np.zeros_like(pos_x, dtype=np.float32)
            
            if pos_x is None or is_eating is None:
                return
                
            # Stack into (N, 4) float32
            # padding can be 0
            # np.column_stack is good
            
            # Optimization: Only upload if count > 0
            # Data: x, z, is_eating, 0.0
            data = np.column_stack((pos_x, pos_z, is_eating, np.zeros_like(pos_x, dtype=np.float32))).astype(np.float32)
            
            # Upload to buffer
            # Check size
            current_bytes = data.nbytes
            if current_bytes > self.entity_buffer.size:
                # Resize buffer
                self.entity_buffer.orphan(current_bytes * 2) # Double it
                self.max_entities = (current_bytes * 2) // 16
                
            self.entity_buffer.write(data.tobytes())
            
            # Bind resources
            density_tex.bind_to_image(0, read=True, write=True)
            self.entity_buffer.bind_to_storage_buffer(1)
            
            # Set uniforms
            if 'entity_count' in self.program:
                self.program['entity_count'].value = count
            if 'eat_amount' in self.program:
                self.program['eat_amount'].value = self.eat_amount
                
            # Dispatch
            # Group size is 64
            groups = (count + 63) // 64
            self.program.run(groups, 1, 1)
            
            # Ensure synchronization? 
            # Usually MemoryBarrier is handled by driver between dispatches if accessing same resource
            # But if next rule reads it, it should be fine.
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Error in GpuVegetationConsumptionRule: {e}")
