import sys
import os
import time
import numpy as np
import moderngl

# Setup path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Engine.Engine import Engine
from Engine.Scene.Camera import Camera
from Engine.Scene.Light import DirectionalLight
from Engine.Math import Vector3
from Engine.Natural.NaturalSystem import NaturalSystem
from Engine.Natural.Visualization.SimpleEntityRenderer import SimpleEntityRenderer
from Engine.Natural.Visualization.SimpleTerrainRenderer import SimpleTerrainRenderer

def main():
    print("Initializing Integrated Ecological Demo...")
    
    # 1. Initialize Engine
    engine = Engine()
    
    # Use Flight Sim style camera for overview
    config = {
        "frontend": {
            "type": "tkinter",
            "ui_type": "game",
            "enable_html": False,
            "enable_tkinter": False # DISABLE DEFAULT UI CREATION
        },
        "renderer": {
            "enable_render_result": True
        }
    }
    
    # Skip standard scene creation to customize
    # But Engine.initialize calls _create_default_scene automatically.
    # We will modify it after init.
    engine.initialize(config)
    
    if not engine.is_initialized:
        print("Engine failed to initialize.")
        return
        
    # 2. Setup Natural System
    print("Setting up Natural System...")
    nat_config = {
        'enable_grazing': True,
        'enable_vegetation_growth': True,
        'enable_erosion': False,
        'use_gpu_vegetation': True,
        'enable_gpu_readback': True, # Essential for GrazingRule to see the grass
        'enable_lazy_gpu': True      # Deferred GPU init to share context with Tkinter
    }
    natural_system = NaturalSystem(nat_config)
    
    # Inject Natural System into Engine (if supported, or just keep reference)
    # Engine.py doesn't have native slot for NaturalSystem yet, so we manage it here.
    
    # Define Deferred Renderer Initialization Hook
    # This will be called by OpenGLViewport.initgl() when the context is ready
    def initialize_renderer_deferred():
        print("Deferred Initialization Hook Called!")
        
        # 1. Get Context from Viewport
        ctx = None
        if hasattr(engine, 'tk_ui') and engine.tk_ui and hasattr(engine.tk_ui, 'viewport'):
             ctx = engine.tk_ui.viewport.ctx
        elif hasattr(engine, 'main_window') and hasattr(engine.main_window, 'viewport'):
             ctx = engine.main_window.viewport.ctx
             
        if ctx:
             print("Found Context in Viewport! Injecting into NaturalSystem...")
             natural_system.gpu_manager.set_context(ctx)
             
             # 2. Register Textures (moved from below)
             print("Registering GPU Textures...")
             
             # Height Texture
             h_tex = ctx.texture((size, size), 1, dtype='f4')
             h_tex.write(height_map.astype('f4').tobytes())
             natural_system.gpu_manager.register_texture("height", h_tex)
             
             # Vegetation Density Texture (Initial)
             d_tex = ctx.texture((size, size), 1, dtype='f4')
             d_tex.write(np.ones(size*size, dtype='f4').tobytes())
             natural_system.gpu_manager.register_texture("vegetation_density", d_tex)
             
             print("Textures registered successfully.")
             return True
        else:
            print("Warning: Could not find OpenGL Context in Viewport.")
            return False

    # Monkey-patch the hook onto the engine instance
    engine.initialize_renderer_deferred = initialize_renderer_deferred

    # 3. Create Terrain & Entities
    print("Creating World...")
    size = 512
    
    # Terrain (Sin wave)
    x = np.linspace(0, 10, size)
    z = np.linspace(0, 10, size)
    X, Z = np.meshgrid(x, z)
    height_map = np.sin(X) * np.cos(Z) * 5.0 + 20.0 # Height 15-25
    natural_system.engine.facts.create_table("terrain_main", size*size, {
        'height': np.float32,
        'water': np.float32,
        'vegetation_density': np.float32,
        'slope': np.float32
    })
    
    # Set count to full size as it is a dense grid
    natural_system.engine.facts.set_count("terrain_main", size*size)
    
    # Load initial data
    # Note: We need to use TerrainLoader or manually set columns.
    # FactBase is SoA.
    natural_system.engine.facts.set_column("terrain_main", "height", height_map.flatten().astype(np.float32))
    # Seed some vegetation
    natural_system.engine.facts.set_column("terrain_main", "vegetation_density", np.ones(size*size, dtype=np.float32))
    
    # Register Height Texture for GPU Rules & Visualization
    # Moved to initialize_renderer_deferred to wait for Context
    # if natural_system.gpu_manager and natural_system.gpu_manager.context: ...
    
    # Entities
    num_entities = 1000
    print(f"Spawning {num_entities} herbivores...")
    
    # Create Table
    schema = {
        'pos_x': np.float32,
        'pos_z': np.float32,
        'vel_x': np.float32,
        'vel_z': np.float32,
        'hunger': np.float32,
        'is_eating': np.float32,
        'heading': np.float32
    }
    natural_system.engine.facts.create_table("herbivore", num_entities, schema)
    natural_system.engine.facts.set_count("herbivore", num_entities)
    
    # Random Positions (Center cluster)
    rng = np.random.default_rng()
    pos_x = rng.uniform(200, 300, num_entities).astype(np.float32) # Grid coords (0..512)
    pos_z = rng.uniform(200, 300, num_entities).astype(np.float32)
    
    natural_system.engine.facts.set_column("herbivore", "pos_x", pos_x)
    natural_system.engine.facts.set_column("herbivore", "pos_z", pos_z)
    natural_system.engine.facts.set_column("herbivore", "hunger", np.ones(num_entities, dtype=np.float32)*0.8) # Very hungry
    
    # 4. Setup Visualization
    print("Setting up Visualization...")
    scene_mgr = engine.scene_mgr
    
    # Disable LightingRule to avoid crash and focus on Grazing
    print("Disabling LightingRule for stability...")
    natural_system.engine.rules.rules = [r for r in natural_system.engine.rules.rules if "Lighting" not in r.name]
    
    # Add Terrain Renderer
    # terrain_node = SimpleTerrainRenderer("Terrain", natural_system, size=512, grid_size=512)
    # scene_mgr.root_node.add_child(terrain_node)
    
    # Add Entity Renderer
    # entity_node = SimpleEntityRenderer("Herbivores", natural_system, "herbivore", color=(1.0, 1.0, 1.0)) # White dots
    # scene_mgr.root_node.add_child(entity_node)
    
    # 4.5 3D Model Placeholder
    from Engine.Scene.MeshRenderer import MeshRenderer
    # Use procedural tree instead of waiting for friend
    model_node = MeshRenderer("ProceduralTree", "procedural_tree.obj")
    # Position explanation:
    # Terrain height at center (256, 256) is roughly 20-30 based on noise.
    # We place it at y=50 to be safe (floating tree is better than buried tree)
    # Scale: 20x to be HUGE landmark
    model_node.set_position(Vector3(256, 40, 256)) 
    model_node.set_scale(Vector3(20, 20, 20)) 
    scene_mgr.root_node.add_child(model_node)
    
    # Adjust Camera for "Travel Photo" Angle
    cam = scene_mgr.active_camera
    # Position: Slightly lower, looking out at the horizon for depth
    # Previous: Vector3(256, 300, 256) -> Too high, like a satellite
    # New: "Drone Shot"
    cam.set_position(Vector3(100, 150, 100)) 
    cam.look_at(Vector3(300, 50, 300)) # Look towards the center/mountains
    
    # 5. Main Loop Injection
    print("Starting Simulation Loop...")
    
    # NEW: Minimal UI Integration
    # We replace the default Engine UI with our custom "Photo Mode" UI
    from demos.minimal_ui import MinimalUI
    
    # We need to ensure engine thinks it has a UI so it doesn't complain
    # But we won't use engine.tk_ui
    
    minimal_ui = MinimalUI(engine)
    
    # Hack: Inject the viewport's context into NaturalSystem once it starts
    # The OpenGLViewport in MinimalUI will initialize GL context asynchronously
    # We need to hook into it.
    
    # The viewport will call engine.initialize_renderer_deferred() if it exists.
    # We can use that hook.
    
    def on_gl_ready():
        print("Minimal UI GL Ready! Injecting Context...")
        if hasattr(minimal_ui.viewport, 'ctx') and minimal_ui.viewport.ctx:
             # Inject into NaturalSystem
             natural_system.gpu_manager.set_context(minimal_ui.viewport.ctx)
             
             # Re-register textures
             if hasattr(natural_system.engine.facts, 'tables'):
                 # ... (texture reg code)
                 pass
             
    # Monkey patch engine to receive the hook
    engine.initialize_renderer_deferred = lambda: (on_gl_ready() or True)
    
    # NEW: Monkey patch engine.render to bypass broken TerrainManager and only render SceneManager
    # This is critical because ui_type="game" triggers engine.render() which crashes on default terrain
    original_render = engine.render
    def safe_render():
        try:
            # Only render Scene Manager nodes that have a render method
            if engine.scene_mgr and engine.scene_mgr.active_camera:
                camera = engine.scene_mgr.active_camera
                
                # Make sure context is available for nodes (lazy init)
                # Some nodes might need engine.renderer.ctx or similar
                # But our SimpleTerrainRenderer uses natural_system.gpu_manager.context
                
                # Helper to render node and children
                def render_node(node):
                    if hasattr(node, 'render') and callable(node.render):
                        try:
                            # Pass camera and light_manager (None for now)
                            node.render(camera, None)
                        except Exception as e:
                            # print(f"Error rendering node {node.name}: {e}")
                            pass
                    
                    # Recurse
                    for child in node.children:
                        render_node(child)
                
                # Start from root
                if engine.scene_mgr.root_node:
                    render_node(engine.scene_mgr.root_node)
        except Exception as e:
            print(f"Render Loop Error: {e}")
            
    engine.render = safe_render
    
    # Animation Loop attached to Tkinter
    def game_loop():
        # Update Natural System
        natural_system.update(0.1)
        
        # Schedule next
        minimal_ui.root.after(16, game_loop)
        
    minimal_ui.root.after(100, game_loop)
    
    print("Launching Minimal UI...")
    minimal_ui.run()

if __name__ == "__main__":
    main()
