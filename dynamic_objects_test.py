#!/usr/bin/env python3
"""
动态物体性能测试 - 100 个动态物体 +10 个动态光源
测试 SDDB 批处理系统在真实场景下的性能
"""
import sys
import os
import time
import math
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from Engine.Engine import Engine
from Engine.Scene.SceneNode import SceneNode
from Engine.Scene.Camera import Camera
from Engine.Scene.Light import DirectionalLight, AmbientLight, PointLight
from Engine.Renderer.Resources.Mesh import Mesh
from Engine.Renderer.Resources.Material import Material
from Engine.Math import Vector3, Vector2

engine = None
dynamic_objects = []
dynamic_lights = []

def main():
    global engine, dynamic_objects, dynamic_lights
    
    print("=" * 100)
    print("动态物体性能测试 - 100 个物体 +10 个动态光源")
    print("=" * 100)
    
    # 创建引擎
    engine = Engine()
    engine.initialize()
    
    if not engine.is_initialized:
        print("引擎初始化失败")
        return
    
    scene_mgr = engine.scene_mgr
    
    # 删除默认物体
    for node_name in ["Cube", "Floor", "Sphere", "Plane"]:
        node = scene_mgr.find_node(node_name)
        if node:
            scene_mgr.root_node.remove_child(node)
    
    # 创建程序化地形（简单平面）
    print("创建程序化地形...")
    terrain_mesh = Mesh.create_plane(200, 200, 50, 50)  # 200x200 单位，50x50 分段
    terrain_node = SceneNode("Terrain")
    terrain_node.mesh = terrain_mesh
    
    terrain_material = Material()
    terrain_material.set_color(Vector3(0.3, 0.5, 0.3))  # 绿色地面
    terrain_node.material = terrain_material
    terrain_node.set_position(Vector3(0, 0, 0))
    
    scene_mgr.root_node.add_child(terrain_node)
    
    # 设置相机
    cam = Camera("TestCamera")
    cam.set_perspective(70, 16.0/9.0, 0.1, 1000.0)
    cam.set_position(Vector3(0, 50, 80))
    cam.look_at(Vector3(0, 0, 0))
    scene_mgr.active_camera = cam
    
    # 设置光照
    sun_light = DirectionalLight()
    sun_light.set_direction(Vector3(-0.5, -1.0, -0.3))
    sun_light.set_intensity(10.0)
    sun_light.set_color(Vector3(1.0, 0.95, 0.9))
    scene_mgr.light_manager.add_light(sun_light)
    
    ambient_light = AmbientLight()
    ambient_light.set_color(Vector3(0.4, 0.4, 0.45))
    ambient_light.set_intensity(1.0)
    scene_mgr.light_manager.add_light(ambient_light)
    
    # ========== 创建 100 个动态物体 ==========
    print("\n创建 100 个动态物体（50 个立方体 +50 个球体）...")
    
    cube_mesh = Mesh.create_cube(2.0)
    sphere_mesh = Mesh.create_sphere(1.5, 16, 16)
    
    colors = [
        Vector3(1.0, 0.2, 0.2), Vector3(0.2, 1.0, 0.2), Vector3(0.2, 0.2, 1.0),
        Vector3(1.0, 1.0, 0.2), Vector3(1.0, 0.2, 1.0), Vector3(0.2, 1.0, 1.0),
        Vector3(1.0, 0.5, 0.0), Vector3(0.5, 0.0, 1.0),
    ]
    
    # 50 个立方体
    for i in range(50):
        angle = (i / 50.0) * math.pi * 2
        radius = 30 + (i % 5) * 8
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = 5 + (i % 3) * 6
        
        cube_node = SceneNode(f"Cube_{i}")
        cube_node.mesh = cube_mesh
        
        material = Material()
        material.set_color(colors[i % len(colors)])
        material.set_emissive(colors[i % len(colors)] * 0.3, strength=0.5)
        cube_node.material = material
        
        cube_node.set_position(Vector3(x, y, z))
        cube_node.set_rotation(math.radians(i * 7.2), math.radians(i * 3.6), math.radians(i * 1.8))
        
        scene_mgr.root_node.add_child(cube_node)
        dynamic_objects.append({
            'node': cube_node, 'type': 'cube',
            'original_pos': Vector3(x, y, z),
            'speed': 0.5 + (i % 5) * 0.2,
            'axis': 'y' if i % 3 == 0 else ('x' if i % 3 == 1 else 'z')
        })
    
    print(f"已创建 50 个立方体")
    
    # 50 个球体
    for i in range(50):
        angle = (i / 50.0) * math.pi * 2 + 0.1
        radius = 50 + (i % 7) * 6
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = 8 + (i % 4) * 5
        
        sphere_node = SceneNode(f"Sphere_{i}")
        sphere_node.mesh = sphere_mesh
        
        material = Material()
        material.set_color(colors[(i + 2) % len(colors)])
        material.set_emissive(colors[(i + 2) % len(colors)] * 0.2, strength=0.3)
        sphere_node.material = material
        
        sphere_node.set_position(Vector3(x, y, z))
        
        scene_mgr.root_node.add_child(sphere_node)
        dynamic_objects.append({
            'node': sphere_node, 'type': 'sphere',
            'original_pos': Vector3(x, y, z),
            'speed': 0.3 + (i % 4) * 0.15,
            'axis': 'y' if i % 4 == 0 else ('x' if i % 4 == 1 else ('z' if i % 4 == 2 else 'xyz'))
        })
    
    print(f"已创建 50 个球体")
    print(f"动态物体总数：{len(dynamic_objects)}")
    
    # ========== 创建 10 个动态点光源 ==========
    print("\n创建 10 个动态点光源...")
    
    for i in range(10):
        angle = (i / 10.0) * math.pi * 2
        radius = 40 + i * 3
        x = math.cos(angle) * radius
        z = math.sin(angle) * radius
        y = 15 + (i % 5) * 8
        
        point_light = PointLight()
        point_light.set_position(Vector3(x, y, z))
        point_light.set_intensity(5.0 + (i % 3) * 2.0)
        point_light.set_color(colors[(i + 3) % len(colors)])
        point_light.set_radius(30.0 + (i % 4) * 10.0)
        
        scene_mgr.light_manager.add_light(point_light)
        dynamic_lights.append({
            'light': point_light,
            'original_pos': Vector3(x, y, z),
            'speed': 0.2 + (i % 3) * 0.1,
            'axis': 'y' if i % 2 == 0 else 'x'
        })
    
    print(f"已创建 {len(dynamic_lights)} 个动态点光源")
    print(f"\n场景总计：{len(dynamic_objects)} 个动态物体 + {len(dynamic_lights)} 个动态光源")
    print("=" * 100 + "\n")
    
    # ========== 主循环 ==========
    frame_count = 0
    last_time = time.time()
    
    def game_update():
        nonlocal frame_count, last_time
        if not engine.is_initialized:
            return
        
        frame_start = time.time()
        current_time = time.time()
        
        # 更新动态物体
        dynamic_start = time.time()
        for obj_data in dynamic_objects:
            node = obj_data['node']
            original_pos = obj_data['original_pos']
            speed = obj_data['speed']
            axis = obj_data['axis']
            
            t = current_time * speed
            offset = math.sin(t) * 8.0
            
            if axis == 'y':
                new_pos = Vector3(original_pos.x, original_pos.y + offset, original_pos.z)
            elif axis == 'x':
                new_pos = Vector3(original_pos.x + offset, original_pos.y, original_pos.z)
            elif axis == 'z':
                new_pos = Vector3(original_pos.x, original_pos.y, original_pos.z + offset)
            else:
                new_pos = Vector3(
                    original_pos.x + math.sin(t) * 6.0,
                    original_pos.y + math.cos(t) * 6.0,
                    original_pos.z + math.sin(t * 0.7) * 6.0
                )
            
            node.set_position(new_pos)
            
            if obj_data['type'] == 'cube':
                node.set_rotation(math.radians(t * 20), math.radians(t * 15), math.radians(t * 10))
            elif obj_data['type'] == 'sphere':
                node.set_rotation(math.radians(t * 10), math.radians(t * 20), math.radians(t * 5))
        
        # 更新动态光源
        for light_data in dynamic_lights:
            light = light_data['light']
            original_pos = light_data['original_pos']
            speed = light_data['speed']
            axis = light_data['axis']
            
            t = current_time * speed
            circle_offset = math.cos(t * 0.5) * 4.0
            height_offset = math.sin(t) * 6.0
            
            if axis == 'y':
                new_pos = Vector3(original_pos.x + circle_offset, original_pos.y + height_offset, original_pos.z + circle_offset)
            else:
                new_pos = Vector3(original_pos.x + height_offset, original_pos.y + circle_offset, original_pos.z + circle_offset)
            
            light.set_position(new_pos)
        
        dynamic_time = time.time() - dynamic_start
        
        # 场景更新
        scene_mgr.update(0.016)
        
        # 渲染
        engine.render()
        
        frame_time = time.time() - frame_start
        frame_count += 1
        
        # 每秒输出性能数据
        if time.time() - last_time >= 1.0:
            fps = frame_count / (time.time() - last_time)
            
            print(f"\n{'='*100}")
            print(f"[性能] FPS: {fps:.1f} | 帧时间：{frame_time*1000:.1f}ms | 动态物体更新：{dynamic_time*1000:.1f}ms")
            print(f"  场景统计：{len(dynamic_objects)} 个动态物体 + {len(dynamic_lights)} 个动态光源")
            
            # SDDB 统计
            if engine.renderer and hasattr(engine.renderer, 'curr_pipeline'):
                pipeline = engine.renderer.curr_pipeline
                if hasattr(pipeline, 'get_performance_stats'):
                    stats = pipeline.get_performance_stats()
                    
                    if 'sddb' in stats:
                        sddb = stats['sddb']
                        print(f"\n  [SDDB 批处理]")
                        print(f"    原始 DC: {sddb.get('original_draw_calls', 'N/A')} → 合并后：{sddb.get('batched_draw_calls', 'N/A')}")
                        print(f"    合并率：{sddb.get('merge_ratio', 0)*100:.1f}%")
                    
                    if 'instancing' in stats:
                        inst = stats['instancing']
                        print(f"\n  [实例化]")
                        print(f"    批次：{inst.get('batches', 'N/A')} | 实例：{inst.get('instances', 'N/A')} | DC: {inst.get('draw_calls', 'N/A')}")
            
            print("=" * 100)
            
            frame_count = 0
            last_time = time.time()
        
        if hasattr(engine, 'tk_ui') and engine.tk_ui:
            engine.tk_ui.root.after(16, game_update)
    
    print("\n测试运行中...（查看控制台输出性能数据）")
    print("=" * 100 + "\n")
    
    if hasattr(engine, 'tk_ui') and engine.tk_ui:
        engine.tk_ui.root.after(100, game_update)
        engine.tk_ui.mainloop()
    
    engine.shutdown()
    print("\n测试完成")

if __name__ == "__main__":
    main()
