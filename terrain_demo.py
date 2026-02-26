import sys
import os
import time
import math
import numpy as np
from collections import deque

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

from Engine.Engine import Engine
from Engine.Scene.SceneNode import SceneNode
from Engine.Scene.Camera import Camera
from Engine.Scene.Light import DirectionalLight, AmbientLight
from Engine.Scene.MeshRenderer import MeshRenderer
from Engine.Renderer.Resources.ModelLoader import ModelLoader
from Engine.Renderer.Resources.Mesh import Mesh
from Engine.Renderer.Resources.Material import Material
from Engine.Math import Vector3, Vector2
from Engine.UI.FirstPersonController import FirstPersonController

engine = None
terrain_mesh = None
terrain_node = None
player_controller = None
test_ended = False

global sun_node

def main():
    global engine, terrain_mesh, terrain_node, player_controller, test_ended
    
    print("=" * 100)
    print("冰岛山测试 - 修复版")
    print("=" * 100)
    
    # 配置文件已在引擎初始化时加载，不再修改
    
    print("创建引擎...")
    engine = Engine()
    
    print("初始化引擎...")
    engine.initialize()
    if not engine.is_initialized:
        print("引擎初始化失败")
        return
    
    # 立即禁用 EffectManager 的自动优化，强制启用所有特效
    if hasattr(engine, 'renderer') and engine.renderer:
        if hasattr(engine.renderer, 'effect_manager') and engine.renderer.effect_manager:
            em = engine.renderer.effect_manager
            
            # 1. 禁用所有自动优化函数
            def no_op(*args, **kwargs):
                pass
            em.optimize_effects_for_performance = no_op
            em._reduce_performance_impact = no_op
            em._optimize_for_gpu_architecture = no_op
            
            # 2. 强制启用所有特效
            for effect_name, effect in em.effects.items():
                effect.is_enabled = True
            
            print("已强制启用所有特效，禁用自动优化")
    
    print("获取场景管理器...")
    scene_mgr = engine.scene_mgr
    
    print("删除默认测试对象...")
    for node_name in ["Cube", "Floor", "Sphere", "Plane"]:
        node = scene_mgr.find_node(node_name)
        if node:
            scene_mgr.root_node.remove_child(node)
            print(f"已删除默认对象: {node_name}")
    
    print("加载干燥岩砂地形模型...")
    model_loader = ModelLoader()
    
    # 使用干燥岩砂地形模型 (~520,000 triangles)
    # 来源: 爱给网
    terrain_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "干燥岩砂地形", "╢α╕±╩╜", "gltf", "free_-_dry_rock_sand_terrain", "scene.gltf")
    terrain_mesh = model_loader.load_model(terrain_path)
    if not terrain_mesh:
        print("模型加载失败")
        return
    print(f"模型加载成功：{len(terrain_mesh.vertices)} 顶点")
    
    print(f"地形网格: {len(terrain_mesh.vertices)} 顶点, {len(terrain_mesh.indices)//3} 面")
    
    # 检查索引数据是否有效
    print("检查索引数据...")
    vertex_count = len(terrain_mesh.vertices)
    invalid_indices = 0
    for i, idx in enumerate(terrain_mesh.indices):
        if idx < 0 or idx >= vertex_count:
            invalid_indices += 1
            # 修复为有效索引
            terrain_mesh.indices[i] = 0
    if invalid_indices > 0:
        print(f"修复了 {invalid_indices} 个无效索引")
    else:
        print("索引数据检查通过")
    
    print("重新计算法线...")
    terrain_mesh.recalculate_normals(flip=False)
    print(f"法线计算完成：{len(terrain_mesh.normals)} 个法线")
    
    # 修复法线 - 强制所有法线朝上（避免黑色三角形）
    print("强制法线朝上...")
    for i in range(len(terrain_mesh.normals)):
        terrain_mesh.normals[i] = Vector3(0, 1, 0)
    print(f"已设置 {len(terrain_mesh.normals)} 个法线朝上")
    
    # 确保切线数据存在（某些渲染器需要）
    if not hasattr(terrain_mesh, 'tangents') or terrain_mesh.tangents is None or len(terrain_mesh.tangents) == 0:
        print("生成切线数据...")
        terrain_mesh.tangents = [Vector3(1, 0, 0) for _ in range(len(terrain_mesh.vertices))]
        print(f"切线数据生成完成: {len(terrain_mesh.tangents)} 个")
    
    # 确保UV数据存在
    if not hasattr(terrain_mesh, 'uvs') or terrain_mesh.uvs is None or len(terrain_mesh.uvs) == 0:
        print("生成默认UV数据...")
        terrain_mesh.uvs = [Vector2(0, 0) for _ in range(len(terrain_mesh.vertices))]
        print(f"UV数据生成完成: {len(terrain_mesh.uvs)} 个")
    
    # 确保顶点颜色数据存在（避免黑色三角形）
    if not hasattr(terrain_mesh, 'colors') or terrain_mesh.colors is None or len(terrain_mesh.colors) == 0:
        print("生成默认顶点颜色...")
        terrain_mesh.colors = [Vector3(0.5, 0.5, 0.5) for _ in range(len(terrain_mesh.vertices))]
        print(f"顶点颜色生成完成: {len(terrain_mesh.colors)} 个")
    
    # 不要在这里设 is_dirty=False，让第一帧正常上传到GPU
    # 我们会在第一帧渲染后，通过修改 Mesh 类来防止后续更新
    
    # 计算地形边界
    vertices = terrain_mesh.vertices
    if len(vertices) > 0:
        min_x = min(v.x for v in vertices)
        max_x = max(v.x for v in vertices)
        min_y = min(v.y for v in vertices)
        max_y = max(v.y for v in vertices)
        min_z = min(v.z for v in vertices)
        max_z = max(v.z for v in vertices)
        print(f"地形边界: X[{min_x:.1f}, {max_x:.1f}] Y[{min_y:.1f}, {max_y:.1f}] Z[{min_z:.1f}, {max_z:.1f}]")
        
        # 计算缩放因子，让地形大小合适（约1000单位）
        terrain_size = max(max_x - min_x, max_z - min_z)
        target_size = 1000.0
        scale = target_size / terrain_size if terrain_size > 0 else 1.0
        print(f"地形缩放: {scale:.2f}x")
    else:
        scale = 1.0
    
    terrain_node = SceneNode("Terrain")
    terrain_node.mesh = terrain_mesh
    
    # 创建材质并加载纹理
    terrain_material = Material()
    terrain_material.set_double_sided(False)
    terrain_material.set_wireframe(False)
    
    # 设置基础颜色（确保有颜色）
    terrain_material.set_color(Vector3(0.5, 0.5, 0.5))  # 中灰色作为基础色
    
    # 设置基础颜色（确保有颜色）
    terrain_material.set_color(Vector3(0.5, 0.5, 0.5))  # 中灰色作为基础色
    
    # 尝试加载纹理图片（新模型可能自带纹理或不需要外部纹理）
    # 新模型纹理路径（干燥岩砂地形）
    texture_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "干燥岩砂地形", "╢α╕±╩╜", "gltf", "free_-_dry_rock_sand_terrain", "textures", "material_0_diffuse.png")
    if os.path.exists(texture_path):
        print(f"加载纹理: {texture_path}")
        try:
            from PIL import Image
            import numpy as np
            img = Image.open(texture_path).convert('RGB')
            img_array = np.array(img)
            terrain_material.base_color_image = img_array
            print(f"纹理加载成功: {img_array.shape}")
        except Exception as e:
            print(f"纹理加载失败: {e}")
            terrain_material.set_color(Vector3(0.8, 0.8, 0.9))  # 雪白色
    else:
        print("纹理文件不存在，使用雪白色")
        terrain_material.set_color(Vector3(0.8, 0.8, 0.9))  # 雪白色
    
    # 禁用阴影以提高性能
    terrain_material.cast_shadows = False
    terrain_material.receive_shadows = False
    
    terrain_node.material = terrain_material
    scene_mgr.root_node.add_child(terrain_node)
    
    # 保持原始方向，不翻转
    print("保持模型原始方向...")
    terrain_node.set_scale(Vector3(scale, scale, scale))
    
    # 计算出生点（地形中心最高点）
    print("计算地形出生点...")
    if len(vertices) > 0:
        # 找到地形中心
        center_x = (min_x + max_x) / 2
        center_z = (min_z + max_z) / 2
        
        # 找中心区域最高的点
        best_y = min_y
        best_x, best_z = center_x, center_z
        search_radius = terrain_size * 0.1  # 中心10%区域
        
        for v in vertices:
            dx = v.x - center_x
            dz = v.z - center_z
            dist = (dx*dx + dz*dz) ** 0.5
            if dist < search_radius and v.y > best_y:
                best_y = v.y
                best_x = v.x
                best_z = v.z
        
        # 应用缩放并设置出生点
        spawn_x = best_x * scale
        spawn_y = best_y * scale + 5.0
        spawn_z = best_z * scale
        
        start_pos = Vector3(spawn_x, spawn_y + 20.0, spawn_z)  # 再抬高20单位确保在上方
        print(f"出生点: ({spawn_x:.2f}, {spawn_y + 20.0:.2f}, {spawn_z:.2f})")
    else:
        start_pos = Vector3(0, 100, 0)
    
    print("设置相机...")
    cam = Camera("PlayerCamera")
    aspect_ratio = 16.0 / 9.0
    cam.set_perspective(70, aspect_ratio, 0.1, 8000.0)  # 增加FOV到70度，更有电影感
    cam.set_position(start_pos)
    cam.look_at(Vector3(start_pos.x + 10, start_pos.y - 5, start_pos.z + 10))
    scene_mgr.active_camera = cam
    
    # 创建可见太阳 - 日出/日落位置（低角度，更有戏剧性）
    print("创建可见太阳...")
    import math
    # 日出/日落方向：太阳在地平线附近
    sun_dir = Vector3(-0.7, -0.3, -0.5)  # 低角度，接近地平线
    sun_dir.normalize()
    sun_dist = 800.0
    sun_pos = Vector3(
        start_pos.x - sun_dir.x * sun_dist,
        start_pos.y - sun_dir.y * sun_dist + 100,  # 更低的位置
        start_pos.z - sun_dir.z * sun_dist
    )
    sun_size = 150.0  # 稍微小一点，更真实
    
    # 生成太阳球体网格
    segments = 32
    sun_vertices = []
    sun_indices = []
    sun_normals = []
    sun_uvs = []
    
    # 生成球体顶点
    for lat in range(segments // 2 + 1):
        theta = lat * math.pi / (segments // 2)
        sin_theta = math.sin(theta)
        cos_theta = math.cos(theta)
        
        for lon in range(segments + 1):
            phi = lon * 2.0 * math.pi / segments
            sin_phi = math.sin(phi)
            cos_phi = math.cos(phi)
            
            x = cos_phi * sin_theta
            y = cos_theta
            z = sin_phi * sin_theta
            
            sun_vertices.append(Vector3(x * sun_size, y * sun_size, z * sun_size))
            sun_normals.append(Vector3(x, y, z))
            sun_uvs.append(Vector2(lon / segments, lat / segments))
    
    # 生成球体索引
    for lat in range(segments // 2):
        for lon in range(segments):
            first = lat * (segments + 1) + lon
            second = first + segments + 1
            
            sun_indices.append(first)
            sun_indices.append(second)
            sun_indices.append(first + 1)
            
            sun_indices.append(second)
            sun_indices.append(second + 1)
            sun_indices.append(first + 1)
    
    # 创建太阳网格
    sun_mesh = Mesh()
    sun_mesh.vertices = sun_vertices
    sun_mesh.normals = sun_normals
    sun_mesh.uvs = sun_uvs
    sun_mesh.indices = sun_indices
    
    # 创建太阳材质（自发光暖色 - 日出/日落的橙红色调）
    sun_material = Material()
    # 日出/日落的暖橙红色：更真实的太阳颜色
    sun_material.set_emissive(Vector3(1.0, 0.6, 0.2), strength=3.0)  # 强烈的橙红色发光
    sun_material.set_color(Vector3(1.0, 0.7, 0.3))  # 橙黄色
    sun_material.set_double_sided(True)
    
    # 创建太阳节点（全局变量，方便后续更新）
    global sun_node
    sun_node = SceneNode("Sun")
    sun_node.mesh = sun_mesh  # 关联网格
    sun_node.material = sun_material  # 关联材质
    sun_node.set_position(sun_pos)
    
    # 确保添加到根节点
    scene_mgr.root_node.add_child(sun_node)
    # 立即添加到可见节点列表
    if sun_node not in scene_mgr.visible_nodes:
        scene_mgr.visible_nodes.append(sun_node)
    
    print(f"可见太阳创建完成: 位置({sun_pos.x:.1f}, {sun_pos.y:.1f}, {sun_pos.z:.1f}), 网格{len(sun_mesh.vertices)}顶点")
    
    # 设置蓝色天空 - 修改天气系统，让get_sky_color()返回蓝色
    if hasattr(engine, 'atmosphere_system') and engine.atmosphere_system:
        engine.atmosphere_system.sky_enabled = False
        engine.atmosphere_system.fog_enabled = False
        engine.atmosphere_system.sky_color = Vector3(0.3, 0.5, 0.8)
        
        # 重写get_sky_color()方法，强制返回蓝色
        def get_blue_sky():
            return Vector3(0.3, 0.5, 0.8)
        engine.atmosphere_system.get_sky_color = get_blue_sky
        
        print("已设置天气系统天空颜色为蓝色")
    
    # 创建地形高度数据表供阴影系统使用
    print("创建地形高度数据表...")
    import numpy as np  # 确保numpy已导入
    if hasattr(engine, 'natural_system') and engine.natural_system:
        # 从模型顶点创建高度图网格
        height_map_size = 256  # 256x256 高度图
        height_map = np.zeros((height_map_size, height_map_size), dtype=np.float32)
        
        # 计算模型边界
        if len(vertices) > 0:
            # 将顶点映射到高度图网格
            for v in vertices:
                # 世界坐标（考虑缩放）
                world_x = -v.x * scale
                world_z = -v.z * scale
                world_y = -v.y * scale
                
                # 映射到网格坐标
                grid_x = int((world_x - min_x * scale) / (max_x - min_x) * scale * height_map_size / target_size)
                grid_z = int((world_z - min_z * scale) / (max_z - min_z) * scale * height_map_size / target_size)
                
                # 边界检查
                if 0 <= grid_x < height_map_size and 0 <= grid_z < height_map_size:
                    # 保存最高点
                    if world_y > height_map[grid_z, grid_x]:
                        height_map[grid_z, grid_x] = world_y
        
        # 创建地形数据表
        engine.natural_system.create_terrain_table("terrain_main", height_map_size, height_map)
        print(f"地形高度数据表创建完成: {height_map_size}x{height_map_size}")
        
        # 设置太阳方向用于阴影计算（与主光源一致）
        engine.natural_system.set_global('sun_direction', np.array([-0.7, -0.3, -0.5], dtype=np.float32))
        
        # 设置相机位置
        engine.natural_system.set_camera_position(np.array([start_pos.x, start_pos.y, start_pos.z], dtype=np.float32))
    
    print("设置艺术光照...")
    
    # 主光源 - 日出/日落暖色调阳光（与太阳位置匹配）
    sun_light = DirectionalLight()
    sun_light.set_direction(Vector3(-0.7, -0.3, -0.5))  # 日出/日落角度
    sun_light.set_intensity(30.0)  # 更强的光照
    sun_light.set_color(Vector3(1.0, 0.6, 0.2))  # 橙红色阳光
    scene_mgr.light_manager.add_light(sun_light)
    
    # 补光 - 更冷的色调（增强雪的质感）
    fill_light = DirectionalLight()
    fill_light.set_direction(Vector3(0.5, -0.3, 0.5))  # 从另一侧补光
    fill_light.set_intensity(6.0)  # 稍微增强
    fill_light.set_color(Vector3(0.5, 0.7, 0.95))  # 冷蓝色
    scene_mgr.light_manager.add_light(fill_light)
    
    # 环境光 - 冷色调基础照明
    ambient_light = AmbientLight()
    ambient_light.set_color(Vector3(0.6, 0.75, 0.9))  # 淡蓝色环境光
    ambient_light.set_intensity(2.0)  # 降低环境光，让主光更突出
    scene_mgr.light_manager.add_light(ambient_light)
    
    # 轮廓光 - 冷色调
    rim_light = DirectionalLight()
    rim_light.set_direction(Vector3(0, 0.5, -1))  # 从背后照射
    rim_light.set_intensity(3.0)  # 稍微增强
    rim_light.set_color(Vector3(0.7, 0.85, 1.0))  # 冷白色轮廓光
    scene_mgr.light_manager.add_light(rim_light)
    
    # 增强丁达尔效应（体积光束效果）
    if hasattr(engine, 'natural_system') and engine.natural_system:
        # 设置太阳方向和颜色用于体积光（日出/日落）
        import numpy as np
        engine.natural_system.set_global('sun_direction', np.array([-0.7, -0.3, -0.5], dtype=np.float32))
        engine.natural_system.set_global('sun_color', np.array([1.0, 0.6, 0.2], dtype=np.float32))  # 橙红色
        engine.natural_system.set_global('sun_intensity', 3.0)  # 增强体积光强度
        
        # 设置体积光参数（丁达尔效应）
        engine.natural_system.set_global('volumetric_steps', 32)  # 增加步数，更精细的光束
        engine.natural_system.set_global('volumetric_intensity', 1.5)  # 增强光束强度
        engine.natural_system.set_global('enable_volumetric_light', True)
        
        print("已增强丁达尔效应（体积光束）")
    
    print("创建自由飞行控制器（上帝模式）...")
    
    # 自由飞行控制器
    class FreeFlightController:
        def __init__(self, camera):
            self.camera = camera
            self.position = Vector3(camera.position.x, camera.position.y, camera.position.z)
            self.yaw = 0.0
            self.pitch = -0.3
            self.move_speed = 30.0   # 降低速度，让人物看起来更小
            self.fast_speed = 100.0  # 加速时也保持相对较慢
            self.mouse_sensitivity = 0.003
            
        def set_position(self, x, y, z):
            self.position.x = x
            self.position.y = y
            self.position.z = z
            
        def set_rotation(self, yaw, pitch):
            self.yaw = yaw
            self.pitch = pitch
            
        def update(self, delta_time, input_state):
            # 鼠标旋转
            mouse_dx = input_state.get('mouse_dx', 0)
            mouse_dy = input_state.get('mouse_dy', 0)
            
            if mouse_dx != 0 or mouse_dy != 0:
                self.yaw -= mouse_dx * self.mouse_sensitivity
                self.pitch -= mouse_dy * self.mouse_sensitivity
                max_pitch = 1.55
                self.pitch = max(-max_pitch, min(max_pitch, self.pitch))
                input_state['mouse_dx'] = 0
                input_state['mouse_dy'] = 0
            
            # 计算方向向量
            sin_yaw = math.sin(self.yaw)
            cos_yaw = math.cos(self.yaw)
            sin_pitch = math.sin(self.pitch)
            cos_pitch = math.cos(self.pitch)
            
            # 前方向量
            forward = Vector3(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
            # 右方向量
            right = Vector3(cos_yaw, 0, -sin_yaw)
            # 上方向量
            up = Vector3(0, 1, 0)
            
            # 选择速度
            speed = self.fast_speed if input_state.get('shift') else self.move_speed
            
            # WASD 移动（水平面）
            move_dir = Vector3(0, 0, 0)
            if input_state.get('w'): move_dir += forward
            if input_state.get('s'): move_dir -= forward
            if input_state.get('d'): move_dir += right
            if input_state.get('a'): move_dir -= right
            
            if move_dir.length_squared() > 0.001:
                move_dir.normalize()
                self.position.x += move_dir.x * speed * delta_time
                self.position.y += move_dir.y * speed * delta_time
                self.position.z += move_dir.z * speed * delta_time
            
            # Space 上升，Ctrl 下降（垂直方向）
            if input_state.get('space'):
                self.position.y += speed * delta_time
            if input_state.get('ctrl') or input_state.get('c'):
                self.position.y -= speed * delta_time
            
            # 更新相机
            self.camera.position = Vector3(self.position.x, self.position.y, self.position.z)
            
            # 计算 LookAt 目标
            look_dir = Vector3(
                -sin_yaw * cos_pitch,
                sin_pitch,
                -cos_yaw * cos_pitch
            )
            target = self.position + look_dir
            self.camera.look_at(target)
    
    player_controller = FreeFlightController(cam)
    player_controller.set_rotation(0.0, -0.3)
    player_controller.set_position(start_pos.x, start_pos.y, start_pos.z)
    
    print("\n" + "="*100)
    print("冰岛山Demo启动! (上帝模式)")
    print("控制：WASD 移动，鼠标旋转视角")
    print("Space 上升，Ctrl/C 下降，Shift 加速")
    print("="*100)
    
    def get_input_state():
        global engine
        if engine and hasattr(engine, 'input_state'):
            return engine.input_state
        if engine and hasattr(engine, 'tk_ui') and hasattr(engine.tk_ui, 'input_state'):
            return engine.tk_ui.input_state
        return {'w':False,'a':False,'s':False,'d':False,'space':False,'shift':False,'ctrl':False,'c':False,'mouse_dx':0,'mouse_dy':0}
    
    # 构建世界坐标到高度的直接映射（简单可靠的方法）
    print("构建地形高度查询表...")
    
    # 创建一个字典，存储每个顶点的世界坐标对应的高度
    # 模型翻转: 世界X = -本地X * scale, 世界Y = -本地Y * scale, 世界Z = -本地Z * scale
    world_height_map = {}
    
    if len(vertices) > 0:
        for v in vertices:
            # 计算世界坐标（考虑模型翻转）
            world_x = -v.x * scale
            world_z = -v.z * scale
            world_y = -v.y * scale
            
            # 使用网格坐标作为键（降低精度，增加容错）
            grid_x = round(world_x / 5.0)  # 5米一个格子
            grid_z = round(world_z / 5.0)
            key = (grid_x, grid_z)
            
            # 保存该格子的最高高度
            if key not in world_height_map or world_y > world_height_map[key]:
                world_height_map[key] = world_y
    
    print(f"地形高度表构建完成: {len(world_height_map)} 个单元")
    
    # 地形高度检测函数（简单直接）
    def get_terrain_height(world_x, world_z):
        """获取指定世界坐标的地形高度"""
        if len(world_height_map) == 0:
            return 0.0
        
        # 计算网格键
        grid_x = round(world_x / 5.0)
        grid_z = round(world_z / 5.0)
        key = (grid_x, grid_z)
        
        # 直接查询
        if key in world_height_map:
            return world_height_map[key]
        
        # 如果没找到，搜索附近的格子
        best_height = None
        best_dist = float('inf')
        
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                near_key = (grid_x + dx, grid_z + dz)
                if near_key in world_height_map:
                    # 计算实际距离
                    near_x = near_key[0] * 5.0
                    near_z = near_key[1] * 5.0
                    dist = (near_x - world_x) ** 2 + (near_z - world_z) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_height = world_height_map[near_key]
        
        if best_height is not None:
            return best_height
        
        # 最后的 fallback
        return -min_y * scale  # 返回最低高度
    
    # 性能分析
    frame_count = 0
    last_time = time.time()
    first_frame_rendered = False
    
    def game_update():
        nonlocal frame_count, last_time, first_frame_rendered
        global test_ended
        if not engine.is_initialized or test_ended:
            return
        
        frame_start = time.time()
        
        input_state = get_input_state()
        
        # 控制器更新
        ctrl_start = time.time()
        if player_controller:
            player_controller.update(0.016, input_state)
        ctrl_time = time.time() - ctrl_start
        
        # 场景更新
        scene_start = time.time()
        if terrain_node not in scene_mgr.visible_nodes:
            scene_mgr.visible_nodes.append(terrain_node)
        # 确保太阳节点也在可见节点列表中
        if 'sun_node' in globals() and sun_node not in scene_mgr.visible_nodes:
            scene_mgr.visible_nodes.append(sun_node)
        scene_mgr.update(0.016)
        scene_time = time.time() - scene_start
        
        # 渲染
        render_start = time.time()
        engine.render()
        render_time = time.time() - render_start
        
        # 第一帧渲染后，禁用地形网格更新以提升性能
        if not first_frame_rendered and terrain_mesh:
            def no_op_update(*args, **kwargs):
                pass
            terrain_mesh.update = no_op_update
            first_frame_rendered = True
        
        frame_time = time.time() - frame_start
        frame_count += 1
        
        # 每秒输出一次性能数据
        if time.time() - last_time >= 1.0:
            fps = frame_count / (time.time() - last_time)
            print(f"[性能] FPS: {fps:.1f} | 帧时间: {frame_time*1000:.1f}ms | "
                  f"控制: {ctrl_time*1000:.1f}ms | 场景: {scene_time*1000:.1f}ms | "
                  f"渲染: {render_time*1000:.1f}ms")
            
            # 输出渲染器详细阶段计时（如果有）
            if engine.renderer and hasattr(engine.renderer, 'curr_pipeline'):
                pipeline = engine.renderer.curr_pipeline
                if hasattr(pipeline, 'render_stage_timings'):
                    timings = pipeline.render_stage_timings
                    print(f"  [渲染阶段] 清屏: {timings.get('clear_buffers', 0):.1f}ms | "
                          f"相机: {timings.get('setup_camera', 0):.1f}ms | "
                          f"着色器: {timings.get('setup_shader', 0):.1f}ms | "
                          f"全局Uniform: {timings.get('setup_global_uniforms', 0):.1f}ms | "
                          f"物体渲染: {timings.get('render_objects', 0):.1f}ms")
                    print(f"  [物体子阶段] 网格更新: {timings.get('obj_mesh_update', 0):.1f}ms | "
                          f"材质更新: {timings.get('obj_material_update', 0):.1f}ms | "
                          f"材质绑定: {timings.get('obj_material_bind', 0):.1f}ms | "
                          f"网格绘制: {timings.get('obj_mesh_draw', 0):.1f}ms")
            
            # 输出Natural系统每个规则的耗时（如果有）
            if hasattr(engine, 'natural_system') and engine.natural_system:
                if hasattr(engine.natural_system, 'get_rule_timings'):
                    rule_timings = engine.natural_system.get_rule_timings()
                    if rule_timings:
                        print(f"  [Natural系统规则]")
                        total_natural = 0.0
                        for rule_name, duration_ms in sorted(rule_timings.items(), key=lambda x: -x[1]):
                            if isinstance(duration_ms, (int, float)) and duration_ms > 0.01:
                                print(f"    {rule_name}: {duration_ms:.2f}ms")
                                total_natural += duration_ms
                        print(f"    [总计] Natural系统: {total_natural:.2f}ms")
            
            frame_count = 0
            last_time = time.time()
        
        if engine.is_initialized and hasattr(engine,'tk_ui') and engine.tk_ui and not test_ended:
            engine.tk_ui.root.after(16, game_update)
    
    if hasattr(engine,'tk_ui') and engine.tk_ui:
        engine.tk_ui.root.after(100, game_update)
        print("启动主循环...")
        engine.tk_ui.mainloop()
    
    engine.shutdown()
    print("\nDemo已关闭")

if __name__ == "__main__":
    main()
