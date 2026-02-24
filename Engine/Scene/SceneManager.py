# -*- coding: utf-8 -*-
"""
游戏世界的总导演！
负责搭建舞台、安排演员、控制镜头和灯光
让整个游戏世界在舞台上完美呈现
"""

import time
import math
from collections import defaultdict
import numpy as np

# 确保正确导入Math模块
from Engine.Math import Vector3, Matrix4x4, Quaternion
from Engine.Renderer.Resources import ResourceManager
from Engine.Renderer.Shaders import ShaderManager
from .SceneNode import SceneNode
from .Camera import Camera
from .Light import LightManager, DirectionalLight, AmbientLight
from Engine.Utils.PerformanceMonitor import PerformanceMonitor
# 导入优化的视锥体剔除系统
try:
    from Engine.Optimizations.FrustumCullingOptimizer import OptimizedFrustumCulling
    USE_OPTIMIZED_FRUSTUM_CULLING = True
except ImportError:
    USE_OPTIMIZED_FRUSTUM_CULLING = False


class SceneManager:
    """游戏世界的总导演！
    负责管理整个舞台：场景图、相机、灯光、演员调度
    让每个元素都在正确的时间出现在正确的位置
    """
    
    def __init__(self, engine=None):
        """搭建游戏世界的舞台
        
        Args:
            engine: 魔法引擎的指挥中心（可选）
        """
        # 导入日志系统
        from Engine.Logger import get_logger
        self.logger = get_logger("SceneManager")
        
        # 引擎引用
        self.engine = engine
        self.renderer = engine.renderer if engine else None
        self.resource_manager = engine.res_mgr if engine else None
        self.shader_manager = engine.renderer.shader_mgr if engine and engine.renderer else None
        
        # 场景根节点
        self.root_node = SceneNode("root")
        
        # 相机管理
        self.active_camera = None
        self.cameras = []
        
        # 灯光管理
        self.light_manager = LightManager()
        
        # 性能监控
        self.perf_monitor = PerformanceMonitor()
        
        # 场景优化配置
        self.optimization_settings = {
            # 视锥体剔除设置
            "frustum_culling": True,
            "backface_culling_only": False,  # 只剔除身后的物体（简单快速）
            "culling_distance": 1000.0,  # 最大可见距离
            "use_occlusion_culling": True,
            
            # 批处理设置
            "static_batching": True,
            "dynamic_batching": True,
            "max_batch_size": 1024,  # 最大批处理顶点数
            
            # 实例化渲染设置
            "instancing_enabled": True,
            "min_instances_for_batching": 4,
            
            # 场景分区设置
            "use_octree": False,  # 暂时禁用八叉树，因为Octree模块缺失
            "octree_depth": 6,
            "octree_max_objects_per_node": 16,
            
            # 视距LOD设置
            "lod_enabled": True,
            "lod_distance_steps": [10.0, 25.0, 50.0, 100.0],
            
            # 视口设置
            "viewport_width": 1280,
            "viewport_height": 720,
            "aspect_ratio": 16.0 / 9.0,
            
            # 低端GPU特定优化
            "max_draw_calls": 1000,   # 针对GTX 750Ti优化的最大绘制调用数
            "max_visible_lights": 8,  # 针对低端GPU的最大可见光源数
            "max_shadow_casters": 4,  # 最大阴影投射光源数
            "shadow_map_resolution": 1024,  # 针对GTX 750Ti的阴影贴图分辨率
        }
        
        # 场景状态
        self.is_loaded = False
        self.is_running = False
        self.frame_count = 0
        self.total_time = 0.0
        
        # 可见节点和批处理数据
        self.visible_nodes = []
        self.last_visible_node_ids = set()  # hysteresis：上一帧可见的节点ID
        self.static_batches = {}
        self.dynamic_batches = {}
        self.instanced_objects = defaultdict(list)
        
        # 八叉树
        self.octree = None
        if self.optimization_settings["use_octree"]:
            self._initialize_octree()
        
        # 初始化优化的视锥体剔除系统
        if USE_OPTIMIZED_FRUSTUM_CULLING:
            self.optimized_frustum_culling = OptimizedFrustumCulling(self)
            print("SceneManager: Using Optimized Frustum Culling System")
        else:
            self.optimized_frustum_culling = None
    
    def _initialize_octree(self):
        """初始化八叉树用于场景分区和加速"""
        # Octree模块缺失，暂时禁用八叉树
        self.octree = None
        self.optimization_settings["use_octree"] = False
    
    def create_node(self, name, position=None, rotation=None, scale=None):
        """在场景中创建一个新节点
        
        Args:
            name: 节点名称
            position: 位置
            rotation: 旋转（四元数）
            scale: 缩放
            
        Returns:
            SceneNode: 创建的节点
        """
        position = position or Vector3(0, 0, 0)
        rotation = rotation or Quaternion.identity()
        scale = scale or Vector3(1, 1, 1)
        
        node = SceneNode(name, position, rotation, scale)
        self.root_node.add_child(node)
        
        # 如果启用了八叉树，将节点添加到八叉树
        if self.optimization_settings["use_octree"] and self.octree:
            self.octree.insert(node)
            
        return node
    
    def find_node(self, name):
        """通过名称查找场景节点
        
        Args:
            name: 节点名称
            
        Returns:
            SceneNode or None: 找到的节点或None
        """
        return self.root_node.find_child(name)
    
    def add_light(self, light):
        """添加光源到场景
        
        Args:
            light: 光源对象
        """
        return self.light_manager.add_light(light)
    
    def create_camera(self, name, position=None, rotation=None):
        """创建并添加相机到场景
        
        Args:
            name: 相机名称
            position: 相机位置
            rotation: 相机旋转（四元数）
            
        Returns:
            Camera: 相机实例
        """
        position = position or Vector3(0, 0, 5)
        rotation = rotation or Quaternion.identity()
        
        # 正确调用Camera构造函数，只传递name参数
        camera = Camera(name)
        
        # 设置相机位置和旋转
        camera.position = position
        camera.rotation = rotation
        camera.target = Vector3(0, 0, -1)
        
        self.cameras.append(camera)
        
        # 如果是第一个相机，设为活动相机
        if not self.active_camera:
            self.active_camera = camera
            
        # 将相机添加到场景图
        camera_node = SceneNode(name + "Node", position, rotation)
        camera_node.attach_camera(camera)
        self.root_node.add_child(camera_node)
        
        return camera
    
    def set_active_camera(self, camera):
        """设置活动相机
        
        Args:
            camera: 要设置为活动的相机
        """
        if camera in self.cameras:
            self.active_camera = camera
    
    def load_scene(self, scene_name):
        """加载场景（这里是基础框架，实际项目中需要实现具体的加载逻辑）
        
        Args:
            scene_name: 场景名称
            
        Returns:
            bool: 是否加载成功
        """
        self.perf_monitor.start("scene_load")
        
        self.logger.info(f"加载场景: {scene_name}")
        
        # 清空当前场景
        self.clear_scene()
        
        try:
            # TODO: 实现实际的场景加载逻辑
            # 这可能包括从文件加载场景数据、创建节点、加载资源等
            
            # 示例：创建一个基础场景结构
            self._create_default_scene()
            
            # 更新场景状态
            self.is_loaded = True
            self.frame_count = 0
            self.total_time = 0.0
            
            self.logger.info(f"场景 '{scene_name}' 加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"加载场景 '{scene_name}' 失败: {e}", exc_info=True)
            return False
        finally:
            self.perf_monitor.stop("scene_load")
    
    def _create_default_scene(self):
        """创建默认场景（用于测试）"""
        # 创建主相机
        camera = self.create_camera("MainCamera", Vector3(0, 2, -5), Quaternion.from_euler(15, 0, 0))
        camera.set_perspective(60, 16/9, 0.1, 1000.0)
        
        # 创建方向光（太阳光）
        sun = DirectionalLight("Sun", direction=Vector3(-1, -1, -1), color=Vector3(1, 0.95, 0.8), intensity=1.0)
        sun.cast_shadows = True
        self.add_light(sun)

        # 创建环境光 - 修复：使用正确的方式初始化AmbientLight
        ambient = AmbientLight("Ambient")
        # 使用setter方法设置颜色和强度
        ambient.set_color(Vector3(0.2, 0.2, 0.2))
        ambient.set_intensity(0.5)
        self.add_light(ambient)
        
        # 创建地面
        ground_node = self.create_node("Ground", Vector3(0, -1, 0))
        # 注意：这里只是创建节点，实际的网格和材质需要在应用层添加
    
    def clear_scene(self):
        """清空当前场景"""
        # 重置场景图
        self.root_node = SceneNode("root")
        
        # 重置相机列表
        self.cameras = []
        self.active_camera = None
        
        # 重置灯光
        self.light_manager.clear()
        
        # 重置场景状态
        self.is_loaded = False
        self.visible_nodes = []
        self.static_batches = {}
        self.dynamic_batches = {}
        self.instanced_objects = defaultdict(list)
        
        # 重置八叉树
        if self.optimization_settings["use_octree"]:
            self._initialize_octree()
    
    def update(self, delta_time):
        """更新场景状态
        
        Args:
            delta_time: 帧间时间（秒）
        """
        self.total_time += delta_time
        self.frame_count += 1
        
        # 更新场景图
        self.root_node.update(delta_time)
        
        # 更新灯光
        self.light_manager.update(delta_time)
        
        # 更新八叉树（如果启用）
        if self.optimization_settings["use_octree"] and self.octree:
            self.octree.update()
        
        # 进行视锥体剔除（如果启用）
        if self.optimization_settings["frustum_culling"] and self.active_camera:
            self._perform_frustum_culling()
    
    def _perform_frustum_culling(self):
        """执行视锥体剔除（带 hysteresis 防止闪烁）"""
        if not self.active_camera:
            return

        # 如果只剔除身后的物体（简单快速模式）
        if self.optimization_settings.get("backface_culling_only", False):
            self._perform_backface_culling_only()
            return

        # 使用优化的视锥体剔除系统
        if USE_OPTIMIZED_FRUSTUM_CULLING and self.optimized_frustum_culling:
            # 获取所有节点用于剔除
            all_nodes = self.get_all_nodes()
            visible_nodes = self.optimized_frustum_culling.perform_frustum_culling(
                self.active_camera, all_nodes
            )
            self.visible_nodes = visible_nodes
        else:
            # 传统的剔除方法
            # 清空可见节点列表
            self.visible_nodes.clear()

            # 获取相机的视锥体
            frustum = self.active_camera.get_frustum()

            # 由于Octree模块缺失，直接遍历所有节点
            self._cull_node(self.root_node, frustum)
        
        # 更新上一帧可见的节点ID（用于 hysteresis）
        self.last_visible_node_ids = set(id(node) for node in self.visible_nodes)
    
    def _perform_backface_culling_only(self):
        """只剔除身后的物体（简单快速，不闪）"""
        if not self.active_camera:
            return
        
        self.visible_nodes.clear()
        
        # 获取相机位置和前向向量
        cam_pos = self.active_camera.position
        cam_forward = self.active_camera.get_forward()
        
        # 遍历所有节点
        all_nodes = self.get_all_nodes()
        
        for node in all_nodes:
            # 跳过根节点
            if node == self.root_node:
                continue
            
            # 如果节点没有网格，直接加入
            if not hasattr(node, 'mesh') or node.mesh is None:
                self.visible_nodes.append(node)
                continue
            
            # 计算节点中心相对于相机的向量
            node_pos = node.world_position
            to_node = node_pos - cam_pos
            to_node_normalized = to_node.normalize()
            
            # 点积判断：如果点积 > 0 → 在身前 → 渲染
            dot = cam_forward.dot(to_node_normalized)
            
            # 稍微宽松一点，点积 > -0.2 就渲染（防止侧面物体被误删）
            if dot > -0.2:
                self.visible_nodes.append(node)
    
    def _cull_node(self, node, frustum):
        """递归剔除节点

        Args:
            node: 要检查的节点
            frustum: 相机视锥体
        """
        if not node or not node.visible:
            return

        # 如果节点有包围盒，使用包围盒进行快速剔除
        if hasattr(node, 'world_bounding_box') and node.world_bounding_box:
            # 使用包围盒进行视锥剔除
            in_frustum = frustum.contains_bounding_box(node.world_bounding_box)
        else:
            # 检查节点是否在视锥体内
            # 对于大物体（如地形），检查多个点
            if node.mesh and len(node.mesh.vertices) > 1000:
                # 大物体，使用宽松检查 - 检查中心点和几个偏移点
                wp = node.world_position
                offsets = [
                    (0, 0, 0),           # 中心
                    (500, 0, 0),         # 右
                    (-500, 0, 0),        # 左
                    (0, 0, 500),         # 前
                    (0, 0, -500),        # 后
                    (0, 200, 0),         # 上
                ]
                in_frustum = False
                for ox, oy, oz in offsets:
                    check_pos = Vector3(wp.x + ox, wp.y + oy, wp.z + oz)
                    if frustum.contains_point(check_pos):
                        in_frustum = True
                        break
            else:
                # 小物体，只检查中心点
                in_frustum = frustum.contains_point(node.world_position)

        # Hysteresis 逻辑：
        # - 如果上一帧可见，这帧就保持可见（防止边缘闪烁）
        # - 如果上一帧不可见，这帧必须完全在视锥内才渲染
        node_id = id(node)
        was_visible_last_frame = node_id in self.last_visible_node_ids
        
        if was_visible_last_frame or in_frustum:
            self.visible_nodes.append(node)

        # 递归检查子节点
        for child in node.children:
            self._cull_node(child, frustum)
    
    def get_visible_nodes(self):
        """获取当前可见的节点列表
        
        Returns:
            list: 可见节点列表
        """
        if not self.visible_nodes and self.optimization_settings["frustum_culling"]:
            self._perform_frustum_culling()
        
        return self.visible_nodes
    
    def get_lights(self):
        """获取场景中的所有光源
        
        Returns:
            list: 光源列表
        """
        # 合并普通光源和环境光
        return self.light_manager.lights + self.light_manager.ambient_lights
    
    def get_active_camera(self):
        """获取当前活动的相机
        
        Returns:
            Camera: 当前活动相机
        """
        return self.active_camera
    
    def set_optimization_setting(self, setting_name, value):
        """设置场景优化参数
        
        Args:
            setting_name: 参数名称
            value: 参数值
        """
        if setting_name in self.optimization_settings:
            self.optimization_settings[setting_name] = value
            
            # 如果更改了八叉树相关设置，重新初始化八叉树
            if setting_name in ["use_octree", "octree_depth", "octree_max_objects_per_node"]:
                if self.optimization_settings["use_octree"]:
                    self._initialize_octree()
                else:
                    self.octree = None
    
    def get_optimization_settings(self):
        """获取当前的优化设置
        
        Returns:
            dict: 优化设置字典
        """
        return self.optimization_settings.copy()
    
    def get_performance_stats(self):
        """获取性能统计信息
        
        Returns:
            dict: 性能统计数据
        """
        stats = self.perf_monitor.get_stats()
        
        # 添加场景特定统计
        stats.update({
            "visible_nodes": len(self.visible_nodes),
            "total_nodes": self.root_node.count_nodes(),
            "lights": len(self.light_manager.get_lights()),
            "cameras": len(self.cameras),
            "draw_calls": 0,  # 将由渲染器填充
            "triangles": 0,   # 将由渲染器填充
        })
        
        return stats
    
    def get_scene_bounds(self):
        """获取场景的边界
        
        Returns:
            tuple: (min, max) 边界值
        """
        return self.root_node.get_bounds()
    
    def export_scene(self, filename):
        """导出场景到文件
        
        Args:
            filename: 导出文件路径
            
        Returns:
            bool: 是否导出成功
        """
        # TODO: 实现场景导出逻辑
        self.logger.info(f"导出场景到文件: {filename}")
        return True
    
    def import_scene(self, filename):
        """从文件导入场景
        
        Args:
            filename: 导入文件路径
            
        Returns:
            bool: 是否导入成功
        """
        # TODO: 实现场景导入逻辑
        self.logger.info(f"从文件导入场景: {filename}")
        return True
    
    def add_node(self, node):
        """添加节点到场景
        
        Args:
            node: 要添加的节点
        """
        if node.parent is None:
            self.root_node.add_child(node)
        
        # 如果启用了八叉树，将节点添加到八叉树
        if self.optimization_settings["use_octree"] and self.octree:
            self.octree.insert(node)
    
    def remove_node(self, node):
        """从场景中移除节点
        
        Args:
            node: 要移除的节点
        """
        if node.parent:
            node.parent.remove_child(node)
        
        # 如果启用了八叉树，从八叉树中移除节点
        if self.optimization_settings["use_octree"] and self.octree:
            self.octree.remove(node)
    
    def add_model(self, model, position=None, rotation=None, scale=None):
        """添加模型到场景
        
        Args:
            model: 要添加的模型
            position: 模型位置
            rotation: 模型旋转
            scale: 模型缩放
            
        Returns:
            SceneNode: 模型所在的场景节点
        """
        # TODO: 实现模型添加逻辑
        return None
    
    def get_node_by_uid(self, uid):
        """通过UID查找节点
        
        Args:
            uid: 节点的唯一标识符
            
        Returns:
            SceneNode or None: 找到的节点或None
        """
        return self.root_node.find_node_by_uid(uid)
    
    def get_all_nodes(self):
        """获取场景中的所有节点
        
        Returns:
            list: 所有节点列表
        """
        nodes = []
        self._get_all_nodes_recursive(self.root_node, nodes)
        return nodes
    
    def _get_all_nodes_recursive(self, node, nodes):
        """递归获取所有节点
        
        Args:
            node: 当前节点
            nodes: 节点列表，用于存储结果
        """
        nodes.append(node)
        for child in node.children:
            self._get_all_nodes_recursive(child, nodes)
    
    def get_nodes_by_type(self, component_type):
        """获取具有特定组件类型的节点
        
        Args:
            component_type: 组件类型
            
        Returns:
            list: 符合条件的节点列表
        """
        nodes = []
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node.has_component(component_type):
                nodes.append(node)
        return nodes
    
    def get_nodes_with_meshes(self):
        """获取所有包含网格的节点
        
        Returns:
            list: 包含网格的节点列表
        """
        nodes = []
        all_nodes = self.get_all_nodes()
        for node in all_nodes:
            if node.mesh:
                nodes.append(node)
        return nodes
    
    def get_render_data(self):
        """获取渲染数据
        
        Returns:
            dict: 用于渲染的数据
        """
        render_data = {
            "nodes": self.get_nodes_with_meshes(),
            "lights": self.get_lights(),
            "camera": self.active_camera,
            "optimization_settings": self.optimization_settings
        }
        return render_data
    
    def __str__(self):
        """返回场景管理器的字符串表示
        
        Returns:
            str: 场景管理器的字符串表示
        """
        return f"SceneManager(nodes={self.root_node.get_children_count()}, cameras={len(self.cameras)}, lights={len(self.light_manager.lights)})"