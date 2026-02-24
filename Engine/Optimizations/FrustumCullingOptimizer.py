"""
优化的视锥体剔除系统
专门针对GTX 1650 Max-Q等移动GPU进行性能优化
"""

import math
from Engine.Math import Vector3, BoundingBox
from Engine.Scene.Camera import Camera
from Engine.Scene.SceneNode import SceneNode


class OptimizedFrustumCulling:
    """
    针对低端GPU优化的视锥体剔除系统
    使用层次化剔除策略，减少不必要的计算
    """
    
    def __init__(self, scene_manager):
        self.scene_manager = scene_manager
        self.culling_cache = {}  # 缓存剔除结果
        self.cache_valid = True
        
    def perform_frustum_culling(self, camera: Camera, nodes):
        """
        执行优化的视锥体剔除
        
        Args:
            camera: 当前相机
            nodes: 待剔除的节点列表
            
        Returns:
            list: 可见节点列表
        """
        if not camera.use_frustum_culling:
            return nodes
            
        # 获取更新的视锥体
        frustum = camera.get_frustum()
        
        # 检查缓存是否有效
        if self.cache_valid:
            cache_key = hash((id(camera), id(nodes)))
            if cache_key in self.culling_cache:
                return self.culling_cache[cache_key]
        
        # 使用层次化剔除策略
        visible_nodes = self._hierarchical_culling(frustum, nodes)
        
        # 缓存结果
        self.culling_cache[hash((id(camera), id(nodes)))] = visible_nodes
        return visible_nodes
    
    def _hierarchical_culling(self, frustum, nodes):
        """
        层次化剔除：先粗略筛选，再精确检查
        """
        # 第一层：基于距离的快速筛选
        distance_culled = self._distance_culling(frustum, nodes)
        
        # 第二层：基于包围盒的精确剔除
        precise_culled = self._precise_culling(frustum, distance_culled)
        
        return precise_culled
    
    def _distance_culling(self, frustum, nodes):
        """
        基于距离的快速筛选
        """
        camera_pos = frustum.camera_position  # 假设有这个属性
        max_distance = self.scene_manager.optimization_settings.get("culling_distance", 1000.0)
        
        filtered_nodes = []
        for node in nodes:
            if hasattr(node, 'world_position') and node.world_position:
                distance = (node.world_position - camera_pos).length()
                if distance <= max_distance:
                    filtered_nodes.append(node)
        
        return filtered_nodes
    
    def _precise_culling(self, frustum, nodes):
        """
        基于包围盒的精确剔除
        """
        visible_nodes = []
        
        for node in nodes:
            # 检查是否有包围盒
            if hasattr(node, 'world_bounding_box') and node.world_bounding_box:
                # 使用包围盒进行精确剔除
                if frustum.contains_bounding_box(node.world_bounding_box):
                    visible_nodes.append(node)
            else:
                # 如果没有包围盒，使用节点位置进行点剔除
                if hasattr(node, 'world_position') and node.world_position:
                    if frustum.contains_point(node.world_position):
                        visible_nodes.append(node)
        
        return visible_nodes
    
    def invalidate_cache(self):
        """
        使缓存失效（当相机或场景发生变化时调用）
        """
        self.culling_cache.clear()
        self.cache_valid = False
    
    def validate_cache(self):
        """
        验证缓存有效性
        """
        self.cache_valid = True


class AdvancedVolumetricLightSettings:
    """
    针对GTX 1650 Max-Q优化的体积光设置
    在性能和视觉效果之间找到最佳平衡点
    """
    
    def __init__(self, gpu_capability="mid-range"):
        self.gpu_capability = gpu_capability
        self.settings = self._get_optimized_settings()
        
    def _get_optimized_settings(self):
        """
        根据GPU能力返回优化的设置
        """
        if self.gpu_capability == "low-end":
            # 适用于GTX 1650 Max-Q等移动GPU
            return {
                # 体积光参数
                "volumetric_steps": 8,          # 减少光线步进次数
                "volumetric_intensity": 0.3,    # 降低强度
                "volumetric_scattering": 0.2,   # 降低散射
                "volumetric_downsample": 8,     # 增加降采样倍数
                
                # 性能优化
                "enable_volumetric_light": True,    # 仍然启用但优化
                "volumetric_quality": "low",        # 低质量
                "volumetric_cache": True,           # 启用缓存
                
                # 替代效果
                "use_directional_shadows": True,    # 使用定向阴影替代
                "use_cheap_volumetrics": True,      # 使用便宜的体积效果
            }
        elif self.gpu_capability == "mid-range":
            # 适用于主流桌面GPU
            return {
                "volumetric_steps": 16,
                "volumetric_intensity": 0.5,
                "volumetric_scattering": 0.3,
                "volumetric_downsample": 4,
                
                "enable_volumetric_light": True,
                "volumetric_quality": "medium",
                "volumetric_cache": True,
                
                "use_directional_shadows": True,
                "use_cheap_volumetrics": False,
            }
        else:  # high-end
            # 适用于高端GPU
            return {
                "volumetric_steps": 32,
                "volumetric_intensity": 0.8,
                "volumetric_scattering": 0.5,
                "volumetric_downsample": 2,
                
                "enable_volumetric_light": True,
                "volumetric_quality": "high",
                "volumetric_cache": True,
                
                "use_directional_shadows": False,
                "use_cheap_volumetrics": False,
            }
    
    def apply_to_natural_system(self, natural_system):
        """
        将优化设置应用到自然系统
        """
        # 更新自然系统的体积光配置
        config_updates = {
            'volumetric_steps': self.settings["volumetric_steps"],
            'volumetric_intensity': self.settings["volumetric_intensity"],
            'volumetric_scattering': self.settings["volumetric_scattering"],
            'volumetric_downsample': self.settings["volumetric_downsample"],
            'enable_volumetric_light': self.settings["enable_volumetric_light"]
        }
        
        # 应用配置更新
        for key, value in config_updates.items():
            natural_system.config[key] = value
            
        return config_updates


class PerformanceOptimizer:
    """
    综合性能优化器
    结合视锥体剔除和体积光优化
    """
    
    def __init__(self, scene_manager, natural_system, gpu_capability="mid-range"):
        self.scene_manager = scene_manager
        self.natural_system = natural_system
        self.gpu_capability = gpu_capability
        
        # 初始化优化组件
        self.frustum_optimizer = OptimizedFrustumCulling(scene_manager)
        self.volumetric_optimizer = AdvancedVolumetricLightSettings(gpu_capability)
        
    def optimize_scene_for_gpu(self):
        """
        针对特定GPU优化整个场景
        """
        # 1. 应用体积光优化设置
        volumetric_updates = self.volumetric_optimizer.apply_to_natural_system(
            self.natural_system
        )
        
        # 2. 优化场景管理器设置
        self._optimize_scene_manager()
        
        # 3. 返回优化摘要
        return {
            "volumetric_updates": volumetric_updates,
            "gpu_capability": self.gpu_capability,
            "optimization_applied": True
        }
    
    def _optimize_scene_manager(self):
        """
        优化场景管理器设置
        """
        # 针对移动GPU优化场景管理器
        if self.gpu_capability == "low-end":
            self.scene_manager.optimization_settings.update({
                "culling_distance": 500.0,  # 减少渲染距离
                "max_draw_calls": 500,      # 限制绘制调用
                "lod_enabled": True,        # 启用LOD
                "lod_distance_steps": [5.0, 15.0, 30.0, 60.0],  # 调整LOD距离
                "use_octree": False,        # 禁用八叉树（如果未实现）
                "static_batching": True,    # 启用静态批处理
                "dynamic_batching": True,   # 启用动态批处理
            })
        elif self.gpu_capability == "mid-range":
            self.scene_manager.optimization_settings.update({
                "culling_distance": 800.0,
                "max_draw_calls": 800,
                "lod_enabled": True,
                "lod_distance_steps": [10.0, 25.0, 50.0, 100.0],
                "use_octree": False,
                "static_batching": True,
                "dynamic_batching": True,
            })
    
    def get_performance_recommendations(self):
        """
        获取性能优化建议
        """
        recommendations = []
        
        if self.gpu_capability == "low-end":
            recommendations.extend([
                "使用低分辨率阴影贴图 (1024x1024)",
                "禁用高成本后期处理效果",
                "减少光源数量 (最多4个动态光源)",
                "使用简化版材质",
                "启用纹理压缩",
                "降低几何细节级别"
            ])
        elif self.gpu_capability == "mid-range":
            recommendations.extend([
                "使用中等分辨率阴影贴图 (2048x2048)",
                "适度使用后期处理效果",
                "限制动态光源数量 (最多8个)",
                "平衡材质复杂度",
                "启用各向异性过滤"
            ])
        
        return recommendations