"""
综合性能优化模块
整合视锥体剔除和体积光优化，针对GTX 1650 Max-Q等移动GPU优化
"""

from Engine.Optimizations.FrustumCullingOptimizer import OptimizedFrustumCulling, PerformanceOptimizer
from Engine.Optimizations.FastFrustum import OptimizedFrustum, FastFrustumCulling
from Engine.Optimizations.VolumetricLightOptimizer import VolumetricLightConfigManager, VolumetricLightOptimizer, GPUAnalyzer
from Engine.Scene.SceneManager import SceneManager
from Engine.Natural.NaturalSystem import NaturalSystem
from Engine.Scene.Camera import Camera


class ComprehensivePerformanceOptimizer:
    """
    综合性能优化器
    整合视锥体剔除和体积光优化，提供一键优化功能
    """
    
    def __init__(self, scene_manager: SceneManager, natural_system: NaturalSystem, gpu_capability: str = "low-end"):
        self.scene_manager = scene_manager
        self.natural_system = natural_system
        self.gpu_capability = gpu_capability
        
        # 初始化各个优化组件
        self.frustum_optimizer = OptimizedFrustumCulling(scene_manager)
        self.volumetric_config_manager = VolumetricLightConfigManager()
        self.volumetric_optimizer = VolumetricLightOptimizer(self.volumetric_config_manager)
        
        # 性能监控
        self.performance_history = []
        
    def optimize_for_gtx_1650_maxq(self):
        """
        针对GTX 1650 Max-Q进行完整优化
        """
        print("开始针对GTX 1650 Max-Q进行性能优化...")
        
        # 1. 优化视锥体剔除
        self._optimize_frustum_culling()
        
        # 2. 优化体积光设置
        self._optimize_volumetric_lighting()
        
        # 3. 优化场景管理器设置
        self._optimize_scene_manager()
        
        # 4. 生成性能报告
        report = self.generate_performance_report()
        
        print("性能优化完成！")
        return report
    
    def _optimize_frustum_culling(self):
        """
        优化视锥体剔除
        """
        # 替换场景管理器中的视锥体类为优化版本
        # 注意：这需要修改Camera类以使用新的视锥体类
        print("  - 优化视锥体剔除算法...")
        
        # 更新场景管理器的优化设置
        self.scene_manager.optimization_settings.update({
            "frustum_culling": True,
            "culling_distance": 500.0,  # 针对移动GPU减少渲染距离
            "use_occlusion_culling": False,  # 禁用遮挡剔除以节省性能
        })
        
        # 确保相机使用优化的视锥体
        if self.scene_manager.active_camera:
            # 重新初始化相机视锥体（如果需要）
            self.scene_manager.active_camera.frustum = OptimizedFrustum()
            self.scene_manager.active_camera.frustum_dirty = True
    
    def _optimize_volumetric_lighting(self):
        """
        优化体积光设置
        """
        print("  - 优化体积光设置...")
        
        # 为GTX 1650 Max-Q设置合适的体积光参数
        gtx_1650_config = {
            "step_count": 8,              # 显著减少步进次数
            "intensity": 0.25,            # 降低强度
            "scattering": 0.15,           # 降低散射
            "absorption": 0.05,           # 降低吸收
            "max_distance": 50.0,         # 减少最大距离
            
            "downsample_factor": 8,       # 增加降采样以提升性能
            "enable_cache": True,         # 启用缓存
            "temporal_reprojection": True, # 时域重投影
            
            "use_directional_shadows": True,  # 使用方向阴影替代
            "use_cheap_volumetrics": True,    # 使用简单体积效果
            "use_fog_simulation": True,       # 使用雾模拟
        }
        
        # 应用配置到自然系统
        self.natural_system.config.update({
            'volumetric_steps': gtx_1650_config["step_count"],
            'volumetric_intensity': gtx_1650_config["intensity"],
            'volumetric_scattering': gtx_1650_config["scattering"],
            'volumetric_downsample': gtx_1650_config["downsample_factor"],
            'enable_volumetric_light': True,  # 仍然启用但已优化
        })
        
        # 查找并更新体积光规则
        for rule in self.natural_system.engine.rules.rules:
            if hasattr(rule, '__class__') and 'volumetric' in rule.__class__.__name__.lower():
                if hasattr(rule, 'set_parameters'):
                    rule.set_parameters(
                        step_count=gtx_1650_config["step_count"],
                        intensity=gtx_1650_config["intensity"],
                        scattering=gtx_1650_config["scattering"]
                    )
                if hasattr(rule, 'downsample_factor'):
                    rule.downsample_factor = gtx_1650_config["downsample_factor"]
    
    def _optimize_scene_manager(self):
        """
        优化场景管理器设置
        """
        print("  - 优化场景管理器设置...")
        
        # 针对移动GPU优化场景管理器设置
        self.scene_manager.optimization_settings.update({
            "culling_distance": 500.0,     # 减少渲染距离
            "max_draw_calls": 500,         # 限制绘制调用
            "max_visible_lights": 4,       # 减少可见光源数量
            "shadow_map_resolution": 1024, # 降低阴影贴图分辨率
            "lod_enabled": True,           # 启用LOD
            "lod_distance_steps": [5.0, 15.0, 30.0, 60.0],  # 调整LOD距离
            "use_octree": False,           # 禁用八叉树（如果未完全实现）
            "static_batching": True,       # 启用静态批处理
            "dynamic_batching": True,      # 启用动态批处理
            "instancing_enabled": True,    # 启用实例化
            "min_instances_for_batching": 2,  # 降低批处理阈值
        })
        
        # 如果相机存在，进一步优化相机设置
        if self.scene_manager.active_camera:
            camera = self.scene_manager.active_camera
            camera.max_render_distance = 500.0  # 同步渲染距离
            camera.optimize_for_low_end_gpu()   # 应用相机专用优化
    
    def adaptive_optimize(self, current_fps: float, target_fps: float = 53.0):
        """
        自适应优化：根据当前帧率调整设置
        
        Args:
            current_fps: 当前帧率
            target_fps: 目标帧率
        """
        print(f"自适应优化: 当前FPS {current_fps:.1f}, 目标FPS {target_fps:.1f}")
        
        # 设置目标帧率
        self.volumetric_optimizer.set_target_fps(target_fps)
        
        # 根据当前性能调整体积光质量
        frame_time = 1000.0 / current_fps if current_fps > 0 else 16.67
        self.volumetric_optimizer.adapt_to_performance(current_fps, frame_time)
        
        # 获取当前配置摘要
        summary = self.volumetric_optimizer.get_current_config_summary()
        
        print(f"  - 调整后质量等级: {summary['current_quality_level']}")
        print(f"  - 性能影响估算: {summary['performance_impact']['compute_cost_estimate']:.2f}")
        
        return summary
    
    def generate_performance_report(self):
        """
        生成性能优化报告
        """
        report = {
            "optimization_applied": True,
            "gpu_target": "GTX 1650 Max-Q",
            "frustum_culling_enabled": self.scene_manager.optimization_settings["frustum_culling"],
            "culling_distance": self.scene_manager.optimization_settings["culling_distance"],
            "volumetric_light_config": {
                "steps": self.natural_system.config.get('volumetric_steps', 'N/A'),
                "intensity": self.natural_system.config.get('volumetric_intensity', 'N/A'),
                "downsample": self.natural_system.config.get('volumetric_downsample', 'N/A'),
            },
            "scene_manager_settings": {
                "max_draw_calls": self.scene_manager.optimization_settings["max_draw_calls"],
                "max_visible_lights": self.scene_manager.optimization_settings["max_visible_lights"],
                "shadow_resolution": self.scene_manager.optimization_settings["shadow_map_resolution"],
                "lod_enabled": self.scene_manager.optimization_settings["lod_enabled"],
            },
            "recommendations": self._get_performance_recommendations()
        }
        
        return report
    
    def _get_performance_recommendations(self):
        """
        获取性能优化建议
        """
        return [
            "使用低分辨率阴影贴图 (1024x1024)",
            "限制动态光源数量 (最多4个)",
            "启用纹理压缩和Mipmap",
            "使用LOD系统减少远处几何体复杂度",
            "优化材质以减少过度绘制",
            "考虑使用固定管线着色器替代复杂PBR",
            "减少透明物体的数量和复杂度",
            "优化地形LOD以减少多边形数量"
        ]


def apply_performance_optimizations(engine):
    """
    应用性能优化到引擎
    """
    if not engine.scene_mgr or not engine.natural:
        print("错误: 引擎缺少必要的组件进行性能优化")
        return None
    
    # 创建综合优化器
    optimizer = ComprehensivePerformanceOptimizer(
        engine.scene_mgr, 
        engine.natural, 
        gpu_capability="low-end"
    )
    
    # 执行优化
    report = optimizer.optimize_for_gtx_1650_maxq()
    
    # 应用优化到引擎组件
    engine.scene_mgr.optimization_settings.update(report["scene_manager_settings"])
    
    print("\n性能优化应用完成!")
    print(f"视锥体剔除: {'已启用' if report['frustum_culling_enabled'] else '已禁用'}")
    print(f"渲染距离: {report['culling_distance']}米")
    print(f"体积光步进: {report['volumetric_light_config']['steps']}步")
    
    return optimizer


# 测试函数
def test_optimization():
    """
    测试优化功能
    """
    print("测试性能优化功能...")
    
    # 模拟GPU检测
    gpu_analyzer = GPUAnalyzer()
    detected_gpu = "NVIDIA GeForce GTX 1650 Max-Q"
    print(f"检测到GPU: {detected_gpu}")
    
    # 获取推荐设置
    recommendation = gpu_analyzer.recommend_settings(detected_gpu)
    print(f"推荐设置: {recommendation['note']}")
    
    # 这里应该集成到实际的游戏引擎中
    print("优化模块准备就绪，等待集成到游戏引擎...")


if __name__ == "__main__":
    test_optimization()