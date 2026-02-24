"""
体积光配置管理器
针对GTX 1650 Max-Q等移动GPU优化体积光设置
"""

import json
import os
from typing import Dict, Any


class VolumetricLightConfigManager:
    """
    体积光配置管理器
    根据硬件能力自动调整体积光参数
    """
    
    def __init__(self, config_file_path=None):
        self.config_file_path = config_file_path or self._get_default_config_path()
        self.default_configs = self._get_default_configs()
        self.current_config = self.load_config()
        
    def _get_default_config_path(self):
        """获取默认配置文件路径"""
        return os.path.join(os.path.dirname(__file__), "volumetric_light_configs.json")
    
    def _get_default_configs(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "low_end_gpu": {
                # 体积光参数
                "step_count": 8,              # 光线步进次数
                "intensity": 0.25,            # 强度
                "scattering": 0.15,           # 散射系数
                "absorption": 0.05,           # 吸收系数
                "max_distance": 50.0,         # 最大距离
                
                # 性能优化
                "downsample_factor": 8,       # 降采样倍数
                "enable_cache": True,         # 启用缓存
                "temporal_reprojection": True, # 时域重投影
                
                # 替代效果
                "use_directional_shadows": True,  # 使用方向阴影
                "use_cheap_volumetrics": True,    # 使用简单体积效果
                "use_fog_simulation": True,       # 使用雾模拟
            },
            "mid_range_gpu": {
                "step_count": 16,
                "intensity": 0.5,
                "scattering": 0.3,
                "absorption": 0.1,
                "max_distance": 100.0,
                
                "downsample_factor": 4,
                "enable_cache": True,
                "temporal_reprojection": True,
                
                "use_directional_shadows": True,
                "use_cheap_volumetrics": False,
                "use_fog_simulation": False,
            },
            "high_end_gpu": {
                "step_count": 32,
                "intensity": 0.8,
                "scattering": 0.5,
                "absorption": 0.15,
                "max_distance": 200.0,
                
                "downsample_factor": 2,
                "enable_cache": True,
                "temporal_reprojection": True,
                
                "use_directional_shadows": False,
                "use_cheap_volumetrics": False,
                "use_fog_simulation": False,
            }
        }
    
    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_file_path):
            try:
                with open(self.config_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                pass  # 如果读取失败，使用默认配置
        
        # 返回默认配置（低配版，适合GTX 1650 Max-Q）
        return self.default_configs["low_end_gpu"]
    
    def save_config(self, config: Dict[str, Any]):
        """保存配置"""
        self.current_config = config
        with open(self.config_file_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    
    def get_config_for_gpu(self, gpu_name: str) -> Dict[str, Any]:
        """根据GPU名称获取合适的配置"""
        gpu_name_lower = gpu_name.lower()
        
        # 检测是否为低配GPU
        if any(keyword in gpu_name_lower for keyword in [
            'gt 1030', 'gt 730', 'intel hd', 'intel iris', 'amd radeon r5',
            'gtx 1650', 'mx150', 'mx250', 'mx350', 'mx450', 'rtx 2060 max-q',
            'gtx 1650 max-q'
        ]):
            return self.default_configs["low_end_gpu"]
        
        # 检测是否为中配GPU
        elif any(keyword in gpu_name_lower for keyword in [
            'gtx 1060', 'gtx 1070', 'rtx 2060', 'rtx 2070', 'rx 580', 'rx 5700',
            'rtx 3050', 'rtx 3060'
        ]):
            return self.default_configs["mid_range_gpu"]
        
        # 高配GPU
        else:
            return self.default_configs["high_end_gpu"]
    
    def apply_to_volumetric_rule(self, volumetric_rule, config: Dict[str, Any] = None):
        """将配置应用到体积光规则"""
        if config is None:
            config = self.current_config
        
        # 应用参数到体积光规则
        if hasattr(volumetric_rule, 'set_parameters'):
            volumetric_rule.set_parameters(
                step_count=config.get('step_count', 8),
                intensity=config.get('intensity', 0.25),
                scattering=config.get('scattering', 0.15)
            )
        
        # 更新降采样参数
        if hasattr(volumetric_rule, 'downsample_factor'):
            volumetric_rule.downsample_factor = config.get('downsample_factor', 8)
    
    def get_performance_impact_estimate(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """估计配置的性能影响"""
        if config is None:
            config = self.current_config
        
        step_count = config.get('step_count', 8)
        downsample = config.get('downsample_factor', 8)
        
        # 估算性能影响（数值越小越好）
        compute_cost = step_count / (downsample * downsample)
        memory_cost = 1 / downsample  # 降采样减少内存使用
        
        return {
            "compute_cost_estimate": compute_cost,
            "memory_cost_estimate": memory_cost,
            "estimated_fps_drop": max(0, compute_cost * 10 - 5),  # 粗略估算
            "recommended_setting": "low" if compute_cost > 2.0 else "medium" if compute_cost > 1.0 else "high"
        }


class VolumetricLightOptimizer:
    """
    体积光优化器
    动态调整体积光参数以维持目标帧率
    """
    
    def __init__(self, config_manager: VolumetricLightConfigManager):
        self.config_manager = config_manager
        self.target_fps = 60
        self.current_quality_level = "low"
        self.adaptation_history = []
        
    def set_target_fps(self, fps: int):
        """设置目标帧率"""
        self.target_fps = fps
    
    def adapt_to_performance(self, current_fps: float, frame_time: float):
        """
        根据当前性能自适应调整体积光质量
        
        Args:
            current_fps: 当前帧率
            frame_time: 当前帧时间（毫秒）
        """
        # 计算性能余量
        performance_margin = current_fps - self.target_fps
        
        # 如果性能不足，降低质量
        if performance_margin < -10:  # 性能严重不足
            self._decrease_quality()
        elif performance_margin < -5:  # 性能稍差
            self._slightly_decrease_quality()
        elif performance_margin > 10:  # 性能充裕
            self._increase_quality()
        elif performance_margin > 5:  # 性能较好
            self._slightly_increase_quality()
        
        # 记录适应历史
        self.adaptation_history.append({
            "fps": current_fps,
            "frame_time": frame_time,
            "quality_level": self.current_quality_level,
            "timestamp": __import__('time').time()
        })
        
        # 保持历史记录不超过100条
        if len(self.adaptation_history) > 100:
            self.adaptation_history = self.adaptation_history[-100:]
    
    def _decrease_quality(self):
        """降低质量"""
        if self.current_quality_level == "high":
            self.current_quality_level = "medium"
            self._apply_quality_level("medium")
        elif self.current_quality_level == "medium":
            self.current_quality_level = "low"
            self._apply_quality_level("low")
    
    def _slightly_decrease_quality(self):
        """轻微降低质量"""
        if self.current_quality_level == "high":
            self._apply_quality_level("medium_low")
        elif self.current_quality_level == "medium":
            self._apply_quality_level("low")
    
    def _increase_quality(self):
        """提高质量"""
        if self.current_quality_level == "low":
            self.current_quality_level = "medium"
            self._apply_quality_level("medium")
        elif self.current_quality_level == "medium":
            self.current_quality_level = "high"
            self._apply_quality_level("high")
    
    def _slightly_increase_quality(self):
        """轻微提高质量"""
        if self.current_quality_level == "low":
            self._apply_quality_level("medium_low")
        elif self.current_quality_level == "medium_low":
            self._apply_quality_level("medium")
    
    def _apply_quality_level(self, level: str):
        """应用质量等级"""
        # 根据等级获取配置
        if level == "low":
            config = self.config_manager.default_configs["low_end_gpu"]
        elif level == "medium_low":
            # 创建介于低和中之间的配置
            low_cfg = self.config_manager.default_configs["low_end_gpu"]
            med_cfg = self.config_manager.default_configs["mid_range_gpu"]
            config = self._interpolate_configs(low_cfg, med_cfg, 0.5)
        elif level == "medium":
            config = self.config_manager.default_configs["mid_range_gpu"]
        elif level == "high":
            config = self.config_manager.default_configs["high_end_gpu"]
        else:
            config = self.config_manager.default_configs["low_end_gpu"]
        
        # 更新当前配置
        self.config_manager.current_config = config
    
    def _interpolate_configs(self, config1: Dict, config2: Dict, ratio: float) -> Dict:
        """在两个配置之间插值"""
        interpolated = {}
        for key in config1.keys():
            val1, val2 = config1[key], config2[key]
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                interpolated[key] = val1 + (val2 - val1) * ratio
            else:
                # 对于非数值类型，选择第一个配置的值
                interpolated[key] = val1
        return interpolated
    
    def get_current_config_summary(self) -> Dict[str, Any]:
        """获取当前配置摘要"""
        performance_impact = self.config_manager.get_performance_impact_estimate()
        
        return {
            "current_quality_level": self.current_quality_level,
            "target_fps": self.target_fps,
            "config": self.config_manager.current_config,
            "performance_impact": performance_impact,
            "adaptation_history_length": len(self.adaptation_history)
        }


# GPU检测和配置工具
class GPUAnalyzer:
    """GPU分析器，用于检测GPU型号并推荐设置"""
    
    @staticmethod
    def detect_gpu():
        """检测当前GPU（简化版实现）"""
        try:
            import subprocess
            # 这是一个简化的方法，实际项目中应使用更可靠的GPU检测
            result = subprocess.run(['wmic', 'path', 'win32_videocontroller', 'get', 'name'], 
                                  capture_output=True, text=True)
            gpu_name = result.stdout.strip().split('\n')[1].strip()  # 获取GPU名称
            return gpu_name or "Unknown GPU"
        except:
            return "Unknown GPU"
    
    @staticmethod
    def recommend_settings(gpu_name: str) -> Dict[str, Any]:
        """根据GPU推荐设置"""
        config_manager = VolumetricLightConfigManager()
        recommended_config = config_manager.get_config_for_gpu(gpu_name)
        
        # 添加推荐说明
        if 'gtx 1650 max-q' in gpu_name.lower():
            recommendation_note = "针对GTX 1650 Max-Q的优化设置，平衡性能与视觉效果"
        elif 'gtx 1650' in gpu_name.lower():
            recommendation_note = "针对GTX 1650的优化设置，注重性能稳定性"
        else:
            recommendation_note = "基于GPU能力的自动优化设置"
        
        return {
            "config": recommended_config,
            "note": recommendation_note,
            "gpu_detected": gpu_name
        }