"""
二级推理规则 - 地形特征识别
"""

import numpy as np
from typing import List
from ..TerrainInferenceSystem import TerrainRuleBase, TerrainFactBase


class RidgeRule(TerrainRuleBase):
    """
    山脊线识别规则
    
    识别山脊线位置
    依赖: slope, curvature
    输出: is_ridge (布尔数组)
    """
    
    def __init__(self, curvature_threshold: float = -0.001, min_slope: float = 0.1):
        super().__init__("RidgeRule", priority=20)
        self.curvature_threshold = curvature_threshold
        self.min_slope = min_slope
    
    def get_dependencies(self) -> List[str]:
        return ["slope", "curvature"]
    
    def evaluate(self, facts: TerrainFactBase):
        slope = facts.get("slope")
        curvature = facts.get("curvature")
        
        if slope is None or curvature is None:
            return
        
        is_ridge = (
            (curvature < self.curvature_threshold) &
            (slope > self.min_slope)
        ).astype(np.float32)
        
        facts.set("is_ridge", is_ridge, "山脊线")


class ValleyRule(TerrainRuleBase):
    """
    河谷线识别规则
    
    识别河谷线位置
    依赖: curvature
    输出: is_valley (布尔数组)
    """
    
    def __init__(self, curvature_threshold: float = 0.001):
        super().__init__("ValleyRule", priority=21)
        self.curvature_threshold = curvature_threshold
    
    def get_dependencies(self) -> List[str]:
        return ["curvature"]
    
    def evaluate(self, facts: TerrainFactBase):
        curvature = facts.get("curvature")
        
        if curvature is None:
            return
        
        is_valley = (curvature > self.curvature_threshold).astype(np.float32)
        
        facts.set("is_valley", is_valley, "河谷线")


class CliffRule(TerrainRuleBase):
    """
    陡崖识别规则
    
    识别陡崖位置
    依赖: slope
    输出: is_cliff (布尔数组)
    """
    
    def __init__(self, slope_threshold: float = 0.785):
        super().__init__("CliffRule", priority=22)
        self.slope_threshold = slope_threshold
    
    def get_dependencies(self) -> List[str]:
        return ["slope"]
    
    def evaluate(self, facts: TerrainFactBase):
        slope = facts.get("slope")
        
        if slope is None:
            return
        
        is_cliff = (slope > self.slope_threshold).astype(np.float32)
        
        facts.set("is_cliff", is_cliff, "陡崖")


class PlateauRule(TerrainRuleBase):
    """
    平台识别规则
    
    识别平台/平坦区域
    依赖: slope, elevation
    输出: is_plateau (布尔数组)
    """
    
    def __init__(self, slope_threshold: float = 0.087):
        super().__init__("PlateauRule", priority=23)
        self.slope_threshold = slope_threshold
    
    def get_dependencies(self) -> List[str]:
        return ["slope", "elevation"]
    
    def evaluate(self, facts: TerrainFactBase):
        slope = facts.get("slope")
        elevation = facts.get("elevation")
        
        if slope is None or elevation is None:
            return
        
        avg_elevation = np.mean(elevation)
        
        is_plateau = (
            (slope < self.slope_threshold) &
            (elevation > avg_elevation * 0.5)
        ).astype(np.float32)
        
        facts.set("is_plateau", is_plateau, "平台")


class WaterBodyRule(TerrainRuleBase):
    """
    水体识别规则
    
    识别可能的湖泊/水体区域
    依赖: elevation, slope
    输出: is_water (布尔数组)
    """
    
    def __init__(self, slope_threshold: float = 0.017):
        super().__init__("WaterBodyRule", priority=24)
        self.slope_threshold = slope_threshold
    
    def get_dependencies(self) -> List[str]:
        return ["elevation", "slope"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        slope = facts.get("slope")
        
        if elevation is None or slope is None:
            return
        
        min_elevation = np.min(elevation)
        
        is_water = (
            (slope < self.slope_threshold) &
            (elevation < min_elevation + 10)
        ).astype(np.float32)
        
        facts.set("is_water", is_water, "水体")


class DrainageRule(TerrainRuleBase):
    """
    排水路径规则 (优化版)
    
    计算水流累积量 (简化版)
    依赖: elevation
    输出: flow_accumulation
    """
    
    def __init__(self, iterations: int = 10):
        super().__init__("DrainageRule", priority=25)
        self.iterations = iterations
    
    def get_dependencies(self) -> List[str]:
        return ["elevation"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        
        if elevation is None:
            return
        
        height, width = elevation.shape
        
        # 使用向量化计算代替循环
        flow = np.ones((height, width), dtype=np.float32)
        
        # 简化的水流累积计算
        for _ in range(self.iterations):
            # 计算梯度方向
            dy, dx = np.gradient(elevation)
            
            # 根据梯度方向累积流量
            flow_x = np.where(dx > 0, np.roll(flow, -1, axis=1), 0)
            flow_x += np.where(dx < 0, np.roll(flow, 1, axis=1), 0)
            flow_y = np.where(dy > 0, np.roll(flow, -1, axis=0), 0)
            flow_y += np.where(dy < 0, np.roll(flow, 1, axis=0), 0)
            
            flow = flow + (flow_x + flow_y) * 0.25
        
        # 归一化
        flow = np.log1p(flow)
        flow = (flow - flow.min()) / (flow.max() - flow.min() + 1e-10)
        
        facts.set("flow_accumulation", flow, "水流累积量")
