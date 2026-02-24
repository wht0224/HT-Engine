"""
一级推理规则 - 地形属性计算
"""

import numpy as np
from typing import List
from ..TerrainInferenceSystem import TerrainRuleBase, TerrainFactBase


class SlopeRule(TerrainRuleBase):
    """
    坡度计算规则 (向量化优化版)
    
    计算每个单元格的坡度（角度）
    依赖: elevation
    输出: slope (弧度)
    """
    
    def __init__(self, cell_size: float = 30.0):
        super().__init__("SlopeRule", priority=10)
        self.cell_size = cell_size
    
    def get_dependencies(self) -> List[str]:
        return ["elevation"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        if elevation is None:
            return
        
        # 使用向量化计算
        dy, dx = np.gradient(elevation, self.cell_size)
        slope = np.arctan(np.sqrt(dx * dx + dy * dy))
        
        facts.set("slope", slope.astype(np.float32), "坡度 (弧度)")


class AspectRule(TerrainRuleBase):
    """
    坡向计算规则 (向量化优化版)
    
    计算每个单元格的坡向（朝向）
    依赖: elevation
    输出: aspect (弧度, 0=东, π/2=北, π=西, 3π/2=南)
    """
    
    def __init__(self, cell_size: float = 30.0):
        super().__init__("AspectRule", priority=11)
        self.cell_size = cell_size
    
    def get_dependencies(self) -> List[str]:
        return ["elevation"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        if elevation is None:
            return
        
        # 使用向量化计算
        dy, dx = np.gradient(elevation, self.cell_size)
        aspect = np.arctan2(-dy, dx)
        
        facts.set("aspect", aspect.astype(np.float32), "坡向 (弧度)")


class CurvatureRule(TerrainRuleBase):
    """
    曲率计算规则 (向量化优化版)
    
    计算每个单元格的曲率
    正值 = 凹陷 (河谷)
    负值 = 凸起 (山脊)
    依赖: elevation
    输出: curvature
    """
    
    def __init__(self, cell_size: float = 30.0):
        super().__init__("CurvatureRule", priority=12)
        self.cell_size = cell_size
    
    def get_dependencies(self) -> List[str]:
        return ["elevation"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        if elevation is None:
            return
        
        h2 = self.cell_size * self.cell_size
        
        # 使用向量化计算二阶导数
        d2z_dx2 = np.zeros_like(elevation)
        d2z_dy2 = np.zeros_like(elevation)
        
        d2z_dx2[:, 1:-1] = (elevation[:, 2:] - 2 * elevation[:, 1:-1] + elevation[:, :-2]) / h2
        d2z_dy2[1:-1, :] = (elevation[2:, :] - 2 * elevation[1:-1, :] + elevation[:-2, :]) / h2
        
        curvature = d2z_dx2 + d2z_dy2
        
        facts.set("curvature", curvature.astype(np.float32), "曲率")


class HillshadeRule(TerrainRuleBase):
    """
    山影计算规则 (向量化优化版)
    
    计算每个单元格的山影值 (用于可视化)
    依赖: elevation, slope, aspect
    输出: hillshade
    """
    
    def __init__(self, sun_azimuth: float = 315.0, sun_altitude: float = 45.0):
        super().__init__("HillshadeRule", priority=13)
        self.sun_azimuth = np.radians(sun_azimuth)
        self.sun_altitude = np.radians(sun_altitude)
    
    def get_dependencies(self) -> List[str]:
        return ["slope", "aspect"]
    
    def evaluate(self, facts: TerrainFactBase):
        slope = facts.get("slope")
        aspect = facts.get("aspect")
        
        if slope is None or aspect is None:
            return
        
        zenith = np.pi / 2 - self.sun_altitude
        azimuth = np.pi - self.sun_azimuth
        
        hillshade = (
            np.cos(zenith) * np.cos(slope) +
            np.sin(zenith) * np.sin(slope) * np.cos(azimuth - aspect)
        )
        
        hillshade = np.clip(hillshade, 0, 1)
        
        facts.set("hillshade", hillshade.astype(np.float32), "山影值")
