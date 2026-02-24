"""
三级推理规则 - 细节增强
"""

import numpy as np
from typing import List
from ..TerrainInferenceSystem import TerrainRuleBase, TerrainFactBase


def _generate_perlin_noise(height: int, width: int, scale: float = 10.0, 
                           octaves: int = 4, persistence: float = 0.5) -> np.ndarray:
    """生成Perlin噪声"""
    try:
        from noise import pnoise2
        noise = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                nx = x / scale
                ny = y / scale
                value = 0
                amplitude = 1
                frequency = 1
                max_value = 0
                for _ in range(octaves):
                    value += pnoise2(nx * frequency, ny * frequency) * amplitude
                    max_value += amplitude
                    amplitude *= persistence
                    frequency *= 2
                noise[y, x] = value / max_value
        return noise
    except ImportError:
        noise = np.random.rand(height, width).astype(np.float32) * 2 - 1
        from scipy.ndimage import gaussian_filter
        for _ in range(3):
            noise = gaussian_filter(noise, sigma=scale / 5)
        return noise


class RockDetailRule(TerrainRuleBase):
    """
    岩石细节规则
    
    在陡崖和山脊添加岩石细节
    依赖: elevation, is_cliff, is_ridge, slope
    输出: elevation_enhanced (更新)
    """
    
    def __init__(self, detail_scale: float = 5.0, intensity: float = 2.0):
        super().__init__("RockDetailRule", priority=30)
        self.detail_scale = detail_scale
        self.intensity = intensity
    
    def get_dependencies(self) -> List[str]:
        return ["elevation", "is_cliff", "is_ridge", "slope"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        is_cliff = facts.get("is_cliff")
        is_ridge = facts.get("is_ridge")
        slope = facts.get("slope")
        
        if elevation is None:
            return
        
        enhanced = elevation.copy()
        height, width = elevation.shape
        
        detail = _generate_perlin_noise(height, width, self.detail_scale)
        
        rock_mask = np.zeros((height, width), dtype=np.float32)
        if is_cliff is not None:
            rock_mask += is_cliff
        if is_ridge is not None:
            rock_mask += is_ridge * 0.5
        
        if slope is not None:
            slope_factor = np.clip(slope / 1.0, 0, 1)
            rock_mask = rock_mask * slope_factor
        
        rock_mask = np.clip(rock_mask, 0, 1)
        
        enhanced += detail * rock_mask * self.intensity
        
        facts.set("elevation_enhanced", enhanced, "增强后的高程")


class ErosionRule(TerrainRuleBase):
    """
    侵蚀痕迹规则
    
    在河谷添加侵蚀痕迹
    依赖: elevation_enhanced, is_valley, flow_accumulation
    输出: elevation_enhanced (更新)
    """
    
    def __init__(self, erosion_depth: float = 3.0):
        super().__init__("ErosionRule", priority=31)
        self.erosion_depth = erosion_depth
    
    def get_dependencies(self) -> List[str]:
        return ["elevation_enhanced", "is_valley", "flow_accumulation"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation_enhanced")
        if elevation is None:
            elevation = facts.get("elevation")
        
        is_valley = facts.get("is_valley")
        flow = facts.get("flow_accumulation")
        
        if elevation is None:
            return
        
        enhanced = elevation.copy()
        
        if is_valley is not None and flow is not None:
            erosion_mask = is_valley * flow
            enhanced -= erosion_mask * self.erosion_depth
        
        facts.set("elevation_enhanced", enhanced, "增强后的高程")


class VegetationDensityRule(TerrainRuleBase):
    """
    植被密度规则
    
    根据地形计算植被密度
    依赖: slope, elevation, is_cliff, is_water
    输出: vegetation_density
    """
    
    def __init__(self, treeline: float = 1000.0, max_slope: float = 0.7):
        super().__init__("VegetationDensityRule", priority=32)
        self.treeline = treeline
        self.max_slope = max_slope
    
    def get_dependencies(self) -> List[str]:
        return ["slope", "elevation", "is_cliff", "is_water"]
    
    def evaluate(self, facts: TerrainFactBase):
        slope = facts.get("slope")
        elevation = facts.get("elevation")
        is_cliff = facts.get("is_cliff")
        is_water = facts.get("is_water")
        
        if slope is None or elevation is None:
            return
        
        density = np.ones_like(elevation, dtype=np.float32)
        
        slope_factor = 1.0 - np.clip(slope / self.max_slope, 0, 1)
        density *= slope_factor
        
        if self.treeline > 0:
            elev_factor = 1.0 - np.clip(
                (elevation - self.treeline * 0.7) / (self.treeline * 0.3),
                0, 1
            )
            density *= elev_factor
        
        if is_cliff is not None:
            density *= (1.0 - is_cliff)
        
        if is_water is not None:
            density *= (1.0 - is_water)
        
        density = np.clip(density, 0, 1)
        
        facts.set("vegetation_density", density, "植被密度")


class SnowCoverageRule(TerrainRuleBase):
    """
    积雪覆盖规则
    
    根据高度和坡向计算积雪覆盖
    依赖: elevation, slope, aspect
    输出: snow_coverage
    """
    
    def __init__(self, snow_line: float = 800.0, season: str = "summer"):
        super().__init__("SnowCoverageRule", priority=33)
        self.snow_line = snow_line
        self.season = season
        
        self.season_factors = {
            "winter": 1.5,
            "spring": 1.2,
            "summer": 1.0,
            "autumn": 1.1
        }
    
    def get_dependencies(self) -> List[str]:
        return ["elevation", "slope", "aspect"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation")
        slope = facts.get("slope")
        aspect = facts.get("aspect")
        
        if elevation is None:
            return
        
        season_factor = self.season_factors.get(self.season, 1.0)
        effective_snow_line = self.snow_line * season_factor
        
        snow = np.zeros_like(elevation, dtype=np.float32)
        
        elev_factor = np.clip(
            (elevation - effective_snow_line * 0.8) / (effective_snow_line * 0.4),
            0, 1
        )
        snow += elev_factor * 0.7
        
        if slope is not None:
            slope_factor = 1.0 - np.clip(slope / 1.0, 0, 1) * 0.3
            snow *= slope_factor
        
        if aspect is not None:
            north_factor = (np.cos(aspect) + 1) / 2
            snow += north_factor * 0.2
        
        snow = np.clip(snow, 0, 1)
        
        facts.set("snow_coverage", snow, "积雪覆盖")


class TerrainSmoothingRule(TerrainRuleBase):
    """
    地形平滑规则
    
    对最终地形进行平滑处理
    依赖: elevation_enhanced
    输出: elevation_final
    """
    
    def __init__(self, iterations: int = 1):
        super().__init__("TerrainSmoothingRule", priority=99)
        self.iterations = iterations
    
    def get_dependencies(self) -> List[str]:
        return ["elevation_enhanced"]
    
    def evaluate(self, facts: TerrainFactBase):
        elevation = facts.get("elevation_enhanced")
        if elevation is None:
            elevation = facts.get("elevation")
        
        if elevation is None:
            return
        
        smoothed = elevation.copy()
        
        for _ in range(self.iterations):
            kernel = np.array([
                [0.0625, 0.125, 0.0625],
                [0.125,  0.25,  0.125],
                [0.0625, 0.125, 0.0625]
            ])
            
            from scipy.ndimage import convolve
            smoothed = convolve(smoothed, kernel, mode='nearest')
        
        facts.set("elevation_final", smoothed, "最终高程")
