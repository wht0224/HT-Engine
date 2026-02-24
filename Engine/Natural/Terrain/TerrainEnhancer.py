"""
地形增强器
整合DEM导入和推理系统的高级接口
"""

import numpy as np
from typing import Optional, Dict, Any, Tuple
from pathlib import Path

from .DEMImporter import DEMImporter
from .TerrainInferenceSystem import TerrainInferenceSystem
from .Rules import (
    SlopeRule, AspectRule, CurvatureRule, HillshadeRule,
    RidgeRule, ValleyRule, CliffRule, PlateauRule, WaterBodyRule, DrainageRule,
    RockDetailRule, ErosionRule, VegetationDensityRule,
    SnowCoverageRule, TerrainSmoothingRule
)


class TerrainEnhancer:
    """
    地形增强器
    
    整合DEM导入和符号主义推理系统，提供一站式地形增强功能。
    
    使用示例:
        enhancer = TerrainEnhancer()
        enhancer.load_dem("terrain.tif")
        enhancer.setup_rules()
        results = enhancer.enhance()
        vertices, normals, indices = enhancer.generate_mesh()
    """
    
    def __init__(self, cell_size: float = 30.0):
        """
        初始化地形增强器
        
        Args:
            cell_size: DEM单元格大小 (米)
        """
        self.cell_size = cell_size
        self.importer = DEMImporter()
        self.inference_system = None
        self._is_loaded = False
        self._is_enhanced = False
    
    def load_dem(self, filepath: str, 
                 bounds: Optional[Tuple[float, float, float, float]] = None,
                 target_resolution: Optional[float] = None) -> np.ndarray:
        """
        加载DEM数据
        
        Args:
            filepath: GeoTIFF文件路径
            bounds: 裁剪边界 (min_x, min_y, max_x, max_y)
            target_resolution: 目标分辨率
            
        Returns:
            高程数据数组
        """
        elevation = self.importer.load(filepath, bounds, target_resolution)
        
        self.inference_system = TerrainInferenceSystem(elevation, self.cell_size)
        self._is_loaded = True
        self._is_enhanced = False
        
        return elevation
    
    def setup_rules(self, 
                    enable_primary: bool = True,
                    enable_secondary: bool = True,
                    enable_tertiary: bool = True,
                    custom_config: Optional[Dict[str, Any]] = None):
        """
        设置推理规则
        
        Args:
            enable_primary: 启用一级规则 (坡度、坡向、曲率)
            enable_secondary: 启用二级规则 (山脊、河谷、陡崖)
            enable_tertiary: 启用三级规则 (细节增强)
            custom_config: 自定义配置
        """
        if not self._is_loaded:
            raise ValueError("请先加载DEM数据")
        
        config = custom_config or {}
        
        if enable_primary:
            self.inference_system.add_rule(SlopeRule(
                cell_size=config.get('cell_size', self.cell_size)
            ))
            self.inference_system.add_rule(AspectRule(
                cell_size=config.get('cell_size', self.cell_size)
            ))
            self.inference_system.add_rule(CurvatureRule(
                cell_size=config.get('cell_size', self.cell_size)
            ))
            self.inference_system.add_rule(HillshadeRule(
                sun_azimuth=config.get('sun_azimuth', 315.0),
                sun_altitude=config.get('sun_altitude', 45.0)
            ))
        
        if enable_secondary:
            self.inference_system.add_rule(RidgeRule(
                curvature_threshold=config.get('ridge_curvature_threshold', -0.001),
                min_slope=config.get('ridge_min_slope', 0.1)
            ))
            self.inference_system.add_rule(ValleyRule(
                curvature_threshold=config.get('valley_curvature_threshold', 0.001)
            ))
            self.inference_system.add_rule(CliffRule(
                slope_threshold=config.get('cliff_slope_threshold', 0.785)
            ))
            self.inference_system.add_rule(PlateauRule(
                slope_threshold=config.get('plateau_slope_threshold', 0.087)
            ))
            self.inference_system.add_rule(WaterBodyRule(
                slope_threshold=config.get('water_slope_threshold', 0.017)
            ))
            self.inference_system.add_rule(DrainageRule(
                iterations=config.get('drainage_iterations', 100)
            ))
        
        if enable_tertiary:
            self.inference_system.add_rule(RockDetailRule(
                detail_scale=config.get('rock_detail_scale', 5.0),
                intensity=config.get('rock_intensity', 2.0)
            ))
            self.inference_system.add_rule(ErosionRule(
                erosion_depth=config.get('erosion_depth', 3.0)
            ))
            self.inference_system.add_rule(VegetationDensityRule(
                treeline=config.get('treeline', 1000.0),
                max_slope=config.get('max_vegetation_slope', 0.7)
            ))
            self.inference_system.add_rule(SnowCoverageRule(
                snow_line=config.get('snow_line', 800.0),
                season=config.get('season', 'summer')
            ))
            self.inference_system.add_rule(TerrainSmoothingRule(
                iterations=config.get('smoothing_iterations', 1)
            ))
    
    def enhance(self, verbose: bool = False) -> Dict[str, Any]:
        """
        执行地形增强
        
        Args:
            verbose: 是否输出详细信息
            
        Returns:
            推理结果统计
        """
        if not self._is_loaded:
            raise ValueError("请先加载DEM数据")
        
        if len(self.inference_system.rules) == 0:
            self.setup_rules()
        
        results = self.inference_system.run(verbose=verbose)
        self._is_enhanced = True
        
        return results
    
    def get_enhanced_elevation(self) -> np.ndarray:
        """获取增强后的高程数据"""
        if not self._is_enhanced:
            raise ValueError("请先执行enhance()")
        
        elevation = self.inference_system.get_result("elevation_final")
        if elevation is None:
            elevation = self.inference_system.get_result("elevation_enhanced")
        if elevation is None:
            elevation = self.inference_system.get_result("elevation")
        
        return elevation
    
    def get_all_results(self) -> Dict[str, np.ndarray]:
        """获取所有推理结果"""
        return self.inference_system.get_all_results()
    
    def generate_mesh(self, 
                      scale: float = 1.0, 
                      height_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成地形网格
        
        Args:
            scale: 水平缩放
            height_scale: 高度缩放
            
        Returns:
            (顶点, 法线, 索引)
        """
        elevation = self.get_enhanced_elevation()
        
        height, width = elevation.shape
        
        vertices = []
        for y in range(height):
            for x in range(width):
                vx = x * scale
                vy = elevation[y, x] * height_scale
                vz = y * scale
                vertices.append([vx, vy, vz])
        
        vertices = np.array(vertices, dtype=np.float32)
        
        normals = self._compute_normals(vertices, width, height)
        
        indices = []
        for y in range(height - 1):
            for x in range(width - 1):
                i0 = y * width + x
                i1 = y * width + (x + 1)
                i2 = (y + 1) * width + x
                i3 = (y + 1) * width + (x + 1)
                
                indices.append([i0, i2, i1])
                indices.append([i1, i2, i3])
        
        indices = np.array(indices, dtype=np.uint32)
        
        return vertices, normals, indices
    
    def _compute_normals(self, vertices: np.ndarray, 
                         width: int, height: int) -> np.ndarray:
        """计算顶点法线"""
        normals = np.zeros_like(vertices)
        
        for y in range(height):
            for x in range(width):
                i = y * width + x
                
                if x > 0 and x < width - 1:
                    dx = vertices[i + 1] - vertices[i - 1]
                elif x == 0:
                    dx = vertices[i + 1] - vertices[i]
                else:
                    dx = vertices[i] - vertices[i - 1]
                
                if y > 0 and y < height - 1:
                    dy = vertices[i + width] - vertices[i - width]
                elif y == 0:
                    dy = vertices[i + width] - vertices[i]
                else:
                    dy = vertices[i] - vertices[i - width]
                
                normal = np.cross(dy, dx)
                norm = np.linalg.norm(normal)
                if norm > 0:
                    normal = normal / norm
                normals[i] = normal
        
        return normals
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取地形统计信息"""
        stats = self.importer.get_statistics()
        
        if self._is_enhanced:
            enhanced = self.get_enhanced_elevation()
            stats['enhanced_min'] = float(np.min(enhanced))
            stats['enhanced_max'] = float(np.max(enhanced))
            stats['enhanced_mean'] = float(np.mean(enhanced))
        
        return stats
    
    def export_heightmap(self, filepath: str, 
                         size: Optional[Tuple[int, int]] = None):
        """
        导出高度图
        
        Args:
            filepath: 输出文件路径
            size: 输出尺寸 (width, height)
        """
        elevation = self.get_enhanced_elevation()
        
        if size:
            from scipy.ndimage import zoom
            scale_y = size[1] / elevation.shape[0]
            scale_x = size[0] / elevation.shape[1]
            elevation = zoom(elevation, (scale_y, scale_x), order=1)
        
        normalized = (elevation - elevation.min()) / (elevation.max() - elevation.min())
        
        import cv2
        heightmap = (normalized * 65535).astype(np.uint16)
        cv2.imwrite(filepath, heightmap)
