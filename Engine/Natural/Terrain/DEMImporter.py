"""
DEM数据导入器
支持GeoTIFF格式的数字高程模型数据导入
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any


class DEMImporter:
    """
    DEM数据导入器
    
    支持:
    - GeoTIFF格式
    - 自动裁剪
    - 重采样
    - 坐标转换
    """
    
    def __init__(self):
        self.elevation_data = None
        self.metadata = {}
        self.bounds = None
        
    def load(self, filepath: str, 
             bounds: Optional[Tuple[float, float, float, float]] = None,
             target_resolution: Optional[float] = None) -> np.ndarray:
        """
        加载DEM数据
        
        Args:
            filepath: GeoTIFF文件路径
            bounds: 裁剪边界 (min_x, min_y, max_x, max_y)
            target_resolution: 目标分辨率 (米/像素)
            
        Returns:
            高程数据数组
        """
        try:
            import rasterio
        except ImportError:
            raise ImportError("请安装rasterio: pip install rasterio")
        
        with rasterio.open(filepath) as src:
            self.metadata = {
                'crs': src.crs,
                'transform': src.transform,
                'width': src.width,
                'height': src.height,
                'bounds': src.bounds,
                'resolution': src.res
            }
            
            # 如果需要降采样，在读取时就指定输出形状
            if target_resolution and target_resolution > src.res[0]:
                scale = target_resolution / float(src.res[0])
                out_height = max(1, int(src.height / scale))
                out_width = max(1, int(src.width / scale))
                
                if bounds:
                    window = rasterio.windows.from_bounds(
                        bounds[0], bounds[1], bounds[2], bounds[3],
                        src.transform
                    )
                    self.elevation_data = src.read(1, window=window)
                    self.bounds = bounds
                    self.elevation_data = self._resample(
                        self.elevation_data,
                        float(src.res[0]),
                        float(target_resolution)
                    )
                else:
                    from rasterio.enums import Resampling
                    self.elevation_data = src.read(
                        1,
                        out_shape=(out_height, out_width),
                        resampling=Resampling.average
                    )
                    self.bounds = (src.bounds.left, src.bounds.bottom,
                                  src.bounds.right, src.bounds.top)
            else:
                if bounds:
                    window = rasterio.windows.from_bounds(
                        bounds[0], bounds[1], bounds[2], bounds[3],
                        src.transform
                    )
                    self.elevation_data = src.read(1, window=window)
                    self.bounds = bounds
                else:
                    self.elevation_data = src.read(1)
                    self.bounds = (src.bounds.left, src.bounds.bottom,
                                  src.bounds.right, src.bounds.top)
        
        self.elevation_data = self._clean_data(self.elevation_data)
        
        return self.elevation_data
    
    def _resample(self, data: np.ndarray, 
                  current_res: float, 
                  target_res: float) -> np.ndarray:
        """重采样数据到目标分辨率"""
        if target_res <= current_res:
            return data
        
        scale = target_res / current_res
        new_shape = (
            int(data.shape[0] / scale),
            int(data.shape[1] / scale)
        )
        
        # 使用skimage的resize，更高效
        try:
            from skimage.transform import resize
            return resize(data, new_shape, order=1, preserve_range=True).astype(np.float32)
        except ImportError:
            # 备用方案：使用scipy.ndimage.zoom
            from scipy.ndimage import zoom
            return zoom(data, 1/scale, order=1).astype(np.float32)
    
    def _clean_data(self, data: np.ndarray) -> np.ndarray:
        """清理无效数据"""
        data = data.astype(np.float32)
        nodata = self.metadata.get('nodata', None)
        if nodata is not None:
            data[data == nodata] = np.nan
        
        if np.any(np.isnan(data)):
            from scipy.ndimage import distance_transform_edt
            nan_mask = np.isnan(data)
            if np.any(nan_mask):
                valid_mask = ~nan_mask
                distances = distance_transform_edt(nan_mask)
                nearest_valid = distance_transform_edt(~valid_mask)
                data[nan_mask] = np.interp(
                    distances[nan_mask],
                    distances[valid_mask],
                    data[valid_mask]
                )
        
        return data
    
    def get_elevation_at(self, x: float, y: float) -> float:
        """获取指定坐标的高程值"""
        if self.elevation_data is None:
            raise ValueError("请先加载数据")
        
        transform = self.metadata.get('transform')
        if transform is None:
            raise ValueError("缺少坐标变换信息")
        
        col = int((x - transform.c) / transform.a)
        row = int((y - transform.f) / transform.e)
        
        if 0 <= row < self.elevation_data.shape[0] and \
           0 <= col < self.elevation_data.shape[1]:
            return float(self.elevation_data[row, col])
        
        return 0.0
    
    def get_statistics(self) -> Dict[str, float]:
        """获取高程统计信息"""
        if self.elevation_data is None:
            return {}
        
        return {
            'min': float(np.nanmin(self.elevation_data)),
            'max': float(np.nanmax(self.elevation_data)),
            'mean': float(np.nanmean(self.elevation_data)),
            'std': float(np.nanstd(self.elevation_data)),
            'range': float(np.nanmax(self.elevation_data) - np.nanmin(self.elevation_data))
        }
    
    def generate_mesh_data(self, 
                           scale: float = 1.0,
                           height_scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        生成地形网格数据
        
        Args:
            scale: 水平缩放
            height_scale: 高度缩放
            
        Returns:
            (顶点, 法线, 索引)
        """
        if self.elevation_data is None:
            raise ValueError("请先加载数据")
        
        height, width = self.elevation_data.shape
        
        vertices = []
        normals = []
        
        for y in range(height):
            for x in range(width):
                vx = x * scale
                vy = y * scale
                vz = self.elevation_data[y, x] * height_scale
                vertices.append([vx, vz, vy])
        
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
