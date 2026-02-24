import numpy as np
import logging
from typing import Optional, Tuple
from ...Core.FactBase import FactBase

# 尝试导入引擎类型以进行类型提示，但要在运行时处理循环依赖或缺失
try:
    from Engine.Renderer.Terrain.TerrainManager import TerrainChunk
except ImportError:
    TerrainChunk = None

class TerrainLoader:
    """
    地形加载器 (TerrainLoader)
    
    负责将 HT Engine 的 TerrainChunk 数据转换为 Natural 模块的 Facts。
    计算并缓存衍生数据（如坡度、法线），供规则使用。
    """
    
    def __init__(self, fact_base: FactBase):
        self.logger = logging.getLogger("Natural.TerrainLoader")
        self.fact_base = fact_base
        
    def load_chunk(self, chunk: 'TerrainChunk', chunk_id: str):
        """
        加载一个地形块到事实库
        
        Args:
            chunk: Engine 的 TerrainChunk 实例
            chunk_id: 唯一标识符 (e.g., "chunk_0_0")
        """
        if chunk.heightmap is None:
            self.logger.warning(f"TerrainChunk {chunk_id} has no heightmap data")
            return

        # 1. 提取基础数据
        heightmap = chunk.heightmap # shape: (res, res)
        resolution = chunk.resolution
        size = chunk.size
        
        # 2. 计算衍生数据 (坡度)
        # 使用 numpy.gradient 计算梯度
        # gradient 返回 (dy, dx)
        # cell_size = size / (resolution - 1)
        cell_size = size / max(1, resolution - 1)
        
        grad_y, grad_x = np.gradient(heightmap, cell_size)
        
        # 坡度 (Slope) = sqrt(dx^2 + dy^2)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # 3. 展平数据以存入 SoA Table
        # Natural 偏好一维数组
        flat_height = heightmap.flatten().astype(np.float32)
        flat_slope = slope.flatten().astype(np.float32)
        flat_grad_x = grad_x.flatten().astype(np.float32)
        flat_grad_y = grad_y.flatten().astype(np.float32)
        
        # 4. 生成世界坐标 (可选，视内存而定，也可以在规则中动态计算)
        # 这里为了规则编写方便，预计算 X, Z 坐标
        # grid_x, grid_z = np.meshgrid(...)
        
        # 5. 创建或更新表
        table_name = f"terrain_{chunk_id}"
        capacity = resolution * resolution
        
        schema = {
            "height": np.float32,
            "slope": np.float32,
            "grad_x": np.float32,
            "grad_y": np.float32
        }
        
        # 检查表是否存在，不存在则创建
        if table_name not in self.fact_base.tables:
            self.fact_base.create_table(table_name, capacity, schema)
            
        # 写入数据 (直接覆盖整列)
        self.fact_base.set_column(table_name, "height", flat_height)
        self.fact_base.set_column(table_name, "slope", flat_slope)
        self.fact_base.set_column(table_name, "grad_x", flat_grad_x)
        self.fact_base.set_column(table_name, "grad_y", flat_grad_y)
        
        # 更新计数
        self.fact_base.counts[table_name] = capacity
        
        self.logger.info(f"Loaded terrain chunk {chunk_id} into FactBase. Size: {resolution}x{resolution}")
