import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class OcclusionRule(Rule):
    """
    遮挡规则 (Occlusion Rule) - 软阴影的因果逻辑
    
    核心思想：阴影的软硬由遮挡物的距离决定。
    不是PCF采样，而是"遮挡程度"的自然推导。
    
    逻辑链条：
    1. 完全遮挡 = 硬阴影（遮挡物很近）
    2. 部分遮挡 = 软阴影（遮挡物有一定距离）
    3. 遮挡物越远，阴影越软
    4. 多个遮挡物会叠加遮挡效果
    
    Attributes:
        hard_shadow_threshold (float): 硬阴影距离阈值
        soft_shadow_range (float): 软阴影最大距离
        penumbra_scale (float): 半影区缩放因子
    """
    
    def __init__(self, hard_shadow_threshold=2.0, soft_shadow_range=15.0, penumbra_scale=1.0):
        super().__init__("Lighting.Occlusion", priority=100)
        self.hard_shadow_threshold = hard_shadow_threshold
        self.soft_shadow_range = soft_shadow_range
        self.penumbra_scale = penumbra_scale
        
    def evaluate(self, facts: FactBase):
        """
        执行遮挡规则，计算软阴影（向量化版本）。"""
        table_name = "terrain_main"
        
        try:
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-5)
            
            # 使用向量化计算
            shadow_map, shadow_softness = self._calculate_occlusion_vectorized(H, sun_dir, size)
            
            facts.set_column(table_name, "shadow_soft", shadow_map.flatten())
            facts.set_column(table_name, "shadow_softness", shadow_softness.flatten())
            facts.set_global("lighting_soft_shadows", shadow_map)
            
        except KeyError:
            pass
    
    def _calculate_occlusion_vectorized(self, height_map, sun_dir, size):
        """
        向量化软阴影计算。
        
        核心优化：
        - 使用numpy数组操作替代Python循环
        - 批量处理所有像素
        - 减少Python解释器开销
        """
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # 太阳方向
        sun_xz = np.array([sun_dir[0], sun_dir[2]])
        sun_xz_norm = np.linalg.norm(sun_xz)
        
        if sun_xz_norm < 1e-5 or sun_dir[1] >= 0:
            # 太阳直射或在下方
            return np.ones_like(height_map), np.zeros_like(height_map)
        
        sun_xz = sun_xz / sun_xz_norm
        sun_height_factor = -sun_dir[1] / sun_xz_norm
        
        # 初始化结果
        shadow_map = np.ones_like(height_map, dtype=np.float32)
        shadow_softness = np.zeros_like(height_map, dtype=np.float32)
        
        # 向量化Ray Marching
        step_size = 1.0
        max_steps = int(self.soft_shadow_range * 2 / step_size)
        
        for step in range(1, max_steps + 1):
            distance = step * step_size
            
            # 计算所有像素的采样位置（向量化）
            sample_x = x_coords + sun_xz[0] * distance
            sample_y = y_coords + sun_xz[1] * distance
            
            # 边界掩码
            valid_mask = (sample_x >= 0) & (sample_x < size - 1) & \
                        (sample_y >= 0) & (sample_y < size - 1)
            
            # 计算射线高度（向量化）
            ray_height = height_map + sun_height_factor * distance
            
            # 双线性插值采样高度（向量化）
            x0 = np.clip(sample_x.astype(np.int32), 0, size - 2)
            y0 = np.clip(sample_y.astype(np.int32), 0, size - 2)
            x1 = x0 + 1
            y1 = y0 + 1
            
            wx = sample_x - x0
            wy = sample_y - y0
            
            # 采样四个角
            h00 = height_map[y0, x0]
            h10 = height_map[y0, x1]
            h01 = height_map[y1, x0]
            h11 = height_map[y1, x1]
            
            sample_height = (1 - wx) * (1 - wy) * h00 + \
                           wx * (1 - wy) * h10 + \
                           (1 - wx) * wy * h01 + \
                           wx * wy * h11
            
            # 检测遮挡（向量化）
            occluded = (sample_height > ray_height) & valid_mask & (shadow_map > 0.5)
            
            # 更新阴影值（只在第一次遮挡时）
            new_shadow = np.where(occluded, 
                                 self._compute_shadow_value(distance, sample_height - ray_height),
                                 shadow_map)
            shadow_map = np.where(shadow_map > 0.99, new_shadow, shadow_map)
            
            # 计算软度
            new_softness = self._compute_softness_vectorized(distance, occluded)
            shadow_softness = np.maximum(shadow_softness, new_softness)
            
            # 如果全部遮挡，提前退出
            if np.all(shadow_map < 0.5):
                break
        
        return shadow_map, shadow_softness
    
    def _compute_shadow_value(self, distance, height_diff):
        """计算阴影值（简化版）"""
        base_shadow = 1.0 - np.minimum(1.0, height_diff * 0.1)
        
        if distance < self.hard_shadow_threshold:
            return base_shadow
        elif distance < self.soft_shadow_range:
            softness = (distance - self.hard_shadow_threshold) / \
                      (self.soft_shadow_range - self.hard_shadow_threshold)
            return base_shadow * (0.3 + 0.7 * (1.0 - softness))
        else:
            return base_shadow * 0.3
    
    def _compute_softness_vectorized(self, distance, occluded):
        """向量化软度计算"""
        softness = np.zeros_like(occluded, dtype=np.float32)
        
        if distance < self.hard_shadow_threshold:
            softness = np.where(occluded, 0.0, softness)
        elif distance < self.soft_shadow_range:
            s = (distance - self.hard_shadow_threshold) / \
                (self.soft_shadow_range - self.hard_shadow_threshold)
            softness = np.where(occluded, s, softness)
        else:
            softness = np.where(occluded, 1.0, softness)
        
        return softness
