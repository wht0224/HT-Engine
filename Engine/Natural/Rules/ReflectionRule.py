import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class ReflectionRule(Rule):
    """
    反射规则 (Reflection Rule)
    
    核心思想：光遇到表面会"反弹"，反弹方式由表面"性格"决定。
    不是BRDF计算，而是表面特性决定光的行为。
    
    逻辑链条：
    1. 光滑表面：光"整齐地"反弹（镜面反射）
    2. 粗糙表面：光"散乱地"反弹（漫反射）
    3. 半透明表面：光"部分穿过"（折射）
    4. 反射的光会继续传播，照亮其他区域
    
    Attributes:
        smoothness_threshold (float): 光滑度阈值，决定镜面/漫反射
        reflection_range (float): 反射光传播的最大距离
        max_bounces (int): 最大反射次数
    """
    
    def __init__(self, smoothness_threshold=0.7, reflection_range=20.0, max_bounces=2):
        super().__init__("Lighting.Reflection", priority=90)
        self.smoothness_threshold = smoothness_threshold
        self.reflection_range = reflection_range
        self.max_bounces = max_bounces
        
    def evaluate(self, facts: FactBase):
        """执行反射规则（向量化版本）。"""
        table_name = "terrain_main"
        
        try:
            flat_height = facts.get_column(table_name, "height")
            flat_roughness = facts.get_column(table_name, "roughness")
            flat_wetness = facts.get_column(table_name, "wetness")
            
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            roughness = flat_roughness.reshape((size, size))
            wetness = flat_wetness.reshape((size, size))
            
            try:
                direct = facts.get_column(table_name, "shadow_mask").reshape((size, size))
            except KeyError:
                direct = np.ones_like(H)
            
            try:
                indirect = facts.get_column(table_name, "indirect_light").reshape((size, size))
            except KeyError:
                indirect = np.zeros_like(H)
            
            total_light = direct + indirect
            
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
            sun_dir = sun_dir / (np.linalg.norm(sun_dir) + 1e-5)
            
            # 向量化计算反射
            reflection_map = self._calculate_reflection_vectorized(
                H, roughness, wetness, total_light, sun_dir, size
            )
            
            facts.set_column(table_name, "reflection", reflection_map.flatten())
            facts.set_global("lighting_reflection", reflection_map)
            
        except KeyError:
            pass
    
    def _calculate_reflection_vectorized(self, height_map, roughness, wetness, 
                                        total_light, sun_dir, size):
        """
        向量化反射计算。
        
        核心优化：
        - 批量计算所有像素的反射
        - 使用numpy数组操作
        - 避免Python循环
        """
        # 计算有效粗糙度（湿润度降低粗糙度）
        effective_roughness = roughness * (1.0 - wetness * 0.5)
        
        # 判断反射类型
        is_specular = effective_roughness < self.smoothness_threshold
        is_diffuse = ~is_specular
        
        # 初始化反射强度图
        reflection_intensity = np.zeros_like(height_map, dtype=np.float32)
        
        # 镜面反射强度
        specular_mask = is_specular & (total_light > 0.1)
        reflection_intensity[specular_mask] = (
            (1.0 - effective_roughness[specular_mask]) * 
            total_light[specular_mask] * 
            (1.0 + wetness[specular_mask] * 0.5)
        )
        
        # 漫反射强度
        diffuse_mask = is_diffuse & (total_light > 0.1)
        reflection_intensity[diffuse_mask] = (
            (1.0 - effective_roughness[diffuse_mask]) * 
            total_light[diffuse_mask] * 
            0.5
        )
        
        # 限制最大值
        reflection_intensity = np.clip(reflection_intensity, 0.0, 1.0)
        
        return reflection_intensity
