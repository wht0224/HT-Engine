import numpy as np
from scipy.ndimage import convolve
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class LightingRule(Rule):
    """
    自然光照规则 (Lighting Rule)
    
    目标: 替代传统光线追踪，实现静态全局光照 (GI) 和动态地形阴影。
    核心哲学: Zero Raytracing (零光追，纯数学推导)
    
    包含:
    1. 环境光遮蔽 (Ambient Occlusion) - 基于曲率计算
    2. 地形阴影 (Terrain Shadows) - 基于高度图的Ray Marching
    """
    
    def __init__(self, shadow_step_size=1.0, max_shadow_steps=100, 
                 ao_strength=1.0, shadow_strength=1.0):
        super().__init__("Lighting.Natural", priority=80)
        self.shadow_step_size = shadow_step_size
        self.max_shadow_steps = max_shadow_steps
        self.ao_strength = ao_strength
        self.shadow_strength = shadow_strength
        
        # 曲率计算卷积核 (Laplacian approximation)
        self.curvature_kernel = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=np.float32)
        
    def evaluate(self, facts: FactBase):
        """执行光照计算"""
        table_name = "terrain_main"
        
        try:
            # 获取地形数据
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            
            # 获取太阳方向
            sun_dir = facts.get_global("sun_direction")
            if sun_dir is None:
                sun_dir = np.array([0.5, -1.0, 0.3], dtype=np.float32)
                sun_dir = sun_dir / np.linalg.norm(sun_dir)
            
            # 1. 计算环境光遮蔽 (AO)
            ao_map = self._calculate_ao(H)
            
            # 2. 计算地形阴影
            shadow_mask = self._calculate_shadows(H, sun_dir, size)
            
            # 3. 计算坡度图 (供其他规则使用)
            slope = self._calculate_slope(H)
            
            # 写回结果
            facts.set_column(table_name, "ao_map", ao_map.flatten())
            facts.set_column(table_name, "shadow_mask", shadow_mask.flatten())
            facts.set_column(table_name, "slope", slope.flatten())
            
            # 存储全局光照信息
            facts.set_global("lighting_ao_map", ao_map)
            facts.set_global("lighting_shadow_mask", shadow_mask)
            facts.set_global("lighting_slope", slope)
            
        except KeyError as e:
            # 地形数据不存在，跳过
            pass
    
    def _calculate_ao(self, height_map: np.ndarray) -> np.ndarray:
        """
        计算环境光遮蔽 (Ambient Occlusion)
        
        原理: 凹陷的地方光进不去，所以暗；凸起的地方亮。
        使用二阶导数 (曲率) 来估算 AO。
        
        Args:
            height_map: 2D高度图
            
        Returns:
            AO图 (0-1范围，1=完全照亮，0=完全遮蔽)
        """
        # 计算曲率 (Laplacian)
        curvature = convolve(height_map, self.curvature_kernel, mode='nearest')
        
        # 归一化曲率
        curvature = curvature / (np.abs(height_map).max() + 1e-5)
        
        # 曲率 < 0 (山谷/坑) -> AO 强 (暗)
        # 曲率 > 0 (山峰/脊) -> AO 弱 (亮)
        # 使用 sigmoid 函数映射
        ao = 1.0 / (1.0 + np.exp(-curvature * 5.0 * self.ao_strength))
        
        # 确保范围在 [0.3, 1.0] 之间 (不完全黑暗)
        ao = np.clip(ao, 0.3, 1.0)
        
        return ao.astype(np.float32)
    
    def _calculate_shadows(self, height_map: np.ndarray, 
                          sun_dir: np.ndarray, size: int) -> np.ndarray:
        """
        计算地形阴影 (Terrain Shadows)
        
        原理: 从当前像素出发，沿着太阳反方向在高度图上 "爬行"。
        如果在爬行路径上发现被挡住了，说明在阴影中。
        
        Args:
            height_map: 2D高度图
            sun_dir: 太阳方向向量 (3D)
            size: 地图尺寸
            
        Returns:
            阴影遮罩 (0=阴影, 1=亮部)
        """
        shadow_mask = np.ones_like(height_map, dtype=np.float32)
        
        # 太阳在XZ平面的投影方向
        sun_xz = np.array([sun_dir[0], sun_dir[2]])
        sun_xz_norm = np.linalg.norm(sun_xz)
        
        if sun_xz_norm < 1e-5:
            # 太阳直射，无地形阴影
            return shadow_mask
        
        sun_xz = sun_xz / sun_xz_norm
        
        # 太阳高度角因子 (用于计算射线高度)
        sun_height_factor = -sun_dir[1] / (sun_xz_norm + 1e-5)
        
        # 使用向量化优化：对每个像素进行Ray Marching
        y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        for step in range(1, self.max_shadow_steps + 1):
            # 计算采样位置
            distance = step * self.shadow_step_size
            sample_x = x_coords + sun_xz[0] * distance
            sample_y = y_coords + sun_xz[1] * distance
            
            # 边界检查
            valid_mask = (sample_x >= 0) & (sample_x < size - 1) & \
                        (sample_y >= 0) & (sample_y < size - 1)
            
            # 双线性插值采样高度
            x0 = np.floor(sample_x).astype(np.int32)
            y0 = np.floor(sample_y).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, size - 1)
            y1 = np.clip(y0 + 1, 0, size - 1)
            
            # 插值权重
            wx = sample_x - x0
            wy = sample_y - y0
            
            # 确保索引在有效范围内
            x0 = np.clip(x0, 0, size - 1)
            y0 = np.clip(y0, 0, size - 1)
            
            # 双线性插值
            h00 = height_map[y0, x0]
            h10 = height_map[y0, x1]
            h01 = height_map[y1, x0]
            h11 = height_map[y1, x1]
            
            sample_height = (1 - wx) * (1 - wy) * h00 + \
                           wx * (1 - wy) * h10 + \
                           (1 - wx) * wy * h01 + \
                           wx * wy * h11
            
            # 计算射线在当前位置的高度
            ray_height = height_map + sun_height_factor * distance
            
            # 如果被遮挡，标记为阴影
            occlusion_mask = valid_mask & (sample_height > ray_height)
            shadow_mask[occlusion_mask] = 0.0
            
            # 优化：如果所有像素都已在阴影中，提前退出
            if np.all(shadow_mask == 0):
                break
        
        return shadow_mask
    
    def _calculate_slope(self, height_map: np.ndarray) -> np.ndarray:
        """
        计算地形坡度
        
        Args:
            height_map: 2D高度图
            
        Returns:
            坡度图 (梯度模长)
        """
        grad_y, grad_x = np.gradient(height_map)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        return slope


class ThermalWeatheringRule(Rule):
    """
    热力风化规则 (Thermal Weathering Rule)
    
    原理: 岩石热胀冷缩崩解，形成碎石坡 (Talus)。
    如果坡度 > 临界角 (休止角)，将高处物质搬运到低处。
    """
    
    def __init__(self, critical_angle=0.8, weathering_rate=0.1, iterations=5):
        super().__init__("Terrain.ThermalWeathering", priority=25)
        # 临界角 (弧度)，约45度
        self.critical_angle = critical_angle
        self.weathering_rate = weathering_rate
        self.iterations = iterations
        
    def evaluate(self, facts: FactBase):
        table_name = "terrain_main"
        
        try:
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            
            # 多次迭代直到稳定
            for _ in range(self.iterations):
                H = self._weathering_step(H, size)
            
            facts.set_column(table_name, "height", H.flatten())
            
        except KeyError:
            pass
    
    def _weathering_step(self, H: np.ndarray, size: int) -> np.ndarray:
        """单次风化迭代"""
        # 计算坡度
        grad_y, grad_x = np.gradient(H)
        slope = np.sqrt(grad_x**2 + grad_y**2)
        
        # 找到超过临界角的区域
        unstable_mask = slope > self.critical_angle
        
        if not np.any(unstable_mask):
            return H
        
        # 计算需要移动的物质数量
        excess_slope = slope - self.critical_angle
        material_to_move = excess_slope * self.weathering_rate
        material_to_move = np.clip(material_to_move, 0, None)
        
        # 创建新高度图
        H_new = H.copy()
        
        # 向四个邻居扩散 (简化版)
        # 向坡度最大的方向移动
        neighbors = [
            (0, 1, grad_x, grad_y),   # 右
            (0, -1, -grad_x, -grad_y), # 左
            (1, 0, grad_y, grad_x),   # 下
            (-1, 0, -grad_y, -grad_x)  # 上
        ]
        
        for dy, dx, grad_component, _ in neighbors:
            # 计算流向该邻居的物质
            flow = material_to_move * np.maximum(0, grad_component / (slope + 1e-5))
            flow = flow * 0.25  # 分配到四个方向
            
            # 应用流动
            if dx > 0:
                H_new[:, :-1] -= flow[:, :-1]
                H_new[:, 1:] += flow[:, :-1]
            elif dx < 0:
                H_new[:, 1:] -= flow[:, 1:]
                H_new[:, :-1] += flow[:, 1:]
            elif dy > 0:
                H_new[:-1, :] -= flow[:-1, :]
                H_new[1:, :] += flow[:-1, :]
            elif dy < 0:
                H_new[1:, :] -= flow[1:, :]
                H_new[:-1, :] += flow[1:, :]
        
        return H_new
