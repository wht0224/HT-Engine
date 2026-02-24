import numpy as np
from scipy.ndimage import label, gaussian_filter
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class HydroVisualRule(Rule):
    """
    水文视觉规则 (Hydro Visual Rule)
    
    目标: 实现低成本的动态反射和湿润效果。
    核心哲学: Data-Driven (一切皆数据，拒绝实时物理探测)
    
    包含:
    1. 湿润表面 (Wetness) - 基于WaterMap调整材质粗糙度
    2. 平面反射检测 (Planar Reflection) - 识别湖泊区域
    3. 河流流动 (River Flow) - 基于水力侵蚀的水流方向
    """
    
    def __init__(self, wetness_smoothness=0.8, reflection_threshold=0.5,
                 min_lake_size=100):
        super().__init__("Hydro.Visual", priority=65)
        self.wetness_smoothness = wetness_smoothness
        self.reflection_threshold = reflection_threshold
        self.min_lake_size = min_lake_size
        
    def evaluate(self, facts: FactBase):
        """执行水文视觉计算"""
        table_name = "terrain_main"
        
        try:
            # 获取地形数据
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            
            # 获取水文数据
            try:
                water = facts.get_column(table_name, "water").reshape((size, size))
            except KeyError:
                water = np.zeros((size, size), dtype=np.float32)
            
            # 获取降雨强度
            rain_intensity = facts.get_global("rain_intensity")
            if rain_intensity is None:
                rain_intensity = 0.0
            
            # 1. 计算湿润度
            wetness, roughness, specular = self._calculate_wetness(
                water, rain_intensity, size
            )
            
            # 2. 检测水体区域 (湖泊/河流)
            water_regions, flow_direction = self._detect_water_regions(
                water, H, size
            )
            
            # 3. 计算反射强度
            reflection_intensity = self._calculate_reflection(
                water, wetness, size
            )
            
            # 写回结果
            facts.set_column(table_name, "wetness", wetness.flatten())
            facts.set_column(table_name, "roughness", roughness.flatten())
            facts.set_column(table_name, "specularity", specular.flatten())
            facts.set_column(table_name, "water_regions", water_regions.flatten())
            facts.set_column(table_name, "flow_direction_x", flow_direction[0].flatten())
            facts.set_column(table_name, "flow_direction_y", flow_direction[1].flatten())
            facts.set_column(table_name, "reflection_intensity", reflection_intensity.flatten())
            
            # 存储全局信息 (始终设置，即使没有水体)
            facts.set_global("hydro_wetness_map", wetness)
            facts.set_global("hydro_roughness_map", roughness)
            facts.set_global("hydro_water_regions", water_regions)
            
        except KeyError:
            # 如果表不存在，静默跳过
            pass
    
    def _calculate_wetness(self, water: np.ndarray, rain_intensity: float,
                          size: int) -> tuple:
        """
        计算湿润表面效果
        
        原理: 水多的地方反光强，且粗糙度低。
        Wetness = clamp(WaterMap + RainIntensity, 0, 1)
        Roughness = OriginalRoughness * (1.0 - Wetness * 0.8)
        Specular = OriginalSpecular + Wetness
        
        Args:
            water: 水量图
            rain_intensity: 当前降雨强度
            size: 地图尺寸
            
        Returns:
            (wetness, roughness, specular) 三个2D数组
        """
        # 基础湿润度 = 积水 + 降雨
        wetness = np.clip(water + rain_intensity * 0.5, 0.0, 1.0)
        
        # 应用平滑使过渡更自然
        wetness = gaussian_filter(wetness, sigma=1.5)
        wetness = np.clip(wetness, 0.0, 1.0)
        
        # 计算粗糙度 (越湿越光滑)
        # 假设基础粗糙度为 0.8
        base_roughness = 0.8
        roughness = base_roughness * (1.0 - wetness * self.wetness_smoothness)
        roughness = np.clip(roughness, 0.1, 1.0)
        
        # 计算高光强度 (越湿越亮)
        base_specular = 0.1
        specular = base_specular + wetness * 0.5
        specular = np.clip(specular, 0.0, 1.0)
        
        return wetness.astype(np.float32), roughness.astype(np.float32), specular.astype(np.float32)
    
    def _detect_water_regions(self, water: np.ndarray, height: np.ndarray,
                             size: int) -> tuple:
        """
        检测水体区域 (湖泊/河流)
        
        原理: 
        - 大面积静止水面 = 湖泊 (连通区域检测)
        - 流动的水 = 河流 (基于梯度计算流向)
        
        Args:
            water: 水量图
            height: 高度图
            size: 地图尺寸
            
        Returns:
            (water_regions, (flow_x, flow_y))
            water_regions: 水体区域标记 (0=无水体, >0=区域ID)
            flow_direction: 水流方向 (2, size, size) 数组
        """
        # 检测积水区域
        water_mask = water > self.reflection_threshold
        
        # 连通区域标记 (识别湖泊)
        labeled_array, num_features = label(water_mask)
        
        # 过滤小区域 (只保留大面积水体)
        water_regions = np.zeros_like(labeled_array, dtype=np.float32)
        for i in range(1, num_features + 1):
            region_size = np.sum(labeled_array == i)
            if region_size >= self.min_lake_size:
                water_regions[labeled_array == i] = i
        
        # 计算水流方向 (基于高度梯度)
        grad_y, grad_x = np.gradient(height)
        
        # 水流方向与高度梯度相反 (从高处流向低处)
        flow_x = -grad_x
        flow_y = -grad_y
        
        # 归一化
        flow_magnitude = np.sqrt(flow_x**2 + flow_y**2) + 1e-5
        flow_x = flow_x / flow_magnitude
        flow_y = flow_y / flow_magnitude
        
        # 只在有水的地方有流向
        flow_x = flow_x * (water > 0.1).astype(np.float32)
        flow_y = flow_y * (water > 0.1).astype(np.float32)
        
        return water_regions.astype(np.float32), (flow_x.astype(np.float32), flow_y.astype(np.float32))
    
    def _calculate_reflection(self, water: np.ndarray, wetness: np.ndarray,
                             size: int) -> np.ndarray:
        """
        计算反射强度
        
        原理: 只有大面积静止水面（湖/海）需要真反射。
        小水坑和湿润表面使用简化反射。
        
        Args:
            water: 水量图
            wetness: 湿润度图
            size: 地图尺寸
            
        Returns:
            反射强度图 (0-1范围)
        """
        # 基础反射强度基于水量
        reflection = np.clip(water * 2.0, 0.0, 1.0)
        
        # 湿润表面也有微弱反射
        wet_reflection = wetness * 0.2
        
        # 合并
        total_reflection = np.maximum(reflection, wet_reflection)
        
        # 平滑
        total_reflection = gaussian_filter(total_reflection, sigma=1.0)
        
        return np.clip(total_reflection, 0.0, 1.0).astype(np.float32)


class PlanarReflectionRule(Rule):
    """
    平面反射规则 (Planar Reflection Rule)
    
    管理大面积静止水面的反射渲染。
    仅在相机位于湖边时启用 "倒置相机渲染"。
    """
    
    def __init__(self, activation_distance=50.0):
        super().__init__("Hydro.PlanarReflection", priority=55)
        self.activation_distance = activation_distance
        self.active_reflections = {}  # 活跃的反射区域
        
    def evaluate(self, facts: FactBase):
        """
        评估是否需要启用平面反射
        
        注意: 此规则主要设置标志位，实际渲染在渲染器中处理
        """
        # 获取水体区域
        water_regions = facts.get_global("hydro_water_regions")
        
        # 检测相机是否靠近大水体
        # 简化：检查全局反射需求
        has_large_water = False
        if water_regions is not None:
            has_large_water = np.any(water_regions > 0)
        
        if has_large_water:
            facts.set_global("planar_reflection_enabled", True)
            facts.set_global("planar_reflection_quality", self._determine_quality(facts))
        else:
            facts.set_global("planar_reflection_enabled", False)
            facts.set_global("planar_reflection_quality", "cubemap_only")
    
    def _determine_quality(self, facts: FactBase) -> str:
        """
        根据硬件性能决定反射质量
        
        Returns:
            质量等级: "high", "medium", "low", "cubemap_only"
        """
        # 获取GPU性能等级
        gpu_tier = facts.get_global("gpu_tier")
        if gpu_tier is None:
            gpu_tier = "medium"
        
        # 根据等级返回质量设置
        quality_map = {
            "high": "high",
            "medium": "medium", 
            "low": "cubemap_only",
            "minimal": "cubemap_only"
        }
        
        return quality_map.get(gpu_tier, "medium")


class WaterCausticsRule(Rule):
    """
    水焦散规则 (Water Caustics Rule)
    
    模拟水下光线折射产生的焦散效果。
    低成本实现：基于噪声纹理的动画图案。
    """
    
    def __init__(self, animation_speed=1.0, intensity=0.5):
        super().__init__("Hydro.Caustics", priority=50)
        self.animation_speed = animation_speed
        self.intensity = intensity
        self._caustics_texture = None
        
    def evaluate(self, facts: FactBase):
        """计算水焦散效果"""
        table_name = "terrain_main"
        
        try:
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            # 获取时间和水体信息
            time = facts.get_global("time")
            if time is None:
                time = 0.0
            
            water_regions = facts.get_global("hydro_water_regions")
            
            # 生成焦散图案
            caustics = self._generate_caustics(size, time)
            
            # 只在水下区域应用
            if water_regions is not None:
                caustics = caustics * (water_regions > 0).astype(np.float32)
            else:
                caustics = np.zeros((size, size), dtype=np.float32)
            
            # 存储结果
            facts.set_column(table_name, "caustics", caustics.flatten())
            facts.set_global("hydro_caustics", caustics)
            
        except KeyError:
            pass
    
    def _generate_caustics(self, size: int, time: float) -> np.ndarray:
        """
        生成焦散图案
        
        使用多层正弦波叠加模拟焦散效果。
        """
        # 创建坐标网格
        y, x = np.meshgrid(np.linspace(0, 4*np.pi, size),
                          np.linspace(0, 4*np.pi, size),
                          indexing='ij')
        
        # 动画偏移
        t = time * self.animation_speed
        
        # 多层正弦波叠加
        caustics = np.zeros((size, size), dtype=np.float32)
        
        # 层1: 基础图案
        caustics += np.sin(x * 2.0 + t) * np.cos(y * 2.0 + t * 0.7)
        
        # 层2: 细节图案
        caustics += np.sin(x * 5.0 - t * 1.2) * np.sin(y * 5.0 + t * 0.5) * 0.5
        
        # 层3: 微小细节
        caustics += np.sin(x * 10.0 + t * 0.3) * np.cos(y * 10.0 - t * 0.8) * 0.25
        
        # 归一化到 [0, 1] 并应用强度
        caustics = (caustics + 1.75) / 3.5  # 归一化
        caustics = np.power(caustics, 3.0)  # 增强对比度
        caustics = caustics * self.intensity
        
        return np.clip(caustics, 0.0, 1.0).astype(np.float32)
