import numpy as np
from scipy.ndimage import gaussian_filter
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class AtmosphereRule(Rule):
    """
    大气光学规则 (Atmosphere Rule)
    
    目标: 实现丁达尔效应 (God Rays) 和云层投影，增强空气感。
    核心哲学: Cheat but Logical (视觉欺骗，但逻辑自洽)
    
    包含:
    1. 丁达尔效应 (Volumetric God Rays) - 径向模糊
    2. 云影漂移 (Cloud Shadows) - Perlin Noise + 风
    """
    
    def __init__(self, god_ray_samples=32, god_ray_density=0.5,
                 cloud_speed=1.0, cloud_scale=0.01):
        super().__init__("Atmosphere.Optical", priority=70)
        self.god_ray_samples = god_ray_samples
        self.god_ray_density = god_ray_density
        self.cloud_speed = cloud_speed
        self.cloud_scale = cloud_scale
        
        # 云图缓存
        self._cloud_texture = None
        self._cloud_size = 0
        
    def evaluate(self, facts: FactBase):
        """执行大气光学计算"""
        table_name = "terrain_main"
        
        try:
            # 获取地形尺寸
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            # 1. 计算丁达尔效应
            god_ray_overlay = self._calculate_god_rays(facts, size)
            
            # 2. 计算云影
            cloud_shadow = self._calculate_cloud_shadows(facts, size)
            
            # 3. 合并大气效果
            atmosphere_intensity = god_ray_overlay * (1.0 - cloud_shadow * 0.5)
            
            # 写回结果
            facts.set_column(table_name, "god_ray", god_ray_overlay.flatten())
            facts.set_column(table_name, "cloud_shadow", cloud_shadow.flatten())
            facts.set_column(table_name, "atmosphere_intensity", atmosphere_intensity.flatten())
            
            # 存储全局大气信息
            facts.set_global("atmosphere_god_ray", god_ray_overlay)
            facts.set_global("atmosphere_cloud_shadow", cloud_shadow)
            
        except KeyError:
            pass
    
    def _calculate_god_rays(self, facts: FactBase, size: int) -> np.ndarray:
        """
        计算丁达尔效应 (Volumetric God Rays)
        
        原理: 强光 + 遮挡物 + 雾 = 光束
        使用径向模糊 (Radial Blur) 模拟光线从太阳发散的效果。
        
        Args:
            facts: 事实库
            size: 地图尺寸
            
        Returns:
            丁达尔效应叠加层 (0-1范围)
        """
        # 获取阴影遮罩 (遮挡物)
        try:
            shadow_mask = facts.get_global("lighting_shadow_mask")
            if shadow_mask is None:
                shadow_mask = np.ones((size, size), dtype=np.float32)
        except:
            shadow_mask = np.ones((size, size), dtype=np.float32)
        
        # 获取雾密度
        fog_density = facts.get_global("fog_density")
        if fog_density is None:
            fog_density = 0.3
        
        # 如果雾太薄，不计算丁达尔效应
        if fog_density < 0.1:
            return np.zeros((size, size), dtype=np.float32)
        
        # 获取太阳屏幕位置 (简化：假设太阳在地平线某处)
        sun_dir = facts.get_global("sun_direction")
        if sun_dir is None:
            sun_dir = np.array([0.5, -1.0, 0.3])
        
        # 将太阳方向转换为屏幕坐标 (简化投影)
        # 假设太阳在屏幕的某个角落
        sun_screen_x = size * (0.5 + sun_dir[0] * 0.5)
        sun_screen_y = size * (0.5 + sun_dir[2] * 0.5)
        
        # 创建径向模糊
        # 从太阳位置向外采样
        god_rays = np.zeros((size, size), dtype=np.float32)
        
        # 采样权重 (越靠近太阳权重越高)
        weights = np.zeros(self.god_ray_samples, dtype=np.float32)
        for i in range(self.god_ray_samples):
            weights[i] = np.power(1.0 - i / self.god_ray_samples, 2.0)
        weights /= weights.sum() + 1e-5
        
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # 计算到太阳的方向向量
        dx = x_coords - sun_screen_x
        dy = y_coords - sun_screen_y
        dist = np.sqrt(dx**2 + dy**2) + 1e-5
        
        # 归一化方向
        dir_x = dx / dist
        dir_y = dy / dist
        
        # 径向模糊采样
        for i in range(self.god_ray_samples):
            sample_dist = i * 2.0  # 采样距离
            sample_x = x_coords - dir_x * sample_dist
            sample_y = y_coords - dir_y * sample_dist
            
            # 双线性插值采样阴影遮罩
            sample_x_clipped = np.clip(sample_x, 0, size - 1)
            sample_y_clipped = np.clip(sample_y, 0, size - 1)
            
            x0 = np.floor(sample_x_clipped).astype(np.int32)
            y0 = np.floor(sample_y_clipped).astype(np.int32)
            x1 = np.clip(x0 + 1, 0, size - 1)
            y1 = np.clip(y0 + 1, 0, size - 1)
            
            wx = sample_x_clipped - x0
            wy = sample_y_clipped - y0
            
            # 采样
            s00 = shadow_mask[y0, x0]
            s10 = shadow_mask[y0, x1]
            s01 = shadow_mask[y1, x0]
            s11 = shadow_mask[y1, x1]
            
            sampled = (1 - wx) * (1 - wy) * s00 + \
                     wx * (1 - wy) * s10 + \
                     (1 - wx) * wy * s01 + \
                     wx * wy * s11
            
            god_rays += sampled * weights[i]
        
        # 反转：阴影区域(0)应该产生光束，亮部(1)不产生
        god_rays = (1.0 - god_rays) * fog_density * self.god_ray_density
        
        # 应用高斯模糊使光束更柔和
        god_rays = gaussian_filter(god_rays, sigma=2.0)
        
        return np.clip(god_rays, 0, 1).astype(np.float32)
    
    def _calculate_cloud_shadows(self, facts: FactBase, size: int) -> np.ndarray:
        """
        计算云影漂移 (Cloud Shadows)
        
        原理: 风吹云动，云遮地暗。
        使用 Perlin Noise 生成云图，根据风速移动。
        
        Args:
            facts: 事实库
            size: 地图尺寸
            
        Returns:
            云影遮罩 (0=无云, 1=完全遮蔽)
        """
        # 获取时间和风
        time = facts.get_global("time")
        if time is None:
            time = 0.0
        
        wind_dir = facts.get_global("wind_direction")
        if wind_dir is None:
            wind_dir = np.array([1.0, 0.0, 0.0])
        
        wind_speed = facts.get_global("wind_speed")
        if wind_speed is None:
            wind_speed = 1.0
        
        # 生成或获取云纹理
        if self._cloud_texture is None or self._cloud_size != size:
            self._cloud_texture = self._generate_cloud_texture(size)
            self._cloud_size = size
        
        # 计算UV偏移
        uv_offset_x = wind_dir[0] * wind_speed * self.cloud_speed * time * 0.1
        uv_offset_y = wind_dir[2] * wind_speed * self.cloud_speed * time * 0.1
        
        # 采样云图 (带偏移)
        cloud_shadow = self._sample_cloud_texture(
            self._cloud_texture, size, uv_offset_x, uv_offset_y
        )
        
        return cloud_shadow
    
    def _generate_cloud_texture(self, size: int) -> np.ndarray:
        """
        生成无缝云纹理 (多层Perlin Noise叠加)
        
        Args:
            size: 纹理尺寸
            
        Returns:
            云纹理 (0-1范围)
        """
        # 使用简单的噪声叠加模拟Perlin Noise
        # 实际应用中可以使用噪声库
        
        cloud = np.zeros((size, size), dtype=np.float32)
        
        # 多层噪声
        octaves = 4
        persistence = 0.5
        amplitude = 1.0
        frequency = self.cloud_scale
        
        for i in range(octaves):
            # 生成随机噪声
            noise = np.random.rand(size, size).astype(np.float32)
            
            # 应用频率缩放 (通过重复平铺模拟)
            scaled_size = int(size * frequency)
            if scaled_size > 0:
                noise_scaled = np.repeat(np.repeat(noise[:scaled_size, :scaled_size], 
                                                   size // scaled_size + 1, axis=0),
                                        size // scaled_size + 1, axis=1)[:size, :size]
            else:
                noise_scaled = noise
            
            # 平滑
            noise_scaled = gaussian_filter(noise_scaled, sigma=2.0 / frequency)
            
            cloud += noise_scaled * amplitude
            amplitude *= persistence
            frequency *= 2.0
        
        # 归一化并应用阈值 (模拟云的离散性)
        cloud = (cloud - cloud.min()) / (cloud.max() - cloud.min() + 1e-5)
        cloud = np.power(cloud, 2.0)  # 增强对比度
        
        return cloud
    
    def _sample_cloud_texture(self, texture: np.ndarray, size: int,
                             offset_x: float, offset_y: float) -> np.ndarray:
        """
        带偏移采样云纹理 (无缝平铺)
        
        Args:
            texture: 云纹理
            size: 采样尺寸
            offset_x: X方向偏移
            offset_y: Y方向偏移
            
        Returns:
            采样结果
        """
        # 计算实际偏移 (取模实现无缝)
        ox = offset_x % size
        oy = offset_y % size
        
        # 创建坐标网格
        y_coords, x_coords = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
        
        # 应用偏移
        sample_x = (x_coords + ox) % size
        sample_y = (y_coords + oy) % size
        
        # 双线性插值
        x0 = np.floor(sample_x).astype(np.int32) % size
        y0 = np.floor(sample_y).astype(np.int32) % size
        x1 = (x0 + 1) % size
        y1 = (y0 + 1) % size
        
        wx = sample_x - np.floor(sample_x)
        wy = sample_y - np.floor(sample_y)
        
        # 采样
        s00 = texture[y0, x0]
        s10 = texture[y0, x1]
        s01 = texture[y1, x0]
        s11 = texture[y1, x1]
        
        result = (1 - wx) * (1 - wy) * s00 + \
                wx * (1 - wy) * s10 + \
                (1 - wx) * wy * s01 + \
                wx * wy * s11
        
        return result.astype(np.float32)


class FogRule(Rule):
    """
    雾效规则 (Fog Rule)
    
    根据水文条件动态调整雾的密度和颜色。
    水多雾大，温度影响雾的高度分布。
    """
    
    def __init__(self, base_density=0.1, humidity_factor=0.5):
        super().__init__("Atmosphere.Fog", priority=60)
        self.base_density = base_density
        self.humidity_factor = humidity_factor
        
    def evaluate(self, facts: FactBase):
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
            
            # 获取全局环境
            temperature = facts.get_global("temperature")
            if temperature is None:
                temperature = 20.0
            
            # 计算雾密度
            # 基础密度 + 湿度影响
            humidity = np.clip(water * self.humidity_factor, 0, 1)
            fog_density = self.base_density + humidity * 0.5
            
            # 温度影响：温度越低，雾越容易在低处聚集
            # 使用高度衰减
            height_factor = np.exp(-H / (100.0 + temperature * 5.0))
            fog_density = fog_density * height_factor
            
            # 存储全局雾密度
            avg_fog = np.mean(fog_density)
            facts.set_global("fog_density", avg_fog)
            facts.set_global("fog_density_map", fog_density)
            
        except KeyError:
            pass
