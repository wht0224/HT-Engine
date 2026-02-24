"""
海浪规则 (Gerstner Waves Rule)

实现动态海浪效果，基于Gerstner波叠加。
用于海岸场景的水面渲染。

核心特性:
- Data-Driven: 基于风向和时间的波浪数据
- Zero Raytracing: 纯数学推导，无实时物理模拟
- Cheat but Logical: 视觉欺骗但逻辑自洽

参考: "Gerstner Waves: A Procedural Wave Model for Real-Time Water Rendering"
"""

import numpy as np
import logging
from typing import Dict, Any, Optional

from .Core.RuleBase import Rule
from .Core.FactBase import FactBase


class GerstnerWavesRule(Rule):
    """
    海浪规则 (Gerstner Waves Rule)
    
    基于Gerstner波叠加实现动态海浪效果。
    """
    
    def __init__(self, 
                 num_waves: int = 5,
                 wave_amplitude: float = 1.0,
                 wave_length: float = 20.0,
                 wave_speed: float = 1.0,
                 foam_threshold: float = 0.5):
        """
        初始化海浪规则
        
        Args:
            num_waves: 波浪数量（建议3-7）
            wave_amplitude: 波浪振幅
            wave_length: 波浪长度
            wave_speed: 波浪速度
            foam_threshold: 白沫阈值（Jacobian）
        """
        super().__init__("Ocean.GerstnerWaves", priority=45)
        
        self.num_waves = num_waves
        self.wave_amplitude = wave_amplitude
        self.wave_length = wave_length
        self.wave_speed = wave_speed
        self.foam_threshold = foam_threshold
        
        # 波浪参数（每个波的方向、频率、相位）
        self.wave_directions = np.zeros(num_waves, dtype=np.float32)
        self.wave_frequencies = np.zeros(num_waves, dtype=np.float32)
        self.wave_phases = np.random.rand(num_waves) * 2 * np.pi
        
        # 输出数据
        self.wave_heights = None  # 波浪高度图
        self.foam_mask = None  # 白沫遮罩
        self.wave_normals = None  # 波浪法线
        
        self.logger = logging.getLogger("Ocean.GerstnerWaves")
    
    def evaluate(self, facts: FactBase):
        """
        计算动态海浪
        
        Args:
            facts: FactBase实例
        """
        # 获取输入数据
        wind_direction = facts.get_global("wind_direction")
        wind_speed = facts.get_global("wind_speed")
        time = facts.get_global("time")
        
        if wind_direction is None or time is None:
            return
        
        # 获取地形尺寸
        terrain_size = int(np.sqrt(facts.get_count("terrain_main")))
        
        # 初始化波浪参数（如果还没初始化）
        if self.wave_directions is None or len(self.wave_directions) != self.num_waves:
            self._init_waves(terrain_size)
        
        # 更新波浪相位
        self._update_wave_phases(time, wind_speed)
        
        # 计算波浪高度
        wave_height = self._calculate_wave_heights(terrain_size)
        
        # 计算法线（Jacobian）用于白沫
        foam_mask = self._calculate_foam_mask(wave_height, terrain_size)
        
        # 计算法线（用于法线）
        wave_normals = self._calculate_wave_normals(terrain_size)
        
        # 存储结果
        facts.set_column("terrain_main", "wave_heights", wave_height.flatten())
        facts.set_column("terrain_main", "foam_mask", foam_mask.flatten())
        facts.set_column("terrain_main", "wave_normals_x", wave_normals[0].flatten())
        facts.set_column("terrain_main", "wave_normals_y", wave_normals[1].flatten())
        facts.set_column("terrain_main", "wave_normals_z", wave_normals[2].flatten())
        
        self.logger.debug(f"Calculated Gerstner waves: {self.num_waves} waves, foam: {foam_mask.mean():.2f}")
    
    def _init_waves(self, terrain_size: int):
        """
        初始化波浪参数
        
        使用Gerstner波的标准参数：
        - 方向：均匀分布在[0, 2π]范围
        - 频率：对数分布
        - 相位：随机初始相位
        """
        # 波浪方向（均匀分布在[0, 2π]）
        angles = np.linspace(0, 2 * np.pi, self.num_waves, dtype=np.float32)
        
        # 波浪频率（对数分布）
        frequencies = np.linspace(1.0, 3.0, self.num_waves, dtype=np.float32)
        
        # 波浪相位（随机）
        self.wave_phases = np.random.rand(self.num_waves) * 2 * np.pi
        
        self.wave_directions = angles
        self.wave_frequencies = frequencies
        
        self.logger.debug(f"Initialized {self.num_waves} Gerstner waves")
    
    def _update_wave_phases(self, time: float, wind_speed: float):
        """
        更新波浪相位
        
        根据风速调整波浪速度
        """
        speed_factor = 1.0 + wind_speed * 0.5
        
        # 更新相位
        self.wave_phases = (self.wave_phases + time * self.wave_frequencies * speed_factor) % (2 * np.pi)
    
    def _calculate_wave_heights(self, terrain_size: int) -> np.ndarray:
        """
        计算波浪高度图
        
        Gerstner波公式：
        H(x, z, t) = Sum(Amplitude * cos(D * (x / wavelength) + Phase))
        """
        size = terrain_size
        x = np.linspace(-1, 1, size)
        z = np.linspace(-1, 1, size)
        X, Z = np.meshgrid(x, z)
        
        # 计算波浪高度
        wave_height = np.zeros((size, size), dtype=np.float32)
        
        for i in range(self.num_waves):
            amplitude = self.wave_amplitude
            wavelength = self.wave_length / (i + 1)
            phase = self.wave_phases[i]
            frequency = self.wave_frequencies[i]
            direction = self.wave_directions[i]
            
            # Gerstner波公式
            wave = amplitude * np.cos(
                direction * X + frequency * Z + phase
            )
            
            wave_height += wave
        
        return wave_height
    
    def _calculate_foam_mask(self, wave_height: np.ndarray, terrain_size: int) -> np.ndarray:
        """
        计算白沫遮罩
        
        使用Jacobian判断波峰：
        如果波浪的梯度（Jacobian）超过阈值，认为是波峰，添加白沫
        """
        size = terrain_size
        
        # 计算波浪的梯度（Jacobian近似）
        # 使用中心差分计算梯度
        dx = np.zeros_like(wave_height)
        dz = np.zeros_like(wave_height)
        
        dx[1:-1, :] = (wave_height[2:, 2:] - wave_height[:, :-2, :]) / 2
        dz[:-1, 1:-1] = (wave_height[2:, 1:, :] - wave_height[:, :-1, :]) / 2
        
        # 计算Jacobian（梯度矩阵）
        jacobian = np.zeros((size, size, 2), dtype=np.float32)
        jacobian[:, :, 0] = dx
        jacobian[:, :, 1] = dz
        
        # 计算波峰检测（Jacobian的行列式）
        jacobian_determinant = jacobian[:, :, 0] * jacobian[:, :, 1] - jacobian[:, :, 0] * jacobian[:, :, 1]
        
        # 波峰 = 负的行列式
        foam_mask = (jacobian_determinant < -self.foam_threshold).astype(np.float32)
        
        return foam_mask
    
    def _calculate_wave_normals(self, terrain_size: int) -> np.ndarray:
        """
        计算法线（用于水面法线）
        
        简化：使用波浪的梯度作为法线
        """
        size = terrain_size
        x = np.linspace(-1, 1, size)
        z = np.linspace(-1, 1, size)
        X, Z = np.meshgrid(x, z)
        
        # 计算波浪的梯度（用于法线）
        dx = np.zeros_like(X)
        dz = np.zeros_like(Z)
        
        dx[1:-1, :] = (X[2:, 2:] - X[:, :-2, :]) / 2
        dz[:-1, 1:-1] = (Z[2:, 1:, :] - Z[:, :-1, :]) / 2
        
        # 法线（归一化的梯度）
        normals = np.zeros((size, size, 3), dtype=np.float32)
        normals[:, :, 0] = -dx
        normals[:, :, 1] = np.zeros_like(dx)
        normals[:, :, 2] = -dz
        
        # 归一化
        norm = np.sqrt(normals[:, :, 0]**2 + normals[:, :, 1]**2 + normals[:, :, 2]**2)
        normals[:, :, 0] /= norm
        normals[:, :, 1] /= norm
        normals[:, :, 2] /= norm
        
        return normals
