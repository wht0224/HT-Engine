import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase


class LightPropagationRule(Rule):
    """
    光传播规则 (Light Propagation Rule)
    
    核心思想：光像水一样从亮处流向暗处。
    不是GI计算，而是"亮度均衡"的自然法则。
    
    逻辑链条：
    1. 亮的地方有"多余"的光
    2. 光会"流"向附近的暗处
    3. 距离越远，流过去的光越少
    4. 障碍物会"阻挡"光的流动
    
    Attributes:
        propagation_range (float): 光传播的最大距离
        propagation_strength (float): 光传播强度
        iterations (int): 传播迭代次数（模拟多次反弹）
    """
    
    def __init__(self, propagation_range=10.0, propagation_strength=0.5, iterations=3):
        super().__init__("Lighting.Propagation", priority=80)
        self.propagation_range = propagation_range
        self.propagation_strength = propagation_strength
        self.iterations = iterations
        
    def evaluate(self, facts: FactBase):
        """执行光传播规则（向量化版本）。"""
        table_name = "terrain_main"
        
        try:
            flat_height = facts.get_column(table_name, "height")
            direct_light = facts.get_column(table_name, "shadow_mask")
            
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return
            
            H = flat_height.reshape((size, size))
            direct = direct_light.reshape((size, size))
            indirect = np.zeros_like(H, dtype=np.float32)
            
            # 迭代传播
            for iteration in range(self.iterations):
                sources = direct if iteration == 0 else indirect
                new_indirect = self._propagate_light_vectorized(H, sources, indirect, size)
                indirect += new_indirect * (self.propagation_strength ** (iteration + 1))
                indirect = np.clip(indirect, 0.0, 1.0)
            
            facts.set_column(table_name, "indirect_light", indirect.flatten())
            facts.set_global("lighting_indirect", indirect)
            
        except KeyError:
            pass
    
    def _propagate_light_vectorized(self, height_map, sources, current_indirect, size):
        """
        向量化光传播计算。
        
        核心优化：
        - 使用卷积模拟光传播
        - 批量处理所有像素
        - 避免Python循环
        """
        # 创建传播核
        radius = int(self.propagation_range)
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        
        y_coords, x_coords = np.meshgrid(
            np.arange(-radius, radius + 1),
            np.arange(-radius, radius + 1),
            indexing='ij'
        )
        distances = np.sqrt(x_coords**2 + y_coords**2)
        
        # 距离衰减
        mask = (distances > 0) & (distances <= radius)
        kernel[mask] = 1.0 / (1.0 + distances[mask]**2 * 0.1)
        
        # 归一化
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum
        
        # 总光源（直接+间接）
        total_sources = sources + current_indirect
        
        # 手动实现卷积（避免scipy依赖）
        new_indirect = self._fast_convolve(total_sources, kernel)
        
        # 只保留暗处区域的间接光
        dark_mask = (sources < 0.5)
        result = np.zeros_like(height_map, dtype=np.float32)
        result[dark_mask] = new_indirect[dark_mask]
        
        return result
    
    def _fast_convolve(self, data, kernel):
        """
        快速卷积实现（向量化）。
        
        使用FFT加速大核卷积，小核使用直接计算。
        """
        kernel_size = kernel.shape[0]
        
        if kernel_size <= 7:
            # 小核：直接计算更快
            return self._direct_convolve(data, kernel)
        else:
            # 大核：使用FFT
            return self._fft_convolve(data, kernel)
    
    def _direct_convolve(self, data, kernel):
        """直接卷积（小核优化）"""
        from scipy.ndimage import convolve
        return convolve(data, kernel, mode='nearest')
    
    def _fft_convolve(self, data, kernel):
        """FFT卷积（大核优化）"""
        from numpy.fft import fft2, ifft2
        
        # 填充到2的幂次
        h, w = data.shape
        kh, kw = kernel.shape
        pad_h = 1 << (h + kh - 1).bit_length()
        pad_w = 1 << (w + kw - 1).bit_length()
        
        data_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
        kernel_pad = np.zeros((pad_h, pad_w), dtype=np.float32)
        
        data_pad[:h, :w] = data
        kernel_pad[:kh, :kw] = kernel
        
        # FFT卷积
        data_fft = fft2(data_pad)
        kernel_fft = fft2(kernel_pad)
        result_fft = data_fft * kernel_fft
        result = np.real(ifft2(result_fft))
        
        # 提取有效区域
        start_h = kh // 2
        start_w = kw // 2
        return result[start_h:start_h+h, start_w:start_w+w]
