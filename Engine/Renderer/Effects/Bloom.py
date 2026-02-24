# -*- coding: utf-8 -*-
"""
泛光效果实现
针对低端GPU优化，支持不同质量级别
"""

import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

class Bloom(EffectBase):
    """泛光效果类"""
    
    def __init__(self, gpu_architecture, quality_level):
        """
        初始化泛光效果
        
        参数:
        - gpu_architecture: GPU架构
        - quality_level: 质量级别
        """
        super().__init__(gpu_architecture, quality_level)
        self.name = "bloom"
        self.performance_cost = {
            EffectQuality.LOW: 0.5,
            EffectQuality.MEDIUM: 1.5,
            EffectQuality.HIGH: 3.0
        }
        
        # 泛光参数
        self.threshold = 0.8
        self.intensity = 0.5
        self.blur_radius = 4
        self.downsample_factor = 4
        
        # 初始化着色器和纹理
        self.shaders = {}
        self.textures = {}
    
    def initialize(self, renderer):
        """初始化泛光效果"""
        super().initialize(renderer)
        # 简化实现，实际需要创建着色器和纹理
        pass
    
    def _apply_effect(self, input_texture, output_texture):
        """应用泛光效果"""
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_bloom_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_bloom_gcn(input_texture, output_texture)
        else:
            return self._apply_bloom_generic(input_texture, output_texture)
    
    def _extract_brightness(self, texture):
        """提取亮部区域"""
        if not isinstance(texture, np.ndarray):
            return texture
        
        # 创建亮部纹理
        bright_texture = np.copy(texture)
        height, width = bright_texture.shape[:2]
        
        # 对每个像素应用亮度阈值
        for y in range(height):
            for x in range(width):
                color = bright_texture[y, x]
                # 计算亮度（使用感知亮度公式）
                brightness = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
                
                # 应用阈值
                if brightness > self.threshold:
                    # 保留超出阈值的部分
                    bright_texture[y, x] = (brightness - self.threshold) * (1.0 / (1.0 - self.threshold)) * color
                else:
                    # 暗部设为黑色
                    bright_texture[y, x] = np.array([0.0, 0.0, 0.0])
        
        return bright_texture
    
    def _downsample(self, texture, factor):
        """降采样纹理"""
        if not isinstance(texture, np.ndarray):
            return texture
        
        height, width = texture.shape[:2]
        new_height = height // factor
        new_width = width // factor
        
        # 使用平均降采样
        downsampled = np.zeros((new_height, new_width, texture.shape[2]), dtype=np.float32)
        
        for y in range(new_height):
            for x in range(new_width):
                sum_color = np.zeros(texture.shape[2], dtype=np.float32)
                count = 0
                
                # 对每个区块进行平均
                for dy in range(factor):
                    for dx in range(factor):
                        orig_y = y * factor + dy
                        orig_x = x * factor + dx
                        if orig_y < height and orig_x < width:
                            sum_color += texture[orig_y, orig_x]
                            count += 1
                
                downsampled[y, x] = sum_color / count
        
        return downsampled
    
    def _gaussian_blur(self, texture, radius):
        """高斯模糊"""
        if not isinstance(texture, np.ndarray) or radius < 1:
            return texture
        
        # 生成高斯核
        sigma = radius / 3.0
        kernel_size = radius * 2 + 1
        
        # 创建高斯核
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        center = radius
        
        for y in range(kernel_size):
            for x in range(kernel_size):
                dx = x - center
                dy = y - center
                kernel[y, x] = np.exp(-(dx*dx + dy*dy) / (2.0 * sigma*sigma))
        
        # 归一化核
        kernel_sum = np.sum(kernel)
        kernel /= kernel_sum
        
        # 应用高斯模糊
        height, width = texture.shape[:2]
        blurred = np.copy(texture)
        channels = texture.shape[2]
        
        for y in range(height):
            for x in range(width):
                sum_color = np.zeros(channels, dtype=np.float32)
                
                for ky in range(-radius, radius + 1):
                    for kx in range(-radius, radius + 1):
                        # 计算原始坐标
                        orig_y = y + ky
                        orig_x = x + kx
                        
                        # 边界检查
                        if orig_y >= 0 and orig_y < height and orig_x >= 0 and orig_x < width:
                            weight = kernel[ky + radius, kx + radius]
                            sum_color += texture[orig_y, orig_x] * weight
                
                blurred[y, x] = sum_color
        
        return blurred
    
    def _upsample(self, texture, factor):
        """上采样纹理"""
        if not isinstance(texture, np.ndarray):
            return texture
        
        height, width = texture.shape[:2]
        new_height = height * factor
        new_width = width * factor
        
        # 使用双线性上采样
        upsampled = np.zeros((new_height, new_width, texture.shape[2]), dtype=np.float32)
        
        for y in range(new_height):
            for x in range(new_width):
                # 计算原始坐标
                orig_y = y / factor
                orig_x = x / factor
                
                # 整数坐标
                y0 = int(orig_y)
                x0 = int(orig_x)
                y1 = min(y0 + 1, height - 1)
                x1 = min(x0 + 1, width - 1)
                
                # 插值权重
                ty = orig_y - y0
                tx = orig_x - x0
                
                # 双线性插值
                color00 = texture[y0, x0]
                color01 = texture[y0, x1]
                color10 = texture[y1, x0]
                color11 = texture[y1, x1]
                
                color0 = color00 * (1 - tx) + color01 * tx
                color1 = color10 * (1 - tx) + color11 * tx
                final_color = color0 * (1 - ty) + color1 * ty
                
                upsampled[y, x] = final_color
        
        return upsampled
    
    def _apply_bloom_maxwell(self, input_texture, output_texture):
        """针对NVIDIA Maxwell架构的优化实现"""
        return self._generic_bloom_implementation(input_texture, output_texture, 
                                               high_performance=True, multi_pass=False)
    
    def _apply_bloom_gcn(self, input_texture, output_texture):
        """针对AMD GCN架构的优化实现"""
        return self._generic_bloom_implementation(input_texture, output_texture, 
                                               high_performance=False, multi_pass=True)
    
    def _apply_bloom_generic(self, input_texture, output_texture):
        """通用实现"""
        return self._generic_bloom_implementation(input_texture, output_texture, 
                                               high_performance=True, multi_pass=False)
    
    def _generic_bloom_implementation(self, input_texture, output_texture, 
                                     high_performance=True, multi_pass=True):
        """通用Bloom实现"""
        if not isinstance(input_texture, np.ndarray):
            return input_texture
        
        # 创建输出纹理
        output = np.copy(input_texture)
        
        # 1. 提取亮部区域
        bright_pass = self._extract_brightness(input_texture)
        
        # 2. 多级降采样
        downsample_factor = self.downsample_factor
        if high_performance:
            # 高性能模式下使用更大的降采样因子
            downsample_factor = max(downsample_factor, 8)
        
        downsampled = self._downsample(bright_pass, downsample_factor)
        
        # 3. 高斯模糊
        blur_radius = self.blur_radius
        if high_performance:
            # 高性能模式下使用更小的模糊半径
            blur_radius = max(1, blur_radius // 2)
        
        blurred = self._gaussian_blur(downsampled, blur_radius)
        
        # 4. 上采样回到原始分辨率
        upsampled = self._upsample(blurred, downsample_factor)
        
        # 5. 与原始图像混合
        height, width = output.shape[:2]
        bloom_height, bloom_width = upsampled.shape[:2]
        
        for y in range(height):
            for x in range(width):
                # 确保坐标在bloom纹理范围内
                bloom_x = min(x, bloom_width - 1)
                bloom_y = min(y, bloom_height - 1)
                
                # 获取bloom颜色
                bloom_color = upsampled[bloom_y, bloom_x]
                
                # 混合到原始图像
                output[y, x] = output[y, x] + bloom_color * self.intensity
        
        # 6. 限制颜色范围
        output = np.clip(output, 0.0, 1.0)
        
        return output
    
    def adjust_quality(self, quality_level):
        """调整泛光质量"""
        super().adjust_quality(quality_level)
        
        if quality_level == EffectQuality.LOW:
            self.threshold = 0.9
            self.intensity = 0.3
            self.blur_radius = 2
            self.downsample_factor = 8
        elif quality_level == EffectQuality.MEDIUM:
            self.threshold = 0.8
            self.intensity = 0.5
            self.blur_radius = 4
            self.downsample_factor = 4
        elif quality_level == EffectQuality.HIGH:
            self.threshold = 0.7
            self.intensity = 0.8
            self.blur_radius = 8
            self.downsample_factor = 2
    
    def set_intensity(self, intensity):
        """设置泛光强度"""
        self.intensity = max(0.0, min(2.0, intensity))
    
    def set_threshold(self, threshold):
        """设置亮度阈值"""
        self.threshold = max(0.0, min(1.0, threshold))
    
    def set_blur_radius(self, radius):
        """设置模糊半径"""
        self.blur_radius = max(1, min(16, radius))
