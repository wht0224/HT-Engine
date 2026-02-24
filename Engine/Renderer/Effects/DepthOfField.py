# -*- coding: utf-8 -*-
"""
景深效果实现
针对低端GPU优化，支持不同质量级别
"""

import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

class DepthOfField(EffectBase):
    """景深效果类"""
    
    def __init__(self, gpu_architecture, quality_level):
        """
        初始化景深效果
        
        参数:
        - gpu_architecture: GPU架构
        - quality_level: 质量级别
        """
        super().__init__(gpu_architecture, quality_level)
        self.name = "depth_of_field"
        self.performance_cost = {
            EffectQuality.LOW: 1.0,
            EffectQuality.MEDIUM: 2.5,
            EffectQuality.HIGH: 5.0
        }
        
        # 景深参数
        self.focus_distance = 5.0
        self.focus_range = 2.0
        self.aperture = 0.5
        self.blur_radius = 8
        self.downsample_factor = 4
        
        # 初始化着色器和纹理
        self.shaders = {}
        self.textures = {}
    
    def initialize(self, renderer):
        """初始化景深效果"""
        super().initialize(renderer)
        # 简化实现，实际需要创建着色器和纹理
        pass
    
    def _apply_effect(self, input_texture, output_texture):
        """应用景深效果"""
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_dof_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_dof_gcn(input_texture, output_texture)
        else:
            return self._apply_dof_generic(input_texture, output_texture)
    
    def _calculate_coc(self, depth_buffer):
        """计算混淆圆(CoC)大小
        
        参数:
        - depth_buffer: 深度缓冲区
        
        返回:
        - CoC贴图
        """
        if not isinstance(depth_buffer, np.ndarray):
            return depth_buffer
        
        height, width = depth_buffer.shape[:2]
        coc_map = np.zeros((height, width), dtype=np.float32)
        
        # 基于物理的CoC计算
        for y in range(height):
            for x in range(width):
                depth = depth_buffer[y, x] if depth_buffer.shape[2] == 1 else depth_buffer[y, x, 0]
                
                # 计算与焦点平面的距离
                delta_depth = abs(depth - self.focus_distance)
                
                # 计算混淆圆大小（基于物理公式）
                # CoC = (aperture * focal_length * delta_depth) / (depth * (focal_length - delta_depth))
                # 简化实现
                coc_size = (self.aperture * delta_depth) / (self.focus_distance + delta_depth)
                
                # 应用对焦范围（过渡区域）
                if delta_depth < self.focus_range:
                    # 在对焦范围内，减少CoC大小
                    coc_size *= delta_depth / self.focus_range
                
                # 限制CoC大小
                coc_size = min(coc_size, self.blur_radius / 100.0)  # 归一化到0-1范围
                
                coc_map[y, x] = coc_size
        
        return coc_map
    
    def _apply_bokeh_blur(self, input_texture, coc_map, blur_radius):
        """应用散景模糊
        
        参数:
        - input_texture: 输入纹理
        - coc_map: 混淆圆贴图
        - blur_radius: 最大模糊半径
        
        返回:
        - 模糊后的纹理
        """
        if not isinstance(input_texture, np.ndarray) or not isinstance(coc_map, np.ndarray):
            return input_texture
        
        height, width = input_texture.shape[:2]
        blurred = np.copy(input_texture)
        
        # 简化的散景模糊实现
        for y in range(height):
            for x in range(width):
                # 获取当前像素的CoC大小
                coc_size = coc_map[y, x]
                if coc_size < 0.01:
                    # CoC太小，跳过模糊
                    continue
                
                # 根据CoC计算实际模糊半径
                current_blur_radius = int(coc_size * blur_radius)
                if current_blur_radius < 1:
                    continue
                
                # 散景模糊（简化实现，实际需要更复杂的算法）
                sum_color = np.zeros(input_texture.shape[2], dtype=np.float32)
                sum_weight = 0.0
                
                for ky in range(-current_blur_radius, current_blur_radius + 1):
                    for kx in range(-current_blur_radius, current_blur_radius + 1):
                        # 计算采样位置
                        sample_y = y + ky
                        sample_x = x + kx
                        
                        # 边界检查
                        if sample_y >= 0 and sample_y < height and sample_x >= 0 and sample_x < width:
                            # 计算距离权重（高斯分布）
                            distance = np.sqrt(ky*ky + kx*kx)
                            if distance > current_blur_radius:
                                continue
                            
                            # 基于距离的权重
                            weight = np.exp(-distance * distance / (2.0 * coc_size * coc_size * blur_radius * blur_radius))
                            
                            # 累积颜色和权重
                            sum_color += input_texture[sample_y, sample_x] * weight
                            sum_weight += weight
                
                if sum_weight > 0.0:
                    blurred[y, x] = sum_color / sum_weight
        
        return blurred
    
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
    
    def _apply_dof_maxwell(self, input_texture, output_texture):
        """针对NVIDIA Maxwell架构的优化实现"""
        return self._generic_dof_implementation(input_texture, output_texture, 
                                             high_performance=True, use_bokeh=False)
    
    def _apply_dof_gcn(self, input_texture, output_texture):
        """针对AMD GCN架构的优化实现"""
        return self._generic_dof_implementation(input_texture, output_texture, 
                                             high_performance=False, use_bokeh=True)
    
    def _apply_dof_generic(self, input_texture, output_texture):
        """通用实现"""
        return self._generic_dof_implementation(input_texture, output_texture, 
                                             high_performance=True, use_bokeh=False)
    
    def _generic_dof_implementation(self, input_texture, output_texture, 
                                  high_performance=True, use_bokeh=True):
        """通用景深实现"""
        if not isinstance(input_texture, np.ndarray):
            return input_texture
        
        # 创建输出纹理
        output = np.copy(input_texture)
        height, width = output.shape[:2]
        
        # 生成简化的深度缓冲区（实际应从渲染器获取）
        # 这里使用R通道作为深度
        depth_buffer = input_texture[:, :, 0:1] if input_texture.shape[2] > 1 else input_texture
        
        # 1. 计算混淆圆贴图
        coc_map = self._calculate_coc(depth_buffer)
        
        # 2. 降采样（提高性能）
        downsample_factor = self.downsample_factor
        if high_performance:
            downsample_factor = max(downsample_factor, 8)
        
        downsampled = self._downsample(input_texture, downsample_factor)
        downsampled_coc = self._downsample(coc_map[:, :, np.newaxis], downsample_factor)[:, :, 0]
        
        # 3. 应用散景模糊
        blur_radius = self.blur_radius
        if high_performance:
            blur_radius = max(1, blur_radius // 2)
        
        blurred = self._apply_bokeh_blur(downsampled, downsampled_coc, blur_radius)
        
        # 4. 上采样回到原始分辨率
        upsampled = self._upsample(blurred, downsample_factor)
        
        # 5. 基于CoC混合原始图像和模糊图像
        for y in range(height):
            for x in range(width):
                # 获取CoC值
                coc_value = coc_map[y, x]
                
                # 确保坐标在upsampled范围内
                upsampled_y = min(y, upsampled.shape[0] - 1)
                upsampled_x = min(x, upsampled.shape[1] - 1)
                
                # 混合图像
                blurred_color = upsampled[upsampled_y, upsampled_x]
                output[y, x] = output[y, x] * (1.0 - coc_value) + blurred_color * coc_value
        
        return output
    
    def adjust_quality(self, quality_level):
        """调整景深质量"""
        super().adjust_quality(quality_level)
        
        if quality_level == EffectQuality.LOW:
            self.blur_radius = 4
            self.downsample_factor = 8
            # 低端GPU使用简化的景深算法
        elif quality_level == EffectQuality.MEDIUM:
            self.blur_radius = 8
            self.downsample_factor = 4
        elif quality_level == EffectQuality.HIGH:
            self.blur_radius = 16
            self.downsample_factor = 2
            # 高端GPU使用更复杂的散景效果
    
    def set_focus_distance(self, distance):
        """设置焦距"""
        self.focus_distance = max(0.1, distance)
    
    def set_focus_range(self, range):
        """设置对焦范围"""
        self.focus_range = max(0.1, range)
    
    def set_aperture(self, aperture):
        """设置光圈大小"""
        self.aperture = max(0.0, min(1.0, aperture))
