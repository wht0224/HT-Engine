import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

class AmbientOcclusion(EffectBase):
    """针对低端GPU优化的环境光遮蔽效果"""
    
    def __init__(self, gpu_architecture=GPUArchitecture.OTHER, quality_level=EffectQuality.LOW):
        super().__init__(gpu_architecture, quality_level)
        self.performance_cost = 1.2  # 基础性能开销估算
        self.radius = 1.0  # AO采样半径
        self.intensity = 0.8  # AO强度
        self.falloff = 0.5  # 衰减因子
        self.sample_count = 8  # 采样点数量
        self.downsample_factor = 2  # 降采样因子
        
        # 根据质量级别设置参数
        self._update_resolution_and_settings()
    
    def _setup_resources(self):
        """设置AO所需的资源"""
        # 深度和法线纹理
        self.depth_texture = None
        self.normal_texture = None
        
        # 中间渲染纹理
        self.ao_texture = None
        self.blur_texture = None
        
        # 着色器程序
        self.ao_shader = None
        self.blur_shader = None
        self.composite_shader = None
        
        # 采样核
        self.sample_kernel = self._generate_sample_kernel()
        self.noise_texture = self._generate_noise_texture()
    
    def _generate_sample_kernel(self):
        """生成半球采样核"""
        # 生成优化的采样核，减少计算量
        kernel = np.zeros((self.sample_count, 4), dtype=np.float32)
        
        for i in range(self.sample_count):
            # 生成半球内的随机向量
            x = np.random.uniform(-1.0, 1.0)
            y = np.random.uniform(-1.0, 1.0)
            z = np.random.uniform(0.0, 1.0)  # 只取上半球
            
            # 归一化
            vec = np.array([x, y, z])
            vec /= np.linalg.norm(vec)
            
            # 按距离加权，近处采样更密集
            scale = float(i) / float(self.sample_count)
            scale = 0.1 + (scale * scale) * 0.9
            vec *= scale
            
            kernel[i] = np.array([vec[0], vec[1], vec[2], scale])
        
        return kernel
    
    def _generate_noise_texture(self):
        """生成随机旋转噪声纹理"""
        # 生成低分辨率噪声纹理以减少采样成本
        size = 4  # 4x4噪声纹理，足够且高效
        noise = np.zeros((size, size, 3), dtype=np.float32)
        
        for i in range(size):
            for j in range(size):
                # 生成切线空间中的随机向量
                x = np.random.uniform(-1.0, 1.0)
                y = np.random.uniform(-1.0, 1.0)
                z = 0.0  # 2D旋转
                
                vec = np.array([x, y, z])
                vec /= np.linalg.norm(vec)
                
                noise[i, j] = vec
        
        return noise
    
    def _update_resolution_and_settings(self):
        """根据质量级别更新参数"""
        if self.quality_level == EffectQuality.LOW:  # GTX 750Ti级别
            self.sample_count = 4  # 减少采样点
            self.radius = 0.5
            self.intensity = 0.6
            self.downsample_factor = 4  # 更大的降采样
            self.performance_cost = 0.6  # 降低性能开销
        elif self.quality_level == EffectQuality.MEDIUM:  # RX 580级别
            self.sample_count = 8
            self.radius = 1.0
            self.intensity = 0.8
            self.downsample_factor = 2
            self.performance_cost = 1.2
        else:  # 高级别
            self.sample_count = 16
            self.radius = 1.5
            self.intensity = 1.0
            self.downsample_factor = 2
            self.performance_cost = 2.0
        
        # 重新生成采样核
        self.sample_kernel = self._generate_sample_kernel()
    
    def _optimize_for_gpu(self):
        """根据GPU架构优化AO"""
        super()._optimize_for_gpu()
        
        # NVIDIA Maxwell特定优化
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            # Maxwell架构对AO的优化
            self._use_horizon_based_ao = True  # 更高效的计算方式
            self._use_voxel_ao = False  # 避免复杂的体素操作
            self._use_half_precision_calculations = True  # 使用半精度计算
        # AMD GCN特定优化
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            # GCN架构对AO的优化
            self._use_horizon_based_ao = False
            self._use_voxel_ao = False
            self._use_compute_shader_optimization = True  # 利用GCN的计算着色器优势
    
    def update(self, delta_time):
        """更新AO参数"""
        # 可以在这里添加动态调整参数的逻辑
        pass
    
    def _apply_effect(self, input_texture, output_texture):
        """应用AO效果"""
        # 模拟AO实现步骤
        # 1. 降采样深度和法线图
        # 2. 执行AO计算
        # 3. 应用高斯模糊减少噪点
        # 4. 上采样并与原始渲染结果混合
        
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_ao_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_ao_gcn(input_texture, output_texture)
        else:
            return self._apply_ao_generic(input_texture, output_texture)
    
    def _apply_ao_maxwell(self, input_texture, output_texture):
        """Maxwell架构特定的AO实现"""
        # 在Maxwell架构上使用高度优化的AO实现
        # 1. 使用超低分辨率计算 (1/4 或 1/8 分辨率)
        # 2. 减少采样点数量
        # 3. 使用简化的模糊算法
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=True, minimal_samples=True)
    
    def _apply_ao_gcn(self, input_texture, output_texture):
        """GCN架构特定的AO实现"""
        # 在GCN架构上使用优化的AO实现
        # 1. 中等分辨率计算 (1/2 分辨率)
        # 2. 标准采样点数量
        # 3. 更精确的模糊算法
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=False, minimal_samples=False)
    
    def _apply_ao_generic(self, input_texture, output_texture):
        """通用的AO实现"""
        # 保守的通用实现
        return self._generic_ao_implementation(input_texture, output_texture, 
                                             very_low_resolution=True, minimal_samples=True)
    
    def _sample_depth(self, depth_buffer, uv):
        """从深度缓冲区采样深度值"""
        if isinstance(depth_buffer, np.ndarray):
            height, width = depth_buffer.shape[:2]
            x = int(uv[0] * width)
            y = int(uv[1] * height)
            # 确保坐标在有效范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            return depth_buffer[y, x]
        return 1.0
    
    def _sample_normal(self, normal_buffer, uv):
        """从法线缓冲区采样法线值"""
        if isinstance(normal_buffer, np.ndarray):
            height, width = normal_buffer.shape[:2]
            x = int(uv[0] * width)
            y = int(uv[1] * height)
            # 确保坐标在有效范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            return normal_buffer[y, x]
        return np.array([0.0, 0.0, 1.0])
    
    def _depth_to_world(self, uv, depth, camera_params):
        """将深度值转换为世界坐标"""
        # 简化实现，实际需要使用相机的逆投影矩阵
        x = uv[0] * 2.0 - 1.0
        y = uv[1] * 2.0 - 1.0
        z = depth
        return np.array([x * z, y * z, z])
    
    def _get_view_direction(self, world_pos, camera_pos):
        """计算视角方向"""
        view_dir = world_pos - camera_pos
        view_dir /= np.linalg.norm(view_dir)
        return view_dir
    
    def _compute_optimized_ao(self, samples, downsample, input_texture):
        """优化的AO计算"""
        if not isinstance(input_texture, np.ndarray):
            return input_texture
        
        # 创建输出AO纹理
        height, width = input_texture.shape[:2]
        low_height = height // downsample
        low_width = width // downsample
        ao_result = np.ones((low_height, low_width), dtype=np.float32)
        
        # 简化的相机参数
        camera_pos = np.array([0.0, 0.0, 0.0])
        camera_params = {"fov": 60.0, "near": 0.1, "far": 100.0}
        
        # 生成简化的深度和法线缓冲区（实际应从渲染器获取）
        depth_buffer = input_texture[:, :, 0]  # 简化实现，使用R通道作为深度
        normal_buffer = np.zeros_like(input_texture)  # 简化实现，生成默认法线
        normal_buffer[:, :, 2] = 1.0
        
        # 对每个低分辨率像素计算AO
        for y in range(low_height):
            for x in range(low_width):
                # 计算UV坐标
                uv = np.array([x / low_width, y / low_height])
                orig_uv = np.array([x * downsample / width, y * downsample / height])
                
                # 采样深度和法线
                depth = self._sample_depth(depth_buffer, orig_uv)
                normal = self._sample_normal(normal_buffer, orig_uv)
                
                # 计算世界坐标
                world_pos = self._depth_to_world(orig_uv, depth, camera_params)
                
                # 计算视角方向
                view_dir = self._get_view_direction(world_pos, camera_pos)
                
                # 计算AO值
                occlusion = 0.0
                
                for i in range(samples):
                    # 获取采样向量
                    sample = self.sample_kernel[i]
                    sample_vec = np.array([sample[0], sample[1], sample[2]])
                    
                    # 转换到世界空间
                    # 简化实现，实际需要使用TBN矩阵
                    tangent = np.cross(view_dir, normal)
                    if np.linalg.norm(tangent) < 0.001:
                        tangent = np.array([1.0, 0.0, 0.0])
                    tangent /= np.linalg.norm(tangent)
                    bitangent = np.cross(normal, tangent)
                    bitangent /= np.linalg.norm(bitangent)
                    
                    # 创建TBN矩阵
                    TBN = np.array([tangent, bitangent, normal]).T
                    sample_world = TBN @ sample_vec
                    
                    # 计算采样位置
                    sample_pos = world_pos + sample_world * self.radius
                    
                    # 转换回屏幕空间
                    sample_uv = np.array([
                        (sample_pos[0] / sample_pos[2] + 1.0) / 2.0,
                        (sample_pos[1] / sample_pos[2] + 1.0) / 2.0
                    ])
                    
                    # 采样参考深度
                    ref_depth = self._sample_depth(depth_buffer, sample_uv)
                    ref_world = self._depth_to_world(sample_uv, ref_depth, camera_params)
                    
                    # 计算深度差
                    depth_diff = sample_pos[2] - ref_world[2]
                    
                    # 应用衰减和深度测试
                    if depth_diff > 0.01 and depth_diff < self.radius:
                        occlusion += 1.0 * (depth_diff / self.falloff)
                
                # 计算最终AO值
                occlusion = 1.0 - (occlusion / samples)
                ao_result[y, x] = occlusion ** self.intensity
        
        return ao_result
    
    def _apply_optimized_blur(self, ao_texture):
        """优化的AO模糊 - 双边滤波"""
        if not isinstance(ao_texture, np.ndarray):
            return ao_texture
        
        height, width = ao_texture.shape
        blurred = np.copy(ao_texture)
        
        # 简单的高斯模糊（双边滤波的简化版）
        kernel_size = 3
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
        kernel /= np.sum(kernel)
        
        for y in range(height):
            for x in range(width):
                sum_val = 0.0
                weight_sum = 0.0
                
                for ky in range(-kernel_size//2, kernel_size//2 + 1):
                    for kx in range(-kernel_size//2, kernel_size//2 + 1):
                        nx = x + kx
                        ny = y + ky
                        
                        if nx >= 0 and nx < width and ny >= 0 and ny < height:
                            weight = kernel[ky + kernel_size//2, kx + kernel_size//2]
                            sum_val += ao_texture[ny, nx] * weight
                            weight_sum += weight
                
                if weight_sum > 0.0:
                    blurred[y, x] = sum_val / weight_sum
        
        return blurred
    
    def _upsample_and_blend(self, ao_texture, downsample, original_texture):
        """上采样并混合AO"""
        if not isinstance(original_texture, np.ndarray) or not isinstance(ao_texture, np.ndarray):
            return original_texture
        
        height, width = original_texture.shape[:2]
        low_height, low_width = ao_texture.shape
        
        # 创建输出纹理
        output = np.copy(original_texture)
        
        # 双线性上采样AO纹理
        for y in range(height):
            for x in range(width):
                # 计算低分辨率坐标
                low_x = x / downsample
                low_y = y / downsample
                
                # 整数坐标
                x0 = int(low_x)
                y0 = int(low_y)
                x1 = min(x0 + 1, low_width - 1)
                y1 = min(y0 + 1, low_height - 1)
                
                # 插值权重
                tx = low_x - x0
                ty = low_y - y0
                
                # 双线性插值
                val00 = ao_texture[y0, x0]
                val01 = ao_texture[y0, x1]
                val10 = ao_texture[y1, x0]
                val11 = ao_texture[y1, x1]
                
                val0 = val00 * (1 - tx) + val01 * tx
                val1 = val10 * (1 - tx) + val11 * tx
                ao_val = val0 * (1 - ty) + val1 * ty
                
                # 应用AO到颜色
                output[y, x] = output[y, x] * ao_val
        
        return output
    
    def _generic_ao_implementation(self, input_texture, output_texture, 
                                  very_low_resolution=True, minimal_samples=True):
        """通用AO实现核心逻辑"""
        # 改进的AO实现，针对低端GPU优化
        
        # 根据参数调整采样点数量和分辨率
        samples = 4 if minimal_samples else self.sample_count
        downsample = 8 if very_low_resolution else self.downsample_factor
        
        # 1. 降采样深度和法线，使用更高效的降采样算法
        # 对于低端GPU，使用更大的降采样因子以提高性能
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL or very_low_resolution:
            downsample = 8
        else:
            downsample = 4
        
        # 2. 计算AO值，使用改进的采样算法
        ao_result = self._compute_optimized_ao(samples, downsample, input_texture)
        
        # 3. 应用改进的模糊算法，减少噪点
        blurred_ao = self._apply_optimized_blur(ao_result)
        
        # 4. 上采样并混合，使用高质量的上采样算法
        final_result = self._upsample_and_blend(blurred_ao, downsample, input_texture)
        
        # 返回处理后的纹理
        return final_result
    
    def set_intensity(self, intensity):
        """设置AO强度"""
        self.intensity = max(0.0, min(2.0, intensity))
    
    def set_radius(self, radius):
        """设置采样半径"""
        self.radius = max(0.1, min(5.0, radius))
    
    def __str__(self):
        return f"AO (Samples: {self.sample_count}, Radius: {self.radius}, Intensity: {self.intensity}, Quality: {self.quality_level.name})"