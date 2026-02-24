import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

class ScreenspaceReflections(EffectBase):
    """针对低端GPU优化的屏幕空间反射效果"""
    
    def __init__(self, gpu_architecture=GPUArchitecture.OTHER, quality_level=EffectQuality.LOW):
        super().__init__(gpu_architecture, quality_level)
        self.performance_cost = 3.0  # 基础性能开销估算
        self.max_steps = 16  # 光线步进最大步数
        self.binary_search_steps = 4  # 二分查找步数
        self.intensity = 0.5  # 反射强度
        self.roughness_factor = 0.2  # 粗糙度因子
        
        # 在低端GPU上默认禁用
        if quality_level == EffectQuality.LOW:
            self.is_enabled = False
        
        # 根据质量级别设置参数
        self._update_resolution_and_settings()
    
    def _setup_resources(self):
        """设置SSR所需的资源"""
        # 深度、法线和颜色纹理
        self.depth_texture = None
        self.normal_texture = None
        self.color_texture = None
        
        # 中间渲染纹理
        self.reflection_texture = None
        self.blur_texture = None
        
        # 着色器程序
        self.ssr_shader = None
        self.blur_shader = None
        self.composite_shader = None
    
    def _update_resolution_and_settings(self):
        """根据质量级别更新参数"""
        if self.quality_level == EffectQuality.LOW:  # GTX 750Ti级别
            self.is_enabled = False  # 在最低质量下默认禁用
            self.max_steps = 8
            self.binary_search_steps = 2
            self.intensity = 0.3
            self.performance_cost = 1.0
        elif self.quality_level == EffectQuality.MEDIUM:  # RX 580级别
            self.max_steps = 16
            self.binary_search_steps = 4
            self.intensity = 0.5
            self.performance_cost = 2.0
        else:  # 高级别
            self.max_steps = 32
            self.binary_search_steps = 8
            self.intensity = 0.8
            self.performance_cost = 4.0
    
    def _optimize_for_gpu(self):
        """根据GPU架构优化SSR"""
        super()._optimize_for_gpu()
        
        # NVIDIA Maxwell特定优化
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            # Maxwell架构对SSR的优化
            self._use_jittered_steps = False  # 避免额外计算
            self._use_lod_bias = True  # 降低纹理采样质量
            self._use_early_termination = True  # 提前终止光线步进
            
            # GTX 750Ti上默认禁用SSR
            if self.quality_level == EffectQuality.LOW:
                self.is_enabled = False
        # AMD GCN特定优化
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            # GCN架构对SSR的优化
            self._use_jittered_steps = True  # GCN可以处理更多计算
            self._use_lod_bias = False
            self._use_early_termination = True
            
            # RX 580可以在中等质量下启用SSR
            if self.quality_level >= EffectQuality.MEDIUM:
                self.is_enabled = True
    
    def update(self, delta_time):
        """更新SSR参数"""
        # 可以在这里添加动态调整参数的逻辑
        pass
    
    def _apply_effect(self, input_texture, output_texture):
        """应用SSR效果"""
        # 模拟SSR实现步骤
        # 1. 从G-Buffer获取法线和深度
        # 2. 为每个像素计算反射光线
        # 3. 执行光线步进找到交点
        # 4. 应用粗糙度模糊
        # 5. 与原始渲染结果混合
        
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_ssr_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_ssr_gcn(input_texture, output_texture)
        else:
            return self._apply_ssr_generic(input_texture, output_texture)
    
    def _apply_ssr_maxwell(self, input_texture, output_texture):
        """Maxwell架构特定的SSR实现"""
        # 在Maxwell架构上使用高度优化的SSR实现
        # 1. 减少光线步进次数
        # 2. 使用更低的分辨率
        # 3. 简化的反射计算
        if self.quality_level >= EffectQuality.MEDIUM and self.is_enabled:
            # 即使启用，也使用最低成本的实现
            return self._generic_ssr_implementation(input_texture, output_texture, 
                                                   low_resolution=True, reduced_steps=True)
        return input_texture  # 否则返回原始纹理
    
    def _apply_ssr_gcn(self, input_texture, output_texture):
        """GCN架构特定的SSR实现"""
        # 在GCN架构上使用优化的SSR实现
        # 1. 中等光线步进次数
        # 2. 中等分辨率
        # 3. 更精确的反射计算
        if self.is_enabled:
            return self._generic_ssr_implementation(input_texture, output_texture, 
                                                   low_resolution=False, reduced_steps=False)
        return input_texture  # 否则返回原始纹理
    
    def _apply_ssr_generic(self, input_texture, output_texture):
        """通用的SSR实现"""
        # 保守的通用实现
        if self.quality_level >= EffectQuality.HIGH and self.is_enabled:
            return self._generic_ssr_implementation(input_texture, output_texture, 
                                                   low_resolution=True, reduced_steps=True)
        return input_texture  # 否则返回原始纹理
    
    def _sample_depth_buffer(self, depth_buffer, uv):
        """从深度缓冲区采样深度值"""
        # 简化实现，实际需要根据UV坐标从深度纹理采样
        # 这里假设depth_buffer是一个numpy数组，存储深度值
        if isinstance(depth_buffer, np.ndarray):
            height, width = depth_buffer.shape[:2]
            x = int(uv[0] * width)
            y = int(uv[1] * height)
            # 确保坐标在有效范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            return depth_buffer[y, x]
        return 1.0  # 默认返回最大深度
    
    def _world_to_depth(self, world_pos):
        """将世界坐标转换为深度值"""
        # 简化实现，实际需要使用相机的投影矩阵
        # 这里假设world_pos的z值即为深度
        return world_pos[2]
    
    def _is_in_viewport(self, world_pos):
        """检查世界坐标是否在视口内"""
        # 简化实现，实际需要使用相机的视锥体
        # 这里只检查z值是否在有效范围内
        return world_pos[2] > 0.01 and world_pos[2] < 1000.0
    
    def _calculate_reflection_vector(self, view_dir, normal, roughness):
        """计算反射向量"""
        # 理想反射方向
        perfect_reflection = view_dir - 2.0 * np.dot(view_dir, normal) * normal
        
        if roughness < 0.01:
            # 几乎光滑的表面，直接返回理想反射
            return perfect_reflection
        
        # 添加基于粗糙度的随机扰动
        def random_in_hemisphere(normal):
            """在法线方向的半球内生成随机向量"""
            while True:
                v = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, 1)])
                v /= np.linalg.norm(v)
                if np.dot(v, normal) > 0:
                    return v
        
        # 使用余弦加权的半球采样
        micro_normal = random_in_hemisphere(normal)
        
        # 根据粗糙度混合理想反射和随机反射
        reflection = perfect_reflection * (1.0 - roughness) + micro_normal * roughness
        reflection /= np.linalg.norm(reflection)
        
        return reflection
    
    def _apply_roughness_blur(self, reflection_texture, roughness_map):
        """应用基于粗糙度的模糊"""
        # 简化实现，实际需要根据粗糙度值应用不同程度的模糊
        # 这里返回原始反射纹理
        return reflection_texture
    
    def _generic_ssr_implementation(self, input_texture, output_texture, 
                                   low_resolution=True, reduced_steps=False):
        """通用SSR实现核心逻辑"""
        if not isinstance(input_texture, np.ndarray):
            return input_texture
            
        # TODO: Implement real OpenGL shader
        # 根据参数调整步进次数
        steps = self.max_steps // 2 if reduced_steps else self.max_steps
        binary_steps = self.binary_search_steps // 2 if reduced_steps else self.binary_search_steps
        
        # 获取G-Buffer数据（简化实现，实际需要从渲染器获取）
        depth_buffer = np.ones_like(input_texture[:, :, 0])  # 临时深度缓冲区
        normal_buffer = np.zeros_like(input_texture)  # 临时法线缓冲区
        normal_buffer[:, :, 2] = 1.0  # 法线指向Z轴正方向
        roughness_buffer = np.ones_like(input_texture[:, :, 0]) * 0.1  # 临时粗糙度缓冲区
        
        # 创建反射结果缓冲区
        reflection_result = np.copy(input_texture)
        
        # 获取纹理尺寸
        height, width = input_texture.shape[:2]
        
        # 降低分辨率处理（如果启用）
        if low_resolution:
            width = width // 2
            height = height // 2
            
        # 对每个像素执行SSR
        for y in range(height):
            for x in range(width):
                # 计算UV坐标
                uv = np.array([x / width, y / height])
                
                # 从G-Buffer获取数据
                depth = self._sample_depth_buffer(depth_buffer, uv)
                normal = normal_buffer[int(uv[1] * depth_buffer.shape[0]), int(uv[0] * depth_buffer.shape[1])] if isinstance(depth_buffer, np.ndarray) else np.array([0.0, 0.0, 1.0])
                roughness = roughness_buffer[int(uv[1] * roughness_buffer.shape[0]), int(uv[0] * roughness_buffer.shape[1])] if isinstance(roughness_buffer, np.ndarray) else 0.1
                
                # 计算世界坐标和视图方向（简化实现）
                view_dir = np.array([uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0, -1.0])
                view_dir /= np.linalg.norm(view_dir)
                world_pos = np.array([uv[0] * 2.0 - 1.0, uv[1] * 2.0 - 1.0, depth])
                
                # 计算反射方向
                reflection_dir = self._calculate_reflection_vector(view_dir, normal, roughness)
                
                # 执行光线步进
                hit_found = False
                hit_pos = None
                hit_depth = 0.0
                
                # 分层光线步进
                step_size = 0.1
                current_pos = world_pos
                
                for step in range(steps):
                    # 从深度缓冲区获取当前位置的深度
                    current_uv = np.array([(current_pos[0] + 1.0) / 2.0, (current_pos[1] + 1.0) / 2.0])
                    scene_depth = self._sample_depth_buffer(depth_buffer, current_uv)
                    
                    # 计算光线位置与场景深度的差异
                    ray_depth = self._world_to_depth(current_pos)
                    depth_diff = ray_depth - scene_depth
                    
                    # 检查是否击中表面
                    if depth_diff > 0.0 and depth_diff < 0.1:
                        # 执行二分查找以获取更精确的交点
                        start_pos = current_pos - reflection_dir * step_size
                        end_pos = current_pos
                        
                        for binary_step in range(binary_steps):
                            mid_pos = (start_pos + end_pos) * 0.5
                            mid_uv = np.array([(mid_pos[0] + 1.0) / 2.0, (mid_pos[1] + 1.0) / 2.0])
                            mid_ray_depth = self._world_to_depth(mid_pos)
                            mid_scene_depth = self._sample_depth_buffer(depth_buffer, mid_uv)
                            
                            mid_depth_diff = mid_ray_depth - mid_scene_depth
                            
                            if abs(mid_depth_diff) < 0.001:
                                hit_found = True
                                hit_pos = mid_pos
                                hit_depth = mid_ray_depth
                                break
                            elif mid_depth_diff > 0.0:
                                end_pos = mid_pos
                            else:
                                start_pos = mid_pos
                        break
                    
                    # 动态调整步进大小
                    if depth_diff < 0.0:
                        step_size *= 0.5  # 靠近表面时减小步进
                    else:
                        step_size *= 1.2  # 远离表面时增大步进
                    
                    # 更新光线位置
                    current_pos += reflection_dir * step_size
                    
                    # 检查是否超出视锥体
                    if not self._is_in_viewport(current_pos):
                        break
                
                # 如果找到交点，采样颜色
                if hit_found:
                    # 将世界坐标转换为UV坐标
                    reflection_uv = np.array([(hit_pos[0] + 1.0) / 2.0, (hit_pos[1] + 1.0) / 2.0])
                    
                    # 确保UV在有效范围内
                    reflection_uv = np.clip(reflection_uv, 0.0, 1.0)
                    
                    # 采样反射颜色（简化实现，实际需要纹理采样）
                    ref_x = int(reflection_uv[0] * input_texture.shape[1])
                    ref_y = int(reflection_uv[1] * input_texture.shape[0])
                    reflection_color = input_texture[ref_y, ref_x]
                    
                    # 应用基于粗糙度的模糊
                    reflection_color = self._apply_roughness_blur(reflection_color, roughness)
                    
                    # 根据粗糙度调整反射强度
                    roughness_factor = 1.0 - min(1.0, roughness * self.roughness_factor)
                    reflection_color *= self.intensity * roughness_factor
                    
                    # 更新反射结果
                    if low_resolution:
                        # 放大到原始分辨率
                        x_orig = x * 2
                        y_orig = y * 2
                        for dy in range(2):
                            for dx in range(2):
                                if y_orig + dy < input_texture.shape[0] and x_orig + dx < input_texture.shape[1]:
                                    reflection_result[y_orig + dy, x_orig + dx] = reflection_color
                    else:
                        reflection_result[y, x] = reflection_color
        
        # 混合反射结果与原始图像
        final_result = input_texture * (1.0 - self.intensity) + reflection_result * self.intensity
        
        return final_result
    
    def set_intensity(self, intensity):
        """设置反射强度"""
        self.intensity = max(0.0, min(1.0, intensity))
    
    def set_roughness_factor(self, factor):
        """设置粗糙度因子"""
        self.roughness_factor = max(0.0, min(1.0, factor))
    
    def __str__(self):
        status = "Enabled" if self.is_enabled else "Disabled"
        return f"SSR ({status}, Steps: {self.max_steps}, Intensity: {self.intensity}, Quality: {self.quality_level.name})"