import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class OceanWaveRule(Rule):
    """
    海洋波浪规则 (Ocean Wave Rule)
    
    基于 Gerstner Wave 理论的过程化海浪生成。
    完全向量化计算，用于生成动态海面高度场和白沫遮罩。
    
    Attributes:
        waves (list): 波浪参数列表 [(wavelength, amplitude, speed, direction_x, direction_z), ...]
        foam_threshold (float): 产生白沫的雅可比行列式阈值
    """
    
    def __init__(self, foam_threshold=0.8):
        super().__init__("Environment.Ocean", priority=40)
        self.foam_threshold = foam_threshold
        
        # 预设的一组波浪参数 (Wavelength, Amplitude, Speed, DirX, DirZ)
        # 这种组合可以产生比较自然的"乱序"感
        self.waves = [
            (20.0, 1.0, 2.0, 1.0, 0.2),   # 主波
            (10.0, 0.5, 3.0, 0.7, 0.7),   # 次波
            (5.0,  0.2, 5.0, -0.2, 1.0),  # 细节波
            (2.5,  0.1, 8.0, 0.5, -0.5),  # 微波
        ]
        
    def evaluate(self, facts: FactBase):
        """
        计算海浪位移和白沫
        """
        try:
            # 1. 获取输入事实
            time = facts.get_global("time")
            wind_dir = facts.get_global("wind_direction") # (3,) vector
            
            # 假设有一个专门的海洋网格或者复用地形网格的一部分
            # 这里我们查找 "ocean_grid" 表
            table_name = "ocean_grid"
            
            # 原始坐标 (Undisplaced Grid)
            base_x = facts.get_column(table_name, "base_x")
            base_z = facts.get_column(table_name, "base_z")
            
            # 准备输出数组
            disp_x = np.zeros_like(base_x)
            disp_y = np.zeros_like(base_x) # Height
            disp_z = np.zeros_like(base_x)
            
            # 雅可比行列式近似值 (用于白沫计算)
            # Jacobian = 1 - sum(Qi * WA * cos(...))
            # 我们累加 sum 部分
            jacobian_sum = np.zeros_like(base_x)
            
            # 2. 叠加 Gerstner Waves
            for (wavelength, amp, speed, dx, dz) in self.waves:
                # 调整波浪方向受风向影响 (简单的点积混合)
                # 这里简化处理：波浪方向 = 预设方向 + 风向偏置
                # 或者直接使用预设方向，假设是"涌"(Swell)
                
                # 计算波数 k
                k = 2 * np.pi / wavelength
                
                # 相速度 c (深水波近似)
                c = np.sqrt(9.8 / k)
                
                # 归一化方向向量 D
                d_len = np.sqrt(dx*dx + dz*dz)
                D_x = dx / d_len
                D_z = dz / d_len
                
                # 陡度 Q (Steepness)
                # 为了避免自交 (Loop), Q * A * k 必须 < 1
                # 这里动态设置 Q
                Q = 0.5 / (k * amp * len(self.waves)) 
                
                # 相位 theta = k * (D . (x, z) - c * t)
                # 注意：这里应该用 base_x/z 还是 displaced x/z?
                # 标准 Gerstner 使用 base coordinates.
                dot_product = base_x * D_x + base_z * D_z
                theta = k * (dot_product - c * time)
                
                # 预计算三角函数
                sin_t = np.sin(theta)
                cos_t = np.cos(theta)
                
                # 累加位移
                # P.x = x + sum(Q * A * D.x * cos(theta))
                # P.y = sum(A * sin(theta))
                # P.z = z + sum(Q * A * D.z * cos(theta))
                
                QA = Q * amp
                disp_x += QA * D_x * cos_t
                disp_y += amp * sin_t
                disp_z += QA * D_z * cos_t
                
                # 累加雅可比分量 (用于白沫)
                # J = 1 - sum(Q * A * k * sin(theta)) 
                # Wait, partial derivatives are needed.
                # Jxx = 1 - sum(Q * A * k * Dx^2 * sin(theta))
                # Jzz = 1 - sum(Q * A * k * Dz^2 * sin(theta))
                # Jxz = - sum(Q * A * k * Dx * Dz * sin(theta))
                # 简化的泡沫判据：垂直位移的拉伸程度
                # 或者直接用波峰判定: height > threshold?
                # 使用标准的 Jacobian 判定: J < 0 表示波峰重叠(破碎)
                # J = Jxx * Jzz - Jxz * Jxz
                
                # 这里用一个简化的近似：只看波峰的挤压程度
                jacobian_sum += Q * amp * k * sin_t

            # 3. 计算结果
            final_x = base_x + disp_x
            final_y = disp_y
            final_z = base_z + disp_z
            
            # 白沫: 当挤压过大 (jacobian_sum 接近 1) 时产生白沫
            # J = 1 - jacobian_sum. If J < threshold, then foam.
            # So if jacobian_sum > (1 - threshold)
            foam_mask = np.clip((jacobian_sum - (1.0 - self.foam_threshold)) * 5.0, 0.0, 1.0)
            
            # 4. 写回 FactBase
            facts.set_column(table_name, "pos_x", final_x)
            facts.set_column(table_name, "pos_y", final_y)
            facts.set_column(table_name, "pos_z", final_z)
            facts.set_column(table_name, "foam_mask", foam_mask)
            
            # 可选：计算法线 (Normal)
            # 稍微复杂一点，需要求导。为了性能暂时略过，或者使用数值差分。
            
        except KeyError:
            # 如果 ocean_grid 表不存在，可能还没初始化
            pass
