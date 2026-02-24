import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class HydraulicErosionRule(Rule):
    """
    水力侵蚀规则 (Hydraulic Erosion Rule)
    
    模拟降雨、水流流动、侵蚀和沉积。
    使用基于网格的 Pipe Model (浅水方程简化版)，全 Numpy 向量化实现。
    
    Attributes:
        rain_rate (float): 每帧降雨量
        evaporation_rate (float): 每帧蒸发率
        solubility (float): 水携带泥沙的能力系数
        erosion_rate (float): 侵蚀速率
        deposition_rate (float): 沉积速率
        dt (float): 模拟时间步长
    """
    
    def __init__(self, rain_rate=0.01, evaporation_rate=0.05, 
                 solubility=0.5, erosion_rate=0.1, deposition_rate=0.1, dt=0.1):
        super().__init__("Evolution.Erosion", priority=20)
        self.rain_rate = rain_rate
        self.evaporation_rate = evaporation_rate
        self.solubility = solubility
        self.erosion_rate = erosion_rate
        self.deposition_rate = deposition_rate
        self.dt = dt

    def evaluate(self, facts: FactBase):
        # 假设存在一个名为 "terrain_main" 的地形表
        # 在实际整合中，应该遍历所有地形 Chunk
        table_name = "terrain_main"
        
        try:
            # 1. 获取数据 (Reshape 为 2D 网格)
            flat_height = facts.get_column(table_name, "height")
            grid_len = len(flat_height)
            size = int(np.sqrt(grid_len))
            
            if size * size != grid_len:
                return # 非方阵，跳过
            
            H = flat_height.reshape((size, size))
            
            # 初始化或获取 Water/Sediment
            try:
                water = facts.get_column(table_name, "water").reshape((size, size))
                sediment = facts.get_column(table_name, "sediment").reshape((size, size))
            except KeyError:
                water = np.zeros_like(H)
                sediment = np.zeros_like(H)
                facts.add_column(table_name, "water", water.flatten())
                facts.add_column(table_name, "sediment", sediment.flatten())

            # ==============================
            # Core Erosion Algorithm (Vectorized)
            # ==============================
            
            # 1. 降雨 (Rain)
            water += self.rain_rate * self.dt
            
            # 2. 侵蚀与沉积 (Erosion & Deposition)
            # -----------------------------------
            # 计算局部梯度
            grad_y, grad_x = np.gradient(H)
            slope = np.sqrt(grad_x**2 + grad_y**2)
            
            # 计算携带能力 (Capacity) = 水量 * 速度(近似坡度) * 溶解度
            # 最小坡度保护，防止除零
            velocity = np.maximum(0.01, slope) 
            capacity = water * velocity * self.solubility
            
            # 决定侵蚀还是沉积
            diff = capacity - sediment
            
            # 侵蚀 (Erode): 如果携带量 < 能力
            erode_mask = diff > 0
            amount_to_erode = diff * self.erosion_rate * self.dt
            # 不能侵蚀超过现有高度的一定比例，防止穿模
            H[erode_mask] -= amount_to_erode[erode_mask]
            sediment[erode_mask] += amount_to_erode[erode_mask]
            
            # 沉积 (Deposit): 如果携带量 > 能力
            deposit_mask = diff < 0
            amount_to_deposit = -diff * self.deposition_rate * self.dt
            H[deposit_mask] += amount_to_deposit[deposit_mask]
            sediment[deposit_mask] -= amount_to_deposit[deposit_mask]
            
            # 3. 水流输送 (Transport) - 简化的 4 邻居流向
            # -----------------------------------
            # 计算四个方向的高度差 (正值代表可以流向该方向)
            # Pad H to handle boundaries (assume infinite walls or clamp)
            # Using roll for periodic (wrap) or manual slice for clamp. 
            # For simplicity in Demo, we assume flat boundary (clamped manually via slicing logic implied by gradient).
            # Let's use roll but zero out boundaries later if needed.
            
            # Left, Right, Up, Down Neighbors
            H_L = np.roll(H, 1, axis=1); H_L[:, 0] = H[:, 0] # Clamp boundary
            H_R = np.roll(H, -1, axis=1); H_R[:, -1] = H[:, -1]
            H_U = np.roll(H, 1, axis=0); H_U[0, :] = H[0, :]
            H_D = np.roll(H, -1, axis=0); H_D[-1, :] = H[-1, :]
            
            # Diffs (Current - Neighbor). Positive means Current is higher.
            d_L = H - H_L
            d_R = H - H_R
            d_U = H - H_U
            d_D = H - H_D
            
            # Only flow downhill
            f_L = np.maximum(0, d_L)
            f_R = np.maximum(0, d_R)
            f_U = np.maximum(0, d_U)
            f_D = np.maximum(0, d_D)
            
            f_total = f_L + f_R + f_U + f_D
            
            # 防止除零
            f_total_safe = np.maximum(1e-5, f_total)
            
            # 计算流出比例 (Flux)
            # 如果 water < f_total * dt, 则按比例流出所有水
            # Scaling factor K
            K = np.minimum(1.0, water / (f_total_safe * self.dt))
            
            # 实际流出的水量
            flow_L = f_L * K
            flow_R = f_R * K
            flow_U = f_U * K
            flow_D = f_D * K
            
            # 更新当前格子水量 (流出)
            total_out = flow_L + flow_R + flow_U + flow_D
            water -= total_out * self.dt
            
            # 也要移动泥沙 (比例相同)
            # Sediment follows water
            sediment_fraction = sediment / (water + total_out * self.dt + 1e-5) # Old water amount approx
            sediment_out = total_out * sediment_fraction * self.dt
            sediment -= sediment_out
            
            # 更新邻居格子水量 (流入)
            # 当前格子的 flow_L 是流向左边，所以左边格子的 inflow 来自右边 (roll -1)
            # We construct Inflow map by rolling the Outflow maps back
            inflow_water = (
                np.roll(flow_L, -1, axis=1) + # Receives from Right (which flowed Left) -> Wait, no.
                # flow_L[x,y] is flow from (x,y) to (x,y-1).
                # So (x,y) receives from (x,y+1) via its flow_L.
                np.roll(flow_L, -1, axis=1) * 0 + # Logic check:
                # If I am at x, my left neighbor is x-1. flow_L moves mass to x-1.
                # So x receives from x+1 (Right neighbor) via x+1's flow_L.
                # x+1 is roll(-1).
                np.roll(flow_L, -1, axis=1) + 
                np.roll(flow_R, 1, axis=1) +
                np.roll(flow_U, -1, axis=0) +
                np.roll(flow_D, 1, axis=0)
            ) * self.dt
            
            # Boundary fix for inflow (first/last col/row rolled in from opposite side)
            # For simplicity, ignore wrap-around artifacts at edges for this demo.
            
            water += inflow_water
            
            # Sediment inflow
            inflow_sediment = (
                np.roll(flow_L * sediment_fraction, -1, axis=1) +
                np.roll(flow_R * sediment_fraction, 1, axis=1) +
                np.roll(flow_U * sediment_fraction, -1, axis=0) +
                np.roll(flow_D * sediment_fraction, 1, axis=0)
            ) * self.dt
            
            sediment += inflow_sediment
            
            # 4. 蒸发 (Evaporation)
            water *= (1.0 - self.evaporation_rate * self.dt)
            
            # ==============================
            # End Logic
            # ==============================
            
            # 写回 FactBase
            facts.set_column(table_name, "height", H.flatten())
            facts.set_column(table_name, "water", water.flatten())
            facts.set_column(table_name, "sediment", sediment.flatten())
            
        except KeyError:
            pass

