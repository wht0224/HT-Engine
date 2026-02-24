import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class VegetationGrowthRule(Rule):
    """
    植被生长规则 (Vegetation Growth Rule)
    
    基于水文分布和地形条件的植被演替。
    
    Logic:
    1. Growth: 水分充足且坡度适宜的地方生长。
    2. Death: 干旱或过陡的地方死亡。
    3. Competition: 密度趋向于承载力 (Carrying Capacity)。
    
    Attributes:
        growth_rate (float): 生长速率
        death_rate (float): 枯死速率
        optimum_water (float): 最适水位
        max_slope (float): 最大生长坡度 (超过此坡度难以生长)
    """
    
    def __init__(self, growth_rate=0.1, death_rate=0.05, optimum_water=1.0, max_slope=1.5):
        super().__init__("Evolution.Vegetation", priority=30)
        self.growth_rate = growth_rate
        self.death_rate = death_rate
        self.optimum_water = optimum_water
        self.max_slope = max_slope
        
    def evaluate(self, facts: FactBase):
        # 此规则通常作用于地形网格 (terrain_main)
        # 假设地形表中包含: water, height, vegetation_density
        
        table_name = "terrain_main"
        try:
            water = facts.get_column(table_name, "water")
            density = facts.get_column(table_name, "vegetation_density")
            height = facts.get_column(table_name, "height")
            
            # 计算坡度 (这里为了简化，可能需要再次计算梯度，或者假设已有 slope 列)
            # 为了保持规则独立性，这里简单计算梯度模长
            # 注意：这里假设是 flattened grid，直接计算梯度比较麻烦
            # 如果没有预计算的 slope 列，我们暂时忽略坡度，或者假设已有 'slope' 列
            try:
                slope = facts.get_column(table_name, "slope")
            except KeyError:
                # Fallback: ignore slope if not present
                slope = np.zeros_like(water)
                
            # 1. 生长 (Growth)
            # 水分越多越好，直到饱和
            # Logistic Growth: rate * density * (1 - density/capacity)
            # 这里简化为: rate * water_factor * (1 - density)
            
            water_factor = np.clip(water / self.optimum_water, 0.0, 1.0)
            slope_factor = np.clip(1.0 - (slope / self.max_slope), 0.0, 1.0)
            
            # 只有在有种子的地方才会长？或者假设到处都有隐式种子
            # 为了让荒漠长出植被，假设有基础生长潜力
            growth_potential = water_factor * slope_factor
            
            # 增加密度
            new_growth = self.growth_rate * growth_potential * (1.0 - density)
            
            # 2. 死亡 (Death)
            # 缺水或过密导致死亡
            # 如果 water < 0.1, 快速死亡
            drought_factor = 1.0 - water_factor
            death = self.death_rate * density * (1.0 + drought_factor * 5.0)
            
            # 更新
            density += new_growth - death
            np.clip(density, 0.0, 1.0, out=density)
            
            # 写回
            facts.set_column(table_name, "vegetation_density", density)
            
        except KeyError:
            pass
