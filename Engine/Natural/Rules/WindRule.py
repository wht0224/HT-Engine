import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class WindRule(Rule):
    """
    风力规则 (Wind Rule) - 优化版
    
    物理层规则：计算植被随风摆动的偏移量。
    完全基于向量化计算 (Numpy)，无 Python 循环。
    
    v3.0 优化：
    - 缓存计算结果，避免重复计算
    - 使用更高效的numpy操作
    - 减少不必要的内存分配
    """
    
    def __init__(self):
        super().__init__("Physics.Wind", priority=100)
        self._cached_count = 0
        self._cached_phase = None
        
    def evaluate(self, facts: FactBase):
        wind_dir = facts.get_global("wind_direction")
        wind_speed = facts.get_global("wind_speed")
        time = facts.get_global("time")
        
        try:
            stiffness = facts.get_column("vegetation", "stiffness")
            pos_x = facts.get_column("vegetation", "pos_x")
            pos_z = facts.get_column("vegetation", "pos_z")
            terrain_height = facts.get_column("vegetation", "terrain_height")
            terrain_grad_x = facts.get_column("vegetation", "terrain_grad_x")
        except KeyError:
            return
            
        count = len(stiffness)
        if count == 0:
            return

        # 缓存phase计算（只在数量变化时重新计算）
        if self._cached_count != count:
            self._cached_phase = pos_x * 0.5 + pos_z * 0.3
            self._cached_count = count
        
        # 合并计算，减少中间数组
        local_wind_speed = wind_speed * (1.0 + terrain_height * 0.005) * \
                          np.clip(1.0 + wind_dir[0] * terrain_grad_x * 2.0, 0.2, 2.0)
        
        # 一次性计算最终结果
        result_offset_x = wind_dir[0] * local_wind_speed * (1.0 - stiffness) * \
                         np.sin(time * 2.0 + self._cached_phase)
        
        facts.set_column("vegetation", "offset_x", result_offset_x)
