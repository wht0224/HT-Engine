import numpy as np
from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase

class GrazingRule(Rule):
    """
    放牧规则 (Grazing Rule) - 生物层
    
    模拟食草动物的觅食行为。
    采用势能场 (Potential Field) 导航，无 A* 寻路。
    
    Update: 
    - 真实读取地形 vegetation_density
    - 基于梯度下降(上升)寻找食物
    """
    
    def __init__(self):
        super().__init__("Bio.Grazing", priority=50)
        
    def evaluate(self, facts: FactBase):
        dt = facts.get_global("dt") or 0.1
        debug = bool(facts.get_global("debug_grazing") or False)
        
        # 1. 获取生物数据
        try:
            pos_x = facts.get_column("herbivore", "pos_x")
            pos_z = facts.get_column("herbivore", "pos_z")
            vel_x = facts.get_column("herbivore", "vel_x")
            vel_z = facts.get_column("herbivore", "vel_z")
            hunger = facts.get_column("herbivore", "hunger")
        except KeyError:
            return
            
        count = len(pos_x)
        if count == 0:
            return

        try:
            herbivore_table = facts.tables["herbivore"]
            if "is_eating" not in herbivore_table:
                facts.add_column("herbivore", "is_eating", np.zeros(len(next(iter(herbivore_table.values()))), dtype=np.float32))
            if "heading" not in herbivore_table:
                facts.add_column("herbivore", "heading", np.zeros(len(next(iter(herbivore_table.values()))), dtype=np.float32))
        except Exception:
            return

        # 2. 获取地形感知数据
        table_name = "terrain_main"
        try:
            # 尝试获取植被密度
            veg_data = facts.get_column(table_name, "vegetation_density")
            grid_len = len(veg_data)
            size = int(np.sqrt(grid_len))
            
            # Reshape for sampling
            veg_map = veg_data.reshape((size, size))
            
            # 同时也需要高度/坡度信息来避险
            try:
                slope_data = facts.get_column(table_name, "slope").reshape((size, size))
            except:
                slope_data = np.zeros((size, size))
                
        except KeyError:
            if not facts.get_global("warned_missing_terrain_main"):
                if debug:
                    print("GrazingRule: Missing vegetation_density or terrain_main")
                facts.set_global("warned_missing_terrain_main", True)
            # 如果没有植被数据，回退到随机游荡
            veg_map = None
            size = 0

        # 3. 计算力 (Forces)
        force_x = np.zeros(count, dtype=np.float32)
        force_z = np.zeros(count, dtype=np.float32)
        
        # --- Force 1: 觅食引力 (Food Attraction) ---
        if veg_map is not None:
            # 将位置映射到网格坐标
            # 假设 pos 是 0..size 范围 (Grid Coordinates)
            grid_x = np.clip(pos_x, 0, size - 2).astype(np.int32)
            grid_z = np.clip(pos_z, 0, size - 2).astype(np.int32)
            
            # 计算梯度 (Gradient) - 简单的差分
            # grad_x = density(x+1) - density(x-1)
            # 为了简单，我们取右边减当前，下边减当前
            
            d00 = veg_map[grid_z, grid_x]
            d10 = veg_map[grid_z, grid_x + 1]
            d01 = veg_map[grid_z + 1, grid_x]
            
            grad_x = (d10 - d00) * 10.0 # 放大梯度
            grad_z = (d01 - d00) * 10.0
            
            # 饥饿驱动
            attraction = hunger * 5.0
            force_x += grad_x * attraction
            force_z += grad_z * attraction
            
            # --- Force 2: 地形排斥 (Slope Repulsion) ---
            s00 = slope_data[grid_z, grid_x]
            
            # 如果坡度太陡 (>0.6)，产生斥力
            # 斥力方向：沿着坡度下降方向 (即 slope gradient 的反方向)
            # 这里简化：假设我们知道 height map gradient，或者简单地避开高坡度区域
            # 暂时简化为：如果在陡坡上，随机乱跑以逃离
            
            slope_mask = s00 > 0.6
            force_x[slope_mask] += np.random.uniform(-5, 5, np.sum(slope_mask))
            force_z[slope_mask] += np.random.uniform(-5, 5, np.sum(slope_mask))
            
            # --- Eating Logic ---
            # 如果在食物丰富的地方且饥饿，进食
            eating_mask = (d00 > 0.1) & (hunger > 0.2)
            
            # Debug stats
            if debug and np.random.random() < 0.05:
                print(f"GrazingRule: Max veg {np.max(d00)}, Max hunger {np.max(hunger)}, Eating count {np.sum(eating_mask)}")
            
            # 记录进食状态，供 GPU 使用
            # 0.0 = not eating, 1.0 = eating
            is_eating = np.zeros(count, dtype=np.float32)
            is_eating[eating_mask] = 1.0
            
            try:
                facts.set_column("herbivore", "is_eating", is_eating)
            except Exception:
                pass

            if np.any(eating_mask):
                # 1. 减少饥饿感
                hunger[eating_mask] -= 0.5 * dt # 吃得快一点
                
                # 2. 消耗 CPU 端植被数据 (Dual Simulation for Logic)
                # 找到正在进食的实体的网格坐标
                e_x = grid_x[eating_mask]
                e_z = grid_z[eating_mask]
                
                # 减少密度 (假设消耗速率 1.0/s)
                consumption = 1.0 * dt
                # 使用 at 处理多个实体吃同一个格子
                np.add.at(veg_map, (e_z, e_x), -consumption)
                
                # 确保不小于0
                np.clip(veg_map, 0, 1.0, out=veg_map)

            # 饥饿自然增长
            hunger[~eating_mask] += 0.05 * dt # 饿得快一点
            np.clip(hunger, 0, 1, out=hunger)
            
            facts.set_column("herbivore", "hunger", hunger)
            
        else:
            try:
                terrain_slope = facts.get_column("herbivore", "terrain_slope")
                terrain_grad_x = facts.get_column("herbivore", "terrain_grad_x")
                terrain_grad_z = facts.get_column("herbivore", "terrain_grad_z")
            except KeyError:
                terrain_slope = np.zeros(count, dtype=np.float32)
                terrain_grad_x = np.zeros(count, dtype=np.float32)
                terrain_grad_z = np.zeros(count, dtype=np.float32)

            food = facts.get_global("food_source")
            if food is None:
                food_x, food_z = 50.0, 50.0
            else:
                try:
                    food_x, food_z = float(food[0]), float(food[1])
                except Exception:
                    food_x, food_z = 50.0, 50.0

            to_food_x = food_x - pos_x
            to_food_z = food_z - pos_z
            dist = np.sqrt(to_food_x * to_food_x + to_food_z * to_food_z) + 1e-6
            dir_x = to_food_x / dist
            dir_z = to_food_z / dist
            attraction = hunger * 5.0
            force_x += dir_x * attraction
            force_z += dir_z * attraction

            slope_mask = terrain_slope > 0.6
            force_x[slope_mask] += -terrain_grad_x[slope_mask] * 10.0
            force_z[slope_mask] += -terrain_grad_z[slope_mask] * 10.0

            eating_mask = (dist < 2.0) & (hunger > 0.2)
            is_eating = np.zeros(count, dtype=np.float32)
            is_eating[eating_mask] = 1.0
            facts.set_column("herbivore", "is_eating", is_eating)

            if np.any(eating_mask):
                hunger[eating_mask] -= 0.5 * dt
            hunger[~eating_mask] += 0.05 * dt
            np.clip(hunger, 0, 1, out=hunger)
            facts.set_column("herbivore", "hunger", hunger)

        # --- Force 3: 随机游荡 (Wander) ---
        # 即使没有食物，也要动一动 (Perlin noise would be better, but random is okay)
        # 增加一个持续的随机力，而不是高频噪音
        wander_strength = 2.0
        wander_x = np.random.uniform(-1, 1, count) * wander_strength
        wander_z = np.random.uniform(-1, 1, count) * wander_strength
        
        force_x += wander_x
        force_z += wander_z
        
        # --- Force 4: 阻力 (Damping) ---
        # 模拟摩擦力，防止无限加速
        force_x -= vel_x * 0.5
        force_z -= vel_z * 0.5
        
        # 4. 积分 (Integration)
        vel_x += force_x * dt
        vel_z += force_z * dt
        
        # 速度限制 (Speed Limit)
        speed = np.sqrt(vel_x**2 + vel_z**2)
        max_speed = 5.0
        limit_mask = speed > max_speed
        if np.any(limit_mask):
            scale = max_speed / speed[limit_mask]
            vel_x[limit_mask] *= scale
            vel_z[limit_mask] *= scale
        
        pos_x += vel_x * dt
        pos_z += vel_z * dt
        
        # 边界约束 (Boundary Constraint)
        if veg_map is not None:
            np.clip(pos_x, 0, size - 1, out=pos_x)
            np.clip(pos_z, 0, size - 1, out=pos_z)
            
        # 5. 动画状态计算 (Animation State)
        # 计算朝向 (Heading) - 简单的 2D 旋转
        # 0 度 = Z+ (Up), 90 度 = X+ (Right)
        heading = np.arctan2(vel_x, vel_z)
        
        # 6. 写回数据
        facts.set_column("herbivore", "pos_x", pos_x)
        facts.set_column("herbivore", "pos_z", pos_z)
        facts.set_column("herbivore", "vel_x", vel_x)
        facts.set_column("herbivore", "vel_z", vel_z)
        facts.set_column("herbivore", "heading", heading)
