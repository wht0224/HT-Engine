# -*- coding: utf-8 -*-
"""
流体物理系统实现
基于粒子的简化流体模拟，适合低端GPU
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode
import random

class Fluid:
    """流体类
    表示物理世界中的一个流体对象"""
    
    def __init__(self, scene_node, particle_count=100, fluid_type="water"):
        """
        初始化流体
        
        Args:
            scene_node: 关联的场景节点
            particle_count: 粒子数量
            fluid_type: 流体类型 ("water", "oil", "lava", "smoke")
        """
        self.scene_node = scene_node
        self.particle_count = particle_count
        self.fluid_type = fluid_type
        
        # 物理引擎ID
        self.physics_id = None
        
        # 流体属性
        self.enabled = True
        self.use_gravity = True
        self.density = 1000.0  # 水的密度
        self.viscosity = 0.001  # 流体粘度
        self.surface_tension = 0.0728  # 表面张力
        self.damping = 0.99
        
        # 粒子属性
        self.particle_radius = 0.1
        self.particle_mass = 0.1
        
        # 碰撞属性
        self.collision_enabled = True
        self.collision_group = 1
        self.collision_mask = -1
        
        # 粒子数据
        self.particles = []
        self.velocities = []
        self.accelerations = []
        self.forces = []
        
        # 邻居列表
        self.neighbors = []
        
        # 初始化粒子
        self._initialize_particles()
        
        # 场景节点关联
        if scene_node:
            scene_node.fluid = self
    
    def _initialize_particles(self):
        """初始化流体粒子"""
        # 根据流体类型设置属性
        if self.fluid_type == "water":
            self.density = 1000.0
            self.viscosity = 0.001
            self.surface_tension = 0.0728
        elif self.fluid_type == "oil":
            self.density = 920.0
            self.viscosity = 0.05
            self.surface_tension = 0.028
        elif self.fluid_type == "lava":
            self.density = 3000.0
            self.viscosity = 100.0
            self.surface_tension = 0.08
        elif self.fluid_type == "smoke":
            self.density = 1.0
            self.viscosity = 0.0001
            self.surface_tension = 0.0
            self.use_gravity = False
        
        # 生成粒子
        for i in range(self.particle_count):
            # 随机初始位置
            x = random.uniform(-1.0, 1.0)
            y = random.uniform(2.0, 4.0)  # 在高处生成
            z = random.uniform(-1.0, 1.0)
            
            self.particles.append(Vector3(x, y, z))
            self.velocities.append(Vector3(0, 0, 0))
            self.accelerations.append(Vector3(0, 0, 0))
            self.forces.append(Vector3(0, 0, 0))
            self.neighbors.append([])
    
    def _calculate_neighbors(self):
        """计算粒子邻居，使用空间划分算法加速"""
        # 重置邻居列表
        for i in range(self.particle_count):
            self.neighbors[i].clear()
        
        # 计算邻居搜索半径
        search_radius = self.particle_radius * 2.5
        cell_size = search_radius
        
        # 空间划分：将空间划分为网格，每个网格存储粒子索引
        grid = {}
        
        # 将粒子分配到网格中
        for i in range(self.particle_count):
            pos = self.particles[i]
            # 计算网格坐标
            grid_x = int(pos.x / cell_size)
            grid_y = int(pos.y / cell_size)
            grid_z = int(pos.z / cell_size)
            grid_key = (grid_x, grid_y, grid_z)
            
            # 添加到网格
            if grid_key not in grid:
                grid[grid_key] = []
            grid[grid_key].append(i)
        
        # 搜索每个粒子的邻居
        for i in range(self.particle_count):
            pos = self.particles[i]
            # 计算当前粒子所在网格
            grid_x = int(pos.x / cell_size)
            grid_y = int(pos.y / cell_size)
            grid_z = int(pos.z / cell_size)
            
            # 搜索当前网格及其周围27个网格
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    for dz in [-1, 0, 1]:
                        # 计算相邻网格坐标
                        neighbor_grid_key = (grid_x + dx, grid_y + dy, grid_z + dz)
                        
                        # 如果网格存在，检查其中的粒子
                        if neighbor_grid_key in grid:
                            for j in grid[neighbor_grid_key]:
                                if i != j:  # 跳过自身
                                    distance = (pos - self.particles[j]).length()
                                    if distance < search_radius:
                                        self.neighbors[i].append(j)
    
    def _kernel_poly6(self, r, h):
        """Poly6核函数，用于计算密度
        
        Args:
            r: 粒子间距离
            h: 核函数半径
        
        Returns:
            float: 核函数值
        """
        if r >= h:
            return 0.0
        
        coeff = 315.0 / (64.0 * 3.14159 * h**9)
        return coeff * (h**2 - r**2)**3
    
    def _kernel_spiky_gradient(self, r, h, direction):
        """Spiky核函数梯度，用于计算压力力
        
        Args:
            r: 粒子间距离
            h: 核函数半径
            direction: 粒子间方向向量
        
        Returns:
            Vector3: 核函数梯度
        """
        if r <= 0.0 or r >= h:
            return Vector3(0, 0, 0)
        
        coeff = -45.0 / (3.14159 * h**6)
        return direction * (coeff * (h - r)**2)
    
    def _kernel_viscosity_laplacian(self, r, h):
        """Viscosity核函数拉普拉斯，用于计算粘性力
        
        Args:
            r: 粒子间距离
            h: 核函数半径
        
        Returns:
            float: 核函数拉普拉斯值
        """
        if r >= h:
            return 0.0
        
        coeff = 45.0 / (3.14159 * h**6)
        return coeff * (h - r)
    
    def _calculate_density_pressure(self):
        """计算粒子密度和压力"""
        # 初始化密度和压力
        self.densities = [0.0] * self.particle_count
        self.pressures = [0.0] * self.particle_count
        
        # 核函数半径
        h = self.particle_radius * 2.5
        
        # 计算每个粒子的密度
        for i in range(self.particle_count):
            # 自身贡献
            self.densities[i] += self.particle_mass * self._kernel_poly6(0.0, h)
            
            # 邻居贡献
            for j in self.neighbors[i]:
                r = (self.particles[i] - self.particles[j]).length()
                self.densities[i] += self.particle_mass * self._kernel_poly6(r, h)
        
        # 计算每个粒子的压力（理想气体状态方程）
        rest_density = 1000.0  # 静止密度
        stiffness = 300.0  # 压力刚度
        
        for i in range(self.particle_count):
            # 压力 = 刚度 * (密度 - 静止密度)
            self.pressures[i] = stiffness * max(self.densities[i] - rest_density, 0.0)
    
    def _calculate_forces(self, gravity):
        """计算粒子受力"""
        # 重置所有力
        for i in range(self.particle_count):
            self.forces[i] = Vector3(0, 0, 0)
        
        # 应用重力
        if self.use_gravity:
            for i in range(self.particle_count):
                self.forces[i] += gravity * self.particle_mass
        
        # 计算密度和压力
        self._calculate_density_pressure()
        
        # 核函数半径
        h = self.particle_radius * 2.5
        
        # 计算压力力和粘性力（SPH模型）
        for i in range(self.particle_count):
            pressure_force = Vector3(0, 0, 0)
            viscosity_force = Vector3(0, 0, 0)
            
            for j in self.neighbors[i]:
                # 粒子间向量和距离
                r_vec = self.particles[i] - self.particles[j]
                r = r_vec.length()
                if r <= 0.0:
                    continue
                
                # 单位方向向量
                direction = r_vec / r
                
                # 计算压力力（SPH）
                pressure = (self.pressures[i] + self.pressures[j]) / (2.0 * self.densities[j])
                pressure_force += self._kernel_spiky_gradient(r, h, direction) * pressure * self.particle_mass
                
                # 计算粘性力（SPH）
                velocity_diff = self.velocities[j] - self.velocities[i]
                viscosity_force += velocity_diff * (self._kernel_viscosity_laplacian(r, h) * self.particle_mass / self.densities[j])
            
            # 应用压力力
            self.forces[i] += pressure_force
            
            # 应用粘性力
            self.forces[i] += viscosity_force * self.viscosity * 100.0
        
        # 应用表面张力（简化实现）
        if self.surface_tension > 0.0:
            self._calculate_surface_tension()
    
    def _calculate_surface_tension(self):
        """计算表面张力"""
        # 核函数半径
        h = self.particle_radius * 2.5
        
        # 计算每个粒子的表面法向量
        surface_normals = [Vector3(0, 0, 0)] * self.particle_count
        
        for i in range(self.particle_count):
            for j in self.neighbors[i]:
                # 粒子间向量和距离
                r_vec = self.particles[i] - self.particles[j]
                r = r_vec.length()
                if r <= 0.0:
                    continue
                
                # 单位方向向量
                direction = r_vec / r
                
                # 计算表面法线贡献
                surface_normals[i] += direction * self._kernel_spiky_gradient(r, h, direction) * self.particle_mass / self.densities[j]
        
        # 应用表面张力
        tension_coeff = self.surface_tension * 0.01
        
        for i in range(self.particle_count):
            # 表面法线长度表示粒子在表面的程度
            normal_length = surface_normals[i].length()
            if normal_length > 0.001:
                # 归一化表面法线
                normal = surface_normals[i] / normal_length
                
                # 应用表面张力
                self.forces[i] -= normal * tension_coeff * normal_length
    
    def _integrate(self, delta_time):
        """积分更新粒子位置和速度"""
        # 时间步长限制，避免数值不稳定
        dt = min(delta_time, 0.016)  # 最大16ms步长
        
        for i in range(self.particle_count):
            # 计算加速度
            self.accelerations[i] = self.forces[i] / self.particle_mass
            
            # 更新速度（Verlet积分，更稳定）
            self.velocities[i] += self.accelerations[i] * dt
            self.velocities[i] *= self.damping
            
            # 更新位置
            self.particles[i] += self.velocities[i] * dt
            
            # 边界碰撞检测
            self._handle_boundary_collisions(i)
    
    def _handle_boundary_collisions(self, particle_index):
        """处理粒子边界碰撞"""
        particle = self.particles[particle_index]
        velocity = self.velocities[particle_index]
        
        # 地面碰撞
        if particle.y < self.particle_radius:
            particle.y = self.particle_radius
            velocity.y *= -0.5  # 反弹
            
            # 摩擦
            velocity.x *= 0.9
            velocity.z *= 0.9
        
        # 简单的立方体边界碰撞（-5.0到5.0）
        boundary = 5.0
        
        # 左边界
        if particle.x < -boundary + self.particle_radius:
            particle.x = -boundary + self.particle_radius
            velocity.x *= -0.5
            velocity.y *= 0.9
            velocity.z *= 0.9
        
        # 右边界
        if particle.x > boundary - self.particle_radius:
            particle.x = boundary - self.particle_radius
            velocity.x *= -0.5
            velocity.y *= 0.9
            velocity.z *= 0.9
        
        # 后边界
        if particle.z < -boundary + self.particle_radius:
            particle.z = -boundary + self.particle_radius
            velocity.z *= -0.5
            velocity.x *= 0.9
            velocity.y *= 0.9
        
        # 前边界
        if particle.z > boundary - self.particle_radius:
            particle.z = boundary - self.particle_radius
            velocity.z *= -0.5
            velocity.x *= 0.9
            velocity.y *= 0.9
        
        # 更新粒子和速度
        self.particles[particle_index] = particle
        self.velocities[particle_index] = velocity
    
    def update(self, delta_time, gravity):
        """更新流体模拟
        
        Args:
            delta_time: 帧时间
            gravity: 重力向量
        """
        if not self.enabled:
            return
        
        # 性能优化：根据硬件能力调整模拟参数
        self._adjust_simulation_parameters()
        
        # 计算邻居
        self._calculate_neighbors()
        
        # 计算力
        self._calculate_forces(gravity)
        
        # 积分更新
        self._integrate(delta_time)
    
    def _adjust_simulation_parameters(self):
        """根据硬件能力调整模拟参数"""
        # 性能模式：自动、低、中、高
        self.performance_mode = getattr(self, 'performance_mode', 'auto')
        
        # 默认参数
        self.density_calculation_enabled = True
        self.surface_tension_enabled = True
        self.max_particles = self.particle_count
        
        # 根据性能模式调整参数
        if self.performance_mode == 'low':
            self.density_calculation_enabled = True
            self.surface_tension_enabled = False
            # 限制最大粒子数量
            self.max_particles = min(self.particle_count, 200)
        elif self.performance_mode == 'medium':
            self.density_calculation_enabled = True
            self.surface_tension_enabled = True
            self.max_particles = min(self.particle_count, 500)
        elif self.performance_mode == 'high':
            self.density_calculation_enabled = True
            self.surface_tension_enabled = True
            self.max_particles = self.particle_count
        else:  # auto
            # 根据粒子数量自动调整
            if self.particle_count > 1000:
                self.surface_tension_enabled = False
            elif self.particle_count > 500:
                self.surface_tension_enabled = True
        
        # 确保max_particles不超过实际粒子数量
        self.max_particles = min(self.max_particles, self.particle_count)
    
    def _calculate_forces(self, gravity):
        """计算粒子受力"""
        # 重置所有力
        for i in range(len(self.forces)):
            self.forces[i] = Vector3(0, 0, 0)
        
        # 应用重力
        if self.use_gravity:
            for i in range(len(self.forces)):
                self.forces[i] += gravity * self.particle_mass
        
        # 计算密度和压力（根据性能模式决定是否启用）
        if self.density_calculation_enabled:
            self._calculate_density_pressure()
        
        # 核函数半径
        h = self.particle_radius * 2.5
        
        # 计算压力力和粘性力（SPH模型）
        for i in range(self.max_particles):  # 只处理前max_particles个粒子
            pressure_force = Vector3(0, 0, 0)
            viscosity_force = Vector3(0, 0, 0)
            
            for j in self.neighbors[i]:
                # 粒子间向量和距离
                r_vec = self.particles[i] - self.particles[j]
                r = r_vec.length()
                if r <= 0.0:
                    continue
                
                # 单位方向向量
                direction = r_vec / r
                
                # 计算压力力（SPH）
                pressure = (self.pressures[i] + self.pressures[j]) / (2.0 * self.densities[j])
                pressure_force += self._kernel_spiky_gradient(r, h, direction) * pressure * self.particle_mass
                
                # 计算粘性力（SPH）
                velocity_diff = self.velocities[j] - self.velocities[i]
                viscosity_force += velocity_diff * (self._kernel_viscosity_laplacian(r, h) * self.particle_mass / self.densities[j])
            
            # 应用压力力
            self.forces[i] += pressure_force
            
            # 应用粘性力
            self.forces[i] += viscosity_force * self.viscosity * 100.0
        
        # 应用表面张力（根据性能模式决定是否启用）
        if self.surface_tension_enabled and self.surface_tension > 0.0:
            self._calculate_surface_tension()
    
    def set_performance_mode(self, mode):
        """设置流体模拟性能模式
        
        Args:
            mode: 性能模式 ('auto', 'low', 'medium', 'high')
        """
        self.performance_mode = mode
    
    def set_particle_count(self, count):
        """设置粒子数量
        
        Args:
            count: 新的粒子数量
        """
        self.particle_count = count
        self._initialize_particles()
    
    def set_fluid_type(self, fluid_type):
        """设置流体类型
        
        Args:
            fluid_type: 流体类型
        """
        self.fluid_type = fluid_type
        # 重新初始化粒子以应用新属性
        self._initialize_particles()
    
    def enable_collision(self, enable):
        """启用或禁用碰撞
        
        Args:
            enable: 是否启用碰撞
        """
        self.collision_enabled = enable