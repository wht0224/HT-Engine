# -*- coding: utf-8 -*-
"""
布料物理系统实现
基于Bullet物理引擎的布料模拟
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
from Engine.Scene.SceneNode import SceneNode

class Cloth:
    """布料类
    表示物理世界中的一个布料对象"""
    
    def __init__(self, scene_node, width=1.0, height=1.0, segments_x=10, segments_y=10, mass=0.1):
        """
        初始化布料
        
        Args:
            scene_node: 关联的场景节点
            width: 布料宽度
            height: 布料高度
            segments_x: X方向的段数
            segments_y: Y方向的段数
            mass: 布料总质量
        """
        self.scene_node = scene_node
        self.width = width
        self.height = height
        self.segments_x = segments_x
        self.segments_y = segments_y
        self.mass = mass
        
        # 物理引擎ID
        self.physics_id = None
        
        # 布料属性
        self.enabled = True
        self.use_gravity = True
        self.damping = 0.01
        self.stiffness = 0.9
        self.friction = 0.5
        self.restitution = 0.0
        
        # 碰撞属性
        self.collision_enabled = True
        self.collision_group = 1
        self.collision_mask = -1
        
        # 顶点数据
        self.vertices = []
        self.normals = []
        self.indices = []
        
        # 物理模拟数据
        self.vertex_mass = mass / ((segments_x + 1) * (segments_y + 1))  # 每个顶点的质量
        self.velocities = []  # 顶点速度
        self.accelerations = []  # 顶点加速度
        self.forces = []  # 顶点受力
        self.original_positions = []  # 原始顶点位置（用于约束）
        
        # 布料约束
        self.pin_constraints = []
        self.structural_constraints = []  # 结构约束（相邻顶点）
        self.shear_constraints = []  # 剪切约束（对角线顶点）
        self.bending_constraints = []  # 弯曲约束（隔行隔列顶点）
        
        # 约束参数
        self.structural_stiffness = 1.0  # 结构约束刚度
        self.shear_stiffness = 0.8  # 剪切约束刚度
        self.bending_stiffness = 0.5  # 弯曲约束刚度
        
        # 初始化布料网格
        self._initialize_cloth_mesh()
        
        # 初始化物理模拟数据
        self._initialize_physics_data()
        
        # 初始化约束
        self._initialize_constraints()
        
        # 场景节点关联
        if scene_node:
            scene_node.cloth = self
    
    def _initialize_cloth_mesh(self):
        """初始化布料网格"""
        # 生成布料顶点
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                # 计算顶点位置
                px = (x / self.segments_x - 0.5) * self.width
                py = 0.0
                pz = (y / self.segments_y - 0.5) * self.height
                self.vertices.append(Vector3(px, py, pz))
                self.normals.append(Vector3(0, 1, 0))
        
        # 生成布料三角形索引
        for y in range(self.segments_y):
            for x in range(self.segments_x):
                # 计算当前顶点索引
                current = y * (self.segments_x + 1) + x
                
                # 添加第一个三角形
                self.indices.append(current)
                self.indices.append(current + self.segments_x + 1)
                self.indices.append(current + 1)
                
                # 添加第二个三角形
                self.indices.append(current + 1)
                self.indices.append(current + self.segments_x + 1)
                self.indices.append(current + self.segments_x + 2)
    
    def add_pin_constraint(self, vertex_index, position=None):
        """
        添加布料固定约束
        
        Args:
            vertex_index: 要固定的顶点索引
            position: 固定位置，如果为None则使用当前顶点位置
        """
        if position is None:
            position = self.vertices[vertex_index]
        
        self.pin_constraints.append({
            "vertex_index": vertex_index,
            "position": position
        })
    
    def set_stiffness(self, stiffness):
        """
        设置布料刚度
        
        Args:
            stiffness: 刚度值，范围[0, 1]
        """
        self.stiffness = max(0.0, min(1.0, stiffness))
    
    def set_damping(self, damping):
        """
        设置布料阻尼
        
        Args:
            damping: 阻尼值，范围[0, 1]
        """
        self.damping = max(0.0, min(1.0, damping))
    
    def set_friction(self, friction):
        """
        设置布料摩擦系数
        
        Args:
            friction: 摩擦系数，范围[0, 1]
        """
        self.friction = max(0.0, min(1.0, friction))
    
    def enable_collision(self, enable):
        """
        启用或禁用碰撞
        
        Args:
            enable: 是否启用碰撞
        """
        self.collision_enabled = enable
    
    def _initialize_physics_data(self):
        """初始化物理模拟数据"""
        # 为每个顶点初始化物理数据
        for i in range(len(self.vertices)):
            self.velocities.append(Vector3(0, 0, 0))
            self.accelerations.append(Vector3(0, 0, 0))
            self.forces.append(Vector3(0, 0, 0))
            self.original_positions.append(self.vertices[i].copy())
    
    def _initialize_constraints(self):
        """初始化布料约束"""
        # 遍历所有顶点，创建约束
        for y in range(self.segments_y + 1):
            for x in range(self.segments_x + 1):
                index = y * (self.segments_x + 1) + x
                
                # 创建结构约束（右、下）
                if x < self.segments_x:
                    # 右邻接点
                    right_index = index + 1
                    rest_length = (self.vertices[index] - self.vertices[right_index]).length()
                    self.structural_constraints.append({
                        "vertex1": index,
                        "vertex2": right_index,
                        "rest_length": rest_length,
                        "stiffness": self.structural_stiffness
                    })
                
                if y < self.segments_y:
                    # 下邻接点
                    down_index = index + (self.segments_x + 1)
                    rest_length = (self.vertices[index] - self.vertices[down_index]).length()
                    self.structural_constraints.append({
                        "vertex1": index,
                        "vertex2": down_index,
                        "rest_length": rest_length,
                        "stiffness": self.structural_stiffness
                    })
                
                # 创建剪切约束（右下、左下）
                if x < self.segments_x and y < self.segments_y:
                    # 右下对角线
                    right_down_index = index + (self.segments_x + 1) + 1
                    rest_length = (self.vertices[index] - self.vertices[right_down_index]).length()
                    self.shear_constraints.append({
                        "vertex1": index,
                        "vertex2": right_down_index,
                        "rest_length": rest_length,
                        "stiffness": self.shear_stiffness
                    })
                
                if x > 0 and y < self.segments_y:
                    # 左下对角线
                    left_down_index = index + (self.segments_x + 1) - 1
                    rest_length = (self.vertices[index] - self.vertices[left_down_index]).length()
                    self.shear_constraints.append({
                        "vertex1": index,
                        "vertex2": left_down_index,
                        "rest_length": rest_length,
                        "stiffness": self.shear_stiffness
                    })
                
                # 创建弯曲约束（右右、下下）
                if x < self.segments_x - 1:
                    # 右右邻接点
                    right_right_index = index + 2
                    rest_length = (self.vertices[index] - self.vertices[right_right_index]).length()
                    self.bending_constraints.append({
                        "vertex1": index,
                        "vertex2": right_right_index,
                        "rest_length": rest_length,
                        "stiffness": self.bending_stiffness
                    })
                
                if y < self.segments_y - 1:
                    # 下下邻接点
                    down_down_index = index + 2 * (self.segments_x + 1)
                    rest_length = (self.vertices[index] - self.vertices[down_down_index]).length()
                    self.bending_constraints.append({
                        "vertex1": index,
                        "vertex2": down_down_index,
                        "rest_length": rest_length,
                        "stiffness": self.bending_stiffness
                    })
    
    def update(self, delta_time, gravity):
        """更新布料物理模拟
        
        Args:
            delta_time: 帧时间
            gravity: 重力向量
        """
        if not self.enabled:
            return
        
        # 性能优化：根据硬件能力调整模拟参数
        self._adjust_simulation_parameters()
        
        # 重置所有力
        for i in range(len(self.forces)):
            self.forces[i] = Vector3(0, 0, 0)
        
        # 应用重力
        if self.use_gravity:
            for i in range(len(self.forces)):
                self.forces[i] += gravity * self.vertex_mass
        
        # 应用阻尼
        for i in range(len(self.velocities)):
            self.forces[i] -= self.velocities[i] * self.damping
        
        # 计算加速度
        for i in range(len(self.accelerations)):
            self.accelerations[i] = self.forces[i] / self.vertex_mass
        
        # 更新速度和位置（半隐式欧拉积分，更稳定）
        for i in range(len(self.vertices)):
            # 更新速度
            self.velocities[i] += self.accelerations[i] * delta_time
            
            # 更新位置
            self.vertices[i] += self.velocities[i] * delta_time
        
        # 应用约束（根据性能设置调整迭代次数）
        for _ in range(self.constraint_iterations):
            # 应用结构约束
            self._apply_constraints(self.structural_constraints)
            
            # 应用剪切约束（仅在高性能模式下启用）
            if self.enable_shear_constraints:
                self._apply_constraints(self.shear_constraints)
            
            # 应用弯曲约束（仅在高性能模式下启用）
            if self.enable_bending_constraints:
                self._apply_constraints(self.bending_constraints)
            
            # 应用固定约束
            self._apply_pin_constraints()
        
        # 更新法线（仅在需要时更新）
        if self._should_update_normals():
            self._update_normals()
        
        # 更新场景节点的网格数据
        self.update_mesh()
    
    def _adjust_simulation_parameters(self):
        """根据硬件能力调整模拟参数"""
        # 性能模式：自动、低、中、高
        self.performance_mode = getattr(self, 'performance_mode', 'auto')
        
        # 默认参数
        self.constraint_iterations = 5
        self.enable_shear_constraints = True
        self.enable_bending_constraints = True
        self.normal_update_frequency = 1  # 每帧更新一次法线
        
        # 根据性能模式调整参数
        if self.performance_mode == 'low':
            self.constraint_iterations = 2
            self.enable_shear_constraints = False
            self.enable_bending_constraints = False
            self.normal_update_frequency = 3  # 每3帧更新一次法线
        elif self.performance_mode == 'medium':
            self.constraint_iterations = 3
            self.enable_shear_constraints = True
            self.enable_bending_constraints = False
            self.normal_update_frequency = 2  # 每2帧更新一次法线
        elif self.performance_mode == 'high':
            self.constraint_iterations = 5
            self.enable_shear_constraints = True
            self.enable_bending_constraints = True
            self.normal_update_frequency = 1  # 每帧更新一次法线
        else:  # auto
            # 根据顶点数量自动调整
            vertex_count = len(self.vertices)
            if vertex_count > 500:
                self.constraint_iterations = 3
                self.enable_shear_constraints = True
                self.enable_bending_constraints = False
            elif vertex_count > 1000:
                self.constraint_iterations = 2
                self.enable_shear_constraints = False
                self.enable_bending_constraints = False
    
    def _should_update_normals(self):
        """检查是否需要更新法线"""
        # 简单实现：根据频率更新
        if not hasattr(self, 'normal_update_counter'):
            self.normal_update_counter = 0
        
        self.normal_update_counter += 1
        if self.normal_update_counter >= self.normal_update_frequency:
            self.normal_update_counter = 0
            return True
        return False
    
    def set_performance_mode(self, mode):
        """设置布料模拟性能模式
        
        Args:
            mode: 性能模式 ('auto', 'low', 'medium', 'high')
        """
        self.performance_mode = mode
    
    def _apply_constraints(self, constraints):
        """应用约束
        
        Args:
            constraints: 约束列表
        """
        for constraint in constraints:
            v1 = constraint["vertex1"]
            v2 = constraint["vertex2"]
            rest_length = constraint["rest_length"]
            stiffness = constraint["stiffness"]
            
            # 计算当前距离
            delta = self.vertices[v2] - self.vertices[v1]
            current_length = delta.length()
            
            if current_length == 0:
                continue
            
            # 计算约束力
            diff = (current_length - rest_length) / current_length
            correction = delta * diff * 0.5 * stiffness
            
            # 应用约束（质量加权）
            if self.vertex_mass > 0:
                self.vertices[v1] += correction
                self.vertices[v2] -= correction
                
                # 同时更新速度
                self.velocities[v1] += correction / self.vertex_mass
                self.velocities[v2] -= correction / self.vertex_mass
    
    def _apply_pin_constraints(self):
        """应用固定约束"""
        for constraint in self.pin_constraints:
            vertex_index = constraint["vertex_index"]
            position = constraint["position"]
            
            # 将顶点固定到指定位置
            self.vertices[vertex_index] = position
            self.velocities[vertex_index] = Vector3(0, 0, 0)
    
    def _update_normals(self):
        """更新布料法线"""
        # 重置所有法线
        for i in range(len(self.normals)):
            self.normals[i] = Vector3(0, 0, 0)
        
        # 遍历所有三角形，计算法线
        for i in range(0, len(self.indices), 3):
            # 获取三角形顶点索引
            v0 = self.indices[i]
            v1 = self.indices[i + 1]
            v2 = self.indices[i + 2]
            
            # 获取顶点位置
            p0 = self.vertices[v0]
            p1 = self.vertices[v1]
            p2 = self.vertices[v2]
            
            # 计算边向量
            edge1 = p1 - p0
            edge2 = p2 - p0
            
            # 计算法线（叉乘）
            normal = edge1.cross(edge2)
            normal.normalize()
            
            # 将法线添加到每个顶点
            self.normals[v0] += normal
            self.normals[v1] += normal
            self.normals[v2] += normal
        
        # 归一化所有顶点法线
        for i in range(len(self.normals)):
            self.normals[i].normalize()
    
    def update_mesh(self):
        """
        更新场景节点的网格数据
        """
        # 更新场景节点的顶点和法线数据
        if self.scene_node and hasattr(self.scene_node, 'mesh') and self.scene_node.mesh:
            self.scene_node.mesh.vertices = self.vertices
            self.scene_node.mesh.normals = self.normals
            self.scene_node.mesh.indices = self.indices
            
            # 通知渲染系统更新网格
            if hasattr(self.scene_node.mesh, 'update'):
                self.scene_node.mesh.update()