# -*- coding: utf-8 -*-
"""
第一人称相机控制器 (FPS Camera Controller)
实现鼠标控制视角、WASD键盘移动、平滑移动插值
"""

import math
from Engine.Math import Vector3, Quaternion, Matrix4x4


class FPSCamera:
    """
    第一人称相机控制器
    
    功能：
    - 鼠标控制视角（水平360度，垂直-90到+90度）
    - WASD键盘移动（W前进，S后退，A左移，D右移）
    - 平滑的相机移动（带惯性插值）
    - 空格跳跃（可选）
    """
    
    def __init__(self, position=None, yaw=0.0, pitch=0.0):
        """
        初始化FPS相机
        
        Args:
            position: 初始位置 (Vector3 或 tuple/list)
            yaw: 初始水平角度（弧度，0表示正Z方向）
            pitch: 初始垂直角度（弧度，0表示水平）
        """
        # 位置
        if position is None:
            self.position = Vector3(0.0, 1.8, 0.0)  # 默认眼睛高度
        elif isinstance(position, (tuple, list)):
            self.position = Vector3(position[0], position[1], position[2])
        else:
            self.position = position.copy()
        
        # 视角角度（弧度）
        self.yaw = yaw      # 水平角度 (0-2π)
        self.pitch = pitch  # 垂直角度 (-π/2 to π/2)
        
        # 限制俯仰角范围
        self._clamp_pitch()
        self._normalize_yaw()
        
        # 移动参数
        self.move_speed = 5.0           # 基础移动速度 (单位/秒)
        self.run_speed = 10.0           # 奔跑速度
        self.acceleration = 10.0        # 加速度 (用于平滑移动)
        self.friction = 10.0            # 摩擦力 (减速系数)
        
        # 鼠标灵敏度
        self.mouse_sensitivity = 0.002  # 鼠标灵敏度
        self.invert_mouse_y = False     # 是否反转Y轴
        
        # 速度向量（用于平滑插值）
        self.velocity = Vector3(0.0, 0.0, 0.0)
        
        # 跳跃参数
        self.enable_jump = True         # 是否启用跳跃
        self.jump_force = 8.0           # 跳跃力度
        self.gravity = -20.0            # 重力加速度
        self.is_grounded = True         # 是否在地面上
        self.ground_height = 0.0        # 地面高度
        self.eye_height = 1.8           # 眼睛高度
        
        # 内部相机对象（用于渲染）
        self._camera = None
        
        # 视图矩阵缓存
        self._view_matrix = Matrix4x4.identity()
        self._view_dirty = True
        
        # 方向向量缓存
        self._forward = Vector3(0.0, 0.0, -1.0)
        self._right = Vector3(1.0, 0.0, 0.0)
        self._up = Vector3(0.0, 1.0, 0.0)
        
        # 更新方向向量
        self._update_direction_vectors()
    
    def _clamp_pitch(self):
        """限制俯仰角在有效范围内 (-89度到+89度)"""
        max_pitch = math.radians(89.0)
        self.pitch = max(-max_pitch, min(max_pitch, self.pitch))
    
    def _normalize_yaw(self):
        """归一化水平角度到 [0, 2π] 范围"""
        two_pi = 2.0 * math.pi
        self.yaw = self.yaw % two_pi
        if self.yaw < 0:
            self.yaw += two_pi
    
    def _update_direction_vectors(self):
        """根据 yaw 和 pitch 更新方向向量"""
        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)
        
        # 前方向（相机看向的方向）
        self._forward = Vector3(
            sin_yaw * cos_pitch,
            sin_pitch,
            cos_yaw * cos_pitch
        ).normalized()
        
        # 右方向（水平面上垂直于前方向）
        # 在右手坐标系中: right = forward × world_up
        # forward = (sin_yaw * cos_pitch, sin_pitch, cos_yaw * cos_pitch)
        # world_up = (0, 1, 0)
        # right = forward × world_up = (-cos_yaw * cos_pitch, 0, sin_yaw * cos_pitch)
        # 归一化后: right = (-cos_yaw, 0, sin_yaw)
        self._right = Vector3(-cos_yaw, 0.0, sin_yaw).normalized()
        
        # 上方向
        self._up = self._right.cross(self._forward).normalized()
        
        # 标记视图矩阵需要更新
        self._view_dirty = True
    
    def update(self, delta_time, keys=None, mouse_dx=0, mouse_dy=0):
        """
        更新相机状态
        
        Args:
            delta_time: 帧时间（秒）
            keys: 键盘输入状态字典，例如 {'w': True, 'a': False, ...}
            mouse_dx: 鼠标X轴移动增量（像素）
            mouse_dy: 鼠标Y轴移动增量（像素）
        
        Returns:
            bool: 相机状态是否发生变化
        """
        if delta_time <= 0:
            return False
        
        keys = keys or {}
        changed = False
        
        # 1. 处理鼠标输入 - 更新视角
        if mouse_dx != 0 or mouse_dy != 0:
            self._handle_mouse_input(mouse_dx, mouse_dy)
            changed = True
        
        # 2. 处理键盘输入 - 计算目标速度
        target_velocity = self._calculate_target_velocity(keys)
        
        # 3. 应用平滑移动插值
        self._apply_smooth_movement(target_velocity, delta_time)
        
        # 4. 处理跳跃和重力
        if self.enable_jump:
            self._apply_gravity_and_jump(keys, delta_time)
        
        # 5. 更新位置
        if self.velocity.length_squared() > 0.0001:
            self.position += self.velocity * delta_time
            changed = True
        
        # 6. 应用地面碰撞
        if self.enable_jump:
            self._apply_ground_collision()
        
        # 7. 更新视图矩阵
        if changed or self._view_dirty:
            self._update_view_matrix()
        
        return changed
    
    def _handle_mouse_input(self, mouse_dx, mouse_dy):
        """处理鼠标输入，更新视角角度"""
        # 水平旋转 (yaw)
        self.yaw -= mouse_dx * self.mouse_sensitivity
        
        # 垂直旋转 (pitch)
        pitch_delta = mouse_dy * self.mouse_sensitivity
        if self.invert_mouse_y:
            self.pitch += pitch_delta
        else:
            self.pitch -= pitch_delta
        
        # 限制角度范围
        self._clamp_pitch()
        self._normalize_yaw()
        
        # 更新方向向量
        self._update_direction_vectors()
    
    def _calculate_target_velocity(self, keys):
        """
        根据键盘输入计算目标速度向量
        
        Args:
            keys: 键盘输入状态字典
        
        Returns:
            Vector3: 目标速度向量
        """
        target_velocity = Vector3(0.0, 0.0, 0.0)
        
        # 检查移动输入
        move_forward = keys.get('w', False) or keys.get('W', False)
        move_backward = keys.get('s', False) or keys.get('S', False)
        move_left = keys.get('a', False) or keys.get('A', False)
        move_right = keys.get('d', False) or keys.get('D', False)
        
        # 计算水平面上的移动方向
        move_dir = Vector3(0.0, 0.0, 0.0)
        
        if move_forward:
            move_dir = move_dir + Vector3(self._forward.x, 0.0, self._forward.z)
        if move_backward:
            move_dir = move_dir - Vector3(self._forward.x, 0.0, self._forward.z)
        if move_right:
            move_dir = move_dir + self._right
        if move_left:
            move_dir = move_dir - self._right
        
        # 归一化移动方向
        if move_dir.length_squared() > 0.001:
            move_dir.normalize()
            
            # 确定移动速度
            is_running = keys.get('shift', False) or keys.get('run', False)
            speed = self.run_speed if is_running else self.move_speed
            
            target_velocity.x = move_dir.x * speed
            target_velocity.z = move_dir.z * speed
        
        return target_velocity
    
    def _apply_smooth_movement(self, target_velocity, delta_time):
        """
        应用平滑移动插值
        使用加速度和摩擦力实现平滑的移动效果
        """
        # X轴平滑插值
        if abs(target_velocity.x) > 0.01:
            # 加速
            self.velocity.x += (target_velocity.x - self.velocity.x) * self.acceleration * delta_time
        else:
            # 减速（摩擦力）
            self.velocity.x -= self.velocity.x * self.friction * delta_time
            if abs(self.velocity.x) < 0.01:
                self.velocity.x = 0.0
        
        # Z轴平滑插值
        if abs(target_velocity.z) > 0.01:
            # 加速
            self.velocity.z += (target_velocity.z - self.velocity.z) * self.acceleration * delta_time
        else:
            # 减速（摩擦力）
            self.velocity.z -= self.velocity.z * self.friction * delta_time
            if abs(self.velocity.z) < 0.01:
                self.velocity.z = 0.0
    
    def _apply_gravity_and_jump(self, keys, delta_time):
        """应用重力和跳跃逻辑"""
        # 跳跃检测
        jump_pressed = keys.get('space', False) or keys.get(' ', False)
        
        if jump_pressed and self.is_grounded:
            self.velocity.y = self.jump_force
            self.is_grounded = False
        
        # 应用重力
        if not self.is_grounded:
            self.velocity.y += self.gravity * delta_time
    
    def _apply_ground_collision(self):
        """应用地面碰撞检测"""
        ground_level = self.ground_height + self.eye_height
        
        if self.position.y <= ground_level:
            self.position.y = ground_level
            self.velocity.y = 0.0
            self.is_grounded = True
    
    def _update_view_matrix(self):
        """更新视图矩阵"""
        # 计算目标点
        target = self.position + self._forward
        
        # 使用 look_at 创建视图矩阵
        self._view_matrix = Matrix4x4.create_look_at(
            self.position,
            target,
            self._up
        )
        
        self._view_dirty = False
    
    def get_view_matrix(self):
        """
        获取视图矩阵
        
        Returns:
            Matrix4x4: 视图矩阵
        """
        if self._view_dirty:
            self._update_view_matrix()
        return self._view_matrix
    
    def get_position(self):
        """
        获取相机位置
        
        Returns:
            Vector3: 相机位置
        """
        return self.position.copy()
    
    def set_position(self, position):
        """
        设置相机位置
        
        Args:
            position: 新位置 (Vector3 或 tuple/list)
        """
        if isinstance(position, (tuple, list)):
            self.position = Vector3(position[0], position[1], position[2])
        else:
            self.position = position.copy()
        self._view_dirty = True
    
    def get_rotation(self):
        """
        获取相机旋转（四元数）
        
        Returns:
            Quaternion: 相机旋转
        """
        return Quaternion.from_euler(self.pitch, self.yaw, 0.0)
    
    def set_rotation(self, yaw, pitch):
        """
        设置相机旋转角度
        
        Args:
            yaw: 水平角度（弧度）
            pitch: 垂直角度（弧度）
        """
        self.yaw = yaw
        self.pitch = pitch
        self._clamp_pitch()
        self._normalize_yaw()
        self._update_direction_vectors()
    
    def get_forward(self):
        """
        获取前方向向量
        
        Returns:
            Vector3: 前方向
        """
        return self._forward.copy()
    
    def get_right(self):
        """
        获取右方向向量
        
        Returns:
            Vector3: 右方向
        """
        return self._right.copy()
    
    def get_up(self):
        """
        获取上方向向量
        
        Returns:
            Vector3: 上方向
        """
        return self._up.copy()
    
    def set_move_speed(self, speed):
        """
        设置移动速度
        
        Args:
            speed: 移动速度 (单位/秒)
        """
        self.move_speed = speed
    
    def set_run_speed(self, speed):
        """
        设置奔跑速度
        
        Args:
            speed: 奔跑速度 (单位/秒)
        """
        self.run_speed = speed
    
    def set_mouse_sensitivity(self, sensitivity):
        """
        设置鼠标灵敏度
        
        Args:
            sensitivity: 灵敏度系数
        """
        self.mouse_sensitivity = sensitivity
    
    def set_ground_height(self, height):
        """
        设置地面高度（用于跳跃碰撞检测）
        
        Args:
            height: 地面高度
        """
        self.ground_height = height
    
    def teleport(self, position, yaw=None, pitch=None):
        """
        瞬移到指定位置和角度
        
        Args:
            position: 新位置
            yaw: 新水平角度（可选）
            pitch: 新垂直角度（可选）
        """
        if isinstance(position, (tuple, list)):
            self.position = Vector3(position[0], position[1], position[2])
        else:
            self.position = position.copy()
        
        if yaw is not None:
            self.yaw = yaw
        if pitch is not None:
            self.pitch = pitch
        
        self.velocity = Vector3(0.0, 0.0, 0.0)
        self._clamp_pitch()
        self._normalize_yaw()
        self._update_direction_vectors()
    
    def get_camera_object(self):
        """
        获取兼容引擎渲染系统的相机对象
        
        Returns:
            Camera: 引擎相机对象
        """
        from Engine.Scene.Camera import Camera
        
        if self._camera is None:
            self._camera = Camera("FPSCamera")
        
        # 同步位置和旋转
        self._camera.position = self.position.copy()
        self._camera.rotation = self.get_rotation()
        self._camera._update_direction_vectors()
        
        return self._camera
    
    def __str__(self):
        return f"FPSCamera(pos=({self.position.x:.2f}, {self.position.y:.2f}, {self.position.z:.2f}), yaw={math.degrees(self.yaw):.1f}°, pitch={math.degrees(self.pitch):.1f}°)"
