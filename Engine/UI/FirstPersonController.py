# -*- coding: utf-8 -*-
"""
第一人称角色控制器
First Person Controller with terrain collision
"""

from Engine.Math import Vector3
import math

class FirstPersonController:
    """第一人称控制器"""
    
    def __init__(self, camera, terrain_height_map=None, world_size=256):
        """
        初始化控制器
        
        Args:
            camera: 相机对象
            terrain_height_map: 地形高度图 (numpy array) - 可选
            world_size: 世界尺寸 - 可选
        """
        self.camera = camera
        
        # 复制相机初始位置
        self.position = Vector3(camera.position.x, camera.position.y, camera.position.z)
        self.velocity = Vector3(0, 0, 0)
        
        # 视角角度 (弧度)
        self.yaw = 0.0
        self.pitch = 0.0
        
        # 默认配置 (标准单位，使用时需根据场景缩放调整)
        self.move_speed = 5.0
        self.run_speed = 10.0
        self.jump_force = 5.0
        self.gravity = -9.8
        self.player_height = 1.8
        self.mouse_sensitivity = 0.003
        
        # 地形数据
        self.height_map = terrain_height_map
        self.world_size = world_size
        self.resolution = 128
        if terrain_height_map is not None:
            self.resolution = terrain_height_map.shape[0]
            
        # 状态
        self.is_grounded = False
        
        # 高度缓存 (用于网格插值)
        self.height_cache = {}
        
    def set_config(self, move_speed=None, run_speed=None, jump_force=None, gravity=None, player_height=None, mouse_sensitivity=None):
        """配置控制器参数"""
        if move_speed is not None: self.move_speed = move_speed
        if run_speed is not None: self.run_speed = run_speed
        if jump_force is not None: self.jump_force = jump_force
        if gravity is not None: self.gravity = gravity
        if player_height is not None: self.player_height = player_height
        if mouse_sensitivity is not None: self.mouse_sensitivity = mouse_sensitivity

    def set_position(self, x, y, z):
        """强制设置位置"""
        self.position.x = x
        self.position.y = y
        self.position.z = z
        self.velocity = Vector3(0, 0, 0)

    def set_rotation(self, yaw, pitch):
        """强制设置旋转"""
        self.yaw = yaw
        self.pitch = pitch

    def _get_terrain_height_from_map(self, x, z):
        """从高度图获取高度 (如果提供了height_map)"""
        if self.height_map is None:
            return 0.0
            
        # 假设world_size是从中心开始 +/- world_size/2 还是 0 到 world_size?
        # 通常高度图是 0..1 映射到 world bounds
        # 这里假设 x, z 是世界坐标，我们需要映射到 0..resolution
        
        # 简单假设: 0,0 在中心，范围 +/- world_size/2
        # u = (x + self.world_size/2) / self.world_size
        # v = (z + self.world_size/2) / self.world_size
        
        # 但 play_hiking.py 里的逻辑似乎 x, z 是直接对应的？
        # play_hiking.py 里 build_terrain_height_grid 是直接遍历顶点
        # 所以我们这里保留接口，具体实现由外部传入的 height_provider 决定更好
        # 暂时保留这个占位符，主要逻辑依赖 update 里的 callback 或 height_cache
        return 0.0

    def update(self, delta_time, input_state, terrain_height_callback=None):
        """
        更新控制器状态
        
        Args:
            delta_time: 帧间隔时间 (秒)
            input_state: 输入状态字典 {'w': bool, 'a': bool, ...}
            terrain_height_callback: 函数(x, z) -> height，用于获取地形高度
        """
        if delta_time <= 0:
            return

        # 1. 鼠标旋转
        mouse_dx = input_state.get('mouse_dx', 0)
        mouse_dy = input_state.get('mouse_dy', 0)
        
        if mouse_dx != 0 or mouse_dy != 0:
            # 鼠标向左(dx<0) -> 增加Yaw -> 向左看 (用户习惯)
            # 鼠标向右(dx>0) -> 减少Yaw -> 向右看
            self.yaw -= mouse_dx * self.mouse_sensitivity
            # 鼠标向下(dy>0) -> 减少Pitch -> 向下看
            self.pitch -= mouse_dy * self.mouse_sensitivity
            
            # 限制俯仰角 (-89 到 89 度)
            max_pitch = 1.55 # 约 89度
            self.pitch = max(-max_pitch, min(max_pitch, self.pitch))
            
            # 消费掉鼠标移动 (防止累积，如果输入系统没有自动清除)
            # 注意：通常不建议在这里修改 input_state，但为了兼容性...
            # 最好是输入系统每帧重置 mouse_dx/dy
            if isinstance(input_state, dict):
                input_state['mouse_dx'] = 0
                input_state['mouse_dy'] = 0

        # 2. 计算移动方向 (基于当前 Yaw)
        # 坐标系: Y向上, -Z向前 (通常OpenGL习惯)
        # 但 Engine.Math 可能不同。play_hiking.py 里：
        # forward = [-sin(yaw), 0, -cos(yaw)]
        # right = [-cos(yaw), 0, sin(yaw)]
        
        sin_yaw = math.sin(self.yaw)
        cos_yaw = math.cos(self.yaw)
        
        # 前方向量 (水平)
        # 假设初始朝向 -Z
        forward = Vector3(-sin_yaw, 0, -cos_yaw)
        # 右方向量
        right = Vector3(cos_yaw, 0, -sin_yaw) 
        # 注意：这里 right 应该是 forward x up (0,1,0)
        # forward(-sin, 0, -cos) x up(0,1,0) = (cos, 0, -sin) -> 看起来是对的
        
        move_dir = Vector3(0, 0, 0)
        if input_state.get('w'): move_dir += forward
        if input_state.get('s'): move_dir -= forward
        if input_state.get('d'): move_dir += right
        if input_state.get('a'): move_dir -= right
        
        if move_dir.length_squared() > 0.001:
            move_dir.normalize()
            
        # 3. 应用移动速度
        target_speed = self.run_speed if input_state.get('shift') else self.move_speed
        
        # 地面移动
        self.position.x += move_dir.x * target_speed * delta_time
        self.position.z += move_dir.z * target_speed * delta_time
        
        if hasattr(self, "world_bounds") and self.world_bounds:
            min_x, max_x, min_z, max_z = self.world_bounds
            clamped_x = min(max(self.position.x, min_x), max_x)
            clamped_z = min(max(self.position.z, min_z), max_z)
            if clamped_x != self.position.x or clamped_z != self.position.z:
                self.position.x = clamped_x
                self.position.z = clamped_z
                if not hasattr(self, "_bounds_warned"):
                    print("[Controller] 已到达地形边界，位置被限制在模型范围内")
                    self._bounds_warned = True
        
        # 4. 跳跃和重力
        if self.is_grounded and input_state.get('space'):
            self.velocity.y = self.jump_force
            self.is_grounded = False
            
        # 应用重力
        self.velocity.y += self.gravity * delta_time
        self.position.y += self.velocity.y * delta_time
        
        # 5. 地形碰撞检测
        ground_height = 0.0
        if terrain_height_callback:
            ground_height = terrain_height_callback(self.position.x, self.position.z)
        elif self.height_map is not None:
            ground_height = self._get_terrain_height_from_map(self.position.x, self.position.z)
            
        # 简单的地面吸附 - 添加额外的安全余量
        target_ground_y = ground_height + self.player_height
        if self.position.y < target_ground_y + 0.01:
            self.position.y = target_ground_y
            self.velocity.y = 0
            self.is_grounded = True
        else:
            self.is_grounded = False
        
        # 绝对安全高度 - 防止掉到无限深的地方
        SAFE_MIN_HEIGHT = -1000.0
        if self.position.y < SAFE_MIN_HEIGHT:
            self.position.y = SAFE_MIN_HEIGHT + 10.0
            self.velocity.y = 0
            self.is_grounded = True
            print("[Controller] 警告: 玩家已到达安全边界，已重置位置")
            
        # 6. 更新相机
        self.camera.position = Vector3(self.position.x, self.position.y, self.position.z)
        
        # 计算 LookAt 目标
        # 俯仰角影响 look_at 的 y 分量
        # 水平分量需要根据 pitch 缩放 (球面坐标)
        # look_dir = (
        #    -sin(yaw) * cos(pitch),
        #    sin(pitch),
        #    -cos(yaw) * cos(pitch)
        # )
        
        cos_pitch = math.cos(self.pitch)
        sin_pitch = math.sin(self.pitch)
        
        look_dir = Vector3(
            -sin_yaw * cos_pitch,
            sin_pitch,
            -cos_yaw * cos_pitch
        )
        
        target = self.position + look_dir
        self.camera.look_at(target)
