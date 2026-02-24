# -*- coding: utf-8 -*-
"""
分块LOD地形渲染系统
用于处理大规模地形的渲染，支持多级LOD和流式加载
"""

from Engine.Math import Vector2, Vector3, BoundingBox
from Engine.Renderer.Resources.Mesh import Mesh
import numpy as np
import math
from collections import deque

class TerrainChunk:
    """地形块类，用于管理单个地形块的几何数据和LOD"""
    
    def __init__(self, chunk_x, chunk_z, size=256, resolution=32):
        """初始化地形块
        
        Args:
            chunk_x: 地形块的X坐标
            chunk_z: 地形块的Z坐标
            size: 地形块的大小（世界单位）
            resolution: 地形块的分辨率（顶点数量）
        """
        self.chunk_x = chunk_x
        self.chunk_z = chunk_z
        self.size = size
        self.resolution = resolution
        
        # 世界空间位置
        self.position = Vector3(chunk_x * size, 0, chunk_z * size)
        
        # LOD设置
        self.current_lod = 0
        self.lod_levels = []  # 不同LOD级别的网格
        self.lod_distances = [50, 100, 200, 400]  # LOD切换距离
        
        # 地形数据
        self.heightmap = None  # 高度图
        self.normalmap = None  # 法线图
        self.texture_map = None  # 纹理图
        
        # 渲染状态
        self.is_visible = False
        self.is_loaded = False
        self.bounding_box = None
        
        # 生成地形数据
        self._generate_terrain_data()
        
        # 生成LOD级别
        self._generate_lod_levels()
        
        # 计算包围盒
        self._calculate_bounding_box()
        
        # 标记地形块为已加载
        self.is_loaded = True
    
    def _generate_terrain_data(self):
        """生成地形数据（高度图、法线图等）"""
        # 生成简单的高度图（使用多层正弦波模拟不同地形类型）
        self.heightmap = np.zeros((self.resolution, self.resolution), dtype=np.float32)
        
        for z in range(self.resolution):
            for x in range(self.resolution):
                # 归一化坐标到[0, 1]
                norm_x = x / (self.resolution - 1)
                norm_z = z / (self.resolution - 1)
                
                # 世界坐标
                world_x = self.chunk_x * self.size + norm_x * self.size
                world_z = self.chunk_z * self.size + norm_z * self.size
                
                # 使用Perlin噪声生成更真实的地形高度
                height = self._perlin_noise(world_x * 0.001, world_z * 0.001) * 50  # 基础地形
                height += self._perlin_noise(world_x * 0.005, world_z * 0.005) * 20  # 中等特征
                height += self._perlin_noise(world_x * 0.02, world_z * 0.02) * 10  # 细节
                height += self._perlin_noise(world_x * 0.05, world_z * 0.05) * 5  # 微小细节
                
                # 根据高度确定地形类型
                terrain_type = self._get_terrain_type(height)
                
                # 根据地形类型调整高度，使地形更加真实
                if terrain_type == 'water':
                    height = max(height, -10)  # 水域最低高度
                    height = min(height, 0)  # 水域最高高度
                elif terrain_type == 'beach':
                    height = max(height, 0)  # 沙滩从0开始
                    height = min(height, 5)  # 沙滩最高5米
                elif terrain_type == 'plains':
                    height = max(height, 5)  # 平原从5米开始
                    height = min(height, 30)  # 平原最高30米
                elif terrain_type == 'hills':
                    height = max(height, 30)  # 丘陵从30米开始
                    height = min(height, 100)  # 丘陵最高100米
                elif terrain_type == 'mountains':
                    height = max(height, 100)  # 山脉从100米开始
                    height = min(height, 200)  # 山脉最高200米
                
                self.heightmap[z, x] = height
    
    def _perlin_noise(self, x, y):
        """
        简单的Perlin噪声实现
        
        Args:
            x: X坐标
            y: Y坐标
            
        Returns:
            float: 噪声值 [-1.0, 1.0]
        """
        # 使用简化的Perlin噪声算法
        def fade(t):
            return t * t * t * (t * (t * 6 - 15) + 10)
        
        def lerp(t, a, b):
            return a + t * (b - a)
        
        def grad(hash, x, y):
            h = hash & 15
            u = x if h < 8 else y
            v = y if h < 4 else (x if h in (12, 14) else 0)
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
        
        # 计算网格坐标
        ix = int(x)
        iy = int(y)
        
        # 计算网格内的局部坐标
        fx = x - ix
        fy = y - iy
        
        # 平滑插值因子
        u = fade(fx)
        v = fade(fy)
        
        # 生成哈希表
        p = [
            151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
            190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54, 65,25,63,161,1,216,80,73,209,76,132,187,208, 89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
        ] * 2
        
        # 计算哈希值
        a = p[ix & 255] + iy
        aa = p[a & 255]
        ab = p[(a + 1) & 255]
        b = p[(ix + 1) & 255] + iy
        ba = p[b & 255]
        bb = p[(b + 1) & 255]
        
        # 计算梯度值
        n00 = grad(aa, fx, fy)
        n01 = grad(ba, fx - 1, fy)
        n10 = grad(ab, fx, fy - 1)
        n11 = grad(bb, fx - 1, fy - 1)
        
        # 线性插值
        x1 = lerp(u, n00, n01)
        x2 = lerp(u, n10, n11)
        
        return lerp(v, x1, x2)
    
    def _generate_lod_levels(self):
        """生成不同LOD级别的网格"""
        # 生成4个LOD级别
        for lod in range(4):
            # 计算当前LOD级别的分辨率（每级减少一半）
            lod_resolution = max(4, self.resolution // (2 ** lod))
            
            # 创建地形网格
            mesh = self._create_terrain_mesh(lod_resolution)
            self.lod_levels.append(mesh)
    
    def _create_terrain_mesh(self, resolution):
        """创建地形网格
        
        Args:
            resolution: 网格分辨率
            
        Returns:
            Mesh: 地形网格
        """
        mesh = Mesh()
        
        vertices = []
        normals = []
        uvs = []
        indices = []
        
        # 采样步长
        step = (self.resolution - 1) / (resolution - 1)
        
        # 生成顶点
        for z in range(resolution):
            for x in range(resolution):
                # 采样高度图
                height = self.heightmap[int(z * step), int(x * step)]
                
                # 计算世界坐标
                world_x = x / (resolution - 1) * self.size + self.position.x
                world_z = z / (resolution - 1) * self.size + self.position.z
                world_y = height
                
                vertices.append(Vector3(world_x, world_y, world_z))
                
                # 简化处理，暂时使用向上法线
                normals.append(Vector3(0, 1, 0))
                
                # UV坐标
                uvs.append(Vector2(x / (resolution - 1), z / (resolution - 1)))
        
        # 生成索引
        for z in range(resolution - 1):
            for x in range(resolution - 1):
                current = z * resolution + x
                next_row = (z + 1) * resolution + x
                
                # 第一个三角形
                indices.append(current)
                indices.append(next_row)
                indices.append(current + 1)
                
                # 第二个三角形
                indices.append(next_row)
                indices.append(next_row + 1)
                indices.append(current + 1)
        
        mesh.vertices = vertices
        mesh.normals = normals
        mesh.uvs = uvs
        mesh.indices = indices
        
        # 计算包围盒
        mesh._calculate_bounding_box()
        
        return mesh
    
    def _get_terrain_type(self, height):
        """根据高度确定地形类型
        
        Args:
            height: 地形高度
            
        Returns:
            str: 地形类型
        """
        if height < 0:
            return 'water'       # 水域
        elif height < 5:
            return 'beach'       # 沙滩
        elif height < 30:
            return 'plains'      # 平原
        elif height < 100:
            return 'hills'       # 丘陵
        else:
            return 'mountains'    # 山脉
    
    def _calculate_bounding_box(self):
        """计算地形块的包围盒"""
        min_y = np.min(self.heightmap)
        max_y = np.max(self.heightmap)
        
        min_point = Vector3(self.position.x, min_y, self.position.z)
        max_point = Vector3(self.position.x + self.size, max_y, self.position.z + self.size)
        
        self.bounding_box = BoundingBox(min_point, max_point)
    
    def update_lod(self, camera_position):
        """根据相机位置更新LOD级别
        
        Args:
            camera_position: 相机位置
        """
        # 计算相机到地形块中心的距离
        chunk_center = Vector3(
            self.position.x + self.size / 2,
            0,
            self.position.z + self.size / 2
        )
        distance = (camera_position - chunk_center).length()
        
        # 更新LOD级别
        new_lod = 0
        for i, lod_distance in enumerate(self.lod_distances):
            if distance > lod_distance:
                new_lod = i + 1
        
        self.current_lod = min(new_lod, len(self.lod_levels) - 1)
    
    def is_visible_from_camera(self, camera):
        """检查地形块是否在相机视锥体内
        
        Args:
            camera: 相机对象
            
        Returns:
            bool: 是否可见
        """
        self.is_visible = camera.is_visible(self.bounding_box)
        return self.is_visible
    
    def draw(self):
        """绘制地形块"""
        if self.is_visible and self.is_loaded:
            # 绘制当前LOD级别的网格
            mesh = self.lod_levels[self.current_lod]
            mesh.draw()

class InterestPoint:
    """兴趣点类，用于定义地形上的兴趣点"""
    
    def __init__(self, position, name, description=""):
        """初始化兴趣点
        
        Args:
            position: 兴趣点位置
            name: 兴趣点名称
            description: 兴趣点描述
        """
        self.position = position
        self.name = name
        self.description = description
        self.is_visible = False
        self.bounding_box = BoundingBox(
            position - Vector3(5, 5, 5),
            position + Vector3(5, 5, 5)
        )
        
        # 渲染属性
        self.color = Vector3(1, 0, 0)  # 默认红色
        self.scale = 1.0
    
    def update_visibility(self, camera):
        """更新兴趣点可见性
        
        Args:
            camera: 相机对象
        """
        self.is_visible = camera.is_visible(self.bounding_box)
    
    def draw(self, renderer):
        """绘制兴趣点
        
        Args:
            renderer: 渲染器实例
        """
        if self.is_visible:
            # 绘制兴趣点标记（简化实现）
            from OpenGL.GL import (
                glPushMatrix, glPopMatrix, glTranslatef, glScalef, glColor3f,
                glBegin, glEnd, glVertex3f, GL_LINES, glLineWidth
            )
            
            glPushMatrix()
            
            # 设置位置和缩放
            glTranslatef(self.position.x, self.position.y, self.position.z)
            glScalef(self.scale, self.scale, self.scale)
            
            # 设置颜色
            glColor3f(self.color.x, self.color.y, self.color.z)
            
            # 绘制简单的十字标记
            glLineWidth(2.0)
            glBegin(GL_LINES)
            
            # 垂直方向
            glVertex3f(0, -5, 0)
            glVertex3f(0, 5, 0)
            
            # 水平X方向
            glVertex3f(-5, 0, 0)
            glVertex3f(5, 0, 0)
            
            # 水平Z方向
            glVertex3f(0, 0, -5)
            glVertex3f(0, 0, 5)
            
            glEnd()
            
            glPopMatrix()

class TerrainManager:
    """地形管理器，用于管理所有地形块和兴趣点"""
    
    def __init__(self, scene_manager, chunk_size=256, chunk_resolution=32, view_distance=1000):
        """初始化地形管理器
        
        Args:
            scene_manager: 场景管理器
            chunk_size: 地形块大小
            chunk_resolution: 地形块分辨率
            view_distance: 视图距离
        """
        self.scene_manager = scene_manager
        self.chunk_size = chunk_size
        self.chunk_resolution = chunk_resolution
        self.view_distance = view_distance
        self.max_chunk_creations_per_update = 8
        self._pending_chunk_keys = deque()
        self._pending_chunk_key_set = set()
        self.force_generate_in_draw = False
        
        # 地形块字典（key: (chunk_x, chunk_z), value: TerrainChunk）
        self.terrain_chunks = {}
        
        # 加载的地形块
        self.loaded_chunks = []
        
        # 兴趣点列表
        self.interest_points = []
        
        # 地形材质
        self.terrain_material = None
        
        # 初始化地形材质
        self._init_terrain_material()
        
        # 初始化兴趣点
        self._init_interest_points()
        
        # 地形类型颜色映射
        self.terrain_colors = {
            'water': Vector3(0.0, 0.3, 0.7),     # 蓝色水域
            'beach': Vector3(0.9, 0.8, 0.6),     # 沙滩色
            'plains': Vector3(0.3, 0.7, 0.3),    # 绿色平原
            'hills': Vector3(0.2, 0.5, 0.2),     # 深绿色丘陵
            'mountains': Vector3(0.5, 0.5, 0.5)  # 灰色山脉
        }
        
        # 立即生成初始地形块
        self._generate_initial_chunks()
    
    def _init_terrain_material(self):
        """初始化地形材质"""
        # TODO: 实现地形材质初始化
        from Engine.Renderer.Resources.Material import Material
        self.terrain_material = Material()
        self.terrain_material.set_color(Vector3(0.3, 0.7, 0.3))
    
    def _init_interest_points(self):
        """初始化兴趣点"""
        # 创建一些示例兴趣点
        interest_points = [
            # 山脉兴趣点
            InterestPoint(Vector3(0, 100, 0), "山脉顶点"),
            InterestPoint(Vector3(500, 150, 500), "高山湖泊"),
            InterestPoint(Vector3(-500, 120, -500), "雪山"),
            
            # 平原兴趣点
            InterestPoint(Vector3(200, 20, 200), "平原湖泊"),
            InterestPoint(Vector3(-300, 15, 300), "森林"),
            InterestPoint(Vector3(300, 25, -300), "草原"),
            
            # 水域兴趣点
            InterestPoint(Vector3(0, -2, 500), "深海"),
            InterestPoint(Vector3(500, 0, 0), "海湾"),
            InterestPoint(Vector3(-500, 0, 0), "河口")
        ]
        
        # 设置不同兴趣点的颜色
        interest_points[0].color = Vector3(1, 0, 0)    # 红色
        interest_points[1].color = Vector3(0, 1, 0)    # 绿色
        interest_points[2].color = Vector3(1, 1, 1)    # 白色
        interest_points[3].color = Vector3(0, 0, 1)    # 蓝色
        interest_points[4].color = Vector3(0.5, 0.5, 0)  # 棕色
        interest_points[5].color = Vector3(0, 0.8, 0)    # 浅绿色
        interest_points[6].color = Vector3(0, 0, 0.8)    # 深蓝色
        interest_points[7].color = Vector3(0, 0.5, 0.5)  # 青色
        interest_points[8].color = Vector3(0.5, 0.5, 0.5)  # 灰色
        
        self.interest_points = interest_points
    
    def update(self, camera):
        """更新地形管理器
        
        Args:
            camera: 相机对象
        """
        # 获取相机位置
        camera_position = camera.get_position()
        
        # 计算当前相机所在的地形块坐标
        current_chunk_x = int(camera_position.x / self.chunk_size)
        current_chunk_z = int(camera_position.z / self.chunk_size)
        
        # 计算需要加载的地形块范围
        load_radius = int(self.view_distance / self.chunk_size) + 1
        
        missing = []
        for z in range(current_chunk_z - load_radius, current_chunk_z + load_radius + 1):
            for x in range(current_chunk_x - load_radius, current_chunk_x + load_radius + 1):
                chunk_key = (x, z)
                if chunk_key not in self.terrain_chunks:
                    if chunk_key not in self._pending_chunk_key_set:
                        dx = x - current_chunk_x
                        dz = z - current_chunk_z
                        missing.append((dx * dx + dz * dz, chunk_key))
                    continue
                chunk = self.terrain_chunks[chunk_key]
                chunk.update_lod(camera_position)
                chunk.is_visible_from_camera(camera)
        
        if missing:
            missing.sort(key=lambda t: t[0])
            for _, chunk_key in missing:
                if chunk_key in self._pending_chunk_key_set or chunk_key in self.terrain_chunks:
                    continue
                self._pending_chunk_keys.append(chunk_key)
                self._pending_chunk_key_set.add(chunk_key)
        
        created = 0
        while created < self.max_chunk_creations_per_update and self._pending_chunk_keys:
            x, z = self._pending_chunk_keys.popleft()
            self._pending_chunk_key_set.discard((x, z))
            if (x, z) in self.terrain_chunks:
                continue
            chunk = TerrainChunk(x, z, self.chunk_size, self.chunk_resolution)
            chunk.update_lod(camera_position)
            chunk.is_visible_from_camera(camera)
            self.terrain_chunks[(x, z)] = chunk
            self.loaded_chunks.append(chunk)
            created += 1
        
        # 移除超出视图距离的地形块（简化实现）
        # TODO: 实现更复杂的卸载策略
    
    def draw(self):
        """
        绘制所有可见地形块和兴趣点
        """
        # 确保至少有地形块
        if not self.terrain_chunks:
            self._generate_initial_chunks()
        
        if self.force_generate_in_draw and self.scene_manager.active_camera:
            self._update_visible_chunks(self.scene_manager.active_camera)
        
        # 绘制所有加载的地形块
        for chunk in self.loaded_chunks:
            if chunk.is_loaded and chunk.is_visible:
                chunk.draw()
        
        # 绘制兴趣点
        for point in self.interest_points:
            point.draw(None)  # 简化实现，直接绘制
    
    def _generate_initial_chunks(self):
        """
        生成初始地形块
        """
        # 生成玩家周围的地形块
        for x in range(-1, 2):
            for z in range(-1, 2):
                chunk_key = (x, z)
                if chunk_key not in self.terrain_chunks:
                    chunk = TerrainChunk(x, z, self.chunk_size, self.chunk_resolution)
                    self.terrain_chunks[chunk_key] = chunk
                    self.loaded_chunks.append(chunk)
    
    def _update_visible_chunks(self, camera):
        """
        更新可见的地形块
        
        Args:
            camera: 相机对象
        """
        # 计算可见范围内的地形块
        camera_pos = camera.get_position()
        chunk_x = int(camera_pos.x // self.chunk_size)
        chunk_z = int(camera_pos.z // self.chunk_size)
        
        # 生成可见范围内的地形块
        view_distance = int(self.view_distance // self.chunk_size)
        for x in range(chunk_x - view_distance, chunk_x + view_distance + 1):
            for z in range(chunk_z - view_distance, chunk_z + view_distance + 1):
                chunk_key = (x, z)
                if chunk_key not in self.terrain_chunks:
                    # 生成新的地形块
                    chunk = TerrainChunk(x, z, self.chunk_size, self.chunk_resolution)
                    self.terrain_chunks[chunk_key] = chunk
                    self.loaded_chunks.append(chunk)
        
        # 更新所有地形块的可见性
        for chunk in self.terrain_chunks.values():
            chunk.is_visible_from_camera(camera)
    
    def get_terrain_height(self, world_x, world_z):
        """获取世界坐标处的地形高度

        Args:
            world_x: 世界X坐标
            world_z: 世界Z坐标

        Returns:
            float: 地形高度
        """
        # 计算地形块坐标
        chunk_x = int(world_x / self.chunk_size)
        chunk_z = int(world_z / self.chunk_size)

        chunk_key = (chunk_x, chunk_z)
        if chunk_key not in self.terrain_chunks:
            # 尝试生成所需地形块（如果不存在）
            if chunk_key not in self._pending_chunk_key_set:
                chunk = TerrainChunk(chunk_x, chunk_z, self.chunk_size, self.chunk_resolution)
                self.terrain_chunks[chunk_key] = chunk
                self.loaded_chunks.append(chunk)
            else:
                return 0.0  # 如果仍在加载中，返回默认值

        chunk = self.terrain_chunks[chunk_key]

        # 计算在地形块内的局部坐标
        local_x = world_x - chunk.position.x
        local_z = world_z - chunk.position.z

        # 归一化坐标
        norm_x = local_x / self.chunk_size
        norm_z = local_z / self.chunk_size

        # 确保归一化坐标在[0, 1]范围内
        norm_x = max(0.0, min(1.0, norm_x))
        norm_z = max(0.0, min(1.0, norm_z))

        # 从高度图采样
        # 计算在高度图中的索引
        idx_x = int(norm_x * (self.chunk_resolution - 1))
        idx_z = int(norm_z * (self.chunk_resolution - 1))

        # 边界检查
        idx_x = max(0, min(self.chunk_resolution - 1, idx_x))
        idx_z = max(0, min(self.chunk_resolution - 1, idx_z))

        # 直接返回高度值
        if chunk.heightmap is not None:
            return chunk.heightmap[idx_z, idx_x]
        else:
            return 0.0
    
    def destroy(self):
        """销毁地形管理器，释放资源"""
        for chunk in self.loaded_chunks:
            # 释放地形块资源
            for mesh in chunk.lod_levels:
                mesh.destroy()
        
        self.terrain_chunks.clear()
        self.loaded_chunks.clear()
