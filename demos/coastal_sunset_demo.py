"""
海岸悬崖日落演示

展示Natural系统在海岸场景中的应用：
- 悬崖地形生成
- 海洋水面
- 日落光照
- 动态天气

控制：
- WASD：移动相机
- 鼠标：旋转视角
- F：切换雾效
- S：截图保存
- ESC：退出
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import glm
from OpenGL.GL import *
from OpenGL.GLU import *
import glfw

from Engine.Natural import NaturalSystem


class CoastalTerrainGenerator:
    """海岸地形生成器"""
    
    @staticmethod
    def generate_coastal_terrain(size=128):
        """
        生成海岸悬崖地形
        - 一侧是海洋（低高度）
        - 一侧是悬崖（陡峭上升）
        - 顶部是草地平台
        """
        from scipy.ndimage import gaussian_filter
        
        # 创建坐标网格
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        
        # 基础形状：一侧高一侧低（悬崖）
        # 使用更陡峭的阶跃
        cliff_edge = np.tanh((X + 0.5) * 4.0)  # 更陡峭的悬崖
        height = cliff_edge * 20 + 5  # 降低高度范围，更明显的悬崖
        
        # 添加悬崖顶部的起伏（草地平台）
        platform_noise = np.sin(X * 6) * np.cos(Y * 4) * 3
        platform_noise += np.random.rand(size, size) * 0.8
        platform_noise = gaussian_filter(platform_noise, sigma=1.0)
        
        # 只在高处添加平台起伏
        height += np.maximum(0, cliff_edge) * platform_noise
        
        # 添加海滩的轻微起伏
        beach_noise = np.sin(Y * 10) * 0.3
        height += np.minimum(0, cliff_edge) * beach_noise
        
        # 整体平滑（减少锯齿）
        height = gaussian_filter(height, sigma=0.5)
        
        return height.astype(np.float32)
    
    @staticmethod
    def generate_water_level(height_map, water_level=8.0):
        """
        生成水面高度图
        海洋一侧有水，悬崖一侧无水
        """
        water = np.zeros_like(height_map)
        
        # 在低于水位的区域填充水
        mask = height_map < water_level
        water[mask] = water_level - height_map[mask]
        
        # 限制最大水深
        water = np.clip(water, 0, 15)
        
        return water.astype(np.float32)


class CoastalMesh:
    """海岸地形网格"""
    
    def __init__(self, height_map, water_map=None):
        self.size = height_map.shape[0]
        self.height_map = height_map
        self.water_map = water_map if water_map is not None else np.zeros_like(height_map)
        
        # OpenGL资源
        self.terrain_vao = None
        self.terrain_vbo = None
        self.terrain_ebo = None
        self.terrain_indices = 0
        
        self.water_vao = None
        self.water_vbo = None
        self.water_ebo = None
        self.water_indices = 0
        
        self._setup_terrain_mesh()
        self._setup_water_mesh()
    
    def _setup_terrain_mesh(self):
        """设置地形网格"""
        vertices = []
        indices = []
        
        for z in range(self.size):
            for x in range(self.size):
                # 位置
                px = (x / (self.size - 1)) * 200 - 100
                py = self.height_map[z, x] * 0.5
                pz = (z / (self.size - 1)) * 200 - 100
                
                # 基于高度的颜色
                height = self.height_map[z, x]
                if height < 5:  # 沙滩
                    color = [0.76, 0.70, 0.50]
                elif height < 15:  # 草地
                    color = [0.2, 0.5, 0.2]
                elif height < 25:  # 岩石
                    color = [0.4, 0.4, 0.4]
                else:  # 顶部草地
                    color = [0.15, 0.45, 0.15]
                
                # 法线
                nx, ny, nz = 0, 1, 0
                if x > 0 and x < self.size - 1 and z > 0 and z < self.size - 1:
                    dx = (self.height_map[z, x + 1] - self.height_map[z, x - 1]) * 0.5
                    dz = (self.height_map[z + 1, x] - self.height_map[z - 1, x]) * 0.5
                    normal = np.array([-dx, 2.0 * (200.0 / self.size), -dz])
                    normal = normal / (np.linalg.norm(normal) + 1e-5)
                    nx, ny, nz = normal
                
                vertices.extend([px, py, pz, color[0], color[1], color[2], nx, ny, nz])
        
        # 生成索引
        for z in range(self.size - 1):
            for x in range(self.size - 1):
                i = z * self.size + x
                indices.extend([i, i + 1, i + self.size])
                indices.extend([i + 1, i + self.size + 1, i + self.size])
        
        vertices = np.array(vertices, dtype=np.float32)
        indices = np.array(indices, dtype=np.uint32)
        self.terrain_indices = len(indices)
        
        # 创建OpenGL缓冲区
        self.terrain_vao = glGenVertexArrays(1)
        self.terrain_vbo = glGenBuffers(1)
        self.terrain_ebo = glGenBuffers(1)
        
        glBindVertexArray(self.terrain_vao)
        
        glBindBuffer(GL_ARRAY_BUFFER, self.terrain_vbo)
        glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
        
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.terrain_ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
        
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(3 * 4))
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(6 * 4))
        glEnableVertexAttribArray(2)
        
        glBindVertexArray(0)
    
    def _setup_water_mesh(self):
        """设置水面网格"""
        vertices = []
        indices = []
        
        water_level = 4.0  # 水面高度
        
        for z in range(self.size):
            for x in range(self.size):
                # 只在水深大于0的地方生成水面
                if self.water_map[z, x] > 0.1:
                    px = (x / (self.size - 1)) * 200 - 100
                    py = water_level
                    pz = (z / (self.size - 1)) * 200 - 100
                    
                    # 水蓝色
                    depth = min(1.0, self.water_map[z, x] / 10.0)
                    color = [0.0, 0.3 + depth * 0.2, 0.5 + depth * 0.3]
                    
                    # 法线向上
                    nx, ny, nz = 0, 1, 0
                    
                    vertices.extend([px, py, pz, color[0], color[1], color[2], nx, ny, nz])
                else:
                    # 占位符（不会被渲染）
                    vertices.extend([0, -1000, 0, 0, 0, 0, 0, 1, 0])
        
        # 生成索引（只包含水面区域）
        for z in range(self.size - 1):
            for x in range(self.size - 1):
                i = z * self.size + x
                # 检查四个角是否都有水
                if (self.water_map[z, x] > 0.1 and 
                    self.water_map[z, x + 1] > 0.1 and
                    self.water_map[z + 1, x] > 0.1 and
                    self.water_map[z + 1, x + 1] > 0.1):
                    indices.extend([i, i + 1, i + self.size])
                    indices.extend([i + 1, i + self.size + 1, i + self.size])
        
        if len(indices) > 0:
            vertices = np.array(vertices, dtype=np.float32)
            indices = np.array(indices, dtype=np.uint32)
            self.water_indices = len(indices)
            
            self.water_vao = glGenVertexArrays(1)
            self.water_vbo = glGenBuffers(1)
            self.water_ebo = glGenBuffers(1)
            
            glBindVertexArray(self.water_vao)
            
            glBindBuffer(GL_ARRAY_BUFFER, self.water_vbo)
            glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.water_ebo)
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL_STATIC_DRAW)
            
            glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(0))
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(3 * 4))
            glEnableVertexAttribArray(1)
            glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, 9 * 4, ctypes.c_void_p(6 * 4))
            glEnableVertexAttribArray(2)
            
            glBindVertexArray(0)
    
    def render_terrain(self):
        """渲染地形"""
        glBindVertexArray(self.terrain_vao)
        glDrawElements(GL_TRIANGLES, self.terrain_indices, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)
    
    def render_water(self):
        """渲染水面"""
        if self.water_indices > 0:
            glBindVertexArray(self.water_vao)
            glDrawElements(GL_TRIANGLES, self.water_indices, GL_UNSIGNED_INT, None)
            glBindVertexArray(0)
    
    def cleanup(self):
        """清理资源"""
        if self.terrain_vao:
            glDeleteVertexArrays(1, [self.terrain_vao])
        if self.terrain_vbo:
            glDeleteBuffers(1, [self.terrain_vbo])
        if self.terrain_ebo:
            glDeleteBuffers(1, [self.terrain_ebo])
        
        if self.water_vao:
            glDeleteVertexArrays(1, [self.water_vao])
        if self.water_vbo:
            glDeleteBuffers(1, [self.water_vbo])
        if self.water_ebo:
            glDeleteBuffers(1, [self.water_ebo])


class NaturalShaderWrapper:
    """Natural自动着色器包装类"""
    
    def __init__(self, natural_system):
        self.natural = natural_system
        self.shader = None
        self.uniform_locations = {}
    
    def init(self):
        """初始化着色器"""
        if self.shader is None:
            # 获取Natural生成的着色器代码
            shader_code = self.natural.get_shader()
            
            # 编译着色器
            self.shader = self._compile_from_source(shader_code)
            
            # 获取uniform位置
            self._cache_uniform_locations()
    
    def _compile_from_source(self, source: str):
        """从源代码编译着色器"""
        # 分离顶点和片段着色器
        if 'FRAGMENT_SHADER' in source:
            parts = source.split('FRAGMENT_SHADER')
            vertex_source = parts[0]
            fragment_source = 'FRAGMENT_SHADER'.join(parts[1:])
        else:
            vertex_source = source
            fragment_source = None
        
        # 编译
        vertex = glCreateShader(GL_VERTEX_SHADER)
        glShaderSource(vertex, vertex_source)
        glCompileShader(vertex)
        
        if fragment_source:
            fragment = glCreateShader(GL_FRAGMENT_SHADER)
            glShaderSource(fragment, fragment_source)
            glCompileShader(fragment)
        else:
            fragment = None
        
        # 链接程序
        program = glCreateProgram()
        glAttachShader(program, vertex)
        if fragment:
            glAttachShader(program, fragment)
        glLinkProgram(program)
        
        glDeleteShader(vertex)
        if fragment:
            glDeleteShader(fragment)
        
        if not program:
            raise RuntimeError("着色器编译失败")
        
        return program
    
    def _cache_uniform_locations(self):
        """缓存uniform位置"""
        self.uniform_locations = self.natural.get_uniform_locations()
    
    def use(self):
        """使用着色器"""
        glUseProgram(self.shader)
    
    def set_uniform(self, name: str, value: Any):
        """设置uniform值"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的uniform: {name}")
        
        self._set_uniform_value(location, value)
    
    def _set_uniform_value(self, location: int, value: Any):
        """设置uniform值"""
        if isinstance(value, (int, float)):
            glUniform1f(location, value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 3:
                glUniform3f(location, value[0], value[1], value[2])
            elif len(value) == 4:
                glUniform4f(location, value[0], value[1], value[2], value[3])
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                glUniform1f(location, value[0])
            elif value.ndim == 2:
                glUniform2f(location, value[0], value[1])
            elif value.ndim == 3:
                glUniform3f(location, value[0], value[1], value[2])
    
    def set_matrix4(self, name: str, matrix):
        """设置4x4矩阵"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的uniform: {name}")
        
        import glm
        glUniformMatrix4fv(location, 1, GL_FALSE, glm.value_ptr(matrix))
    
    def set_texture(self, name: str, texture_id: int):
        """设置纹理"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的纹理uniform: {name}")
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(self.shader, location), 0)
    
    def cleanup(self):
        """清理资源"""
        if self.shader:
            glDeleteProgram(self.shader)


class Camera:
    """相机控制器"""
    
    def __init__(self):
        # 设置在悬崖顶部，面向海洋
        self.position = glm.vec3(-30, 25, 0)
        self.front = glm.normalize(glm.vec3(1, -0.3, 0))
        self.up = glm.vec3(0, 1, 0)
        self.yaw = 0
        self.pitch = -20
        self.speed = 10.0
        self.mouse_sensitivity = 0.1
    
    def get_view_matrix(self):
        return glm.lookAt(self.position, self.position + self.front, self.up)
    
    def process_keyboard(self, direction, delta_time):
        velocity = self.speed * delta_time
        if direction == 'FORWARD':
            self.position += self.front * velocity
        if direction == 'BACKWARD':
            self.position -= self.front * velocity
        if direction == 'LEFT':
            self.position -= glm.normalize(glm.cross(self.front, self.up)) * velocity
        if direction == 'RIGHT':
            self.position += glm.normalize(glm.cross(self.front, self.up)) * velocity
        if direction == 'UP':
            self.position.y += velocity
        if direction == 'DOWN':
            self.position.y -= velocity
    
    def process_mouse(self, xoffset, yoffset):
        xoffset *= self.mouse_sensitivity
        yoffset *= self.mouse_sensitivity
        
        self.yaw += xoffset
        self.pitch -= yoffset
        
        self.pitch = max(-89, min(89, self.pitch))
        
        front = glm.vec3()
        front.x = np.cos(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        front.y = np.sin(np.radians(self.pitch))
        front.z = np.sin(np.radians(self.yaw)) * np.cos(np.radians(self.pitch))
        self.front = glm.normalize(front)


class CoastalSunsetDemo:
    """海岸日落演示主类"""
    
    def __init__(self):
        self.window = None
        self.width, self.height = 1280, 720
        
        self.natural = None
        self.coastal_mesh = None
        self.shader_wrapper = None  # 使用Natural自动着色器
        self.camera = Camera()
        
        self.show_fog = True
        self.time = 0
        
        self.keys = {}
        self.first_mouse = True
        self.last_x, self.last_y = self.width / 2, self.height / 2
    
    def init(self):
        if not glfw.init():
            return False
        
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        
        self.window = glfw.create_window(self.width, self.height, "Coastal Sunset Demo", None, None)
        if not self.window:
            glfw.terminate()
            return False
        
        glfw.make_context_current(self.window)
        glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_DISABLED)
        
        glfw.set_key_callback(self.window, self._key_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_callback)
        glfw.set_framebuffer_size_callback(self.window, self._resize_callback)
        
        glEnable(GL_DEPTH_TEST)
        glClearColor(0.9, 0.5, 0.3, 1.0)  # 日落橙色背景
        
        self._init_natural()
        self._init_terrain()
        
        # 初始化Natural自动着色器
        self.shader_wrapper = NaturalShaderWrapper(self.natural)
        self.shader_wrapper.init()
        
        return True
    
    def _init_natural(self):
        """初始化Natural系统"""
        self.natural = NaturalSystem(config={
            'enable_lighting': True,
            'enable_atmosphere': True,
            'enable_hydro_visual': True,
            'enable_wind': True,
            'enable_vegetation_growth': False,
            'enable_thermal_weathering': False,
            'enable_erosion': False,
            'god_ray_samples': 16,
            'god_ray_density': 0.5,
        })
        
        # 日落光照设置
        self.natural.set_sun_direction([0.8, -0.3, 0.5])  # 低角度夕阳
        self.natural.set_weather(rain_intensity=0.0, temperature=20.0)
        self.natural.set_wind([0.5, 0.0, 0.3], speed=3.0)
        self.natural.set_gpu_tier('medium')
    
    def _init_terrain(self):
        """初始化海岸地形"""
        print("生成海岸地形...")
        height_map = CoastalTerrainGenerator.generate_coastal_terrain(size=128)
        water_map = CoastalTerrainGenerator.generate_water_level(height_map, water_level=4.0)
        
        # 创建Natural地形表
        self.natural.create_terrain_table('terrain_main', 128, height_map)
        self.natural.set_terrain_data('terrain_main', 'water', water_map.flatten())
        
        self.coastal_mesh = CoastalMesh(height_map, water_map)
        
        print(f"地形范围: [{height_map.min():.1f}, {height_map.max():.1f}]")
        print(f"地形尺寸: {height_map.shape[0]}x{height_map.shape[1]}")
    
    def update(self, dt):
        self._process_input(dt)
        self.natural.update(dt)
        self.time += dt
        
        self.natural.set_camera_position([
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        ])
    
    def render(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # 使用Natural自动着色器
        self.shader_wrapper.use()
        
        model = glm.mat4(1.0)
        view = self.camera.get_view_matrix()
        projection = glm.perspective(glm.radians(60), self.width / self.height, 0.1, 500)
        
        # 设置基础矩阵
        self.shader_wrapper.set_matrix4('uModelMatrix', model)
        self.shader_wrapper.set_matrix4('uViewMatrix', view)
        self.shader_wrapper.set_matrix4('uProjectionMatrix', projection)
        
        # 从Natural获取光照数据
        lighting_data = self.natural.get_lighting_data()
        
        # 设置光照uniform
        if 'ao_map' in lighting_data and lighting_data['ao_map'] is not None:
            self.shader_wrapper.set_texture('uAOTexture', self._create_texture_from_array(lighting_data['ao_map']))
        
        if 'shadow_mask' in lighting_data and lighting_data['shadow_mask'] is not None:
            self.shader_wrapper.set_texture('uShadowTexture', self._create_texture_from_array(lighting_data['shadow_mask']))
        
        # 设置光照参数
        self.shader_wrapper.set_vec3('uLightDir', [0.8, -0.3, 0.5])
        self.shader_wrapper.set_vec3('uLightColor', [1.0, 0.7, 0.4])
        self.shader_wrapper.set_vec3('uAmbientColor', [0.3, 0.2, 0.3])
        
        # 从Natural获取大气数据
        atmosphere_data = self.natural.get_atmosphere_data()
        
        # 设置大气uniform
        if 'god_ray' in atmosphere_data and atmosphere_data['god_ray'] is not None:
            self.shader_wrapper.set_texture('uGodRayTexture', self._create_texture_from_array(atmosphere_data['god_ray']))
        
        fog_density = 0.015 if self.show_fog else 0.0
        self.shader_wrapper.set_float('uFogDensity', fog_density)
        
        # 设置雾色（日落橙）
        fog_color = [0.9, 0.5, 0.3] if self.show_fog else [0.9, 0.5, 0.3]
        self.shader_wrapper.set_vec3('uFogColor', fog_color)
        self.shader_wrapper.set_vec3('uCameraPos', [
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        ])
        self.shader_wrapper.set_float('uTime', self.time)
        
        # 渲染地形
        if self.coastal_mesh:
            self.coastal_mesh.render_terrain()
            self.coastal_mesh.render_water()
        
        glfw.swap_buffers(self.window)
    
    def _create_texture_from_array(self, data: np.ndarray):
        """从numpy数组创建OpenGL纹理"""
        if data is None:
            return 0
        
        # 确保数据在0-1范围
        data = np.clip(data, 0.0, 1.0)
        
        # 创建纹理
        texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        
        # 设置纹理参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        
        # 上传纹理数据
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, data.shape[1], data.shape[0], GL_FLOAT, data.tobytes())
        
        return texture_id
    
    def _process_input(self, dt):
        if glfw.get_key(self.window, glfw.KEY_W) == glfw.PRESS:
            self.camera.process_keyboard('FORWARD', dt)
        if glfw.get_key(self.window, glfw.KEY_S) == glfw.PRESS:
            self.camera.process_keyboard('BACKWARD', dt)
        if glfw.get_key(self.window, glfw.KEY_A) == glfw.PRESS:
            self.camera.process_keyboard('LEFT', dt)
        if glfw.get_key(self.window, glfw.KEY_D) == glfw.PRESS:
            self.camera.process_keyboard('RIGHT', dt)
        if glfw.get_key(self.window, glfw.KEY_Q) == glfw.PRESS:
            self.camera.process_keyboard('UP', dt)
        if glfw.get_key(self.window, glfw.KEY_E) == glfw.PRESS:
            self.camera.process_keyboard('DOWN', dt)
    
    def _key_callback(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_F:
                self.show_fog = not self.show_fog
                print(f"雾效: {'开启' if self.show_fog else '关闭'}")
            
            elif key == glfw.KEY_S:
                self._save_screenshot()
            
            elif key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
    
    def _mouse_callback(self, window, xpos, ypos):
        if self.first_mouse:
            self.last_x, self.last_y = xpos, ypos
            self.first_mouse = False
        
        xoffset = xpos - self.last_x
        yoffset = self.last_y - ypos
        self.last_x, self.last_y = xpos, ypos
        
        self.camera.process_mouse(xoffset, yoffset)
    
    def _resize_callback(self, window, width, height):
        self.width, self.height = width, height
        glViewport(0, 0, width, height)
    
    def _save_screenshot(self):
        import datetime
        from PIL import Image
        
        glPixelStorei(GL_PACK_ALIGNMENT, 1)
        data = glReadPixels(0, 0, self.width, self.height, GL_RGB, GL_UNSIGNED_BYTE)
        
        image = Image.frombytes("RGB", (self.width, self.height), data)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        
        filename = f"coastal_sunset_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        image.save(filename)
        print(f"截图已保存: {filename}")
    
    def run(self):
        print("\n" + "="*50)
        print("Coastal Sunset Demo")
        print("="*50)
        print("控制:")
        print("  WASD - 移动相机")
        print("  鼠标 - 旋转视角")
        print("  F - 切换雾效")
        print("  S - 截图保存")
        print("  ESC - 退出")
        print("="*50 + "\n")
        
        last_time = glfw.get_time()
        
        while not glfw.window_should_close(self.window):
            current_time = glfw.get_time()
            dt = current_time - last_time
            last_time = current_time
            
            self.update(dt)
            self.render()
            
            glfw.poll_events()
    
    def cleanup(self):
        if self.coastal_mesh:
            self.coastal_mesh.cleanup()
        glfw.terminate()


def main():
    demo = CoastalSunsetDemo()
    
    if demo.init():
        try:
            demo.run()
        finally:
            demo.cleanup()
    else:
        print("初始化失败")


if __name__ == '__main__':
    main()
