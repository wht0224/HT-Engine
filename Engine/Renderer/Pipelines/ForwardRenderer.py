"""
前向渲染器 - 极简版本
支持 GPU 实例化和符号驱动批处理（SDDB）
"""

import logging
import os
import numpy as np
from typing import Optional, Dict, Any

from Engine.Math import Matrix4x4, Vector3
from OpenGL.GL import *

# 导入实例化渲染器
try:
    from Engine.Renderer.InstancedRenderer import InstancedRenderer
    INSTANCING_AVAILABLE = True
except ImportError:
    INSTANCING_AVAILABLE = False

# 导入 SSE2 批处理系统（Colony）
try:
    from Engine.Math.SSE2MathAdapter import ColonyRenderer, SSE2MathAdapter
    SDDB_AVAILABLE = True
    # 兼容旧代码
    SymbolDrivenBatcher = ColonyRenderer
except ImportError:
    SDDB_AVAILABLE = False


class ForwardRenderer:
    """
    前向渲染器 - 极简版
    
    只负责基础渲染：
    - 清除缓冲区
    - 设置投影和视图矩阵
    - 渲染不透明物体
    - 渲染透明物体
    
    所有特效（光照、阴影、后处理等）由 Natural 系统负责
    """
    
    def __init__(self, platform, resource_manager=None, shader_manager=None, config=None):
        """
        初始化前向渲染器
        
        Args:
            platform: 平台接口
            resource_manager: 资源管理器
            shader_manager: 着色器管理器
            config: 配置字典
        """
        self.platform = platform
        self.resource_manager = resource_manager
        self.shader_manager = shader_manager
        
        self.logger = logging.getLogger("ForwardRenderer")
        
        # 分辨率
        self.resolution = (1920, 1080)
        if platform and hasattr(platform, "width") and hasattr(platform, "height"):
            self.resolution = (int(platform.width or 1920), int(platform.height or 1080))
        
        # 清除颜色 - 蓝色天空
        self.clear_color = (0.3, 0.5, 0.8, 1.0)
        
        # 渲染特性 - 全部禁用，由 Natural 系统负责
        self.features = {
            "ssr": False,
            "shadow_mapping": False,
            "ambient_occlusion": False,
            "volumetric_lighting": False,
            "bloom": False,
            "motion_blur": False,
            "skybox": False,
            "fog": False
        }
        
        # 性能统计
        self.performance_stats = {
            "draw_calls": 0,
            "triangles": 0,
            "render_time_ms": 0.0,
            "shader_switches": 0,
            "texture_switches": 0,
            "visible_objects": 0,
            "culled_objects": 0
        }
        
        # 详细渲染阶段计时
        self.render_stage_timings = {
            "clear_buffers": 0.0,
            "setup_camera": 0.0,
            "setup_shader": 0.0,
            "setup_render_state": 0.0,
            "setup_global_uniforms": 0.0,
            "render_objects": 0.0,
            "total_objects_rendered": 0,
            "obj_mesh_update": 0.0,
            "obj_set_uniforms": 0.0,
            "obj_material_update": 0.0,
            "obj_material_bind": 0.0,
            "obj_mesh_draw": 0.0
        }
        
        # 着色器程序
        self.shader_programs = {}
        
        # 兼容旧代码
        self.use_instancing = INSTANCING_AVAILABLE
        self.use_frustum_culling = True
        self.batching_threshold = 8
        self.max_draw_calls = 1000
        self.max_visible_lights = 8
        
        # 实例化渲染器
        self.instanced_renderer = None
        if INSTANCING_AVAILABLE:
            self.instanced_renderer = InstancedRenderer()
        
        # 符号驱动批处理系统（Colony）
        self.sddb_batcher = None
        if SDDB_AVAILABLE:
            try:
                sse2_adapter = SSE2MathAdapter()
                self.sddb_batcher = sse2_adapter.get_sddb_batcher()  # 返回 ColonyRenderer
                self.logger.info("Colony 渲染系统已初始化（支持 GPU 实例化）")
            except Exception as e:
                self.logger.warning(f"Colony 初始化失败：{e}，将使用传统渲染")
        
        self.logger.info("极简前向渲染器初始化完成")
    
    def initialize(self, config=None):
        """
        初始化渲染器
        
        Args:
            config: 配置字典
        """
        # 尝试加载 basic 着色器
        if self.platform and hasattr(self.platform, "create_shader_program"):
            try:
                vs_path = os.path.join(os.path.dirname(__file__), "..", "..", "Shaders", "BasicVertexShader.glsl")
                fs_path = os.path.join(os.path.dirname(__file__), "..", "..", "Shaders", "BasicFragmentShader.glsl")
                
                vs_path = os.path.normpath(vs_path)
                fs_path = os.path.normpath(fs_path)
                
                if os.path.exists(vs_path) and os.path.exists(fs_path):
                    with open(vs_path, 'r', encoding='utf-8') as f:
                        vs_code = f.read()
                    with open(fs_path, 'r', encoding='utf-8') as f:
                        fs_code = f.read()
                    
                    prog_id = self.platform.create_shader_program(vs_code, fs_code)
                    self.shader_programs['basic'] = prog_id
                    self.logger.info(f"加载 basic 着色器: ID={prog_id}")
            except Exception as e:
                self.logger.warning(f"加载着色器失败: {e}")
    
    def _reset_performance_stats(self):
        """重置性能统计"""
        for key in self.performance_stats:
            self.performance_stats[key] = 0
    
    def _get_current_time_ms(self):
        """获取当前时间（毫秒）"""
        import time
        return time.time() * 1000.0
    
    def resize(self, width: int, height: int):
        """
        调整渲染分辨率
        
        Args:
            width: 新宽度
            height: 新高度
        """
        self.resolution = (width, height)
        self.logger.info(f"渲染分辨率调整为: {width}x{height}")
    
    def _clear_buffers(self):
        """清除缓冲区"""
        glClearColor(*self.clear_color)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    def _setup_projection(self, camera):
        """设置投影矩阵"""
        pass
    
    def _setup_modelview(self, camera):
        """设置模型视图矩阵"""
        pass
    
    def _setup_lights(self):
        """设置光源"""
        pass
    
    def _render_opaque_objects(self, scene):
        """渲染不透明物体"""
        pass
    
    def _render_transparent_objects(self, scene):
        """渲染透明物体"""
        pass
    
    def _render_skybox(self, scene):
        """渲染天空盒"""
        pass
    
    def render(self, scene, depth_buffer=None):
        """
        渲染场景
        
        Args:
            scene: 要渲染的场景
            depth_buffer: 可选的深度缓冲
            
        Returns:
            (颜色缓冲, 深度缓冲)
        """
        if self.platform and hasattr(self.platform, "width") and hasattr(self.platform, "height"):
            target_w = int(self.platform.width or 0)
            target_h = int(self.platform.height or 0)
            if target_w > 0 and target_h > 0 and (target_w, target_h) != tuple(self.resolution):
                self.resize(target_w, target_h)
        
        # 重置性能统计
        self._reset_performance_stats()
        
        # 重置渲染阶段计时
        for key in self.render_stage_timings:
            self.render_stage_timings[key] = 0.0
        
        # 记录渲染开始时间
        start_time = self._get_current_time_ms()
        
        # --- 阶段 1: 清除缓冲区 ---
        t0 = self._get_current_time_ms()
        self._clear_buffers()
        self.render_stage_timings["clear_buffers"] = self._get_current_time_ms() - t0
        
        # --- 阶段 2: 设置相机 ---
        t0 = self._get_current_time_ms()
        camera = getattr(scene, 'active_camera', None)
        
        # 缓存视图投影矩阵
        view_proj_matrix = None
        camera_pos = Vector3(0, 0, 0)
        if camera:
            if hasattr(camera, 'view_matrix') and hasattr(camera, 'projection_matrix'):
                view_proj_matrix = camera.projection_matrix * camera.view_matrix
            if hasattr(camera, 'position'):
                camera_pos = camera.position
        self.render_stage_timings["setup_camera"] = self._get_current_time_ms() - t0
        
        # --- 阶段 3: 设置着色器 ---
        t0 = self._get_current_time_ms()
        program = 0
        if 'basic' in self.shader_programs:
            prog_val = self.shader_programs['basic']
            if prog_val is not None and isinstance(prog_val, int):
                program = prog_val
                glUseProgram(program)
        self.render_stage_timings["setup_shader"] = self._get_current_time_ms() - t0
        
        # --- 阶段 4: 设置渲染状态 ---
        t0 = self._get_current_time_ms()
        
        # 关闭背面剔除 - 确保双面渲染
        glDisable(GL_CULL_FACE)
        
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)  # 使用 GL_LESS 而不是 GL_LEQUAL，避免 Z-fighting
        
        # 启用深度偏移 - 防止地形 Z-fighting（闪烁）
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)  # factor=1.0, units=1.0
        
        # 确保是填充模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        
        # 禁用混合
        glDisable(GL_BLEND)
        
        self.render_stage_timings["setup_render_state"] = self._get_current_time_ms() - t0
        
        # --- 阶段 5: 设置全局 Uniform ---
        t0 = self._get_current_time_ms()
        # 设置全局光照参数（只设置一次）
        if program > 0:
            # 环境光
            loc = glGetUniformLocation(program, "u_ambientLight")
            if loc != -1:
                glUniform3f(loc, 0.3, 0.3, 0.35)  # 柔和的环境光
            
            # 方向光（模拟太阳）
            loc = glGetUniformLocation(program, "u_directionalLightDir")
            if loc != -1:
                glUniform3f(loc, 0.5, -0.8, 0.3)  # 从斜上方照射
            
            loc = glGetUniformLocation(program, "u_directionalLightColor")
            if loc != -1:
                glUniform3f(loc, 1.0, 0.95, 0.8)  # 暖白色阳光
            
            loc = glGetUniformLocation(program, "u_directionalLightIntensity")
            if loc != -1:
                glUniform1f(loc, 1.0)
            
            # 相机位置
            loc = glGetUniformLocation(program, "u_cameraPos")
            if loc != -1:
                glUniform3f(loc, camera_pos.x, camera_pos.y, camera_pos.z)
            
            # 雾效
            loc = glGetUniformLocation(program, "u_fogEnabled")
            if loc != -1:
                glUniform1i(loc, 0)  # 默认禁用雾
            
            loc = glGetUniformLocation(program, "u_fogColor")
            if loc != -1:
                glUniform3f(loc, *self.clear_color[:3])
            
            loc = glGetUniformLocation(program, "u_fogDensity")
            if loc != -1:
                glUniform1f(loc, 0.001)
        self.render_stage_timings["setup_global_uniforms"] = self._get_current_time_ms() - t0
        
        # --- 阶段 6: 渲染物体 ---
        t0 = self._get_current_time_ms()
        # 获取可见节点
        all_nodes = getattr(scene, 'visible_nodes', [])
        
        # 如果 visible_nodes 为空（视锥剔关闭），直接获取所有节点
        if not all_nodes and hasattr(scene, 'get_all_nodes'):
            all_nodes = scene.get_all_nodes()
        
        rendered_count = 0
        
        # 简单状态批处理！按材质+网格分组，减少绑定/解绑
        rendered_count = self._render_objects_simple_batching(all_nodes, program, scene, view_proj_matrix)
        
        self.render_stage_timings["render_objects"] = self._get_current_time_ms() - t0
        self.render_stage_timings["total_objects_rendered"] = rendered_count
        
        # 更新性能统计
        self.performance_stats["draw_calls"] = rendered_count
        self.performance_stats["visible_objects"] = len(all_nodes)
        self.performance_stats["render_time_ms"] = self._get_current_time_ms() - start_time
        
        return (None, None)
    
    def enable_feature(self, feature_name, enabled):
        """启用/禁用特性（兼容旧代码，实际不做任何事）"""
        pass
    
    def get_performance_stats(self):
        """获取性能统计（包含详细渲染阶段）"""
        stats = self.performance_stats.copy()
        # 添加详细渲染阶段计时
        stats["render_stages"] = self.render_stage_timings.copy()
        
        # 添加 SDDB 统计
        if self.sddb_batcher:
            stats["sddb"] = self.sddb_batcher.get_performance_stats()
        
        # 添加实例化统计
        if self.instanced_renderer:
            stats["instancing"] = self.instanced_renderer.get_performance_stats()
        
        return stats
    
    def _render_with_sddb(self, all_nodes, program, scene, view_proj_matrix):
        """
        使用 SDDB（符号驱动动态批处理）渲染
        使用 SSE2/AVX2 加速矩阵变换，批量提交到 GPU
        
        Args:
            all_nodes: 物体列表
            program: 着色器程序
            scene: 场景
            view_proj_matrix: 视图投影矩阵
            
        Returns:
            渲染的物体数量
        """
        if not self.sddb_batcher:
            return self._render_objects_traditional(all_nodes, program, scene, view_proj_matrix)
        
        # 使用 SDDB 批处理系统
        rendered_count = self.sddb_batcher.render_batch(
            all_nodes,
            program,
            view_proj_matrix,
            self
        )
        
        return rendered_count
    
    def _render_static_terrain_and_sddb(self, all_nodes, program, scene, view_proj_matrix):
        """
        渲染静态地形 + SDDB 动态物体
        地形完全缓存到 GPU，第一帧上传后不再更新
        
        Args:
            all_nodes: 物体列表
            program: 着色器程序
            scene: 场景
            view_proj_matrix: 视图投影矩阵
            
        Returns:
            渲染的物体数量
        """
        from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform3f, glUniform1i, GL_TRUE
        import numpy as np
        
        rendered_count = 0
        
        # 1. 先渲染地形（完全静态，只渲染一次）
        terrain_nodes = [obj for obj in all_nodes if hasattr(obj, 'name') and 'Terrain' in str(obj.name)]
        
        if terrain_nodes:
            terrain = terrain_nodes[0]  # 只渲染第一个地形
            if hasattr(terrain, 'mesh') and terrain.mesh is not None:
                mesh = terrain.mesh
                material = getattr(terrain, 'material', None)
                
                # 设置着色器
                if program > 0:
                    glUseProgram(program)
                    
                    # 设置模型矩阵（关键修复！地形也需要模型矩阵！）
                    if hasattr(terrain, 'world_matrix'):
                        loc = glGetUniformLocation(program, "u_model")
                        if loc != -1:
                            data = np.array(terrain.world_matrix.data, dtype=np.float32)
                            glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                    
                    # 设置视图投影矩阵
                    if view_proj_matrix is not None:
                        loc = glGetUniformLocation(program, "u_viewProj")
                        if loc != -1:
                            data = np.array(view_proj_matrix.data, dtype=np.float32)
                            glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                    
                    # 设置地形材质和纹理
                    if material:
                        # 第一帧上传纹理到 GPU，之后就不再更新
                        if not hasattr(material, '_terrain_texture_uploaded'):
                            material.update()
                            material._terrain_texture_uploaded = True
                        
                        # 绑定纹理（每帧都要绑定，这是正确的！）
                        if material.base_color_texture:
                            material.base_color_texture.bind(0)
                        
                        # 设置纹理采样器
                        loc = glGetUniformLocation(program, "u_baseColorTexture")
                        if loc != -1:
                            glUniform1i(loc, 0)
                        
                        # 基础颜色
                        loc = glGetUniformLocation(program, "u_baseColor")
                        if loc != -1 and hasattr(material, 'base_color'):
                            c = material.base_color
                            glUniform3f(loc, c.x, c.y, c.z)
                        
                        # 设置是否有纹理
                        loc = glGetUniformLocation(program, "u_hasTexture")
                        if loc != -1:
                            has_texture = 1 if material.base_color_texture is not None else 0
                            glUniform1i(loc, has_texture)
                
                # 绘制地形（完全缓存到 GPU）
                mesh.draw()
                rendered_count += 1
        
        # 2. 使用 SDDB 渲染动态物体（不包括地形）
        dynamic_nodes = [obj for obj in all_nodes if not (hasattr(obj, 'name') and 'Terrain' in str(obj.name))]
        
        if dynamic_nodes and self.sddb_batcher:
            dynamic_count = self.sddb_batcher.render_batch(
                dynamic_nodes,
                program,
                view_proj_matrix,
                self
            )
            rendered_count += dynamic_count
        
        return rendered_count
    
    def _render_with_instancing(self, all_nodes, program, scene, view_proj_matrix):
        """
        使用 GPU 硬件实例化渲染
        
        Args:
            all_nodes: 物体列表
            program: 着色器程序
            scene: 场景
            view_proj_matrix: 视图投影矩阵
            
        Returns:
            渲染的物体数量
        """
        if not self.instanced_renderer:
            return self._render_objects_traditional(all_nodes, program, scene, view_proj_matrix)
        
        # 按网格和材质分组
        batches = {}
        for obj in all_nodes:
            if not hasattr(obj, 'mesh') or obj.mesh is None:
                continue
            
            mesh = obj.mesh
            material = getattr(obj, 'material', None)
            
            # 创建批次键
            key = (id(mesh), id(material) if material else None)
            
            if key not in batches:
                batches[key] = []
            
            # 获取世界矩阵
            if hasattr(obj, 'world_matrix'):
                model_matrix = np.array(obj.world_matrix.data, dtype=np.float32)
                batches[key].append((model_matrix, obj, mesh, material))
        
        # 渲染每个批次
        rendered_count = 0
        for key, instances in batches.items():
            if len(instances) == 0:
                continue
            
            mesh = instances[0][2]
            material = instances[0][3]
            
            # 使用实例化渲染器
            batch = self.instanced_renderer.create_batch(mesh, material)
            
            # 添加所有实例
            for model_matrix, obj, _, _ in instances:
                batch.add_instance(model_matrix)
            
            # 渲染批次
            if program > 0:
                glUseProgram(program)
                
                # 设置材质 uniform
                if material:
                    material.update()
                    material.bind()
                    
                    # 基础颜色
                    loc = glGetUniformLocation(program, "u_baseColor")
                    if loc != -1 and hasattr(material, 'base_color'):
                        c = material.base_color
                        glUniform3f(loc, c.x, c.y, c.z)
                
                # 设置视图投影矩阵
                if view_proj_matrix is not None:
                    loc = glGetUniformLocation(program, "u_viewProj")
                    if loc != -1:
                        data = np.array(view_proj_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
            
            # 绘制实例
            batch.draw()
            rendered_count += len(instances)
        
        return rendered_count
    
    def _render_objects_traditional(self, all_nodes, program, scene, view_proj_matrix):
        """
        传统渲染模式：逐物体渲染（fallback）
        
        Args:
            all_nodes: 物体列表
            program: 着色器程序
            scene: 场景
            view_proj_matrix: 视图投影矩阵
            
        Returns:
            渲染的物体数量
        """
        rendered_count = 0
        
        for obj in all_nodes:
            if not hasattr(obj, 'mesh') or obj.mesh is None:
                continue
            
            mesh = obj.mesh
            
            # 网格更新
            if mesh.is_dirty:
                mesh.update()
            
            # 设置 Uniforms
            if program > 0:
                if hasattr(obj, 'world_matrix'):
                    loc = glGetUniformLocation(program, "u_model")
                    if loc != -1:
                        data = np.array(obj.world_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                
                if view_proj_matrix is not None:
                    loc = glGetUniformLocation(program, "u_viewProj")
                    if loc != -1:
                        data = np.array(view_proj_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
            
            # 材质
            material = getattr(obj, 'material', None)
            if material:
                material.update()
                material.bind()
                
                # 设置材质 uniforms
                loc = glGetUniformLocation(program, "u_baseColor")
                if loc != -1 and hasattr(material, 'base_color'):
                    c = material.base_color
                    glUniform3f(loc, c.x, c.y, c.z)
                
                # 高光颜色和光泽度
                loc = glGetUniformLocation(program, "u_specularColor")
                if loc != -1:
                    glUniform3f(loc, 0.2, 0.2, 0.2)

                loc = glGetUniformLocation(program, "u_shininess")
                if loc != -1:
                    if hasattr(material, 'roughness'):
                        shininess = max(1.0, (1.0 - material.roughness) * 128.0)
                        glUniform1f(loc, shininess)
                    else:
                        glUniform1f(loc, 32.0)
                
                # 透明度
                loc = glGetUniformLocation(program, "u_alpha")
                if loc != -1:
                    glUniform1f(loc, getattr(material, 'alpha', 1.0))
                
                # 纹理相关
                loc = glGetUniformLocation(program, "u_hasTexture")
                if loc != -1:
                    has_texture = 1 if hasattr(material, 'base_color_texture') and material.base_color_texture is not None else 0
                    glUniform1i(loc, has_texture)
            
            # 绘制
            mesh.draw()
            rendered_count += 1
        
        return rendered_count
    
    def _render_objects_simple_batching(self, all_nodes, program, scene, view_proj_matrix):
        """
        最简单的状态批处理！按材质+网格分组，减少绑定/解绑
        
        Args:
            all_nodes: 物体列表
            program: 着色器程序
            scene: 场景
            view_proj_matrix: 视图投影矩阵
            
        Returns:
            渲染的物体数量
        """
        from OpenGL.GL import glUseProgram, glGetUniformLocation, glUniformMatrix4fv, glUniform3f, glUniform1i, glUniform1f, GL_TRUE
        import numpy as np
        
        rendered_count = 0
        
        if not all_nodes or len(all_nodes) == 0:
            return 0
        
        # 按材质和网格分组
        batches = {}
        
        for obj in all_nodes:
            if not hasattr(obj, 'mesh') or obj.mesh is None:
                continue
            
            mesh = obj.mesh
            material = getattr(obj, 'material', None)
            
            # 网格更新
            if mesh.is_dirty:
                mesh.update()
            
            # 批次键：(材质ID, 网格ID)
            material_id = id(material) if material else 0
            mesh_id = id(mesh)
            key = (material_id, mesh_id)
            
            if key not in batches:
                batches[key] = []
            batches[key].append(obj)
        
        # 设置全局 Uniform（只设置一次）
        if program > 0:
            glUseProgram(program)
            
            # 视图投影矩阵（只设置一次）
            if view_proj_matrix is not None:
                loc = glGetUniformLocation(program, "u_viewProj")
                if loc != -1:
                    data = np.array(view_proj_matrix.data, dtype=np.float32)
                    glUniformMatrix4fv(loc, 1, GL_TRUE, data)
        
        # 渲染每个批次
        for (material_id, mesh_id), objects in batches.items():
            if not objects:
                continue
            
            base_obj = objects[0]
            mesh = base_obj.mesh
            material = getattr(base_obj, 'material', None)
            
            # 1. 设置材质（只设置一次，每批次）
            if program > 0 and material:
                material.update()
                material.bind()
                
                # 材质基础颜色
                loc = glGetUniformLocation(program, "u_baseColor")
                if loc != -1 and hasattr(material, 'base_color'):
                    c = material.base_color
                    glUniform3f(loc, c.x, c.y, c.z)
                
                # 高光
                loc = glGetUniformLocation(program, "u_specularColor")
                if loc != -1:
                    glUniform3f(loc, 0.2, 0.2, 0.2)
                
                loc = glGetUniformLocation(program, "u_shininess")
                if loc != -1:
                    shininess = 32.0
                    if hasattr(material, 'roughness'):
                        shininess = max(1.0, (1.0 - material.roughness) * 128.0)
                    glUniform1f(loc, shininess)
                
                # 透明度
                loc = glGetUniformLocation(program, "u_alpha")
                if loc != -1:
                    glUniform1f(loc, getattr(material, 'alpha', 1.0))
                
                # 纹理
                loc = glGetUniformLocation(program, "u_hasTexture")
                if loc != -1:
                    has_texture = 1 if hasattr(material, 'base_color_texture') and material.base_color_texture is not None else 0
                    glUniform1i(loc, has_texture)
            
            # 2. 绑定网格 VAO（只绑定一次，每批次）
            mesh.bind()
            
            # 3. 渲染批次中的所有物体（只改模型矩阵）
            for obj in objects:
                if program > 0 and hasattr(obj, 'world_matrix'):
                    loc = glGetUniformLocation(program, "u_model")
                    if loc != -1:
                        data = np.array(obj.world_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                
                # 绘制
                mesh.draw()
                rendered_count += 1
            
            # 4. 解绑（可选，但好习惯）
            mesh.unbind()
        
        return rendered_count
    
    def shutdown(self):
        """关闭渲染器"""
        pass

