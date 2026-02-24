"""
前向渲染器 - 极简版本
只保留基础渲染，所有特效由 Natural 系统负责
"""

import logging
import os
import numpy as np
from typing import Optional, Dict, Any

from Engine.Math import Matrix4x4, Vector3
from OpenGL.GL import *


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
        self.use_instancing = True
        self.use_frustum_culling = True
        self.batching_threshold = 8
        self.max_draw_calls = 1000
        self.max_visible_lights = 8
        
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
        # 禁用背面剔除，确保双面渲染
        glDisable(GL_CULL_FACE)
        
        # 启用深度测试
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        
        # 确保是填充模式，不是线框模式
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
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
        
        for obj in all_nodes:
            if not hasattr(obj, 'mesh') or obj.mesh is None:
                continue
            
            mesh = obj.mesh
            
            # --- 子阶段 1: 网格更新 ---
            t_obj0 = self._get_current_time_ms()
            # 确保网格数据已上传到 GPU
            if mesh.is_dirty:
                mesh.update()
            self.render_stage_timings["obj_mesh_update"] += self._get_current_time_ms() - t_obj0
            
            # --- 子阶段 2: 设置 Uniforms ---
            t_obj0 = self._get_current_time_ms()
            if program > 0:
                # 模型矩阵
                if hasattr(obj, 'world_matrix'):
                    loc = glGetUniformLocation(program, "u_model")
                    if loc != -1:
                        data = np.array(obj.world_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                
                # 视图投影矩阵
                if view_proj_matrix is not None:
                    loc = glGetUniformLocation(program, "u_viewProj")
                    if loc != -1:
                        data = np.array(view_proj_matrix.data, dtype=np.float32)
                        glUniformMatrix4fv(loc, 1, GL_TRUE, data)
                
                # 法线矩阵（模型矩阵的逆转置的3x3部分）
                loc = glGetUniformLocation(program, "u_normalMatrix")
                if loc != -1:
                    # 从模型矩阵提取3x3旋转/缩放部分并计算法线矩阵
                    # 对于均匀缩放，法线矩阵就是旋转部分；对于非均匀缩放，需要逆转置
                    if hasattr(obj, 'world_matrix'):
                        wm = obj.world_matrix
                        # 提取3x3部分并计算逆转置
                        # 简化：对于地形，通常只需要处理旋转和均匀缩放
                        # 提取旋转部分（假设没有非均匀缩放或忽略缩放）
                        normal_matrix = np.array([
                            [wm[0,0], wm[0,1], wm[0,2]],
                            [wm[1,0], wm[1,1], wm[1,2]],
                            [wm[2,0], wm[2,1], wm[2,2]]
                        ], dtype=np.float32)
                        glUniformMatrix3fv(loc, 1, GL_TRUE, normal_matrix.flatten())
                    else:
                        identity = np.eye(3, dtype=np.float32)
                        glUniformMatrix3fv(loc, 1, GL_TRUE, identity)
            self.render_stage_timings["obj_set_uniforms"] += self._get_current_time_ms() - t_obj0
            
            # --- 子阶段 3: 材质 ---
            t_obj0 = self._get_current_time_ms()
            material = getattr(obj, 'material', None)
            if material:
                # 更新材质（这会触发纹理从CPU上传到GPU）
                t_mat0 = self._get_current_time_ms()
                material.update()
                self.render_stage_timings["obj_material_update"] += self._get_current_time_ms() - t_mat0
                # 绑定材质（设置纹理单元和渲染状态）
                t_mat0 = self._get_current_time_ms()
                material.bind()
                self.render_stage_timings["obj_material_bind"] += self._get_current_time_ms() - t_mat0

                # 基础颜色
                if program > 0:
                    loc = glGetUniformLocation(program, "u_baseColor")
                    if loc != -1:
                        if hasattr(material, 'base_color'):
                            c = material.base_color
                            glUniform3f(loc, c.x, c.y, c.z)
                        else:
                            glUniform3f(loc, 0.8, 0.8, 0.8)

                    # 高光颜色和光泽度
                    loc = glGetUniformLocation(program, "u_specularColor")
                    if loc != -1:
                        glUniform3f(loc, 0.2, 0.2, 0.2)

                    loc = glGetUniformLocation(program, "u_shininess")
                    if loc != -1:
                        if hasattr(material, 'roughness'):
                            # 粗糙度转换为光泽度 (roughness 0-1 -> shininess 1-128)
                            shininess = max(1.0, (1.0 - material.roughness) * 128.0)
                            glUniform1f(loc, shininess)
                        else:
                            glUniform1f(loc, 32.0)

                    # 透明度
                    loc = glGetUniformLocation(program, "u_alpha")
                    if loc != -1:
                        glUniform1f(loc, 1.0)

                    # 纹理
                    loc = glGetUniformLocation(program, "u_hasTexture")
                    if loc != -1:
                        # 检查材质是否有基础颜色纹理
                        has_texture = 0
                        if hasattr(material, 'base_color_texture') and material.base_color_texture is not None:
                            has_texture = 1
                        glUniform1i(loc, has_texture)

                        # 如果有纹理，绑定纹理到正确的纹理单元并设置采样器
                        if has_texture and hasattr(material, 'base_color_texture'):
                            # 绑定基础颜色纹理到纹理单元0
                            if material.base_color_texture:
                                material.base_color_texture.bind(0)

                                # 设置纹理采样器到着色器
                                tex_loc = glGetUniformLocation(program, "u_baseColorTexture")
                                if tex_loc != -1:
                                    glUniform1i(tex_loc, 0)  # 纹理单元0
                        else:
                            # 明确设置无纹理
                            glUniform1i(loc, 0)
                            
                    # 绑定岩石纹理（纹理单元1）
                    if hasattr(material, 'textures') and 'rock' in material.textures:
                        rock_tex = material.textures['rock']
                        if rock_tex:
                            rock_tex.bind(1)
                            tex_loc = glGetUniformLocation(program, "u_rockTexture")
                            if tex_loc != -1:
                                glUniform1i(tex_loc, 1)
                    
                    # 绑定雪地纹理（纹理单元2）
                    if hasattr(material, 'textures') and 'snow' in material.textures:
                        snow_tex = material.textures['snow']
                        if snow_tex:
                            snow_tex.bind(2)
                            tex_loc = glGetUniformLocation(program, "u_snowTexture")
                            if tex_loc != -1:
                                glUniform1i(tex_loc, 2)
                    
                    # 设置地形参数
                    terrain_min_loc = glGetUniformLocation(program, "u_terrainMinHeight")
                    if terrain_min_loc != -1:
                        glUniform1f(terrain_min_loc, -50.0)  # 假设最低高度
                    
                    terrain_max_loc = glGetUniformLocation(program, "u_terrainMaxHeight")
                    if terrain_max_loc != -1:
                        glUniform1f(terrain_max_loc, 300.0)  # 假设最高高度
                    
                    snow_line_loc = glGetUniformLocation(program, "u_snowLine")
                    if snow_line_loc != -1:
                        glUniform1f(snow_line_loc, 0.7)  # 雪线在70%高度
                    
                    # 设置emissive（自发光）参数
                    emissive_enabled = 0.0
                    emissive_color = (0.0, 0.0, 0.0)
                    emissive_strength = 0.0
                    
                    if hasattr(material, 'emissive'):
                        emissive_enabled = 1.0
                        emissive_color = (material.emissive.x, material.emissive.y, material.emissive.z)
                        if hasattr(material, 'emissive_strength'):
                            emissive_strength = material.emissive_strength
                    
                    emissive_enabled_loc = glGetUniformLocation(program, "u_emissiveEnabled")
                    if emissive_enabled_loc != -1:
                        glUniform1f(emissive_enabled_loc, emissive_enabled)
                    
                    emissive_color_loc = glGetUniformLocation(program, "u_emissiveColor")
                    if emissive_color_loc != -1:
                        glUniform3f(emissive_color_loc, *emissive_color)
                    
                    emissive_strength_loc = glGetUniformLocation(program, "u_emissiveStrength")
                    if emissive_strength_loc != -1:
                        glUniform1f(emissive_strength_loc, emissive_strength)
            else:
                # 默认材质
                if program > 0:
                    loc = glGetUniformLocation(program, "u_baseColor")
                    if loc != -1:
                        glUniform3f(loc, 0.8, 0.8, 0.8)
                    
                    loc = glGetUniformLocation(program, "u_specularColor")
                    if loc != -1:
                        glUniform3f(loc, 0.2, 0.2, 0.2)
                    
                    loc = glGetUniformLocation(program, "u_shininess")
                    if loc != -1:
                        glUniform1f(loc, 32.0)
                    
                    loc = glGetUniformLocation(program, "u_alpha")
                    if loc != -1:
                        glUniform1f(loc, 1.0)
                    
                    loc = glGetUniformLocation(program, "u_hasTexture")
                    if loc != -1:
                        # 默认材质没有纹理
                        glUniform1i(loc, 0)
            
            # --- 子阶段 4: 绘制网格 ---
            t_obj0 = self._get_current_time_ms()
            mesh.draw()
            self.render_stage_timings["obj_mesh_draw"] += self._get_current_time_ms() - t_obj0
            rendered_count += 1
        
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
        return stats
    
    def shutdown(self):
        """关闭渲染器"""
        pass

