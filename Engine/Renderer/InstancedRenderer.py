# -*- coding: utf-8 -*-
"""
GPU 实例化渲染器
支持硬件实例化绘制，大幅降低 Draw Calls
"""

from OpenGL.GL import *
import numpy as np
from typing import List, Dict, Optional
import time


class InstancedRenderBatch:
    """
    实例化渲染批次
    管理一组共享相同网格和材质的实例
    """
    
    def __init__(self, mesh, material=None):
        self.mesh = mesh
        self.material = material
        self.instances = []  # 实例数据列表
        self.instance_buffer = None
        self.instance_vao = None
        self.instance_count = 0
        self.is_dirty = True
        
        # 实例数据格式：每个实例 4x4 矩阵 (16 个 float) + 可选的额外数据
        self.instance_stride = 16 * 4  # 16 floats * 4 bytes = 64 bytes
    
    def add_instance(self, model_matrix: np.ndarray):
        """添加实例"""
        self.instances.append(model_matrix.flatten())
        self.instance_count = len(self.instances)
        self.is_dirty = True
    
    def set_instances(self, matrices: np.ndarray):
        """
        批量设置实例数据
        
        Args:
            matrices: [N, 16] numpy array, dtype=np.float32
        """
        if matrices.dtype != np.float32:
            matrices = matrices.astype(np.float32)
        
        self.instances = matrices
        self.instance_count = matrices.shape[0]
        self.is_dirty = True
    
    def _ensure_buffer(self):
        """确保 GPU 缓冲区存在"""
        if not self.instances or len(self.instances) == 0:
            return
        
        # 转换为 numpy 数组
        if isinstance(self.instances, list):
            instance_data = np.array(self.instances, dtype=np.float32)
        else:
            instance_data = self.instances
        
        # 创建实例缓冲区
        if self.instance_buffer is None:
            self.instance_buffer = glGenBuffers(1)
        
        # 上传数据到 GPU
        glBindBuffer(GL_ARRAY_BUFFER, self.instance_buffer)
        glBufferData(GL_ARRAY_BUFFER, instance_data.nbytes, instance_data, GL_DYNAMIC_DRAW)
        
        self.is_dirty = False
    
    def bind(self):
        """绑定实例缓冲区"""
        if self.is_dirty:
            self._ensure_buffer()
        
        if self.instance_buffer is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.instance_buffer)
    
    def draw(self):
        """绘制实例"""
        if self.instance_count == 0:
            return
        
        # 绑定实例缓冲区
        self.bind()
        
        # 绘制实例
        if hasattr(self.mesh, 'index_count') and self.mesh.index_count > 0:
            # 索引绘制
            glDrawElementsInstanced(
                GL_TRIANGLES,
                self.mesh.index_count,
                GL_UNSIGNED_INT,
                None,
                self.instance_count
            )
        else:
            # 非索引绘制
            glDrawArraysInstanced(
                GL_TRIANGLES,
                0,
                getattr(self.mesh, 'vertex_count', 0),
                self.instance_count
            )
    
    def dispose(self):
        """释放资源"""
        if self.instance_buffer is not None:
            glDeleteBuffers(1, [self.instance_buffer])
            self.instance_buffer = None


class InstancedRenderer:
    """
    实例化渲染器
    管理多个实例化批次，自动合并相同网格/材质的物体
    """
    
    def __init__(self):
        self.batches: Dict[int, InstancedRenderBatch] = {}
        self.batch_count = 0
        self.total_instances = 0
        
        # 性能统计
        self.stats = {
            'batches': 0,
            'instances': 0,
            'draw_calls': 0,
            'preparation_time_ms': 0.0,
            'rendering_time_ms': 0.0
        }
    
    def create_batch(self, mesh, material=None):
        """
        创建或获取一个实例化批次
        
        Args:
            mesh: 网格对象
            material: 材质对象（可选）
            
        Returns:
            batch: InstancedRenderBatch 实例
        """
        # 创建批次键
        key = (id(mesh), id(material) if material else None)
        
        if key not in self.batches:
            self.batches[key] = InstancedRenderBatch(mesh, material)
            self.batch_count = len(self.batches)
        
        # 清除旧实例数据（每帧重新填充）
        self.batches[key].instances = []
        self.batches[key].instance_count = 0
        
        return self.batches[key]
    
    def prepare_batches(self, scene_objects: List) -> int:
        """
        准备实例化批次
        
        Args:
            scene_objects: 场景物体列表，每个物体包含：
                - mesh: 网格对象
                - material: 材质对象
                - world_matrix: 世界矩阵
        
        Returns:
            batch_count: 批次数量
        """
        start_time = time.perf_counter()
        
        # 清空旧批次
        self.batches.clear()
        self.batch_count = 0
        self.total_instances = 0
        
        # 按网格和材质分组
        groups = {}
        for obj in scene_objects:
            mesh = getattr(obj, 'mesh', None)
            material = getattr(obj, 'material', None)
            
            if mesh is None:
                continue
            
            # 创建分组键
            mesh_id = id(mesh)
            mat_id = hash(str(material)) if material else 0
            group_key = (mesh_id, mat_id)
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(obj)
        
        # 为每个分组创建实例化批次
        for (mesh_id, mat_id), objects in groups.items():
            if not objects:
                continue
            
            base_obj = objects[0]
            mesh = base_obj.mesh
            material = base_obj.material
            
            batch = InstancedRenderBatch(mesh, material)
            
            # 添加所有实例
            matrices = []
            for obj in objects:
                if hasattr(obj, 'world_matrix'):
                    mat = obj.world_matrix
                    if hasattr(mat, 'data'):
                        matrices.append(np.array(mat.data, dtype=np.float32))
                    else:
                        matrices.append(np.eye(4, dtype=np.float32))
                else:
                    matrices.append(np.eye(4, dtype=np.float32))
            
            if matrices:
                matrices_array = np.vstack(matrices)
                batch.set_instances(matrices_array)
                
                self.batches[(mesh_id, mat_id)] = batch
                self.batch_count += 1
                self.total_instances += len(objects)
        
        self.stats['batches'] = self.batch_count
        self.stats['instances'] = self.total_instances
        self.stats['preparation_time_ms'] = (time.perf_counter() - start_time) * 1000
        
        return self.batch_count
    
    def render(self):
        """渲染所有批次"""
        start_time = time.perf_counter()
        
        draw_calls = 0
        for batch in self.batches.values():
            if batch.instance_count > 0:
                batch.draw()
                draw_calls += 1
        
        self.stats['draw_calls'] = draw_calls
        self.stats['rendering_time_ms'] = (time.perf_counter() - start_time) * 1000
    
    def get_performance_stats(self) -> dict:
        """获取性能统计"""
        return self.stats.copy()
    
    def dispose(self):
        """释放所有资源"""
        for batch in self.batches.values():
            batch.dispose()
        self.batches.clear()


# 导出类
__all__ = ['InstancedRenderBatch', 'InstancedRenderer']
