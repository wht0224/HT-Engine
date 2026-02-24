# -*- coding: utf-8 -*-
"""
地形分块工具类
用于将单个大地形 Mesh 切分成多个 Chunk
"""

from Engine.Math import Vector3, Vector2
from Engine.Renderer.Resources.Mesh import Mesh


class TerrainChunkGenerator:
    """地形分块生成器"""
    
    def __init__(self, chunk_size=250.0, terrain_half_size=1000.0):
        """
        初始化地形分块生成器
        
        参数:
            chunk_size: 每个 Chunk 的大小（米）
            terrain_half_size: 地形半长（总大小 = terrain_half_size * 2）
        """
        self.chunk_size = chunk_size
        self.terrain_half_size = terrain_half_size
        self.terrain_size = terrain_half_size * 2.0
        self.grid_size = int(self.terrain_size / chunk_size)
    
    def split_mesh(self, terrain_mesh):
        """
        将地形 Mesh 切分成多个 Chunk
        
        参数:
            terrain_mesh: 原始地形 Mesh 对象
        
        返回:
            chunks: 字典，键为 (chunk_x, chunk_z)，值为 Chunk Mesh 对象
        """
        chunks = {}
        
        if not terrain_mesh.indices:
            print("警告：地形 Mesh 没有索引数据，无法分块")
            return chunks
        
        vertex_count = len(terrain_mesh.vertices)
        
        for i in range(0, len(terrain_mesh.indices), 3):
            if i + 2 >= len(terrain_mesh.indices):
                break
            
            idx0 = terrain_mesh.indices[i]
            idx1 = terrain_mesh.indices[i + 1]
            idx2 = terrain_mesh.indices[i + 2]
            
            if idx0 >= vertex_count or idx1 >= vertex_count or idx2 >= vertex_count:
                continue
            
            v0 = terrain_mesh.vertices[idx0]
            v1 = terrain_mesh.vertices[idx1]
            v2 = terrain_mesh.vertices[idx2]
            
            center_x = (v0.x + v1.x + v2.x) / 3.0
            center_z = (v0.z + v1.z + v2.z) / 3.0
            
            chunk_x = int((center_x + self.terrain_half_size) / self.chunk_size)
            chunk_z = int((center_z + self.terrain_half_size) / self.chunk_size)
            
            chunk_x = max(0, min(chunk_x, self.grid_size - 1))
            chunk_z = max(0, min(chunk_z, self.grid_size - 1))
            
            key = (chunk_x, chunk_z)
            
            if key not in chunks:
                chunks[key] = Mesh()
            
            self._add_triangle_to_chunk(chunks[key], terrain_mesh, idx0, idx1, idx2)
        
        for chunk_mesh in chunks.values():
            chunk_mesh._calculate_bounding_box()
            chunk_mesh.is_dirty = True
        
        return chunks
    
    def _add_triangle_to_chunk(self, chunk_mesh, source_mesh, idx0, idx1, idx2):
        """
        将三角形添加到 Chunk Mesh（带顶点去重）
        """
        if not hasattr(chunk_mesh, '_vertex_map'):
            chunk_mesh._vertex_map = {}
        
        new_indices = []
        
        for idx in [idx0, idx1, idx2]:
            vertex = source_mesh.vertices[idx]
            normal = source_mesh.normals[idx] if (source_mesh.normals and idx < len(source_mesh.normals)) else Vector3(0, 1, 0)
            uv = source_mesh.uvs[idx] if (source_mesh.uvs and idx < len(source_mesh.uvs)) else Vector2(0, 0)
            
            key = (round(vertex.x, 6), round(vertex.y, 6), round(vertex.z, 6))
            
            if key not in chunk_mesh._vertex_map:
                chunk_mesh.vertices.append(vertex)
                chunk_mesh.normals.append(normal)
                chunk_mesh.uvs.append(uv)
                new_idx = len(chunk_mesh.vertices) - 1
                chunk_mesh._vertex_map[key] = new_idx
                new_indices.append(new_idx)
            else:
                new_indices.append(chunk_mesh._vertex_map[key])
        
        chunk_mesh.indices.extend(new_indices)
