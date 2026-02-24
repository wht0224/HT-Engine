#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gaea地形转换为GLB
将Gaea导出的高度图和颜色图转换为GLB模型
"""

import numpy as np
import struct
import json
import os

def read_exr(filepath):
    """读取EXR文件"""
    import OpenEXR
    import Imath
    
    exr_file = OpenEXR.InputFile(filepath)
    header = exr_file.header()
    
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    
    channels = header['channels']
    
    if 'R' in channels and 'G' in channels and 'B' in channels:
        r_str = exr_file.channel('R', pt)
        g_str = exr_file.channel('G', pt)
        b_str = exr_file.channel('B', pt)
        
        r = np.frombuffer(r_str, dtype=np.float32).reshape(height, width)
        g = np.frombuffer(g_str, dtype=np.float32).reshape(height, width)
        b = np.frombuffer(b_str, dtype=np.float32).reshape(height, width)
        
        img = np.stack([r, g, b], axis=-1)
    elif 'Y' in channels:
        y_str = exr_file.channel('Y', pt)
        y = np.frombuffer(y_str, dtype=np.float32).reshape(height, width)
        img = y
    else:
        first_channel = list(channels.keys())[0]
        channel_str = exr_file.channel(first_channel, pt)
        img = np.frombuffer(channel_str, dtype=np.float32).reshape(height, width)
    
    return img.astype(np.float32)

def load_heightmap(filepath):
    """加载高度图"""
    print(f"加载高度图: {filepath}")
    
    height = read_exr(filepath)
    
    if len(height.shape) == 3:
        height = height[:, :, 0]
    
    print(f"高度图尺寸: {height.shape}")
    print(f"高度范围: {height.min():.4f} - {height.max():.4f}")
    
    return height

def load_colormap(filepath):
    """加载颜色图"""
    print(f"加载颜色图: {filepath}")
    
    color = read_exr(filepath)
    
    if len(color.shape) == 2:
        color = np.stack([color, color, color], axis=-1)
    elif color.shape[2] == 4:
        color = color[:, :, :3]
    
    if color.max() <= 1.0:
        color = (color * 255).astype(np.uint8)
    else:
        color = color.astype(np.uint8)
    
    print(f"颜色图尺寸: {color.shape}")
    
    return color

def generate_terrain_mesh(heightmap, target_triangles=1000000):
    """生成地形网格"""
    h, w = heightmap.shape
    
    total_pixels = h * w
    max_triangles = (h - 1) * (w - 1) * 2
    
    if max_triangles <= target_triangles:
        step = 1
    else:
        ratio = target_triangles / max_triangles
        step = max(1, int(1.0 / np.sqrt(ratio)))
    
    new_h = (h - 1) // step + 1
    new_w = (w - 1) // step + 1
    
    print(f"原始分辨率: {w}x{h}")
    print(f"降采样步长: {step}")
    print(f"目标分辨率: {new_w}x{new_h}")
    
    sampled_h = heightmap[::step, ::step]
    if sampled_h.shape[0] != new_h or sampled_h.shape[1] != new_w:
        sampled_h = heightmap[:new_h*step:step, :new_w*step:step]
    
    actual_h, actual_w = sampled_h.shape
    
    height_scale = 400.0  # 调整高度缩放
    terrain_size = 1000.0
    
    x_scale = terrain_size / (actual_w - 1)
    z_scale = terrain_size / (actual_h - 1)
    
    vertices = []
    normals = []
    uvs = []
    indices = []
    
    h_min, h_max = sampled_h.min(), sampled_h.max()
    h_range = h_max - h_min if h_max > h_min else 1.0
    
    print(f"生成顶点...")
    for j in range(actual_h):
        for i in range(actual_w):
            x = i * x_scale - terrain_size / 2
            z = j * z_scale - terrain_size / 2
            y = (sampled_h[j, i] - h_min) / h_range * height_scale
            
            vertices.append([x, y, z])
            uvs.append([i / (actual_w - 1), j / (actual_h - 1)])
    
    print(f"计算法线...")
    for j in range(actual_h):
        for i in range(actual_w):
            idx = j * actual_w + i
            
            if i == 0:
                dx = vertices[idx + 1][0] - vertices[idx][0]
                dy = vertices[idx + 1][1] - vertices[idx][1]
            elif i == actual_w - 1:
                dx = vertices[idx][0] - vertices[idx - 1][0]
                dy = vertices[idx][1] - vertices[idx - 1][1]
            else:
                dx = vertices[idx + 1][0] - vertices[idx - 1][0]
                dy = vertices[idx + 1][1] - vertices[idx - 1][1]
            
            if j == 0:
                dz = vertices[idx + actual_w][2] - vertices[idx][2]
                dw = vertices[idx + actual_w][1] - vertices[idx][1]
            elif j == actual_h - 1:
                dz = vertices[idx][2] - vertices[idx - actual_w][2]
                dw = vertices[idx][1] - vertices[idx - actual_w][1]
            else:
                dz = vertices[idx + actual_w][2] - vertices[idx - actual_w][2]
                dw = vertices[idx + actual_w][1] - vertices[idx - actual_w][1]
            
            tx = np.array([dx, dy, 0])
            tz = np.array([0, dw, dz])
            n = np.cross(tx, tz)
            n_len = np.linalg.norm(n)
            if n_len > 0:
                n = n / n_len
            normals.append(n.tolist())
    
    print(f"生成三角形...")
    for j in range(actual_h - 1):
        for i in range(actual_w - 1):
            v0 = j * actual_w + i
            v1 = j * actual_w + i + 1
            v2 = (j + 1) * actual_w + i
            v3 = (j + 1) * actual_w + i + 1
            
            indices.extend([v0, v1, v2])
            indices.extend([v1, v3, v2])
    
    vertices = np.array(vertices, dtype=np.float32)
    normals = np.array(normals, dtype=np.float32)
    uvs = np.array(uvs, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    
    print(f"顶点数: {len(vertices)}")
    print(f"三角形数: {len(indices) // 3}")
    
    return vertices, normals, uvs, indices, actual_w, actual_h

def create_texture(colormap, target_width=2048):
    """创建纹理"""
    import cv2
    
    h, w = colormap.shape[:2]
    
    if w != target_width:
        scale = target_width / w
        new_h = int(h * scale)
        texture = cv2.resize(colormap, (target_width, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        texture = colormap
    
    texture = cv2.cvtColor(texture, cv2.COLOR_BGR2RGB)
    
    print(f"纹理尺寸: {texture.shape}")
    
    return texture

def save_glb(filepath, vertices, normals, uvs, indices, texture=None):
    """保存为GLB文件"""
    print(f"保存GLB: {filepath}")
    
    vertex_count = len(vertices)
    index_count = len(indices)
    
    position_bytes = vertices.tobytes()
    normal_bytes = normals.tobytes()
    uv_bytes = uvs.tobytes()
    index_bytes = indices.tobytes()
    
    position_min = vertices.min(axis=0).tolist()
    position_max = vertices.max(axis=0).tolist()
    
    gltf = {
        "asset": {"version": "2.0", "generator": "Gaea Terrain Converter"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "Terrain"}],
        "meshes": [{
            "primitives": [{
                "attributes": {
                    "POSITION": 0,
                    "NORMAL": 1,
                    "TEXCOORD_0": 2
                },
                "indices": 3,
                "material": 0
            }],
            "name": "TerrainMesh"
        }],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3",
                "min": position_min,
                "max": position_max
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC3"
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": vertex_count,
                "type": "VEC2"
            },
            {
                "bufferView": 3,
                "componentType": 5125,
                "count": index_count,
                "type": "SCALAR"
            }
        ],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(position_bytes)},
            {"buffer": 0, "byteOffset": len(position_bytes), "byteLength": len(normal_bytes)},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes), "byteLength": len(uv_bytes)},
            {"buffer": 0, "byteOffset": len(position_bytes) + len(normal_bytes) + len(uv_bytes), "byteLength": len(index_bytes)}
        ],
        "buffers": [{"byteLength": 0}]
    }
    
    buffer_data = position_bytes + normal_bytes + uv_bytes + index_bytes
    
    if texture is not None:
        texture_bytes = texture.tobytes()
        texture_width, texture_height = texture.shape[1], texture.shape[0]
        
        gltf["textures"] = [{"sampler": 0, "source": 0}]
        gltf["images"] = [{"bufferView": 4, "mimeType": "image/png"}]
        gltf["materials"] = [{
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.8
            },
            "name": "TerrainMaterial"
        }]
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}]
        
        from PIL import Image
        import io
        
        pil_img = Image.fromarray(texture, mode='RGB')
        png_buffer = io.BytesIO()
        pil_img.save(png_buffer, format='PNG')
        png_data = png_buffer.getvalue()
        texture_bytes = png_data
        
        gltf["bufferViews"].append({
            "buffer": 0,
            "byteOffset": len(buffer_data),
            "byteLength": len(texture_bytes)
        })
        
        buffer_data += texture_bytes
    
    gltf["buffers"][0]["byteLength"] = len(buffer_data)
    
    gltf_json = json.dumps(gltf, separators=(',', ':'))
    gltf_json_padded = gltf_json + ' ' * (4 - len(gltf_json) % 4) if len(gltf_json) % 4 != 0 else gltf_json
    
    json_bytes = gltf_json_padded.encode('utf-8')
    binary_bytes = buffer_data + b'\x00' * (4 - len(buffer_data) % 4) if len(buffer_data) % 4 != 0 else buffer_data
    
    header = struct.pack('<4sII', b'glTF', 2, 12 + 8 + len(json_bytes) + 8 + len(binary_bytes))
    json_chunk_header = struct.pack('<II', len(json_bytes), 0x4E4F534A)
    binary_chunk_header = struct.pack('<II', len(binary_bytes), 0x004E4942)
    
    with open(filepath, 'wb') as f:
        f.write(header)
        f.write(json_chunk_header)
        f.write(json_bytes)
        f.write(binary_chunk_header)
        f.write(binary_bytes)
    
    print(f"GLB保存成功: {filepath}")
    print(f"文件大小: {os.path.getsize(filepath) / 1024 / 1024:.2f} MB")

def convert_gaea_to_glb(heightmap_path, colormap_path, output_path, target_triangles=1000000):
    """转换Gaea地形为GLB"""
    print("=" * 60)
    print("Gaea地形转GLB转换器")
    print("=" * 60)
    
    heightmap = load_heightmap(heightmap_path)
    colormap = load_colormap(colormap_path)
    
    vertices, normals, uvs, indices, w, h = generate_terrain_mesh(heightmap, target_triangles)
    
    texture = create_texture(colormap)
    
    save_glb(output_path, vertices, normals, uvs, indices, texture)
    
    print("=" * 60)
    print("转换完成!")
    print("=" * 60)
    
    return {
        "vertices": len(vertices),
        "triangles": len(indices) // 3,
        "output_path": output_path
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="转换Gaea地形为GLB")
    parser.add_argument("--height", default="e:/新建文件夹 (3)/Gaea/Lake_Out.exr", help="高度图路径")
    parser.add_argument("--color", default="e:/新建文件夹 (3)/Gaea/SuperColor_Out.exr", help="颜色图路径")
    parser.add_argument("--output", default="e:/新建文件夹 (3)/output/terrain.glb", help="输出GLB路径")
    parser.add_argument("--triangles", type=int, default=1000000, help="目标三角形数")
    
    args = parser.parse_args()
    
    convert_gaea_to_glb(args.height, args.color, args.output, args.triangles)
