"""
程序化地形生成器 - 为HT_Engine Demo生成高质量地形
生成150万面以上的地形，带程序化解法线
"""

import numpy as np
import os

def generate_terrain_mesh(width=200, depth=200, height_scale=30.0, noise_scale=0.02):
    """
    生成程序化地形网格
    width, depth: 网格分辨率（越高面数越多）
    height_scale: 高度缩放
    noise_scale: 噪声频率
    """
    print(f"生成地形网格: {width}x{depth} = {width*depth*2} 个三角形...")
    
    # 使用Simplex噪声生成高度图
    try:
        from noise import snoise2
        has_noise = True
    except ImportError:
        has_noise = False
        print("警告: 未安装noise库，使用简化噪声")
    
    vertices = []
    indices = []
    normals = []
    uvs = []
    
    # 生成顶点
    for z in range(depth):
        for x in range(width):
            # 世界坐标
            world_x = (x - width/2) * 0.5
            world_z = (z - depth/2) * 0.5
            
            # 生成高度（多层噪声）
            if has_noise:
                # 主地形
                h1 = snoise2(x * noise_scale, z * noise_scale) * height_scale
                # 细节
                h2 = snoise2(x * noise_scale * 4, z * noise_scale * 4) * height_scale * 0.3
                # 大起伏
                h3 = snoise2(x * noise_scale * 0.3, z * noise_scale * 0.3) * height_scale * 2
                height = h1 + h2 + h3
            else:
                # 简化噪声
                import math
                height = (math.sin(x * noise_scale) * math.cos(z * noise_scale) * height_scale +
                         math.sin(x * noise_scale * 3) * math.cos(z * noise_scale * 3) * height_scale * 0.3)
            
            vertices.append([world_x, height, world_z])
            uvs.append([x / width, z / depth])
    
    # 生成索引（三角形）
    for z in range(depth - 1):
        for x in range(width - 1):
            # 当前顶点索引
            i0 = z * width + x
            i1 = z * width + (x + 1)
            i2 = (z + 1) * width + x
            i3 = (z + 1) * width + (x + 1)
            
            # 两个三角形组成一个四边形
            indices.extend([i0, i2, i1])  # 第一个三角形
            indices.extend([i1, i2, i3])  # 第二个三角形
    
    vertices = np.array(vertices, dtype=np.float32)
    indices = np.array(indices, dtype=np.uint32)
    uvs = np.array(uvs, dtype=np.float32)
    
    print(f"顶点数: {len(vertices)}")
    print(f"三角形数: {len(indices)//3}")
    
    return vertices, indices, uvs

def calculate_normals(vertices, indices):
    """计算法线（正确方向）"""
    print("计算法线...")
    
    normals = np.zeros_like(vertices)
    
    # 遍历所有三角形
    for i in range(0, len(indices), 3):
        i0, i1, i2 = indices[i], indices[i+1], indices[i+2]
        
        v0 = vertices[i0]
        v1 = vertices[i1]
        v2 = vertices[i2]
        
        # 计算两条边
        edge1 = v1 - v0
        edge2 = v2 - v0
        
        # 叉乘得到法线
        normal = np.cross(edge1, edge2)
        
        # 归一化
        length = np.linalg.norm(normal)
        if length > 0:
            normal = normal / length
        
        # 累加到顶点法线
        normals[i0] += normal
        normals[i1] += normal
        normals[i2] += normal
    
    # 归一化所有法线
    for i in range(len(normals)):
        length = np.linalg.norm(normals[i])
        if length > 0:
            normals[i] = normals[i] / length
    
    print(f"法线计算完成: {len(normals)} 个")
    return normals

def generate_procedural_texture(width=512, height=512):
    """生成程序化解纹理（基于高度和坡度）"""
    print(f"生成程序化解纹理: {width}x{height}...")
    
    texture = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 简单的噪声纹理
    for y in range(height):
        for x in range(width):
            # 基础颜色（草地）
            base_color = np.array([34, 139, 34], dtype=np.float32)  # 森林绿
            
            # 添加噪声变化
            import math
            noise = math.sin(x * 0.05) * math.cos(y * 0.05) * 20
            noise += math.sin(x * 0.1) * math.cos(y * 0.1) * 10
            
            color = base_color + noise
            color = np.clip(color, 0, 255).astype(np.uint8)
            
            texture[y, x] = color
    
    print("纹理生成完成")
    return texture

def export_to_glb(vertices, indices, normals, uvs, texture, output_path):
    """导出为GLB格式"""
    print(f"导出到: {output_path}")
    
    try:
        import pygltflib
        from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Material, Texture, Image
        from pygltflib.utils import gltf2_to_glb
        
        # 创建GLTF对象
        gltf = GLTF2()
        
        # 这里简化处理，实际应该完整构建GLTF结构
        # 为了快速生成，我们使用trimesh
        print("使用trimesh导出...")
        
    except ImportError:
        print("pygltflib未安装，尝试使用trimesh...")
    
    try:
        import trimesh
        
        # 创建mesh
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=indices.reshape(-1, 3),
            vertex_normals=normals,
            visual=trimesh.visual.TextureVisuals(
                uv=uvs,
                image=texture
            )
        )
        
        # 导出GLB
        mesh.export(output_path)
        print(f"导出成功: {output_path}")
        print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
        
    except ImportError:
        print("错误: 需要安装trimesh: pip install trimesh")
        return False
    
    return True

def main():
    """主函数"""
    print("="*60)
    print("HT_Engine 程序化地形生成器")
    print("="*60)
    
    # 生成地形参数（150万面以上）
    # 1000x1000 网格 = 200万个三角形
    width, depth = 1000, 1000
    
    # 生成网格
    vertices, indices, uvs = generate_terrain_mesh(
        width=width,
        depth=depth,
        height_scale=40.0,
        noise_scale=0.015
    )
    
    # 计算法线
    normals = calculate_normals(vertices, indices)
    
    # 生成纹理
    texture = generate_procedural_texture(1024, 1024)
    
    # 导出
    output_path = os.path.join(os.path.dirname(__file__), "output", "terrain_procedural.glb")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    success = export_to_glb(vertices, indices, normals, uvs, texture, output_path)
    
    if success:
        print("\n" + "="*60)
        print("地形生成完成!")
        print(f"输出文件: {output_path}")
        print(f"总面数: {len(indices)//3}")
        print("="*60)
    else:
        print("\n生成失败，请安装依赖:")
        print("pip install trimesh pillow noise")

if __name__ == "__main__":
    main()
