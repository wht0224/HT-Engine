"""
Natural光照规则演示

展示纯符号主义光照系统的能力：
- 软阴影的因果推导
- 光传播的间接光照
- 反射规则的效果

不需要高端GPU，只需要规则推导。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import sys
import os

# 添加引擎路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Engine.Natural.NaturalSystem import NaturalSystem


def create_test_terrain(size=64):
    """创建测试地形"""
    # 创建简单的山丘地形
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # 多个山丘
    height = (
        3 * np.exp(-(X**2 + Y**2) / 4) +  # 中心山丘
        2 * np.exp(-((X-3)**2 + (Y-2)**2) / 3) +  # 右侧山丘
        1.5 * np.exp(-((X+2)**2 + (Y-3)**2) / 2)   # 左上小山丘
    )
    
    return height


def create_material_maps(size=64):
    """创建材质属性图"""
    # 粗糙度：0=光滑（水），1=粗糙（岩石）
    roughness = np.ones((size, size)) * 0.8  # 默认粗糙
    
    # 湿润度：0=干燥，1=湿润
    wetness = np.zeros((size, size))
    
    # 创建一些水域（左下角）
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # 水域：左下角圆形区域
    water_mask = ((X - 0.2)**2 + (Y - 0.2)**2) < 0.15**2
    roughness[water_mask] = 0.1  # 水面光滑
    wetness[water_mask] = 1.0    # 水面湿润
    
    return roughness, wetness


def main():
    print("=" * 60)
    print("Natural光照规则演示")
    print("纯符号主义光照系统 - 无需高端GPU")
    print("=" * 60)
    
    # 创建Natural系统
    print("\n[1] 初始化Natural系统...")
    config = {
        'enable_advanced_lighting': True,
        'enable_lighting': True,
        'hard_shadow_threshold': 2.0,
        'soft_shadow_range': 15.0,
        'propagation_range': 10.0,
        'propagation_iterations': 3,
        'reflection_range': 20.0,
    }
    
    natural = NaturalSystem(config)
    print("✓ Natural系统初始化完成")
    
    # 创建测试地形
    print("\n[2] 创建测试地形...")
    size = 64
    height_map = create_test_terrain(size)
    roughness, wetness = create_material_maps(size)
    
    # 创建地形表
    natural.create_terrain_table("terrain_main", size, initial_height=height_map.flatten())
    
    # 设置材质属性
    natural.set_terrain_data("terrain_main", "roughness", roughness.flatten())
    natural.set_terrain_data("terrain_main", "wetness", wetness.flatten())
    
    print(f"✓ 创建{size}x{size}地形")
    
    # 设置光源
    print("\n[3] 设置光源...")
    sun_direction = np.array([0.5, -0.8, 0.3], dtype=np.float32)
    sun_direction = sun_direction / np.linalg.norm(sun_direction)
    natural.set_sun_direction(sun_direction)
    print(f"✓ 太阳方向: {sun_direction}")
    
    # 执行光照计算
    print("\n[4] 执行光照规则推导...")
    print("    - OcclusionRule: 计算软阴影...")
    print("    - ReflectionRule: 计算反射...")
    print("    - LightPropagationRule: 计算间接光照...")
    
    natural.update(0.016)  # 一帧更新
    
    print("✓ 光照推导完成")
    
    # 获取结果
    print("\n[5] 获取光照结果...")
    
    # 软阴影
    try:
        soft_shadows = natural.get_terrain_data("terrain_main", "shadow_soft").reshape((size, size))
        shadow_softness = natural.get_terrain_data("terrain_main", "shadow_softness").reshape((size, size))
        print("✓ 软阴影计算完成")
    except:
        soft_shadows = np.ones((size, size))
        shadow_softness = np.zeros((size, size))
        print("⚠ 软阴影数据未生成")
    
    # 间接光照
    try:
        indirect_light = natural.get_terrain_data("terrain_main", "indirect_light").reshape((size, size))
        print("✓ 间接光照计算完成")
    except:
        indirect_light = np.zeros((size, size))
        print("⚠ 间接光照数据未生成")
    
    # 反射
    try:
        reflection = natural.get_terrain_data("terrain_main", "reflection").reshape((size, size))
        print("✓ 反射计算完成")
    except:
        reflection = np.zeros((size, size))
        print("⚠ 反射数据未生成")
    
    # 可视化结果
    print("\n[6] 生成可视化...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Natural光照规则演示 - 纯符号主义光照系统', fontsize=16)
    
    # 1. 地形高度
    ax = axes[0, 0]
    im = ax.imshow(height_map, cmap='terrain', origin='lower')
    ax.set_title('地形高度')
    plt.colorbar(im, ax=ax)
    
    # 2. 材质属性
    ax = axes[0, 1]
    # 组合显示：粗糙度+湿润度
    material_vis = np.stack([
        roughness,  # R: 粗糙度
        wetness,    # G: 湿润度
        np.zeros_like(roughness)  # B: 0
    ], axis=-1)
    ax.imshow(material_vis, origin='lower')
    ax.set_title('材质属性 (R=粗糙度, G=湿润度)')
    
    # 3. 软阴影
    ax = axes[0, 2]
    im = ax.imshow(soft_shadows, cmap='gray', origin='lower', vmin=0, vmax=1)
    ax.set_title('软阴影 (1=亮, 0=暗)')
    plt.colorbar(im, ax=ax)
    
    # 4. 阴影软度
    ax = axes[1, 0]
    im = ax.imshow(shadow_softness, cmap='hot', origin='lower', vmin=0, vmax=1)
    ax.set_title('阴影软度 (0=硬阴影, 1=软阴影)')
    plt.colorbar(im, ax=ax)
    
    # 5. 间接光照
    ax = axes[1, 1]
    im = ax.imshow(indirect_light, cmap='viridis', origin='lower', vmin=0, vmax=1)
    ax.set_title('间接光照 (GI)')
    plt.colorbar(im, ax=ax)
    
    # 6. 反射
    ax = axes[1, 2]
    im = ax.imshow(reflection, cmap='plasma', origin='lower', vmin=0, vmax=1)
    ax.set_title('反射强度')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    # 保存图片
    output_path = 'natural_lighting_demo.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ 可视化结果已保存: {output_path}")
    
    # 显示解释信息
    print("\n[7] 光照解释示例:")
    try:
        explanations = natural.get_global("occlusion_explanations")
        if explanations:
            # 显示几个示例
            sample_points = list(explanations.keys())[:3]
            for point in sample_points:
                info = explanations[point]
                print(f"    位置{point}: {info['reason']}")
    except:
        print("    (无解释数据)")
    
    print("\n" + "=" * 60)
    print("演示完成！")
    print("=" * 60)
    print("\n核心特性:")
    print("  ✓ 纯符号主义 - 无需训练，规则即逻辑")
    print("  ✓ 因果可解释 - 每个光照结果都有原因")
    print("  ✓ 低端友好 - CPU推导，无需高端GPU")
    print("  ✓ 软阴影 - 基于遮挡物距离的自然推导")
    print("  ✓ 间接光照 - 光传播规则的GI效果")
    print("  ✓ 反射 - 表面材质决定反射行为")
    
    plt.show()


if __name__ == "__main__":
    main()
