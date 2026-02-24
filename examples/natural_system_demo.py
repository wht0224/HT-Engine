"""
Natural System 演示脚本

展示如何使用Natural系统创建环境模拟。
"""

import numpy as np
import sys
import os

# 添加Engine到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from Engine.Natural import NaturalSystem


def generate_test_terrain(size: int = 128) -> np.ndarray:
    """生成测试地形"""
    x = np.linspace(-5, 5, size)
    y = np.linspace(-5, 5, size)
    X, Y = np.meshgrid(x, y)
    
    # 使用多个正弦波叠加创建地形
    height = (
        np.sin(X * 0.5) * np.cos(Y * 0.5) * 10 +
        np.sin(X * 1.5) * np.sin(Y * 1.2) * 3 +
        np.cos(X * 0.3 + Y * 0.4) * 5 +
        np.random.rand(size, size) * 0.5
    )
    
    return height.astype(np.float32)


def main():
    print("=" * 60)
    print("Natural System Demo")
    print("=" * 60)
    
    # 1. 创建Natural系统
    print("\n[1] 初始化Natural系统...")
    natural = NaturalSystem(config={
        'enable_lighting': True,
        'enable_atmosphere': True,
        'enable_hydro_visual': True,
        'enable_wind': True,
        'enable_vegetation_growth': True,
        'enable_thermal_weathering': True,
        'enable_erosion': False,  # 默认关闭，计算量较大
    })
    
    # 2. 生成并创建地形
    print("\n[2] 生成测试地形...")
    terrain_size = 128
    height_map = generate_test_terrain(terrain_size)
    natural.create_terrain_table('terrain_main', terrain_size, height_map)
    print(f"    地形尺寸: {terrain_size}x{terrain_size}")
    print(f"    高度范围: [{height_map.min():.2f}, {height_map.max():.2f}]")
    
    # 3. 设置环境参数
    print("\n[3] 设置环境参数...")
    natural.set_sun_direction([0.5, -1.0, 0.3])
    natural.set_wind([1.0, 0.0, 0.5], speed=5.0)
    natural.set_weather(rain_intensity=0.2, temperature=15.0)
    natural.set_camera_position([0.0, 50.0, 0.0])
    natural.set_gpu_tier('high')
    
    print("    太阳方向: [0.5, -1.0, 0.3]")
    print("    风速: 5.0 m/s")
    print("    降雨强度: 0.2")
    print("    温度: 15.0°C")
    
    # 4. 模拟多帧
    print("\n[4] 运行模拟...")
    num_frames = 10
    dt = 0.016  # 60 FPS
    
    for frame in range(num_frames):
        natural.update(dt)
        
        if frame % 5 == 0:
            print(f"    帧 {frame}/{num_frames}")
    
    # 5. 获取并显示结果
    print("\n[5] 获取渲染数据...")
    
    # 光照数据
    lighting = natural.get_lighting_data()
    print("\n    光照数据:")
    for key, value in lighting.items():
        if value is not None:
            if isinstance(value, np.ndarray):
                print(f"      {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"      {key}: {value}")
        else:
            print(f"      {key}: None")
    
    # 大气数据
    atmosphere = natural.get_atmosphere_data()
    print("\n    大气数据:")
    for key, value in atmosphere.items():
        if value is not None:
            if isinstance(value, np.ndarray):
                print(f"      {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"      {key}: {value}")
        else:
            print(f"      {key}: None")
    
    # 水文数据
    hydro = natural.get_hydro_data()
    print("\n    水文数据:")
    for key, value in hydro.items():
        if value is not None:
            if isinstance(value, np.ndarray):
                print(f"      {key}: shape={value.shape}, range=[{value.min():.3f}, {value.max():.3f}]")
            else:
                print(f"      {key}: {value}")
        else:
            print(f"      {key}: None")
    
    # 6. 获取地形数据
    print("\n[6] 地形数据列:")
    terrain_data = natural.get_terrain_data('terrain_main', 'height')
    if terrain_data is not None:
        print(f"    height: {terrain_data.shape}")
    
    ao_data = natural.get_terrain_data('terrain_main', 'ao_map')
    if ao_data is not None:
        print(f"    ao_map: {ao_data.shape}, range=[{ao_data.min():.3f}, {ao_data.max():.3f}]")
    
    shadow_data = natural.get_terrain_data('terrain_main', 'shadow_mask')
    if shadow_data is not None:
        print(f"    shadow_mask: {shadow_data.shape}, mean={shadow_data.mean():.3f}")
    
    wetness_data = natural.get_terrain_data('terrain_main', 'wetness')
    if wetness_data is not None:
        print(f"    wetness: {wetness_data.shape}, range=[{wetness_data.min():.3f}, {wetness_data.max():.3f}]")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
