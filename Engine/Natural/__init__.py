"""
Natural Module - HT_Engine Environment Simulation System

基于符号主义AI的环境模拟系统，实现"低资源、高画质"的自然环境渲染。

核心特性:
- Data-Driven: 一切皆数据，拒绝实时物理探测
- Zero Raytracing: 零光追，纯数学推导
- Cheat but Logical: 视觉欺骗，但逻辑自洽

架构:
- Core: 核心组件 (FactBase, RuleBase, InferenceEngine)
- Rules: 环境规则 (光照、大气、水文、地形演化等)
- Pipelines: 渲染管线 (NaturalRenderPipeline)
- Integration: 与渲染器的集成

使用示例:
    from Engine.Natural import NaturalSystem
    from Engine.Natural.Pipelines import NaturalRenderPipeline, RenderConfig
    
    # 创建渲染管线
    config = RenderConfig(
        resolution=(1920, 1080),
        target_fps=60,
        enable_bloom=True,
        enable_ssr=True,
        enable_path_trace=True
    )
    pipeline = NaturalRenderPipeline(config=config)
    
    # 或使用完整Natural系统
    natural = NaturalSystem(config={
        'enable_lighting': True,
        'enable_atmosphere': True,
        'enable_hydro_visual': True,
        'enable_render_pipeline': True,
        'enable_bloom': True,
        'enable_ssr': True,
        'enable_volumetric_light': True,
        'enable_path_trace': True,
    })
    
    # 创建地形
    natural.create_terrain_table('terrain_main', size=256, initial_height=height_map)
    
    # 设置环境参数
    natural.set_sun_direction([0.5, -1.0, 0.3])
    natural.set_wind([1.0, 0.0, 0.0], speed=5.0)
    natural.set_weather(rain_intensity=0.2, temperature=15.0)
    
    # 每帧更新
    natural.update(dt=0.016)
    
    # 获取渲染数据
    lighting = natural.get_lighting_data()
    atmosphere = natural.get_atmosphere_data()
    hydro = natural.get_hydro_data()
"""

from .NaturalSystem import NaturalSystem
from .Core.FactBase import FactBase
from .Core.RuleBase import Rule, RuleBase
from .Core.InferenceEngine import InferenceEngine
from .Pipelines import NaturalRenderPipeline, RenderConfig, RenderStats

__all__ = [
    'NaturalSystem',
    'FactBase',
    'Rule',
    'RuleBase',
    'InferenceEngine',
    'NaturalRenderPipeline',
    'RenderConfig',
    'RenderStats',
]

__version__ = '2.0.0'
