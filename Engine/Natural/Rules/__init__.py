"""
Natural Rules Module

基于符号主义AI的环境模拟规则系统。
所有规则遵循Event-Condition-Action模式，无状态设计。
"""

# 核心规则
from .WindRule import WindRule
from .GrazingRule import GrazingRule

# 地形演化规则
from .HydraulicErosionRule import HydraulicErosionRule
from .VegetationGrowthRule import VegetationGrowthRule
from .GpuVegetationRule import GpuVegetationRule
from .LightingRule import ThermalWeatheringRule
from .GpuWeatheringRule import GpuWeatheringRule
from .GpuVegetationConsumptionRule import GpuVegetationConsumptionRule

# 光照规则
from .LightingRule import LightingRule
from .GpuLightingRule import GpuLightingRule

# 高级光照规则（纯符号主义）
from .OcclusionRule import OcclusionRule
from .ReflectionRule import ReflectionRule
from .LightPropagationRule import LightPropagationRule

# GPU加速的高级光照规则
from .GpuOcclusionRule import GpuOcclusionRule
from .GpuReflectionRule import GpuReflectionRule
from .GpuLightPropagationRule import GpuLightPropagationRule

# 大气光学规则
from .AtmosphereRule import AtmosphereRule, FogRule
from .GpuAtmosphereRule import GpuAtmosphereRule
from .GpuFogRule import GpuFogRule

# 水文视觉规则
from .HydroVisualRule import HydroVisualRule, PlanarReflectionRule, WaterCausticsRule
from .GpuHydroRule import GpuHydroRule
from .OceanWaveRule import OceanWaveRule
from .SimplePhysicsRule import SimpleRigidBodyRule

# 渲染管线规则（新增）
from .GpuCullingRule import GpuCullingRule
from .GpuLodRule import GpuLodRule
from .GpuBloomRule import GpuBloomRule
from .GpuSsrRule import GpuSsrRule
from .GpuVolumetricLightRule import GpuVolumetricLightRule
from .GpuMotionBlurRule import GpuMotionBlurRule
from .GpuPathTraceRule import GpuPathTraceRule
from .GpuVolumetricCloudRule import GpuVolumetricCloudRule

__all__ = [
    # 基础规则
    'WindRule',
    'GrazingRule',
    
    # 地形演化
    'HydraulicErosionRule',
    'VegetationGrowthRule',
    'GpuVegetationRule',
    'ThermalWeatheringRule',
    'GpuWeatheringRule',
    'GpuVegetationConsumptionRule',
    
    # 光照（基础）
    'LightingRule',
    'GpuLightingRule',
    
    # 高级光照规则（纯符号主义）
    'OcclusionRule',
    'ReflectionRule',
    'LightPropagationRule',
    
    # GPU加速的高级光照规则
    'GpuOcclusionRule',
    'GpuReflectionRule',
    'GpuLightPropagationRule',
    
    # 大气
    'AtmosphereRule',
    'GpuAtmosphereRule',
    'FogRule',
    'GpuFogRule',
    
    # 水文
    'HydroVisualRule',
    'GpuHydroRule',
    'PlanarReflectionRule',
    'WaterCausticsRule',
    'OceanWaveRule',
    'SimpleRigidBodyRule',
    
    # 渲染管线规则（新增）
    'GpuCullingRule',
    'GpuLodRule',
    'GpuBloomRule',
    'GpuSsrRule',
    'GpuVolumetricLightRule',
    'GpuMotionBlurRule',
    'GpuPathTraceRule',
]
