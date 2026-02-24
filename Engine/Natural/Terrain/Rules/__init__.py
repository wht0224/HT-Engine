"""
地形推理规则
"""

from .PrimaryRules import SlopeRule, AspectRule, CurvatureRule, HillshadeRule
from .SecondaryRules import RidgeRule, ValleyRule, CliffRule, PlateauRule, WaterBodyRule, DrainageRule
from .TertiaryRules import (
    RockDetailRule, ErosionRule, VegetationDensityRule, 
    SnowCoverageRule, TerrainSmoothingRule
)

__all__ = [
    # 一级规则
    'SlopeRule', 'AspectRule', 'CurvatureRule', 'HillshadeRule',
    # 二级规则
    'RidgeRule', 'ValleyRule', 'CliffRule', 'PlateauRule', 'WaterBodyRule', 'DrainageRule',
    # 三级规则
    'RockDetailRule', 'ErosionRule', 'VegetationDensityRule',
    'SnowCoverageRule', 'TerrainSmoothingRule'
]
