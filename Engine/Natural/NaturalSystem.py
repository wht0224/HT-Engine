import numpy as np
import logging
import time
from typing import Dict, List, Optional, Any, TYPE_CHECKING

from .Core.InferenceEngine import InferenceEngine
from .Core.FactBase import FactBase
from .Core.RuleBase import Rule
from .Core.ShaderGenerator import ShaderGenerator
from .Core.GpuContextManager import GpuContextManager

if TYPE_CHECKING:
    from .Core.ShaderFragmentGenerator import DynamicShaderSystem

from .Rules import (
    WindRule,
    GrazingRule,
    HydraulicErosionRule,
    VegetationGrowthRule,
    GpuVegetationRule,
    ThermalWeatheringRule,
    GpuWeatheringRule,
    LightingRule,
    GpuLightingRule,
    AtmosphereRule,
    GpuAtmosphereRule,
    FogRule,
    GpuFogRule,
    HydroVisualRule,
    GpuHydroRule,
    PlanarReflectionRule,
    WaterCausticsRule,
    OceanWaveRule,
    GpuVegetationConsumptionRule,
    SimpleRigidBodyRule,
    # 新增光照规则（CPU版本）
    LightPropagationRule,
    ReflectionRule,
    OcclusionRule,
    # 新增光照规则（GPU版本）
    GpuOcclusionRule,
    GpuReflectionRule,
    GpuLightPropagationRule,
    # 新增渲染管线规则
    GpuCullingRule,
    GpuLodRule,
    GpuBloomRule,
    GpuSsrRule,
    GpuVolumetricLightRule,
    GpuMotionBlurRule,
    GpuPathTraceRule,
    # 新增高级效果规则
    GpuVolumetricCloudRule,
    # GpuAdvancedWaterRule,
)


class NaturalSystem:
    """
    Natural 系统管理器
    
    HT_Engine的环境模拟核心，基于符号主义AI架构。
    整合所有环境规则，提供统一的接口。
    
    核心特性:
    - Data-Driven: 一切皆数据，无实时物理探测
    - Zero Raytracing: 纯数学推导的光影计算
    - Cheat but Logical: 视觉欺骗但逻辑自洽
    
    规则优先级 (高到低):
    100: OcclusionRule (遮挡 - 基础阴影)
     95: ReflectionRule (反射 - 表面响应)
     90: LightPropagationRule (光传播 - 间接光照)
     85: LightingRule (基础光照 - AO和直接阴影)
     80: AtmosphereRule (大气)
     70: WindRule (物理层)
     65: HydroVisualRule (水文视觉)
     60: FogRule (雾效)
     55: PlanarReflectionRule (平面反射)
     50: GrazingRule, WaterCausticsRule (生物层, 焦散)
     30: VegetationGrowthRule (植被生长)
     25: ThermalWeatheringRule (热力风化)
     20: HydraulicErosionRule (水力侵蚀)
     
    新增规则优先级 (按执行顺序):
     60: GpuVolumetricCloudsRule (体积云)
     50: GpuAdvancedWaterRule (高级水面)
     40: GpuLightingRule (增强光照 - 最后执行)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化Natural系统
        
        Args:
            config: 配置字典，可包含各规则的参数
        """
        self.logger = logging.getLogger("Natural.System")
        self.config = config or {}

        profile = str(self.config.get("quality_profile") or "").strip().lower()
        if profile in ("lowspec", "igpu"):
            profile = "low"
        if profile == "low":
            self.config.setdefault("enable_advanced_lighting", False)
            self.config.setdefault("use_gpu_advanced_lighting", False)
            self.config.setdefault("sim_preset", "tourism")
            
            # Low profile 后处理质量配置 - 1x全分辨率（降采样对性能影响极小）
            self.config.setdefault("bloom_downsample", 1)
            self.config.setdefault("ssr_downsample", 1)
            self.config.setdefault("volumetric_downsample", 1)
            self.config.setdefault("path_trace_downsample", 1)
            self.config.setdefault("ssr_max_steps", 24)  # 提升SSR质量
            self.config.setdefault("volumetric_steps", 16)  # 提升体积光质量
            self.config.setdefault("path_trace_samples", 4)
            self.config.setdefault("path_trace_bounces", 2)

            default_rule_enabled = {
                "Hydro.PlanarReflection": False,
            }
            existing_enabled = self.config.get("sim_rule_enabled")
            if isinstance(existing_enabled, dict):
                merged_enabled = dict(default_rule_enabled)
                merged_enabled.update(existing_enabled)
                self.config["sim_rule_enabled"] = merged_enabled
            else:
                self.config["sim_rule_enabled"] = dict(default_rule_enabled)

            default_rule_intervals = {
                "Lighting.Propagation": 4,
                "Lighting.Occlusion": 4,
                "Lighting.Reflection": 4,
                "Atmosphere.Fog": 2,
                "Hydro.Visual": 2,
                "Evolution.Vegetation": 2,
                "Terrain.ThermalWeathering": 4,
                "Bio.Grazing": 2,
            }
            existing_intervals = self.config.get("sim_rule_intervals")
            if isinstance(existing_intervals, dict):
                merged_intervals = dict(default_rule_intervals)
                merged_intervals.update(existing_intervals)
                self.config["sim_rule_intervals"] = merged_intervals
            else:
                self.config["sim_rule_intervals"] = dict(default_rule_intervals)
        
        # 创建推理机
        self.engine = InferenceEngine()
        
        # 初始化 GPU 上下文管理器
        deferred_gpu = self.config.get('enable_lazy_gpu', False)
        self.gpu_manager = GpuContextManager(deferred=deferred_gpu)
        
        # 初始化动态着色器系统
        from .Core.ShaderFragmentGenerator import DynamicShaderSystem
        self.shader_system = DynamicShaderSystem()
        
        self._adaptive_enabled = bool(self.config.get("adaptive_quality", False))
        self._adaptive_threshold_fps = float(self.config.get("adaptive_quality_fps", 45.0))
        self._adaptive_high_size = int(self.config.get("adaptive_quality_high", 512))
        self._adaptive_low_size = int(self.config.get("adaptive_quality_low", 384))
        self._adaptive_hysteresis_fps = float(self.config.get("adaptive_quality_hysteresis_fps", 2.0))
        self._adaptive_ema_alpha = float(self.config.get("adaptive_quality_ema_alpha", 0.1))
        self._adaptive_cooldown_frames = int(self.config.get("adaptive_quality_cooldown_frames", 30))
        self._adaptive_warmup_frames = int(self.config.get("adaptive_quality_warmup_frames", 30))
        
        self._adaptive_fps_ema: Optional[float] = None
        self._adaptive_frames_since_switch = 0
        self._adaptive_quality_level: Optional[str] = None
        self._adaptive_frame_index = 0
        
        self._quality_table_for_level: Dict[str, str] = {}
        self._quality_rules: List[Any] = []
        
        # 自动检测GPU性能并应用设置
        if self.config.get('auto_detect_gpu', True):
            gpu_tier = self._detect_gpu_model()
            self._apply_gpu_performance_settings(gpu_tier)
        
        # 初始化所有规则
        self._init_rules()

        preset = str(self.config.get("sim_preset") or "full").strip().lower()
        rule_enabled = self.config.get("sim_rule_enabled")
        rule_intervals = self.config.get("sim_rule_intervals")

        if preset == "tourism":
            default_enabled = {
                "Bio.Grazing": False,
                "Interaction.VegetationConsumption": False,
                "Evolution.Vegetation": False,
                "Terrain.ThermalWeathering": False,
                "Hydro.PlanarReflection": False,
            }
            if isinstance(rule_enabled, dict):
                merged = dict(default_enabled)
                merged.update(rule_enabled)
                rule_enabled = merged
            else:
                rule_enabled = default_enabled

            if not isinstance(rule_intervals, dict):
                rule_intervals = {}

        if isinstance(rule_enabled, dict):
            self.engine.facts.set_global("rule_enabled", dict(rule_enabled))
        if isinstance(rule_intervals, dict):
            self.engine.facts.set_global("rule_intervals", dict(rule_intervals))
        
        self.logger.info("Natural System initialized")
    
    def _init_shared_textures(self, table_name: str):
        """
        预先创建共享纹理，避免因规则禁用导致纹理链断裂
        
        当某些规则被禁用时（如GpuWeatheringRule），它们创建的共享纹理
        将不可用，导致后续规则每帧都需要从CPU上传数据。
        此方法在规则初始化前预先创建这些纹理。
        """
        if not self.gpu_manager or not self.gpu_manager.context:
            return
        
        ctx = self.gpu_manager.context
        
        # 检查是否已有height纹理
        if self.gpu_manager.get_texture("height"):
            return
        
        # 尝试从FactBase获取地形数据
        try:
            height_data = self.engine.facts.get_column(table_name, "height")
            if height_data is None or len(height_data) == 0:
                return
            
            grid_len = len(height_data)
            size = int(np.sqrt(grid_len))
            if size * size != grid_len:
                return
            
            # 创建height纹理
            height_tex = ctx.texture((size, size), 1, dtype='f4')
            height_tex.write(height_data.astype(np.float32).tobytes())
            self.gpu_manager.register_texture("height", height_tex)
            
            # 尝试创建water纹理
            try:
                water_data = self.engine.facts.get_column(table_name, "water")
                if water_data is not None and len(water_data) == grid_len:
                    water_tex = ctx.texture((size, size), 1, dtype='f4')
                    water_tex.write(water_data.astype(np.float32).tobytes())
                    self.gpu_manager.register_texture("water", water_tex)
            except KeyError:
                pass
            
            self.logger.info(f"Pre-initialized shared textures: height, water ({size}x{size})")
            
        except Exception as e:
            self.logger.debug(f"Could not pre-initialize shared textures: {e}")
    
    def _init_rules(self):
        """初始化并注册所有规则"""
        base_table_name = self.config.get("adaptive_quality_source_table", "terrain_main")
        use_shared_textures = not self._adaptive_enabled
        
        # 预先创建共享纹理（避免因规则禁用导致纹理链断裂）
        self._init_shared_textures(base_table_name)
        
        # 注册着色器片段到动态着色器系统
        if self.config.get('enable_lighting', True):
            from .Core.ShaderFragmentGenerator import LIGHTING_FRAGMENT
            self.shader_system.register_fragment('LightingRule', LIGHTING_FRAGMENT)
        
        if self.config.get('enable_atmosphere', True):
            from .Core.ShaderFragmentGenerator import ATMOSPHERE_FRAGMENT
            self.shader_system.register_fragment('AtmosphereRule', ATMOSPHERE_FRAGMENT)
        
        if self.config.get('enable_hydro_visual', True):
            from .Core.ShaderFragmentGenerator import HYDRO_FRAGMENT
            self.shader_system.register_fragment('HydroVisualRule', HYDRO_FRAGMENT)
        
        if self.config.get('enable_hydro_visual', True):
            from .Core.ShaderFragmentGenerator import HYDRO_FRAGMENT
            self.shader_system.register_fragment('HydroVisualRule', HYDRO_FRAGMENT)
        
        # 新增：高级光照规则（纯符号主义）
        if self.config.get('enable_advanced_lighting', True):
            # 判断是否使用GPU版本
            use_gpu_advanced = self.config.get('use_gpu_advanced_lighting', True)
            
            if use_gpu_advanced and self.gpu_manager and self.gpu_manager.context:
                # GPU版本
                # 1. GPU遮挡规则 - 软阴影（优先级100）
                occlusion_rule = GpuOcclusionRule(
                    hard_shadow_threshold=self.config.get('hard_shadow_threshold', 2.0),
                    soft_shadow_range=self.config.get('soft_shadow_range', 15.0),
                    penumbra_scale=self.config.get('penumbra_scale', 1.0),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name,
                    use_shared_textures=use_shared_textures
                )
                self.engine.rules.register(occlusion_rule)
                self._quality_rules.append(occlusion_rule)
                self.logger.info("Registered GpuOcclusionRule (GPU Soft Shadows)")
                
                # 2. GPU反射规则（优先级95）
                reflection_rule = GpuReflectionRule(
                    smoothness_threshold=self.config.get('reflection_smoothness_threshold', 0.7),
                    reflection_range=self.config.get('reflection_range', 20.0),
                    max_bounces=self.config.get('reflection_max_bounces', 2),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name,
                    use_shared_textures=use_shared_textures
                )
                self.engine.rules.register(reflection_rule)
                self._quality_rules.append(reflection_rule)
                self.logger.info("Registered GpuReflectionRule")
                
                # 3. GPU光传播规则（优先级90）
                propagation_rule = GpuLightPropagationRule(
                    propagation_range=self.config.get('propagation_range', 10.0),
                    propagation_strength=self.config.get('propagation_strength', 0.5),
                    iterations=self.config.get('propagation_iterations', 3),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name
                )
                self.engine.rules.register(propagation_rule)
                self._quality_rules.append(propagation_rule)
                self.logger.info("Registered GpuLightPropagationRule (GPU Indirect GI)")
            else:
                # CPU版本
                # 1. 遮挡规则 - 软阴影的因果逻辑（优先级100）
                occlusion_rule = OcclusionRule(
                    hard_shadow_threshold=self.config.get('hard_shadow_threshold', 2.0),
                    soft_shadow_range=self.config.get('soft_shadow_range', 15.0),
                    penumbra_scale=self.config.get('penumbra_scale', 1.0)
                )
                self.engine.rules.register(occlusion_rule)
                self.logger.info("Registered OcclusionRule (Soft Shadows)")
                
                # 2. 反射规则 - 表面响应的逻辑（优先级95）
                reflection_rule = ReflectionRule(
                    smoothness_threshold=self.config.get('reflection_smoothness_threshold', 0.7),
                    reflection_range=self.config.get('reflection_range', 20.0),
                    max_bounces=self.config.get('reflection_max_bounces', 2)
                )
                self.engine.rules.register(reflection_rule)
                self.logger.info("Registered ReflectionRule")
                
                # 3. 光传播规则 - 间接光照的推导（优先级90）
                propagation_rule = LightPropagationRule(
                    propagation_range=self.config.get('propagation_range', 10.0),
                    propagation_strength=self.config.get('propagation_strength', 0.5),
                    iterations=self.config.get('propagation_iterations', 3)
                )
                self.engine.rules.register(propagation_rule)
                self.logger.info("Registered LightPropagationRule (Indirect GI)")
        
        # 基础光照规则（优先级85）
        if self.config.get('enable_lighting', True):
            if self.config.get('use_gpu_lighting', True):
                light_rule = GpuLightingRule(
                    ao_strength=self.config.get('ao_strength', 1.0),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name,
                    use_shared_textures=use_shared_textures
                )
                self._quality_rules.append(light_rule)
                self.logger.info("Using GPU Lighting Rule")
            else:
                light_rule = LightingRule(
                    shadow_step_size=self.config.get('shadow_step_size', 1.0),
                    max_shadow_steps=self.config.get('max_shadow_steps', 100),
                    ao_strength=self.config.get('ao_strength', 1.0),
                    shadow_strength=self.config.get('shadow_strength', 1.0)
                )
                self.logger.info("Using CPU Lighting Rule")
            self.engine.rules.register(light_rule)
            self.logger.debug("Registered LightingRule")
        
        # 大气规则
        if self.config.get('enable_atmosphere', True):
            if self.config.get('use_gpu_atmosphere', True):
                atmosphere_rule = GpuAtmosphereRule(
                    god_ray_samples=self.config.get('god_ray_samples', 32),
                    god_ray_density=self.config.get('god_ray_density', 0.5),
                    cloud_speed=self.config.get('cloud_speed', 1.0),
                    cloud_scale=self.config.get('cloud_scale', 0.01),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True)
                )
                self.logger.info("Using GPU Atmosphere Rule")
            else:
                atmosphere_rule = AtmosphereRule(
                    god_ray_samples=self.config.get('god_ray_samples', 32),
                    god_ray_density=self.config.get('god_ray_density', 0.5),
                    cloud_speed=self.config.get('cloud_speed', 1.0),
                    cloud_scale=self.config.get('cloud_scale', 0.01)
                )
                self.logger.info("Using CPU Atmosphere Rule")
            self.engine.rules.register(atmosphere_rule)
            self.logger.debug("Registered AtmosphereRule")
            
            if self.config.get('use_gpu_fog', True):
                fog_rule = GpuFogRule(
                    base_density=self.config.get('fog_base_density', 0.1),
                    humidity_factor=self.config.get('fog_humidity_factor', 0.5),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name,
                    use_shared_textures=use_shared_textures
                )
                self._quality_rules.append(fog_rule)
                self.logger.info("Using GPU Fog Rule")
            else:
                fog_rule = FogRule(
                    base_density=self.config.get('fog_base_density', 0.1),
                    humidity_factor=self.config.get('fog_humidity_factor', 0.5)
                )
            self.engine.rules.register(fog_rule)
            self.logger.debug("Registered FogRule")
        else:
            self.logger.info("Atmosphere rules disabled for performance")
        
        # 体积云规则（冰岛风格）
        if self.config.get('enable_volumetric_clouds', True):
            cloud_rule = GpuVolumetricCloudRule(
                cloud_height=self.config.get('cloud_height', 300.0),
                cloud_thickness=self.config.get('cloud_thickness', 150.0),
                cloud_coverage=self.config.get('cloud_coverage', 0.6),
                wind_speed=self.config.get('cloud_wind_speed', 15.0),
                context=self.gpu_manager.context,
                manager=self.gpu_manager,
                readback=self.config.get('enable_gpu_readback', False),
                use_shared_textures=use_shared_textures
            )
            self._quality_rules.append(cloud_rule)
            self.engine.rules.register(cloud_rule)
            self.logger.info("Registered GpuVolumetricCloudRule (Iceland Style)")
        
        # 增强规则：大气散射增强
        if self.config.get('enable_enhanced_atmosphere', False):
            from .Rules.GpuAtmosphereEnhancedRule import GpuAtmosphereEnhancedRule
            atmosphere_enhanced = GpuAtmosphereEnhancedRule(
                context=self.gpu_manager.context if self.gpu_manager else None,
                manager=self.gpu_manager,
                readback=self.config.get('enable_gpu_readback', False),
                quality=self.config.get('atmosphere_quality', 'medium')
            )
            self.engine.rules.register(atmosphere_enhanced)
            self._quality_rules.append(atmosphere_enhanced)
            self.logger.info("Registered GpuAtmosphereEnhancedRule (Rayleigh + Mie Scattering)")
        
        # 增强规则：屏幕空间阴影
        if self.config.get('enable_screen_space_shadows', False):
            from .Rules.GpuScreenSpaceShadowsRule import GpuScreenSpaceShadowsRule
            sss_rule = GpuScreenSpaceShadowsRule(
                context=self.gpu_manager.context if self.gpu_manager else None,
                manager=self.gpu_manager,
                readback=self.config.get('enable_gpu_readback', False),
                quality=self.config.get('sss_quality', 'medium')
            )
            self.engine.rules.register(sss_rule)
            self._quality_rules.append(sss_rule)
            self.logger.info("Registered GpuScreenSpaceShadowsRule (Contact Shadows)")
        
        # 增强规则：全局光照探针
        if self.config.get('enable_gi_probes', False):
            from .Rules.GpuGlobalIlluminationProbesRule import GpuGlobalIlluminationProbesRule
            gi_probes_rule = GpuGlobalIlluminationProbesRule(
                context=self.gpu_manager.context if self.gpu_manager else None,
                manager=self.gpu_manager,
                readback=self.config.get('enable_gpu_readback', False),
                quality=self.config.get('gi_quality', 'medium')
            )
            self.engine.rules.register(gi_probes_rule)
            self._quality_rules.append(gi_probes_rule)
            self.logger.info("Registered GpuGlobalIlluminationProbesRule (Probe-based GI)")
        
        # 水文视觉规则
        if self.config.get('enable_hydro_visual', True):
            if self.config.get('use_gpu_hydro', True):
                hydro_rule = GpuHydroRule(
                    wetness_smoothness=self.config.get('wetness_smoothness', 0.8),
                    reflection_threshold=self.config.get('reflection_threshold', 0.5),
                    caustics_speed=self.config.get('caustics_animation_speed', 1.0),
                    caustics_intensity=self.config.get('caustics_intensity', 0.5),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True),
                    table_name=base_table_name,
                    use_shared_textures=use_shared_textures
                )
                self.engine.rules.register(hydro_rule)
                self._quality_rules.append(hydro_rule)
                self.logger.info("Using GPU Hydro Rule")
            else:
                hydro_rule = HydroVisualRule(
                    wetness_smoothness=self.config.get('wetness_smoothness', 0.8),
                    reflection_threshold=self.config.get('reflection_threshold', 0.5),
                    min_lake_size=self.config.get('min_lake_size', 100)
                )
                self.engine.rules.register(hydro_rule)
                self.logger.debug("Registered HydroVisualRule")
                
                caustics_rule = WaterCausticsRule(
                    animation_speed=self.config.get('caustics_animation_speed', 1.0),
                    intensity=self.config.get('caustics_intensity', 0.5)
                )
                self.engine.rules.register(caustics_rule)
                self.logger.debug("Registered WaterCausticsRule")
            
            planar_reflection_rule = PlanarReflectionRule(
                activation_distance=self.config.get('reflection_activation_distance', 50.0)
            )
            self.engine.rules.register(planar_reflection_rule)
            self.logger.debug("Registered PlanarReflectionRule")
            
            # 海洋波浪规则
            ocean_rule = OceanWaveRule(
                foam_threshold=self.config.get('ocean_foam_threshold', 0.8)
            )
            self.engine.rules.register(ocean_rule)
            self.logger.debug("Registered OceanWaveRule")
        else:
            self.logger.info("Hydro rules disabled for performance")
        
        if self.config.get("enable_simple_physics", True):
            physics_rule = SimpleRigidBodyRule(
                tables=self.config.get("simple_physics_tables", None),
                gravity=self.config.get("simple_physics_gravity", None),
                drag=self.config.get("simple_physics_drag", 0.01),
                ground_y=self.config.get("simple_physics_ground_y", 0.0),
                enable_collisions=self.config.get("simple_physics_enable_collisions", True),
            )
            self.engine.rules.register(physics_rule)
            self.logger.debug("Registered SimpleRigidBodyRule")
        
        # 风力规则
        if self.config.get('enable_wind', True):
            wind_rule = WindRule()
            self.engine.rules.register(wind_rule)
            self.logger.debug("Registered WindRule")
        
        # 生物规则
        if self.config.get('enable_grazing', True):
            grazing_rule = GrazingRule()
            self.engine.rules.register(grazing_rule)
            self.logger.debug("Registered GrazingRule")
        
        # 植被生长规则
        if self.config.get('enable_vegetation_growth', True):
            if self.config.get('use_gpu_vegetation', True):
                vegetation_rule = GpuVegetationRule(
                    growth_rate=self.config.get('vegetation_growth_rate', 0.1),
                    death_rate=self.config.get('vegetation_death_rate', 0.05),
                    optimum_water=self.config.get('vegetation_optimum_water', 1.0),
                    max_slope=self.config.get('vegetation_max_slope', 1.5),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True)
                )
                self.engine.rules.register(vegetation_rule)
                self.logger.info("Using GPU Vegetation Rule")
            else:
                vegetation_rule = VegetationGrowthRule(
                    growth_rate=self.config.get('vegetation_growth_rate', 0.1),
                    death_rate=self.config.get('vegetation_death_rate', 0.05),
                    optimum_water=self.config.get('vegetation_optimum_water', 1.0),
                    max_slope=self.config.get('vegetation_max_slope', 1.5)
                )
                self.engine.rules.register(vegetation_rule)
                self.logger.debug("Registered VegetationGrowthRule")
        
        # 热力风化规则
        if self.config.get('enable_thermal_weathering', True):
            if self.config.get('use_gpu_weathering', True):
                weathering_rule = GpuWeatheringRule(
                    critical_angle=self.config.get('weathering_critical_angle', 0.8),
                    weathering_rate=self.config.get('weathering_rate', 0.1),
                    iterations=self.config.get('weathering_iterations', 5),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    readback=self.config.get('enable_gpu_readback', True)
                )
                self.engine.rules.register(weathering_rule)
                self.logger.info("Using GPU ThermalWeathering Rule")
            else:
                weathering_rule = ThermalWeatheringRule(
                    critical_angle=self.config.get('weathering_critical_angle', 0.8),
                    weathering_rate=self.config.get('weathering_rate', 0.1),
                    iterations=self.config.get('weathering_iterations', 5)
                )
                self.engine.rules.register(weathering_rule)
                self.logger.debug("Registered ThermalWeatheringRule")
        
        # 植被消耗规则 (Interaction)
        if self.config.get('enable_grazing', True): # Grazing implies consumption
            consumption_rule = GpuVegetationConsumptionRule(
                manager=self.gpu_manager
            )
            self.engine.rules.register(consumption_rule)
            self.logger.debug("Registered GpuVegetationConsumptionRule")
        
        # 水力侵蚀规则 (低优先级，每帧计算量较大)
        if self.config.get('enable_erosion', False):  # 默认关闭，需要显式启用
            erosion_rule = HydraulicErosionRule(
                rain_rate=self.config.get('erosion_rain_rate', 0.01),
                evaporation_rate=self.config.get('erosion_evaporation_rate', 0.05),
                solubility=self.config.get('erosion_solubility', 0.5),
                erosion_rate=self.config.get('erosion_erosion_rate', 0.1),
                deposition_rate=self.config.get('erosion_deposition_rate', 0.1),
                dt=self.config.get('erosion_dt', 0.1)
            )
            self.engine.rules.register(erosion_rule)
            self.logger.debug("Registered HydraulicErosionRule")
        
        # 新增：渲染管线规则（针对GTX 1650 Max-Q优化）
        # 默认禁用，需要显式启用以避免与现有测试冲突
        if self.config.get('enable_render_pipeline', False) and self.gpu_manager and self.gpu_manager.context:
            # GPU遮挡剔除规则（优先级110 - 最高）
            if self.config.get('enable_gpu_culling', True):
                culling_rule = GpuCullingRule(
                    max_objects=self.config.get('max_scene_objects', 100000),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('scene_table', 'scene_objects')
                )
                self.engine.rules.register(culling_rule)
                self.logger.info("Registered GpuCullingRule (Hi-Z Occlusion Culling)")
            
            # GPU LOD计算规则（优先级105）
            if self.config.get('enable_gpu_lod', True):
                lod_rule = GpuLodRule(
                    max_objects=self.config.get('max_scene_objects', 100000),
                    lod_levels=self.config.get('lod_levels', 5),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('scene_table', 'scene_objects')
                )
                self.engine.rules.register(lod_rule)
                self.logger.info("Registered GpuLodRule (GPU LOD Selection)")
            
            # GPU泛光效果规则（优先级70）
            if self.config.get('enable_bloom', True):
                bloom_rule = GpuBloomRule(
                    threshold=self.config.get('bloom_threshold', 0.8),
                    intensity=self.config.get('bloom_intensity', 0.5),
                    blur_radius=self.config.get('bloom_blur_radius', 4),
                    downsample_factor=self.config.get('bloom_downsample', 4),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('postprocess_table', 'postprocess')
                )
                self.engine.rules.register(bloom_rule)
                self._quality_rules.append(bloom_rule)
                self.logger.info("Registered GpuBloomRule")
            
            # GPU屏幕空间反射规则（优先级75）
            if self.config.get('enable_ssr', True):
                ssr_rule = GpuSsrRule(
                    max_steps=self.config.get('ssr_max_steps', 16),
                    binary_search_steps=self.config.get('ssr_binary_steps', 4),
                    intensity=self.config.get('ssr_intensity', 0.5),
                    downsample_factor=self.config.get('ssr_downsample', 2),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('postprocess_table', 'postprocess')
                )
                self.engine.rules.register(ssr_rule)
                self._quality_rules.append(ssr_rule)
                self.logger.info("Registered GpuSsrRule")
            
            # GPU体积光规则（优先级65）
            if self.config.get('enable_volumetric_light', True):
                volumetric_rule = GpuVolumetricLightRule(
                    step_count=self.config.get('volumetric_steps', 16),
                    intensity=self.config.get('volumetric_intensity', 0.5),
                    scattering=self.config.get('volumetric_scattering', 0.3),
                    downsample_factor=self.config.get('volumetric_downsample', 4),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('postprocess_table', 'postprocess')
                )
                self.engine.rules.register(volumetric_rule)
                self._quality_rules.append(volumetric_rule)
                self.logger.info("Registered GpuVolumetricLightRule")
            
            # GPU运动模糊规则（优先级60）
            if self.config.get('enable_motion_blur', False):
                motion_blur_rule = GpuMotionBlurRule(
                    intensity=self.config.get('motion_blur_intensity', 0.5),
                    sample_count=self.config.get('motion_blur_samples', 8),
                    max_velocity=self.config.get('motion_blur_max_velocity', 32),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('postprocess_table', 'postprocess')
                )
                self.engine.rules.register(motion_blur_rule)
                self.logger.info("Registered GpuMotionBlurRule")
            
            # GPU路径追踪模拟规则（优先级50 - 最后执行）
            if self.config.get('enable_path_trace', False):
                path_trace_rule = GpuPathTraceRule(
                    sample_count=self.config.get('path_trace_samples', 1),
                    max_bounces=self.config.get('path_trace_bounces', 2),
                    downsample_factor=self.config.get('path_trace_downsample', 4),
                    enable_denoising=self.config.get('path_trace_denoise', True),
                    context=self.gpu_manager.context,
                    manager=self.gpu_manager,
                    table_name=self.config.get('postprocess_table', 'postprocess')
                )
                self.engine.rules.register(path_trace_rule)
                self.logger.info("Registered GpuPathTraceRule (Path Tracing Simulation)")
        
        # ========== 新增高级效果规则（暂时禁用，避免初始化错误） ==========
        # 规则按优先级顺序注册: 体积云(60) -> 水面(50) -> 光照(40)
        
        # # 1. GPU体积云规则（优先级60）
        # if self.config.get('enable_volumetric_clouds', False):
        #     if self.gpu_manager and self.gpu_manager.context:
        #         volumetric_clouds_rule = GpuVolumetricCloudsRule(
        #             ray_march_steps=self.config.get('volumetric_clouds_ray_steps', 64),
        #             light_march_steps=self.config.get('volumetric_clouds_light_steps', 8),
        #             cloud_scale=self.config.get('volumetric_clouds_scale', 1.0),
        #             cloud_density=self.config.get('volumetric_clouds_density', 0.5),
        #             wind_speed=self.config.get('volumetric_clouds_wind', 1.0),
        #             downsample_factor=self.config.get('volumetric_clouds_downsample', 4),
        #             context=self.gpu_manager.context,
        #             manager=self.gpu_manager,
        #             table_name=self.config.get('atmosphere_table', 'atmosphere'),
        #             use_shared_textures=use_shared_textures
        #         )
        #         self.engine.rules.register(volumetric_clouds_rule)
        #         self._quality_rules.append(volumetric_clouds_rule)
        #         self.logger.info("Registered GpuVolumetricCloudsRule (Priority 60)")
        
        # # 2. GPU高级水面规则（优先级50）
        # if self.config.get('enable_advanced_water', False):
        #     if self.gpu_manager and self.gpu_manager.context:
        #         advanced_water_rule = GpuAdvancedWaterRule(
        #             num_waves=self.config.get('water_num_waves', 6),
        #             wave_amplitude=self.config.get('water_wave_amplitude', 0.8),
        #             wave_length=self.config.get('water_wave_length', 15.0),
        #             wave_speed=self.config.get('water_wave_speed', 1.2),
        #             ssr_max_steps=self.config.get('water_ssr_max_steps', 32),
        #             ssr_binary_search_steps=self.config.get('water_ssr_binary_steps', 6),
        #             ssr_intensity=self.config.get('water_ssr_intensity', 0.6),
        #             caustics_intensity=self.config.get('water_caustics_intensity', 1.0),
        #             caustics_scale=self.config.get('water_caustics_scale', 2.0),
        #             ripple_decay=self.config.get('water_ripple_decay', 0.95),
        #             max_ripples=self.config.get('water_max_ripples', 32),
        #             context=self.gpu_manager.context,
        #             manager=self.gpu_manager,
        #             readback=self.config.get('enable_gpu_readback', False),
        #             use_shared_textures=use_shared_textures
        #         )
        #         self.engine.rules.register(advanced_water_rule)
        #         self._quality_rules.append(advanced_water_rule)
        #         self.logger.info("Registered GpuAdvancedWaterRule (Priority 50)")
        
        # # 3. 增强的GPU光照规则（优先级40 - 最后执行）
        # if self.config.get('enable_enhanced_gpu_lighting', False):
        #     if self.gpu_manager and self.gpu_manager.context:
        #         enhanced_lighting_rule = GpuLightingRule(
        #             shadow_step_size=self.config.get('enhanced_shadow_step', 1.0),
        #             max_shadow_steps=self.config.get('enhanced_max_shadow_steps', 100),
        #             ao_strength=self.config.get('enhanced_ao_strength', 1.0),
        #             context=self.gpu_manager.context,
        #             manager=self.gpu_manager,
        #             readback=self.config.get('enable_gpu_readback', False),
        #             table_name=base_table_name,
        #             use_shared_textures=use_shared_textures,
        #             quality=self.config.get('enhanced_lighting_quality', 'high'),
        #             enable_gi=self.config.get('enhanced_enable_gi', True),
        #             enable_atmosphere=self.config.get('enhanced_enable_atmosphere', True),
        #             enable_lut=self.config.get('enhanced_enable_lut', True),
        #             enable_aces=self.config.get('enhanced_enable_aces', True)
        #         )
        #         self.engine.rules.register(enhanced_lighting_rule)
        #         self._quality_rules.append(enhanced_lighting_rule)
        #         self.logger.info("Registered GpuLightingRule (Enhanced, Priority 40)")
    
    def _resample_grid(self, src_2d: np.ndarray, dst_size: int) -> np.ndarray:
        src_h, src_w = src_2d.shape
        if src_h == dst_size and src_w == dst_size:
            return src_2d.astype(np.float32, copy=False)
        
        ys = np.linspace(0.0, float(src_h - 1), dst_size, dtype=np.float32)
        xs = np.linspace(0.0, float(src_w - 1), dst_size, dtype=np.float32)
        
        y0 = np.floor(ys).astype(np.int32)
        x0 = np.floor(xs).astype(np.int32)
        y1 = np.minimum(y0 + 1, src_h - 1)
        x1 = np.minimum(x0 + 1, src_w - 1)
        
        wy = (ys - y0).astype(np.float32)
        wx = (xs - x0).astype(np.float32)
        
        Ia = src_2d[y0[:, None], x0[None, :]]
        Ib = src_2d[y0[:, None], x1[None, :]]
        Ic = src_2d[y1[:, None], x0[None, :]]
        Id = src_2d[y1[:, None], x1[None, :]]
        
        wa = (1.0 - wy)[:, None] * (1.0 - wx)[None, :]
        wb = (1.0 - wy)[:, None] * wx[None, :]
        wc = wy[:, None] * (1.0 - wx)[None, :]
        wd = wy[:, None] * wx[None, :]
        
        out = Ia * wa + Ib * wb + Ic * wc + Id * wd
        return out.astype(np.float32, copy=False)
    
    def _apply_quality_level(self, level: str):
        table_name = self._quality_table_for_level.get(level)
        if not table_name:
            return
        
        for rule in self._quality_rules:
            if hasattr(rule, "table_name"):
                rule.table_name = table_name
        
        self._adaptive_quality_level = level
        self.engine.facts.set_global("natural_quality_level", level)
        self.engine.facts.set_global("natural_quality_table", table_name)
        self.logger.info(f"AdaptiveQuality switched to {level} ({table_name})")
    
    def update(self, dt: float):
        t0 = time.perf_counter()
        self.engine.step(dt)
        t1 = time.perf_counter()
        
        if not self._adaptive_enabled or not self._quality_table_for_level:
            return
        
        fps_override = self.engine.facts.get_global("frame_fps")
        if fps_override is not None:
            try:
                fps = float(fps_override)
            except Exception:
                fps = 0.0
        else:
            frame_ms = self.engine.facts.get_global("frame_ms")
            if frame_ms is not None:
                try:
                    ms = float(frame_ms)
                except Exception:
                    ms = 0.0
                fps = (1000.0 / ms) if ms > 0.0 else 0.0
            else:
                elapsed = t1 - t0
                fps = (1.0 / elapsed) if elapsed > 0.0 else 0.0
        
        self._adaptive_frame_index += 1
        if self._adaptive_frame_index <= self._adaptive_warmup_frames:
            return
        
        if fps <= 0.0:
            return
        if self._adaptive_fps_ema is None:
            self._adaptive_fps_ema = fps
        else:
            a = self._adaptive_ema_alpha
            self._adaptive_fps_ema = (1.0 - a) * self._adaptive_fps_ema + a * fps
        
        
        self.engine.facts.set_global("natural_fps_ema", float(self._adaptive_fps_ema))
        self._adaptive_frames_since_switch += 1
        
        if self._adaptive_frames_since_switch < self._adaptive_cooldown_frames:
            return
        
        thr = self._adaptive_threshold_fps
        hys = self._adaptive_hysteresis_fps
        
        want_high = self._adaptive_fps_ema >= (thr + hys * 0.5)
        want_low = self._adaptive_fps_ema < (thr - hys * 0.5)
        
        if want_high and self._adaptive_quality_level != "high" and "high" in self._quality_table_for_level:
            self._apply_quality_level("high")
            self._adaptive_frames_since_switch = 0
        elif want_low and self._adaptive_quality_level != "low" and "low" in self._quality_table_for_level:
            self._apply_quality_level("low")
            self._adaptive_frames_since_switch = 0
    
    def evaluate(self):
        """执行一帧的规则评估"""
        self.update(0.016)

    def set_global(self, key: str, value: Any):
        """
        设置全局事实
            key: 事实名称
            value: 事实值
        """
        self.engine.facts.set_global(key, value)
    
    def get_global(self, key: str) -> Any:
        """
        获取全局事实
        
        Args:
            key: 事实名称
            
        Returns:
            事实值
        """
        return self.engine.facts.get_global(key)
    
    def get_rule_timings(self) -> dict:
        """
        获取所有规则的执行耗时
        
        Returns:
            dict: {rule_name: duration_ms}
        """
        return getattr(self.engine, 'rule_timings', {})
    
    def _create_terrain_table(self, name: str, size: int, initial_height: Optional[np.ndarray]):
        """
        创建地形数据表
        
        Args:
            name: 表名
            size: 地形尺寸 (size x size)
            initial_height: 初始高度图 (可选)
        """
        capacity = size * size
        
        schema = {
            'height': np.float32,
            'water': np.float32,
            'sediment': np.float32,
            'slope': np.float32,
            'ao_map': np.float32,
            'shadow_mask': np.float32,
            'vegetation_density': np.float32,
            'wetness': np.float32,
            'roughness': np.float32,
            'specularity': np.float32,
            'water_regions': np.float32,
            'flow_direction_x': np.float32,
            'flow_direction_y': np.float32,
            'god_ray': np.float32,
            'cloud_shadow': np.float32,
            'atmosphere_intensity': np.float32,
            'reflection_intensity': np.float32,
            'caustics': np.float32,
        }
        
        self.engine.facts.create_table(name, capacity, schema)
        
        # 如果有初始高度，填充数据
        if initial_height is not None:
            # 接受1D数组（flattened）或2D数组
            if initial_height.ndim == 1:
                if initial_height.shape[0] != capacity:
                    raise ValueError(f"Initial height shape {initial_height.shape} doesn't match capacity {capacity}")
                self.engine.facts.set_count(name, capacity)
                self.engine.facts.set_column(name, 'height', initial_height)
            elif initial_height.ndim == 2:
                if initial_height.shape != (size, size):
                    raise ValueError(f"Initial height shape {initial_height.shape} doesn't match size {size}x{size}")
                self.engine.facts.set_count(name, capacity)
                self.engine.facts.set_column(name, 'height', initial_height.flatten())
            else:
                raise ValueError(f"Initial height must be 1D or 2D array, got {initial_height.ndim}D")
            
            # 初始化其他列为0
            for col in schema.keys():
                if col != 'height':
                    self.engine.facts.set_column(
                        name, col, 
                        np.zeros(capacity, dtype=schema[col])
                    )
        
        self.logger.info(f"Created terrain table '{name}' ({size}x{size})")
    
    def create_terrain_table(self, name: str, size: int, 
                            initial_height: Optional[np.ndarray] = None):
        self._create_terrain_table(name, size, initial_height)
        
        if not self._adaptive_enabled:
            return
        
        base_name = self.config.get("adaptive_quality_source_table", "terrain_main")
        if name != base_name:
            return
        
        high_size = min(self._adaptive_high_size, size)
        low_size = min(self._adaptive_low_size, size)
        
        levels = [("low", low_size), ("high", high_size)]
        created = {}
        
        for level, q_size in levels:
            if q_size <= 0:
                continue
            if q_size == size:
                created[level] = name
                continue
            
            q_name = f"{name}_q{q_size}"
            created[level] = q_name
            
            q_height = None
            if initial_height is not None:
                if initial_height.ndim == 1:
                    src = initial_height.reshape((size, size))
                else:
                    src = initial_height
                q_height = self._resample_grid(src.astype(np.float32, copy=False), q_size)
            
            self._create_terrain_table(q_name, q_size, q_height)
        
        if created:
            self._quality_table_for_level = created
            if self._adaptive_quality_level is None:
                self._apply_quality_level("high" if "high" in created else next(iter(created.keys())))
            else:
                self._apply_quality_level(self._adaptive_quality_level)
    
    def create_vegetation_table(self, name: str, capacity: int):
        """
        创建植被数据表
        
        Args:
            name: 表名
            capacity: 最大容量
        """
        schema = {
            'pos_x': np.float32,
            'pos_y': np.float32,
            'pos_z': np.float32,
            'stiffness': np.float32,
            'terrain_height': np.float32,
            'terrain_grad_x': np.float32,
            'terrain_grad_z': np.float32,
            'offset_x': np.float32,
            'offset_y': np.float32,
            'offset_z': np.float32,
            'scale': np.float32,
            'type_id': np.int32,
        }
        
        self.engine.facts.create_table(name, capacity, schema)
        self.logger.info(f"Created vegetation table '{name}' (capacity: {capacity})")
    
    def create_herbivore_table(self, name: str, capacity: int):
        """
        创建食草动物数据表
        
        Args:
            name: 表名
            capacity: 最大容量
        """
        schema = {
            'pos_x': np.float32,
            'pos_z': np.float32,
            'vel_x': np.float32,
            'vel_z': np.float32,
            'hunger': np.float32,
            'terrain_slope': np.float32,
            'terrain_grad_x': np.float32,
            'terrain_grad_z': np.float32,
            'heading': np.float32,
            'is_eating': np.float32,
        }
        
        self.engine.facts.create_table(name, capacity, schema)
        self.logger.info(f"Created herbivore table '{name}' (capacity: {capacity})")

    def create_physics_body_table(self, name: str, capacity: int):
        schema = {
            "pos_x": np.float32,
            "pos_y": np.float32,
            "pos_z": np.float32,
            "vel_x": np.float32,
            "vel_y": np.float32,
            "vel_z": np.float32,
            "radius": np.float32,
            "mass": np.float32,
            "restitution": np.float32,
        }
        self.engine.facts.create_table(name, capacity, schema)
        self.logger.info(f"Created physics body table '{name}' (capacity: {capacity})")
        
    def create_ocean_table(self, name: str, size: int):
        """
        创建海洋数据表
        
        Args:
            name: 表名
            size: 网格尺寸 (size x size)
        """
        count = size * size
        schema = {
            'base_x': np.float32,
            'base_z': np.float32,
            'pos_x': np.float32,
            'pos_y': np.float32,
            'pos_z': np.float32,
            'foam_mask': np.float32,
        }
        
        self.engine.facts.create_table(name, count, schema)
        self.engine.facts.set_count(name, count)
        
        # 初始化基础网格
        x = np.linspace(0, 100, size) # 默认 100米范围，后续可参数化
        z = np.linspace(0, 100, size)
        base_x, base_z = np.meshgrid(x, z)
        
        self.engine.facts.set_column(name, 'base_x', base_x.flatten())
        self.engine.facts.set_column(name, 'base_z', base_z.flatten())
        
        self.logger.info(f"Created ocean table '{name}' ({size}x{size})")
    
    def get_terrain_data(self, table_name: str, column: str) -> Optional[np.ndarray]:
        """
        获取地形数据
        
        Args:
            table_name: 表名
            column: 列名
            
        Returns:
            数据数组，如果不存在返回None
        """
        try:
            return self.engine.facts.get_column(table_name, column)
        except KeyError:
            return None
    
    def set_terrain_data(self, table_name: str, column: str, data: np.ndarray):
        """
        设置地形数据
        
        Args:
            table_name: 表名
            column: 列名
            data: 数据数组
        """
        self.engine.facts.set_column(table_name, column, data)
    
    def get_lighting_data(self) -> Dict[str, Any]:
        """
        获取光照相关数据
        
        Returns:
            包含AO图、阴影遮罩等的字典
        """
        return {
            'ao_map': self.get_global('lighting_ao_map'),
            'shadow_mask': self.get_global('lighting_shadow_mask'),
            'slope': self.get_global('lighting_slope'),
        }
    
    def get_atmosphere_data(self) -> Dict[str, Any]:
        """
        获取大气相关数据
        
        Returns:
            包含丁达尔效应、云影等的字典
        """
        return {
            'god_ray': self.get_global('atmosphere_god_ray'),
            'cloud_shadow': self.get_global('atmosphere_cloud_shadow'),
            'fog_density': self.get_global('fog_density'),
            'fog_density_map': self.get_global('fog_density_map'),
        }
    
    def get_hydro_data(self) -> Dict[str, Any]:
        """
        获取水文相关数据
        
        Returns:
            包含湿润度、粗糙度、反射等的字典
        """
        return {
            'wetness': self.get_global('hydro_wetness_map'),
            'roughness': self.get_global('hydro_roughness_map'),
            'water_regions': self.get_global('hydro_water_regions'),
            'caustics': self.get_global('hydro_caustics'),
            'planar_reflection_enabled': self.get_global('planar_reflection_enabled'),
            'planar_reflection_quality': self.get_global('planar_reflection_quality'),
        }
    
    def get_shader(self) -> 'DynamicShaderSystem':
        """
        获取Natural动态生成的着色器系统
        
        Returns:
            DynamicShaderSystem实例，可动态编译着色器
        """
        return self.shader_system
    
    def set_sun_direction(self, direction: np.ndarray):
        """
        设置太阳方向
        
        Args:
            direction: 太阳方向向量 (3,)
        """
        direction = np.array(direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-5)
        self.set_global('sun_direction', direction)
    
    def set_wind(self, direction: np.ndarray, speed: float):
        """
        设置风
        
        Args:
            direction: 风向向量 (3,)
            speed: 风速
        """
        direction = np.array(direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-5)
        self.set_global('wind_direction', direction)
        self.set_global('wind_speed', float(speed))
    
    def set_weather(self, rain_intensity: float = 0.0, 
                   temperature: float = 20.0):
        """
        设置天气
        
        Args:
            rain_intensity: 降雨强度 (0-1)
            temperature: 温度 (摄氏度)
        """
        self.set_global('rain_intensity', float(rain_intensity))
        self.set_global('temperature', float(temperature))
    
    def set_camera_position(self, position: np.ndarray):
        """
        设置相机位置
        
        Args:
            position: 相机位置 (3,)
        """
        self.set_global('camera_position', np.array(position, dtype=np.float32))
    
    def set_gpu_tier(self, tier: str):
        """
        设置GPU性能等级
        
        Args:
            tier: 性能等级 ('high', 'medium', 'low', 'minimal')
        """
        self.set_global('gpu_tier', tier)
    
    def _detect_gpu_model(self) -> str:
        """
        检测GPU型号并返回性能等级
        
        Returns:
            性能等级: 'high', 'medium', 'low', 'minimal'
        """
        try:
            import moderngl
            
            # 创建临时上下文获取GPU信息
            ctx = None
            if self.gpu_manager and self.gpu_manager.context:
                ctx = self.gpu_manager.context
            else:
                try:
                    ctx = moderngl.create_context(standalone=True)
                except Exception:
                    return 'medium'
            
            # 获取GPU渲染器信息
            renderer = ctx.info.get('GL_RENDERER', '').lower()
            vendor = ctx.info.get('GL_VENDOR', '').lower()
            
            self.logger.info(f"Detected GPU: {renderer} ({vendor})")
            
            # 检测低端GPU (GTX 750 Ti, 集成显卡等)
            low_end_keywords = [
                'gtx 750', 'gtx 650', 'gt 1030', 'gt 710', 'gt 730',
                'intel', 'intel(r) uhd', 'intel hd', 'iris xe',
                'amd radeon r5', 'amd radeon r7', 'vega 3', 'vega 8'
            ]
            
            # 检测高端GPU
            high_end_keywords = [
                'rtx 40', 'rtx 3090', 'rtx 3080', 'rtx 3070', 'rtx 3060 ti',
                'rtx 2080', 'rtx 2070 super', 'gtx 1080 ti', 'gtx 1080',
                'rx 7900', 'rx 7800', 'rx 6900', 'rx 6800'
            ]
            
            # 检测中端GPU
            medium_end_keywords = [
                'gtx 1660', 'gtx 1650', 'gtx 1060', 'gtx 1070',
                'rtx 3060', 'rtx 3050', 'rx 6600', 'rx 6700',
                'rx 580', 'rx 570', 'rx 5600'
            ]
            
            # 检查GPU型号
            for keyword in low_end_keywords:
                if keyword in renderer:
                    self.logger.info(f"Detected low-end GPU, using minimal settings")
                    return 'minimal'
            
            for keyword in high_end_keywords:
                if keyword in renderer:
                    self.logger.info(f"Detected high-end GPU, enabling all effects")
                    return 'high'
            
            for keyword in medium_end_keywords:
                if keyword in renderer:
                    self.logger.info(f"Detected mid-range GPU, using balanced settings")
                    return 'medium'
            
            # 默认返回中等设置
            self.logger.info(f"Unknown GPU model, using default medium settings")
            return 'medium'
            
        except Exception as e:
            self.logger.warning(f"Failed to detect GPU: {e}, using default settings")
            return 'medium'
    
    def _apply_gpu_performance_settings(self, gpu_tier: str):
        """
        根据GPU性能等级自动调整设置
        
        Args:
            gpu_tier: 性能等级 ('high', 'medium', 'low', 'minimal')
        """
        self.logger.info(f"Applying GPU performance settings for tier: {gpu_tier}")
        
        if gpu_tier == 'minimal' or gpu_tier == 'low':
            # GTX 750 Ti 等低端GPU - 降低质量设置
            self.config['enable_volumetric_clouds'] = False
            self.config['enable_advanced_water'] = False
            self.config['use_gpu_advanced_lighting'] = False
            self.config['enable_advanced_lighting'] = False
            self.config['enable_render_pipeline'] = False
            self.config['enable_bloom'] = False
            self.config['enable_ssr'] = False
            self.config['enable_volumetric_light'] = False
            
            # 降低分辨率和采样
            self.config['volumetric_downsample'] = 8
            self.config['bloom_downsample'] = 4
            self.config['ssr_downsample'] = 4
            
            self.logger.info("Low-end GPU detected: disabled volumetric clouds, advanced water, and GPU lighting")
            
        elif gpu_tier == 'medium':
            # 中端GPU - 平衡设置
            self.config['enable_volumetric_clouds'] = True
            self.config['enable_advanced_water'] = True
            self.config['use_gpu_advanced_lighting'] = True
            self.config['enable_advanced_lighting'] = True
            self.config['enable_render_pipeline'] = True
            self.config['enable_bloom'] = True
            self.config['enable_ssr'] = True
            self.config['enable_volumetric_light'] = True
            
            # 中等降采样
            self.config['volumetric_downsample'] = 4
            self.config['bloom_downsample'] = 2
            self.config['ssr_downsample'] = 2
            
            self.logger.info("Mid-range GPU detected: enabled balanced settings")
            
        elif gpu_tier == 'high':
            # 高端GPU - 开启全部效果
            self.config['enable_volumetric_clouds'] = True
            self.config['enable_advanced_water'] = True
            self.config['use_gpu_advanced_lighting'] = True
            self.config['enable_advanced_lighting'] = True
            self.config['enable_render_pipeline'] = True
            self.config['enable_bloom'] = True
            self.config['enable_ssr'] = True
            self.config['enable_volumetric_light'] = True
            self.config['enable_motion_blur'] = True
            self.config['enable_path_trace'] = True
            
            # 高质量设置
            self.config['volumetric_downsample'] = 2
            self.config['bloom_downsample'] = 1
            self.config['ssr_downsample'] = 1
            self.config['path_trace_samples'] = 4
            self.config['path_trace_bounces'] = 3
            
            self.logger.info("High-end GPU detected: enabled all premium effects")
        
        # 设置全局GPU等级
        self.set_global('gpu_tier', gpu_tier)
