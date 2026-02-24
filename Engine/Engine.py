import sys
import os
import json

# 添加引擎根目录到Python路径，确保能正确导入内部模块
# Add engine root directory to Python path for proper internal module imports
engine_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, engine_root)

# 导入日志系统
# Import logging system
from .Logger import get_logger, LogLevel

# 导入全流程性能追踪器
try:
    from .Utils.FullPerformanceProfiler import get_full_profiler
    profiler = get_full_profiler()
    profiler.enable()
except Exception as e:
    profiler = None

class Engine:
    """
    游戏世界的魔法引擎！
    专为让老显卡也能施展光影魔法而设计
    就像给GTX 750Ti和RX 580装上了魔法加速器
"""
    
    def __init__(self):
        self.logger = get_logger("Engine")
        self.is_initialized = False
        
        # 引擎配置
        self.config = {
            "frontend": {
                "type": "html",
                "enable_html": True,
                "enable_tkinter": False
            },
            "logging": {
                "level": "WARNING",
                "enable_file_logging": False,
                "log_file": "engine.log"
            },
            "renderer": {
                "enable_render_result": True,
                "render_result_interval": 33
            },
            "physics": {
                "engine": "auto"
            }
        }
        
        # 引擎的魔法核心组件
        self.renderer = None  # 光影魔法师
        self.res_mgr = None   # 资源宝库守护者
        self.scene_mgr = None # 场景导演
        self.plt = None       # 平台魔法阵
        
        # 多核并行处理指挥官
        self.mcp_mgr = None

        # MCP服务器（Model Context Protocol）- AI建模API
        self.mcp_server = None
    
    def _load_config(self, config_override=None):
        """
        加载引擎配置文件
        
        参数:
            config_override: 配置覆盖，优先级高于配置文件
        """
        import json
        import os
        
        # 默认配置文件路径
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "engine_config.json")
        
        # 尝试读取配置文件
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    file_config = json.load(f)
                # 更新配置
                self.config.update(file_config)
                self.logger.info(f"成功加载配置文件: {config_path}")
        except Exception as e:
            self.logger.warning(f"加载配置文件失败: {e}")
        
        # 应用配置覆盖
        if config_override:
            self.config.update(config_override)
            self.logger.info(f"应用配置覆盖: {config_override}")
        
        # 强制设置日志级别为WARNING，以减少调试输出
        self.config["logging"] = {
            "level": "WARNING",
            "enable_file_logging": False,
            "log_file": "engine.log"
        }
        
        # 应用日志配置
        self._apply_logging_config()
    
    def _apply_logging_config(self):
        """
        应用日志配置
        """
        from Engine.Logger import set_global_log_level, LogLevel, get_logger
        
        log_level_str = self.config["logging"]["level"].upper()
        try:
            log_level = LogLevel[log_level_str]
            set_global_log_level(log_level)
            # 同时更新当前日志器的级别
            self.logger.set_log_level(log_level)
            # 获取全局日志器并更新其级别
            global_logger = get_logger()
            global_logger.set_log_level(log_level)
            self.logger.info(f"日志级别设置为: {log_level_str}")
        except KeyError:
            self.logger.warning(f"无效的日志级别: {log_level_str}，使用默认级别")
        
    def initialize(self, config=None, skip_renderer=False):
        """
        启动魔法引擎的仪式！
        让所有组件都活起来，准备创造游戏世界
        
        参数:
            config: 魔法配方（引擎配置）
            skip_renderer: 是否跳过光影魔法师的召唤
            
        返回值:
            bool: 仪式成功返回True，否则返回False
        """
        # 加载配置文件
        self._load_config(config)
        
        self.logger.info("正在初始化高性能渲染引擎...")
        
        # 初始化平台模块
        try:
            self.logger.debug("1. 导入平台模块...")
            from .Platform.Platform import Platform
            self.plt = Platform()
            self.plt.initialize()
            self.logger.info("平台模块初始化完成")
        except Exception as e:
            self.logger.error(f"平台模块初始化失败: {e}")
            return False
        
        # 初始化资源管理器
        try:
            self.logger.debug("2. 导入并初始化资源管理器...")
            from .Renderer.Resources.ResourceManager import ResourceManager
            self.res_mgr = ResourceManager()
            self.res_mgr.initialize()
            self.logger.info("资源管理器初始化完成")
        except Exception as e:
            self.logger.error(f"资源管理器初始化失败: {e}")
            return False
        
        # 只有在非无头模式下才初始化渲染器相关组件
        if not skip_renderer:
            # 延迟渲染器初始化（等待OpenGL上下文创建）
            # 注意：渲染器将在OpenGLViewport.initgl()回调中初始化
            self.renderer = None
            self.renderer_config = config  # 保存配置供后续使用
            self.logger.info("渲染器将在OpenGL上下文创建后初始化")
        
        # 初始化场景管理器
        try:
            # 初始化场景管理器
            self.logger.debug("4. 导入并初始化场景管理器...")
            from .Scene.SceneManager import SceneManager
            self.scene_mgr = SceneManager(self)
            self.logger.info("场景管理器初始化完成")
        except ImportError as e:
            self.logger.error(f"场景管理器导入失败: {e}")
            self.logger.error(f"Python路径: {sys.path}")
            return False
        except Exception as e:
            self.logger.error(f"场景管理器初始化失败: {e}")
            return False
        
        try:
            # 初始化地形管理器
            self.logger.debug("4.1 导入并初始化地形管理器...")
            from .Renderer.Terrain.TerrainManager import TerrainManager
            self.terrain_mgr = TerrainManager(self.scene_mgr)
            self.logger.info("地形管理器初始化完成")
        except Exception as e:
            self.logger.error(f"地形管理器初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            # 地形管理器失败不影响引擎运行
            self.terrain_mgr = None
        
        try:
            # 初始化天气和大气系统
            self.logger.debug("4.2 初始化天气和大气系统...")
            from .Renderer.WeatherSystem import AtmosphereSystem, WeatherSystem
            self.atmosphere_system = AtmosphereSystem(None)  # 渲染器尚未初始化，稍后设置
            self.weather_system = WeatherSystem(self.atmosphere_system)
            self.logger.info("天气和大气系统初始化完成")
        except Exception as e:
            self.logger.error(f"天气和大气系统初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            # 天气系统失败不影响引擎运行
        
        # 初始化物理系统（可选，失败不影响引擎运行）
        try:
            self.logger.debug("5. 导入并初始化物理系统...")
            from .Physics.PhysicsSystem import PhysicsSystem
            self.physics_system = PhysicsSystem(self)
            self.logger.info("物理系统初始化完成")
        except Exception as e:
            self.logger.error(f"物理系统初始化失败: {e}")
        
        try:
            # 创建默认测试场景
            self.logger.debug("6. 创建默认测试场景...")
            self._create_default_scene()
            self.logger.info("默认测试场景创建完成")
        except Exception as e:
            self.logger.error(f"创建默认测试场景失败: {e}")
            return False
        
        try:
            # 导入并初始化相机控制器
            self.logger.debug("7. 导入并初始化相机控制器...")
            from .UI.CameraController import CameraController
            self.camera_controller = CameraController(self.scene_mgr.active_camera)
            self.plt.camera_controller = self.camera_controller
            self.logger.info("相机控制器初始化完成")
        except Exception as e:
            self.logger.error(f"相机控制器初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
        
        try:
            # 初始化音频管理器
            self.logger.debug("7.3 导入并初始化音频管理器...")
            from .Audio.AudioManager import AudioManager
            self.audio_manager = AudioManager()
            self.logger.info("音频管理器初始化完成")
        except Exception as e:
            self.logger.error(f"音频管理器初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self.audio_manager = None

        # 初始化自然系统（Natural System）- 用于高级环境模拟
        try:
            self.logger.debug("7.4 导入并初始化自然系统...")
            from .Natural.NaturalSystem import NaturalSystem
            # 配置自然系统 - 暂时禁用以排查黑色三角形问题
            # 从配置文件中读取Natural系统配置
            natural_config = {
                "quality_profile": self.config.get("quality_profile", "high"),
                "enable_advanced_lighting": self.config.get("enable_advanced_lighting", True),
                "use_gpu_advanced_lighting": self.config.get("use_gpu_advanced_lighting", True),
                "use_gpu_lighting": self.config.get("use_gpu_lighting", True),
                "use_gpu_atmosphere": self.config.get("use_gpu_atmosphere", True),
                "use_gpu_fog": self.config.get("use_gpu_fog", True),
                "use_gpu_hydro": self.config.get("use_gpu_hydro", False),
                "enable_hydro_visual": self.config.get("enable_hydro_visual", False),
                "use_gpu_vegetation": self.config.get("use_gpu_vegetation", False),
                "use_gpu_weathering": self.config.get("use_gpu_weathering", False),
                "sim_preset": self.config.get("sim_preset", "full"),
                "adaptive_quality": self.config.get("adaptive_quality", True),
                "enable_gpu_culling": self.config.get("enable_gpu_culling", True),
                "enable_gpu_lod": self.config.get("enable_gpu_lod", True),
                "enable_bloom": self.config.get("enable_bloom", False),
                "enable_ssr": self.config.get("enable_ssr", False),
                "enable_volumetric_light": self.config.get("enable_volumetric_light", False),
                "enable_render_pipeline": self.config.get("enable_render_pipeline", False),
                "auto_detect_gpu": self.config.get("auto_detect_gpu", True),
                "enable_advanced_water": self.config.get("enable_advanced_water", False),
                "enable_volumetric_clouds": self.config.get("enable_volumetric_clouds", False),
                "enable_path_trace": self.config.get("enable_path_trace", False),
                "enable_motion_blur": self.config.get("enable_motion_blur", False),
                "enable_enhanced_gpu_lighting": self.config.get("enable_enhanced_gpu_lighting", False),
                "enable_enhanced_atmosphere": self.config.get("enable_enhanced_atmosphere", True),
                "enable_screen_space_shadows": self.config.get("enable_screen_space_shadows", True),
                "enable_gi_probes": self.config.get("enable_gi_probes", False),
                "enable_grazing": self.config.get("enable_grazing", False),
                "enable_vegetation_growth": self.config.get("enable_vegetation_growth", False),
                "enable_thermal_weathering": self.config.get("enable_thermal_weathering", False),
                "enable_gpu_readback": self.config.get("enable_gpu_readback", False),
                "sim_rule_enabled": self.config.get("sim_rule_enabled", {}),
                "bloom_downsample": self.config.get("bloom_downsample", 4),
                "ssr_downsample": self.config.get("ssr_downsample", 4),
                "volumetric_downsample": self.config.get("volumetric_downsample", 4),
                "ssr_max_steps": self.config.get("ssr_max_steps", 32),
                "volumetric_steps": self.config.get("volumetric_steps", 32),
            }
            self.natural_system = NaturalSystem(natural_config)
            self.logger.info("自然系统初始化完成（已禁用大部分功能）")
        except Exception as e:
            self.logger.error(f"自然系统初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self.natural_system = None
        
        # 只有在非无头模式下才初始化变换操纵器
        if not skip_renderer:
            ui_type = self.config["frontend"].get("ui_type", "default")
            try:
                # 导入并初始化变换操纵器
                self.logger.debug("8. 导入并初始化变换操纵器...")
                from .UI.Controls.TransformManipulator import TransformManipulator
                self.transform_manipulator = TransformManipulator(self.scene_mgr.active_camera)
                self.logger.info("变换操纵器初始化完成")
            except Exception as e:
                self.logger.error(f"变换操纵器初始化失败: {e}")
                return False
        
        # 初始化UI相关组件
        ui_enabled = self.config["frontend"]["enable_html"] or self.config["frontend"]["enable_tkinter"]
        if ui_enabled and not skip_renderer:
            try:
                # 编辑器UI - 初始化完整的UI管理器和默认UI
                self.logger.debug("9. 导入并初始化UI管理器...")
                from .UI.UIManager import UIManager
                self.ui_mgr = UIManager(self.plt)
                
                # 创建默认UI（编辑器UI）
                self._create_default_ui()
                
                self.logger.info("UI管理器初始化完成")
            except Exception as e:
                self.logger.error(f"UI管理器初始化失败: {e}")
                # UI失败不应导致引擎初始化失败
            
            try:
                # 初始化Tkinter UI
                if self.config["frontend"]["enable_tkinter"] or self.config["frontend"]["type"] == "tkinter":
                    self.logger.debug("10. 导入并初始化Tkinter UI...")
                    
                    # 使用默认编辑器UI
                    # 初始化主窗口
                    self.logger.debug("10.1 导入并初始化主窗口...")
                    from .UI.MainWindow import MainWindow
                    self.main_window = MainWindow(self.plt)
                    if hasattr(self, 'ui_mgr'):
                        self.ui_mgr.add_control(self.main_window)
                    self.logger.info("主窗口初始化完成")
                    
                    # 检查是否使用飞行模拟器UI
                    ui_type = self.config["frontend"].get("ui_type", "default")
                    
                    # 创建编辑器UI
                    if ui_type == "game":
                        # 游戏模式UI
                        from .UI.TkUI.game_ui import GameUI
                        self.tk_ui = GameUI(self)
                        self.logger.info("游戏模式UI已启动")
                    else:
                        # 编辑器UI
                        from .UI.TkUI.tk_main_window import TkMainWindow
                        self.tk_ui = TkMainWindow(self)
                        self.logger.info("编辑器Tkinter UI已启动")
                else:
                    self.logger.info("Tkinter UI未启用，引擎将以无UI模式运行")
            except Exception as e:
                self.logger.error(f"Tkinter UI启动失败: {e}")
                import traceback
                self.logger.error(f"错误详情: {traceback.format_exc()}")
                # UI失败不应导致引擎初始化失败

        # 检查是否启用MCP服务器
        enable_mcp = self.config.get("mcp", {}).get("enable_mcp_server", True)
        if enable_mcp:
            try:
                # 初始化MCP服务器（Model Context Protocol）- AI建模API
                self.logger.debug("12. 初始化MCP服务器（AI建模API）...")
                from .MCP import MCPServer
                self.mcp_server = MCPServer(self)
                self.logger.info("MCP服务器已启动 - AI可以通过代码建模了！")
            except Exception as e:
                self.logger.warning(f"MCP服务器初始化失败: {e}")
                # MCP服务器失败不影响引擎运行

        # 检查是否启用HTTP API服务器
        enable_api = self.config.get("mcp", {}).get("enable_api_server", True)
        if enable_api:
            try:
                # 启动HTTP API服务器（让运行中的引擎接收MCP命令）
                self.logger.debug("12.1 启动HTTP API服务器...")
                from .MCP.api_server import EngineAPIServer
                self.api_server = EngineAPIServer(self)
                self.api_server.start()
                self.logger.info("HTTP API服务器已启动 - AI可以通过HTTP调用引擎了！")
            except Exception as e:
                self.logger.warning(f"HTTP API服务器启动失败: {e}")
                # API服务器失败不影响引擎运行

        
        # 检查是否需要初始化MCP架构管理器
        enable_mcp_server = self.config.get("mcp", {}).get("enable_mcp_server", False)
        enable_api_server = self.config.get("mcp", {}).get("enable_api_server", False)
        
        if enable_mcp_server or enable_api_server:
            try:
                # 初始化MCP架构管理器
                self.logger.debug("13. 初始化MCP架构管理器...")
                from .Core.MCP.MCPManager import MCPManager
                self.mcp_mgr = MCPManager(self)
                self.logger.info("MCP架构初始化完成")
            except Exception as e:
                self.logger.error(f"MCP架构初始化失败: {e}")
                return False
        else:
            # 禁用MCP架构
            self.mcp_mgr = None
            self.logger.info("MCP架构已禁用")
        
        # 只有在非无头模式下才初始化增强渲染模块
        if not skip_renderer:
            try:
                # 初始化增强渲染模块（可选）
                self.logger.debug("14. 尝试初始化增强渲染模块...")
                from EnhancedRenderingModule.Integration.EngineIntegration import integrate_enhanced_rendering_module
                success = integrate_enhanced_rendering_module(self)
                if success:
                    self.logger.info("增强渲染模块初始化完成")
                else:
                    self.logger.warning("增强渲染模块初始化失败")
            except ImportError as e:
                self.logger.info(f"增强渲染模块未找到，跳过初始化: {e}")
            except Exception as e:
                self.logger.warning(f"增强渲染模块初始化失败: {e}")
                import traceback
                self.logger.warning(f"错误详情: {traceback.format_exc()}")
        
        # 只有在非无头模式下才初始化场景编辑器
        if not skip_renderer:
            try:
                # 导入并初始化场景编辑器（可选）
                self.logger.debug("14. 尝试导入并初始化场景编辑器...")
                # 检查Editor目录是否存在
                import os
                editor_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Editor")
                if not os.path.exists(editor_dir):
                    self.logger.info("Editor目录不存在，跳过场景编辑器初始化")
                else:
                    from Editor.SceneEditor import SceneEditor
                    self.scene_editor = SceneEditor(self)
                    self.logger.info("场景编辑器初始化完成")
            except ImportError as e:
                self.logger.info(f"场景编辑器模块不存在，跳过初始化: {e}")
            except Exception as e:
                self.logger.error(f"场景编辑器初始化失败: {e}")
        
        self.is_initialized = True
        self.logger.info("引擎初始化完成")
        return True
    
    def _create_default_ui(self):
        """
        搭建游戏世界的控制台
        给玩家提供各种魔法按钮和工具
        """
        self.logger.debug("创建默认UI...")
        
        from .UI.Controls.Button import Button
        from .UI.Controls.Label import Label
        from .UI.Controls.Slider import Slider
        from .UI.Controls.Menu import Menu
        from .UI.Controls.PropertyPanel import PropertyPanel
        from .UI.Controls.Toolbar import Toolbar
        from .Math import Vector3
        
        # 创建工具栏
        self.toolbar = Toolbar(0, 0, self.plt.width, 40)
        self.ui_mgr.add_control(self.toolbar)
        
        def _toggle_view_mode():
            """切换视角模式"""
            if hasattr(self, 'camera_controller'):
                if self.camera_controller.view_mode == "THIRD_PERSON":
                    self.camera_controller.set_view_mode("FIRST_PERSON")
                    self.logger.info("视角模式切换到: FIRST_PERSON")
                elif self.camera_controller.view_mode == "FIRST_PERSON":
                    self.camera_controller.set_view_mode("TOP_DOWN")
                    self.logger.info("视角模式切换到: TOP_DOWN")
                else:
                    self.camera_controller.set_view_mode("THIRD_PERSON")
                    self.logger.info("视角模式切换到: THIRD_PERSON")
        
        def _toggle_auto_level():
            """切换自动水平模式"""
            # 相机控制器没有auto_level属性，保留函数但留空或移除
            pass
        
        def _reset_camera():
            """重置相机位置"""
            if hasattr(self, 'scene_mgr') and hasattr(self, 'camera_controller'):
                self.scene_mgr.active_camera.set_position(Vector3(5, 3, 5))
                self.scene_mgr.active_camera.look_at(Vector3(0, 1, 0))
                self.camera_controller.reset()
                self.logger.info("相机位置已重置")
        
        def _toggle_weather():
            """切换天气"""
            if hasattr(self, 'weather_system'):
                current_weather = self.weather_system.current_weather
                weather_types = ["clear", "cloudy", "rain", "snow"]
                current_index = weather_types.index(current_weather)
                next_index = (current_index + 1) % len(weather_types)
                self.weather_system.set_weather(weather_types[next_index])
                self.logger.info(f"天气切换到: {weather_types[next_index]}")
        
        def _toggle_day_night():
            """切换昼夜"""
            if hasattr(self, 'atmosphere_system'):
                current_time = self.atmosphere_system.time_of_day
                new_time = 12.0 if current_time < 12.0 else 0.0  # 切换到中午或午夜
                self.atmosphere_system.set_time_of_day(new_time)
                self.logger.info(f"时间设置为: {new_time:.1f}:00")
        
        def _toggle_hud():
            """切换HUD显示"""
            # 移除HUD切换功能
            pass
        
        def _toggle_audio():
            """切换音频"""
            if hasattr(self, 'audio_manager'):
                self.audio_manager.enabled = not self.audio_manager.enabled
                self.logger.info(f"音频: {'开启' if self.audio_manager.enabled else '关闭'}")
        
        # 主菜单
        main_menu_items = [
            {"text": "视角模式", "callback": _toggle_view_mode},
            {"text": "重置相机", "callback": _reset_camera},
            {"text": "切换天气", "callback": _toggle_weather},
            {"text": "切换昼夜", "callback": _toggle_day_night},
            {"text": "音频开关", "callback": _toggle_audio},
            {"text": "退出", "callback": self.shutdown}
        ]
        
        main_menu = Menu(10, 50, 150, 30, main_menu_items, text="游戏引擎")
        self.ui_mgr.add_control(main_menu)
        
        # 创建属性面板
        self.property_panel = PropertyPanel(self.plt.width - 300, 50, 290, self.plt.height - 60, "引擎设置")
        self.ui_mgr.add_control(self.property_panel)
        
        self.logger.debug("默认UI创建完成")
    
    def _create_cube(self):
        """创建立方体"""
        self.logger.info("创建立方体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 创建立方体网格
        cube_mesh = Mesh.create_cube(1.0)
        
        # 创建材质
        cube_material = Material()
        cube_material.set_color(Vector3(0.8, 0.2, 0.2))
        
        # 创建场景节点
        cube_node = SceneNode("Cube")
        cube_node.set_position(Vector3(0, 0, 0))
        cube_node.mesh = cube_mesh
        cube_node.material = cube_material
        
        # 添加到场景
        self.scene_mgr.root_node.add_child(cube_node)
        self.logger.info(f"立方体 '{cube_node.name}' 已创建")
    
    def _create_sphere(self):
        """创建球体"""
        self.logger.info("创建球体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 创建球体网格
        sphere_mesh = Mesh.create_sphere(0.5, 32, 32)
        
        # 创建材质
        sphere_material = Material()
        sphere_material.set_color(Vector3(0.2, 0.8, 0.2))
        
        # 创建场景节点
        sphere_node = SceneNode("Sphere")
        sphere_node.set_position(Vector3(2, 0, 0))
        sphere_node.mesh = sphere_mesh
        sphere_node.material = sphere_material
        
        # 添加到场景
        self.scene_mgr.root_node.add_child(sphere_node)
        self.logger.info(f"球体 '{sphere_node.name}' 已创建")
    
    def _create_cylinder(self):
        """创建圆柱体"""
        self.logger.info("创建圆柱体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 创建圆柱体网格
        cylinder_mesh = Mesh.create_cylinder(0.5, 1.0, 32)
        
        # 创建材质
        cylinder_material = Material()
        cylinder_material.set_color(Vector3(0.2, 0.2, 0.8))
        
        # 创建场景节点
        cylinder_node = SceneNode("Cylinder")
        cylinder_node.set_position(Vector3(-2, 0, 0))
        cylinder_node.mesh = cylinder_mesh
        cylinder_node.material = cylinder_material
        
        # 添加到场景
        self.scene_mgr.root_node.add_child(cylinder_node)
        self.logger.info(f"圆柱体 '{cylinder_node.name}' 已创建")
    
    def _create_plane(self):
        """创建平面"""
        self.logger.info("创建平面")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 创建平面网格
        plane_mesh = Mesh.create_plane(2.0, 2.0, 20, 20)
        
        # 创建材质
        plane_material = Material()
        plane_material.set_color(Vector3(0.8, 0.8, 0.8))
        
        # 创建场景节点
        plane_node = SceneNode("Plane")
        plane_node.set_position(Vector3(0, 0, 2))
        plane_node.set_rotation(Vector3(-90, 0, 0))
        plane_node.mesh = plane_mesh
        plane_node.material = plane_material
        
        # 添加到场景
        self.scene_mgr.root_node.add_child(plane_node)
        self.logger.info(f"平面 '{plane_node.name}' 已创建")
    
    def _create_cone(self):
        """创建圆锥体"""
        self.logger.info("创建圆锥体")
        from .Scene.SceneNode import SceneNode
        from .Math import Vector3
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 创建圆锥体网格
        cone_mesh = Mesh.create_cone(0.5, 1.0, 32)
        
        # 创建材质
        cone_material = Material()
        cone_material.set_color(Vector3(0.8, 0.2, 0.8))
        
        # 创建场景节点
        cone_node = SceneNode("Cone")
        cone_node.set_position(Vector3(2, 0, 2))
        cone_node.mesh = cone_mesh
        cone_node.material = cone_material
        
        # 添加到场景
        self.scene_mgr.root_node.add_child(cone_node)
        self.logger.info(f"圆锥体 '{cone_node.name}' 已创建")
    
    def _import_model(self):
        """导入模型"""
        self.logger.info("导入模型")
        from .Assets.ModelImporter import ModelImporter
        
        # 创建模型导入器
        importer = ModelImporter(self.res_mgr)
        
        # 这里可以替换为实际的模型文件路径，或者添加文件选择对话框
        # 目前使用一个示例模型文件路径
        model_file_path = "example_model.obj"
        
        try:
            # 导入模型
            imported_nodes = importer.import_model(model_file_path)
            
            # 将导入的节点添加到场景
            for node in imported_nodes:
                self.scene_mgr.root_node.add_child(node)
                self.logger.info(f"模型节点 '{node.name}' 已添加到场景")
            
            if imported_nodes:
                self.logger.info(f"成功导入 {len(imported_nodes)} 个模型节点")
            else:
                self.logger.warning("未导入任何模型节点")
        except Exception as e:
            self.logger.error(f"导入模型失败: {e}")
            
            # 如果导入失败，创建一个简单的占位符节点
            from .Scene.SceneNode import SceneNode
            from .Math import Vector3
            from .Renderer.Resources.Mesh import Mesh
            from .Renderer.Resources.Material import Material
            
            # 创建一个简单的模型节点作为占位符
            model_node = SceneNode("ImportedModel")
            model_node.set_position(Vector3(0, 0, -2))
            
            # 创建一个临时网格（实际导入时会替换为真实模型）
            temp_mesh = Mesh.create_cube(1.0)
            temp_material = Material()
            temp_material.set_color(Vector3(0.8, 0.8, 0.2))
            
            model_node.mesh = temp_mesh
            model_node.material = temp_material
            
            # 添加到场景
            self.scene_mgr.root_node.add_child(model_node)
            self.logger.info(f"创建模型占位符 '{model_node.name}'")
    
    def _create_default_scene(self):
        """
        创建默认场景，根据UI类型决定场景内容
        Create default scene based on UI type
        """
        self.logger.debug("创建默认场景...")
        
        # 导入场景相关类
        # Import scene-related classes
        from .Scene.Camera import Camera
        from .Scene.Light import DirectionalLight
        from .Math import Vector3
        from .Scene.SceneNode import SceneNode
        from .Renderer.Resources.Mesh import Mesh
        from .Renderer.Resources.Material import Material
        
        # 检查是否为飞行模拟器UI
        ui_type = self.config["frontend"].get("ui_type", "default")
        
        # 默认测试场景：保持原有的测试场景
        # Default test scene: keep existing test scene
            
        # 创建相机
        camera = Camera()
        camera.set_position(Vector3(5, 3, 5))
        camera.look_at(Vector3(0, 1, 0))
        self.scene_mgr.active_camera = camera
        self.scene_mgr.cameras.append(camera)
        
        # 创建方向光
        dir_light = DirectionalLight()
        dir_light.set_direction(Vector3(1, -1, -1))
        dir_light.set_intensity(1.0)
        self.scene_mgr.light_manager.add_light(dir_light)
        
        # 创建一个立方体
        cube_node = SceneNode("Cube")
        cube_mesh = Mesh.create_cube()
        cube_node.mesh = cube_mesh
        cube_node.material = Material()
        cube_node.material.set_color(Vector3(1.0, 0.5, 0.31)) # 橙色
        cube_node.set_position(Vector3(0, 0, 0))
        self.scene_mgr.root_node.add_child(cube_node)
        
        # 创建地面
        floor_node = SceneNode("Floor")
        floor_mesh = Mesh.create_plane(20, 20)
        floor_node.mesh = floor_mesh
        floor_node.material = Material()
        floor_node.material.set_color(Vector3(0.5, 0.5, 0.5)) # 灰色
        floor_node.set_position(Vector3(0, -1, 0))
        self.scene_mgr.root_node.add_child(floor_node)
        
        self.logger.debug("默认场景创建完成")

    def initialize_renderer_deferred(self):
        """
        延迟初始化渲染器（在OpenGL上下文创建后调用）
        Deferred renderer initialization (called after OpenGL context is created)
        """
        if self.renderer is not None:
            self.logger.warning("渲染器已经初始化，跳过重复初始化")
            return True

        try:
            self.logger.info("开始延迟初始化渲染器...")
            from .Renderer.Renderer import Renderer
            self.renderer = Renderer(self.plt, self.res_mgr)

            # 使用保存的配置初始化渲染器
            config = self.renderer_config if hasattr(self, 'renderer_config') else None
            self.renderer.initialize(config)

            # 更新大气系统的渲染器引用
            if hasattr(self, 'atmosphere_system') and self.atmosphere_system:
                self.atmosphere_system.renderer = self.renderer

            self.logger.info("渲染器延迟初始化成功")
            return True
        except Exception as e:
            self.logger.error(f"渲染器延迟初始化失败: {e}")
            import traceback
            self.logger.error(f"错误详情: {traceback.format_exc()}")
            self.renderer = None
            return False

    def shutdown(self):
        """
        关闭引擎，释放资源
        """
        if not self.is_initialized:
            return
            
        self.logger.info("正在关闭引擎...")
        
        # 关闭音频系统
        if hasattr(self, 'audio_manager') and self.audio_manager:
            self.audio_manager.shutdown()
            self.audio_manager = None
        
        # 停止HTML UI服务器
        if hasattr(self, 'html_ui_server') and self.html_ui_server:
            self.html_ui_server.stop()
            self.html_ui_server = None
        
        # 停止Python UI
        if hasattr(self, 'python_ui') and self.python_ui:
            self.python_ui.shutdown()
            self.python_ui = None
        
        # 关闭物理系统
        if hasattr(self, 'physics_system') and self.physics_system:
            self.physics_system.shutdown()
            self.physics_system = None
        
        if self.scene_mgr:
            # SceneManager没有shutdown方法，直接置为None
            self.scene_mgr = None
            
        if self.renderer:
            self.renderer.shutdown()
            self.renderer = None
            
        if self.res_mgr:
            self.res_mgr.shutdown()
            self.res_mgr = None
            
        if self.plt:
            self.plt.shutdown()
            self.plt = None
            
        self.is_initialized = False
        self.logger.info("引擎已关闭")
        
    def update(self, delta_time):
        """
        更新引擎状态
        
        Args:
            delta_time: 帧间隔时间
        """
        if not self.is_initialized:
            return
            
        # 使用MCP管理器更新
        if self.mcp_mgr:
            self.mcp_mgr.update(delta_time)
        else:
            # 兼容旧的更新方式
            # 更新物理系统
            if hasattr(self, 'physics_system') and self.physics_system:
                self.physics_system.update(delta_time)
            
            # 更新场景
            if self.scene_mgr:
                self.scene_mgr.update(delta_time)
            
            # 更新地形管理器
            if hasattr(self, 'terrain_mgr') and self.terrain_mgr:
                self.terrain_mgr.update(self.scene_mgr.active_camera)
            
            # 更新天气和大气系统
            if hasattr(self, 'atmosphere_system') and self.atmosphere_system:
                self.atmosphere_system.update(delta_time)
            if hasattr(self, 'weather_system') and self.weather_system:
                self.weather_system.update(delta_time)
            
            # 更新UI
            ui_type = self.config["frontend"].get("ui_type", "default")
            if ui_type != "flight_simulator" and hasattr(self, 'ui_mgr') and self.ui_mgr:
                self.ui_mgr.update(delta_time)
            
            # 更新音频系统
            if hasattr(self, 'audio_manager') and self.audio_manager:
                self.audio_manager.update(delta_time)
                
                # 更新天气相关声音
                if hasattr(self, 'weather_system') and self.weather_system:
                    weather_info = self.weather_system.get_weather_info()
                    self.audio_manager.update_weather_sounds(weather_info)
                
                # 更新引擎声音（从相机控制器获取速度信息）
                if hasattr(self, 'camera_controller') and self.camera_controller:
                    pass

        # 更新相机控制器（如果存在且有update方法）
        if hasattr(self, 'camera_controller') and self.camera_controller:
            if hasattr(self.camera_controller, 'update'):
                try:
                    self.camera_controller.update(delta_time)
                except Exception:
                    pass
        
        # 更新场景编辑器
        if hasattr(self, 'scene_editor') and self.scene_editor:
            self.scene_editor.update(delta_time)

        # 更新自然系统（如果已初始化）
        if hasattr(self, 'natural_system') and self.natural_system:
            try:
                self.natural_system.update(delta_time)
            except Exception as e:
                self.logger.error(f"自然系统更新失败: {e}")
    
    def render(self):
        """
        渲染当前帧
        """
        if not self.is_initialized:
            return

        # 检查渲染器是否已初始化，避免在无头模式下调用未初始化的渲染器
        has_renderer = hasattr(self, 'renderer') and self.renderer

        # 检查是否为飞行模拟器UI
        ui_type = self.config["frontend"].get("ui_type", "default")
        is_flight_simulator = (ui_type == "flight_simulator")

        # 飞行模拟器模式下，总是设置OpenGL投影和视图矩阵
        # 这确保即使没有完整的渲染器，地形和飞机也能正确渲染
        if self.scene_mgr and self.scene_mgr.active_camera:
            from OpenGL.GL import glMatrixMode, glLoadIdentity, GL_PROJECTION, GL_MODELVIEW, glViewport
            from OpenGL.GLU import gluPerspective, gluLookAt

            # 设置视口
            width, height = 800, 600  # 默认大小
            if hasattr(self, 'plt') and hasattr(self.plt, 'width') and hasattr(self.plt, 'height'):
                width, height = self.plt.width, self.plt.height
            glViewport(0, 0, width, height)

            # 设置投影矩阵
            glMatrixMode(GL_PROJECTION)
            glLoadIdentity()
            aspect = width / height if height > 0 else 1.0
            gluPerspective(60.0, aspect, 0.1, 10000.0)  # 远裁剪面10000米，适合飞行

            # 设置视图矩阵
            glMatrixMode(GL_MODELVIEW)
            glLoadIdentity()

            camera = self.scene_mgr.active_camera
            cam_pos = camera.get_position()
            forward = camera.get_forward()
            up = camera.get_up()
            target = cam_pos + forward

            gluLookAt(
                cam_pos.x, cam_pos.y, cam_pos.z,
                target.x, target.y, target.z,
                up.x, up.y, up.z
            )

        # 使用MCP管理器渲染
        if hasattr(self, 'mcp_mgr') and self.mcp_mgr:
            stats = self.mcp_mgr.render()

            # MCP渲染后，仍然需要渲染飞机模型（MCP不知道飞机的存在）
            if hasattr(self, 'aircraft_model') and self.aircraft_model:
                # 获取当前视图模式
                view_mode = 0  # 默认第三人称视角
                if hasattr(self, 'camera_controller') and hasattr(self.camera_controller, 'view_mode'):
                    view_mode = self.camera_controller.view_mode
                # 渲染飞机模型
                self.aircraft_model.render(self.renderer if hasattr(self, 'renderer') else None, view_mode)

        else:
            # 飞行模拟器模式：无论渲染器是否初始化，都渲染所有组件
            # 渲染顺序：天空 -> 地形 -> 场景（飞机等） -> UI
            
            # 1. 渲染天空和大气（仅在启用时）
            if hasattr(self, 'atmosphere_system') and self.atmosphere_system:
                if hasattr(self.atmosphere_system, 'sky_enabled') and getattr(self.atmosphere_system, 'sky_enabled', True):
                    self.atmosphere_system.render_sky(self.scene_mgr.active_camera)
            
            # 2. 渲染地形
            if hasattr(self, 'terrain_mgr') and self.terrain_mgr:
                self.terrain_mgr.draw()
            
            # 3. 渲染天气效果（仅在启用时）
            if hasattr(self, 'weather_system') and self.weather_system:
                if hasattr(self.weather_system, 'rain_enabled') and self.weather_system.rain_enabled:
                    self.weather_system.render_weather(self.scene_mgr.active_camera)
            
            # 4. 渲染飞行器模型
            if hasattr(self, 'aircraft_model') and self.aircraft_model:
                # 获取当前视图模式
                view_mode = 0  # 默认第三人称视角
                if hasattr(self, 'camera_controller') and hasattr(self.camera_controller, 'view_mode'):
                    view_mode = self.camera_controller.view_mode
                # 渲染飞机模型
                self.aircraft_model.render(self.renderer if hasattr(self, 'renderer') else None, view_mode)
            
            # 5. 只有在渲染器可用时才渲染场景
            if has_renderer and self.scene_mgr:
                if hasattr(self, 'atmosphere_system') and self.atmosphere_system and hasattr(self.atmosphere_system, "get_sky_color"):
                    try:
                        sky = self.atmosphere_system.get_sky_color()
                        if hasattr(self.renderer, "curr_pipeline") and self.renderer.curr_pipeline and hasattr(self.renderer.curr_pipeline, "set_clear_color"):
                            self.renderer.curr_pipeline.set_clear_color(float(sky.x), float(sky.y), float(sky.z), 1.0)
                    except Exception:
                        pass
                
                # 应用自然系统优化
                if hasattr(self, 'natural_system') and self.natural_system:
                    try:
                        if profiler:
                            profiler.start_step("自然系统评估", "natural_system.evaluate() - 所有特效/规则执行", level=1)
                        # 使用自然系统进行场景优化（LOD、剔除等）
                        camera_pos = self.scene_mgr.active_camera.get_position() if self.scene_mgr.active_camera else None
                        if camera_pos is not None:
                            import numpy as np
                            # 转换为 numpy 数组
                            camera_pos_np = np.array([camera_pos.x, camera_pos.y, camera_pos.z], dtype=np.float32)
                            # 如果自然系统有设置相机位置的方法，使用它
                            if hasattr(self.natural_system, 'set_camera_position'):
                                self.natural_system.set_camera_position(camera_pos_np)
                            else:
                                # 否则使用全局事实设置
                                self.natural_system.set_global('camera_position', camera_pos_np)

                        # 执行自然系统的规则评估
                        if hasattr(self.natural_system, 'evaluate'):
                            self.natural_system.evaluate()
                        else:
                            # 如果没有evaluate方法，使用update方法
                            self.natural_system.update(0.016)  # 使用固定的delta time
                        
                        # 获取自然系统规则耗时并记录
                        if profiler and hasattr(self.natural_system, 'get_rule_timings'):
                            rule_timings = self.natural_system.get_rule_timings()
                            for rule_name, duration_ms in rule_timings.items():
                                if isinstance(duration_ms, (int, float)):
                                    profiler.record_frame_metric(f"自然规则_{rule_name}", duration_ms)
                        
                        if profiler:
                            profiler.end_step()
                    except Exception as e:
                        if profiler:
                            profiler.end_step()
                        self.logger.error(f"自然系统评估失败: {e}")
                
                # 绑定 Natural 系统的阴影纹理给着色器
                try:
                    from OpenGL.GL import glUseProgram, glGetUniformLocation, glActiveTexture, GL_TEXTURE0, glBindTexture, glUniform1i
                    
                    # 获取当前着色器程序（假设用的是basic着色器）
                    if hasattr(self.renderer, 'shader_programs') and 'basic' in self.renderer.shader_programs:
                        program = self.renderer.shader_programs['basic']
                        if isinstance(program, int) and program > 0:
                            glUseProgram(program)
                            
                            # 获取 Natural 系统的阴影纹理
                            if (hasattr(self, 'natural_system') and 
                                self.natural_system and 
                                hasattr(self.natural_system, 'gpu_manager')):
                                shadow_texture = self.natural_system.gpu_manager.get_texture("shadow_soft")
                                if shadow_texture:
                                    # 绑定到纹理单元 3
                                    glActiveTexture(GL_TEXTURE0 + 3)
                                    # 注意：ModernGL 的纹理需要用 .glo 或类似方法获取 OpenGL ID
                                    if hasattr(shadow_texture, 'glo'):
                                        glBindTexture(shadow_texture.target, shadow_texture.glo)
                                    loc = glGetUniformLocation(program, "uShadowTexture")
                                    if loc != -1:
                                        glUniform1i(loc, 3)
                except Exception as e:
                    self.logger.debug(f"绑定Natural系统纹理失败（可能还未准备好）: {e}")
                
                # 简化的渲染调用，直接传递场景管理器
                if profiler:
                    profiler.start_step("渲染器渲染", "self.renderer.render() - 完整渲染管线", level=1)
                self.renderer.render(self.scene_mgr)
                if profiler:
                    # 获取渲染器性能统计并记录
                    try:
                        stats = self.renderer.get_performance_stats()
                        if stats and "render_stages" in stats:
                            stages = stats["render_stages"]
                            for stage_name, duration_ms in stages.items():
                                if isinstance(duration_ms, (int, float)):
                                    if stage_name == "total_objects_rendered":
                                        profiler.record_frame_metric("渲染物体数量", duration_ms)
                                    elif stage_name.startswith("obj_"):
                                        profiler.record_frame_metric(f"渲染物体_{stage_name}", duration_ms)
                                    else:
                                        profiler.record_frame_metric(f"渲染阶段_{stage_name}", duration_ms)
                        if "draw_calls" in stats:
                            profiler.record_frame_metric("Draw Calls", stats["draw_calls"])
                        if "render_time_ms" in stats:
                            profiler.record_frame_metric("总渲染时间", stats["render_time_ms"])
                    except Exception as e:
                        pass
                    profiler.end_step()
            
            # 6. 渲染场景编辑器（如果启用）
            if hasattr(self, 'scene_editor') and self.scene_editor:
                self.scene_editor.render()
            
            # 7. 渲染UI和HUD
            # 在飞行模拟器或游戏模式下，只渲染特定HUD，不渲染编辑器UI
            if ui_type == "flight_simulator":
                # 仅渲染飞行HUD
                if hasattr(self, 'flight_hud') and self.flight_hud:
                    self.flight_hud.render()
            elif ui_type == "game":
                # 游戏模式：可以渲染游戏特有的HUD (如果有)
                pass 
            else:
                # 编辑器模式：渲染UI管理器和HUD
                if hasattr(self, 'ui_mgr') and self.ui_mgr:
                    # 在渲染UI之前重置渲染状态，确保UI能够正确渲染
                    from OpenGL.GL import (
                        glEnable, glDisable, glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA,
                        glDepthFunc, GL_DEPTH_TEST, GL_ALWAYS, glMatrixMode, GL_PROJECTION, GL_MODELVIEW,
                        glLoadIdentity, glOrtho, glViewport
                    )

                    # 启用混合模式，以便正确显示半透明UI组件
                    glEnable(GL_BLEND)
                    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

                    # 禁用深度测试，或者将深度函数设置为GL_ALWAYS
                    glDisable(GL_DEPTH_TEST)

                    # 保存当前的矩阵模式
                    glMatrixMode(GL_PROJECTION)
                    glLoadIdentity()
                    # 设置正交投影，用于渲染2D UI
                    glOrtho(0, width, 0, height, -1, 1)

                    # 设置模型视图矩阵
                    glMatrixMode(GL_MODELVIEW)
                    glLoadIdentity()

                    # 设置正确的视口
                    glViewport(0, 0, width, height)

                    # 渲染UI
                    self.ui_mgr.render()

                    # 渲染飞行HUD
                    if hasattr(self, 'flight_hud') and self.flight_hud:
                        self.flight_hud.render()

                    # 恢复深度测试，以便下一帧渲染3D场景
                    glEnable(GL_DEPTH_TEST)
                else:
                    # 如果没有UI管理器，直接渲染飞行HUD
                    if hasattr(self, 'flight_hud') and self.flight_hud:
                        self.flight_hud.render()
        
        # 将渲染结果发送到HTML UI，只有在渲染器可用时才执行
        if has_renderer and hasattr(self, 'html_ui_server') and self.html_ui_server:
            try:
                # 获取渲染结果
                # 渲染结果现在通过HTML UI服务器的状态更新线程自动发送
                pass
            except Exception as e:
                self.logger.error(f"发送渲染结果失败: {e}")



if __name__ == "__main__":
    # 导入日志系统
    from Logger import get_logger, LogLevel
    logger = get_logger("Main")
    
    # 引擎启动示例
    logger.info("开始启动引擎...")
    engine = Engine()
    
    try:
        # 初始化引擎
        logger.info("初始化引擎...")
        engine.initialize()
        
        # 检查引擎是否初始化成功
        if not engine.is_initialized:
            logger.error("引擎初始化失败！")
            exit(1)
        
        # 检查平台是否有图形支持
        if hasattr(engine.platform, 'has_graphics') and not engine.platform.has_graphics:
            logger.warning("警告：没有图形支持！")
        else:
            logger.info("图形支持已启用")
        
        # 检查窗口是否创建成功
        if hasattr(engine.platform, 'window_created') and not engine.platform.window_created:
            logger.warning("警告：窗口创建失败！")
        else:
            logger.info("窗口创建成功")
        
        # 检查HTML UI服务器是否启动成功
        if hasattr(engine, 'html_ui_server') and engine.html_ui_server:
            logger.info("HTML UI服务器已启动")
            logger.info(f"HTTP服务器运行在 http://{engine.html_ui_server.host}:{engine.html_ui_server.http_port}")
            logger.info(f"WebSocket服务器运行在 ws://{engine.html_ui_server.host}:{engine.html_ui_server.websocket_port}")
        else:
            logger.warning("警告：HTML UI服务器未启动！")
        
        # 真实主循环，持续运行直到窗口关闭
        logger.info("引擎启动成功！按 ESC 键或关闭窗口退出。")
        
        import time
        frame_count = 0
        last_time = time.time()
        target_fps = 60
        frame_time = 1.0 / target_fps
        
        while True:
            try:
                current_time = time.time()
                delta_time = current_time - last_time
                
                # 帧率控制，限制为target_fps
                if delta_time < frame_time:
                    time.sleep(frame_time - delta_time)
                    current_time = time.time()
                    delta_time = current_time - last_time
                
                last_time = current_time
                
                # 更新和渲染
                engine.update(delta_time)
                engine.render()
                
                # 交换缓冲区以显示渲染结果
                if hasattr(engine.platform, 'swap_buffers'):
                    engine.platform.swap_buffers()
                
                frame_count += 1
                
                # 移除帧计数输出，减少终端冗余信息
                if frame_count % 60 == 0:
                    pass
                
                # 检查窗口是否需要关闭
                if hasattr(engine.platform, 'is_window_open') and not engine.platform.is_window_open():
                    logger.info("窗口关闭请求，正在关闭引擎...")
                    break
                
                # 检查ESC键是否被按下
                if hasattr(engine.platform, 'is_key_pressed') and engine.platform.is_key_pressed(27):  # ESC键
                    logger.info("ESC键被按下，正在关闭引擎...")
                    break
            except KeyboardInterrupt:
                logger.info("\n接收到中断信号，正在关闭引擎...")
                break
            except Exception as e:
                logger.error(f"渲染循环错误: {e}", exc_info=True)
                # 短暂暂停，避免日志刷屏
                time.sleep(1)
    except Exception as e:
        logger.error(f"引擎运行错误：{e}")
        import traceback
        traceback.print_exc()
    finally:
        logger.info("开始关闭引擎...")
        engine.shutdown()
        logger.info("引擎已关闭")
