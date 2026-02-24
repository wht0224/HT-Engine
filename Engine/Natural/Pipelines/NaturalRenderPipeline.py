import numpy as np
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from ..Core.RuleBase import Rule
from ..Core.FactBase import FactBase
from ..Core.GpuContextManager import GpuContextManager


@dataclass
class RenderStats:
    frame_time_ms: float = 0.0
    culling_time_ms: float = 0.0
    lod_time_ms: float = 0.0
    lighting_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    visible_objects: int = 0
    total_triangles: int = 0
    draw_calls: int = 0
    fps: float = 0.0


@dataclass
class RenderConfig:
    resolution: Tuple[int, int] = (1920, 1080)
    target_fps: int = 60
    max_triangles: int = 30_000_000
    enable_bloom: bool = True
    enable_ssr: bool = True
    enable_volumetric_light: bool = True
    enable_motion_blur: bool = False
    enable_path_trace: bool = False
    adaptive_quality: bool = True
    quality_level: str = "high"


class NaturalRenderPipeline:
    """
    Natural 渲染管线 (Natural Render Pipeline)
    
    整合所有GPU渲染规则，提供统一的渲染接口。
    针对 GTX 1650 Max-Q 优化，支持30M三角形场景。
    """
    
    def __init__(self, config: Optional[RenderConfig] = None, gpu_manager: Optional[GpuContextManager] = None):
        self.logger = logging.getLogger("Natural.RenderPipeline")
        self.config = config or RenderConfig()
        self.gpu_manager = gpu_manager
        
        if self.gpu_manager is None:
            self.gpu_manager = GpuContextManager()
        
        self.ctx = self.gpu_manager.context
        self.facts = FactBase()
        
        self.stats = RenderStats()
        self.frame_count = 0
        
        self._init_facts()
        self._init_quality_profiles()
    
    def _init_facts(self):
        self.facts.set_global("screen_size", self.config.resolution)
        self.facts.set_global("target_fps", self.config.target_fps)
        self.facts.set_global("max_triangles", self.config.max_triangles)
        
        self.facts.set_global("enable_bloom", self.config.enable_bloom)
        self.facts.set_global("enable_ssr", self.config.enable_ssr)
        self.facts.set_global("enable_volumetric_light", self.config.enable_volumetric_light)
        self.facts.set_global("enable_motion_blur", self.config.enable_motion_blur)
        self.facts.set_global("enable_path_trace", self.config.enable_path_trace)
        
        self.facts.set_global("camera_position", np.array([0.0, 5.0, 10.0], dtype=np.float32))
        self.facts.set_global("camera_target", np.array([0.0, 0.0, 0.0], dtype=np.float32))
        self.facts.set_global("camera_up", np.array([0.0, 1.0, 0.0], dtype=np.float32))
        
        self.facts.set_global("sun_direction", np.array([0.5, -1.0, 0.3], dtype=np.float32))
        self.facts.set_global("sun_color", np.array([1.0, 0.95, 0.9], dtype=np.float32))
        self.facts.set_global("sun_intensity", 1.0)
        
        self.facts.set_global("view_matrix", np.eye(4, dtype=np.float32))
        self.facts.set_global("projection_matrix", np.eye(4, dtype=np.float32))
        
        self.facts.set_global("lod_distances", np.array([50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0, 6400.0], dtype=np.float32))
        self.facts.set_global("pixel_error_threshold", 4.0)
    
    def _init_quality_profiles(self):
        self.quality_profiles = {
            "ultra": {
                "bloom_intensity": 0.8,
                "bloom_threshold": 0.7,
                "ssr_max_steps": 32,
                "ssr_intensity": 0.8,
                "volumetric_steps": 32,
                "path_trace_samples": 4,
                "path_trace_bounces": 3,
                "downsample_factor": 1,
            },
            "high": {
                "bloom_intensity": 0.6,
                "bloom_threshold": 0.8,
                "ssr_max_steps": 16,
                "ssr_intensity": 0.5,
                "volumetric_steps": 16,
                "path_trace_samples": 2,
                "path_trace_bounces": 2,
                "downsample_factor": 2,
            },
            "medium": {
                "bloom_intensity": 0.4,
                "bloom_threshold": 0.85,
                "ssr_max_steps": 8,
                "ssr_intensity": 0.3,
                "volumetric_steps": 8,
                "path_trace_samples": 1,
                "path_trace_bounces": 1,
                "downsample_factor": 2,
            },
            "low": {
                "bloom_intensity": 0.3,
                "bloom_threshold": 0.9,
                "ssr_max_steps": 4,
                "ssr_intensity": 0.2,
                "volumetric_steps": 4,
                "path_trace_samples": 1,
                "path_trace_bounces": 1,
                "downsample_factor": 4,
            },
            "potato": {
                "bloom_intensity": 0.0,
                "bloom_threshold": 1.0,
                "ssr_max_steps": 0,
                "ssr_intensity": 0.0,
                "volumetric_steps": 0,
                "path_trace_samples": 0,
                "path_trace_bounces": 0,
                "downsample_factor": 8,
            }
        }
    
    def set_quality(self, level: str):
        if level not in self.quality_profiles:
            self.logger.warning(f"Unknown quality level: {level}, using 'medium'")
            level = "medium"
        
        self.config.quality_level = level
        profile = self.quality_profiles[level]
        
        self.facts.set_global("quality_level", level)
        self.facts.set_global("bloom_intensity", profile["bloom_intensity"])
        self.facts.set_global("bloom_threshold", profile["bloom_threshold"])
        self.facts.set_global("ssr_max_steps", profile["ssr_max_steps"])
        self.facts.set_global("ssr_intensity", profile["ssr_intensity"])
        self.facts.set_global("volumetric_steps", profile["volumetric_steps"])
        self.facts.set_global("path_trace_samples", profile["path_trace_samples"])
        self.facts.set_global("path_trace_bounces", profile["path_trace_bounces"])
        self.facts.set_global("downsample_factor", profile["downsample_factor"])
        
        self.logger.info(f"Quality set to: {level}")
    
    def set_camera(self, position: np.ndarray, target: np.ndarray, up: np.ndarray = None):
        if up is None:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        
        self.facts.set_global("camera_position", np.array(position, dtype=np.float32))
        self.facts.set_global("camera_target", np.array(target, dtype=np.float32))
        self.facts.set_global("camera_up", np.array(up, dtype=np.float32))
        
        view_matrix = self._compute_view_matrix(position, target, up)
        self.facts.set_global("view_matrix", view_matrix)
    
    def set_projection(self, fov: float, aspect: float, near: float, far: float):
        proj_matrix = self._compute_projection_matrix(fov, aspect, near, far)
        self.facts.set_global("projection_matrix", proj_matrix)
        self.facts.set_global("fov", fov)
        self.facts.set_global("aspect", aspect)
        self.facts.set_global("near", near)
        self.facts.set_global("far", far)
    
    def set_sun(self, direction: np.ndarray, color: np.ndarray = None, intensity: float = 1.0):
        direction = np.array(direction, dtype=np.float32)
        direction = direction / (np.linalg.norm(direction) + 1e-5)
        
        self.facts.set_global("sun_direction", direction)
        if color is not None:
            self.facts.set_global("sun_color", np.array(color, dtype=np.float32))
        self.facts.set_global("sun_intensity", intensity)
    
    def _compute_view_matrix(self, eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
        forward = target - eye
        forward = forward / (np.linalg.norm(forward) + 1e-5)
        
        right = np.cross(forward, up)
        right = right / (np.linalg.norm(right) + 1e-5)
        
        up = np.cross(right, forward)
        
        view = np.eye(4, dtype=np.float32)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([np.dot(right, eye), np.dot(up, eye), np.dot(-forward, eye)])
        
        return view
    
    def _compute_projection_matrix(self, fov: float, aspect: float, near: float, far: float) -> np.ndarray:
        proj = np.zeros((4, 4), dtype=np.float32)
        
        tan_half_fov = np.tan(np.radians(fov) / 2.0)
        
        proj[0, 0] = 1.0 / (aspect * tan_half_fov)
        proj[1, 1] = 1.0 / tan_half_fov
        proj[2, 2] = -(far + near) / (far - near)
        proj[2, 3] = -(2.0 * far * near) / (far - near)
        proj[3, 2] = -1.0
        
        return proj
    
    def create_scene_table(self, name: str, max_objects: int):
        schema = {
            'positions': np.float32,
            'aabbs': np.float32,
            'lod_params': np.float32,
            'material_ids': np.int32,
        }
        
        self.facts.create_table(name, max_objects, schema)
        self.facts.set_global("scene_table", name)
        
        self.logger.info(f"Created scene table '{name}' with capacity {max_objects}")
    
    def create_postprocess_table(self, name: str, width: int, height: int):
        schema = {
            'color_buffer': np.float32,
            'depth_buffer': np.float32,
            'normal_buffer': np.float32,
            'roughness_buffer': np.float32,
        }
        
        capacity = width * height
        self.facts.create_table(name, capacity, schema)
        self.facts.set_global("postprocess_table", name)
        
        self.logger.info(f"Created postprocess table '{name}' ({width}x{height})")
    
    def update_frame_stats(self, frame_time_ms: float):
        self.stats.frame_time_ms = frame_time_ms
        self.stats.fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0.0
        
        self.facts.set_global("frame_ms", frame_time_ms)
        self.facts.set_global("frame_fps", self.stats.fps)
        
        if self.config.adaptive_quality:
            self._adaptive_quality_adjust()
    
    def _adaptive_quality_adjust(self):
        target_fps = self.config.target_fps
        current_fps = self.stats.fps
        
        if current_fps < target_fps * 0.8:
            current_level = self.config.quality_level
            levels = ["ultra", "high", "medium", "low", "potato"]
            current_idx = levels.index(current_level) if current_level in levels else 2
            
            if current_idx < len(levels) - 1:
                new_level = levels[current_idx + 1]
                self.set_quality(new_level)
                self.logger.info(f"Adaptive quality: {current_level} -> {new_level} (FPS: {current_fps:.1f})")
        
        elif current_fps > target_fps * 1.1:
            current_level = self.config.quality_level
            levels = ["ultra", "high", "medium", "low", "potato"]
            current_idx = levels.index(current_level) if current_level in levels else 2
            
            if current_idx > 0:
                new_level = levels[current_idx - 1]
                self.set_quality(new_level)
                self.logger.info(f"Adaptive quality: {current_level} -> {new_level} (FPS: {current_fps:.1f})")
    
    def get_stats(self) -> RenderStats:
        return self.stats
    
    def get_facts(self) -> FactBase:
        return self.facts
    
    def begin_frame(self):
        self._frame_start_time = time.perf_counter()
        self.frame_count += 1
        self.facts.set_global("frame_index", self.frame_count)
    
    def end_frame(self):
        frame_time = (time.perf_counter() - self._frame_start_time) * 1000.0
        self.update_frame_stats(frame_time)
    
    def get_performance_report(self) -> str:
        report = f"""
=== Natural Render Pipeline Performance Report ===
Frame: {self.frame_count}
Quality: {self.config.quality_level}
Resolution: {self.config.resolution}

Timing:
  Frame Time: {self.stats.frame_time_ms:.2f}ms
  FPS: {self.stats.fps:.1f}
  Target FPS: {self.config.target_fps}

Scene:
  Visible Objects: {self.stats.visible_objects}
  Total Triangles: {self.stats.total_triangles:,}
  Draw Calls: {self.stats.draw_calls}

GPU: {self.ctx.info.get('GL_RENDERER', 'Unknown') if self.ctx else 'N/A'}
=================================================
"""
        return report
