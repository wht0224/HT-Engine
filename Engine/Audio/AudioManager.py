# -*- coding: utf-8 -*-
"""
音频管理器
用于处理游戏中的各种音频效果
"""

import pyglet
from Engine.Math import Vector3
import numpy as np
import os

class AudioManager:
    """音频管理器，用于管理所有音频资源和播放"""
    
    def __init__(self):
        """初始化音频管理器"""
        # 初始化日志器
        from ..Logger import get_logger
        self.logger = get_logger("AudioManager")
        
        # 音频设备初始化
        try:
            # 初始化pyglet音频
            pyglet.options['audio'] = ('openal', 'directsound', 'silent')
            self.audio_player = pyglet.media.Player()
            self.audio_player.loop = True
            
            # 音频资源字典
            self.audio_resources = {}
            
            # 音效实例字典
            self.sound_instances = {}
            
            # 引擎声音参数
            self.engine_volume = 0.5  # 引擎音量
            self.target_engine_volume = 0.5  # 目标引擎音量
            self.engine_pitch = 1.0  # 引擎音调（随速度变化）
            self.target_engine_pitch = 1.0  # 目标引擎音调
            
            # 环境声音参数
            self.ambient_volume = 0.3  # 环境音量
            self.wind_volume = 0.0  # 风音量
            self.rain_volume = 0.0  # 雨音量
            self.snow_volume = 0.0  # 雪音量
            
            # 初始化音频资源
            self._init_audio_resources()
            
            self.logger.info("音频管理器初始化完成")
        except Exception as e:
            print(f"音频初始化失败: {e}")
            self.logger.error(f"音频初始化失败: {e}")
            # 音频初始化失败时，使用空实现
            self.audio_player = None
            self.audio_resources = {}
            self.sound_instances = {}
    
    def _init_audio_resources(self):
        """初始化音频资源"""
        # 这里应该加载实际的音频文件，但目前简化实现，使用空白音频
        # 后续可以添加实际的音频文件
        
        # 创建空白音频源（用于引擎声等）
        # 实际项目中，应该替换为真实的音频文件
        self.audio_resources['engine_idle'] = None  # 引擎怠速声
        self.audio_resources['engine_full'] = None  # 引擎全速声
        self.audio_resources['wind'] = None  # 风声
        self.audio_resources['rain'] = None  # 雨声
        self.audio_resources['snow'] = None  # 雪声
        self.audio_resources['ambient'] = None  # 环境声
    
    def load_sound(self, name, file_path):
        """加载音频资源
        
        Args:
            name: 音频资源名称
            file_path: 音频文件路径
        """
        try:
            if os.path.exists(file_path):
                # 加载音频文件
                sound = pyglet.media.load(file_path, streaming=False)
                self.audio_resources[name] = sound
                return sound
            else:
                self.logger.error(f"音频文件不存在: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"加载音频文件失败: {e}")
            return None
    
    def play_sound(self, name, loop=False):
        """播放音效
        
        Args:
            name: 音频资源名称
            loop: 是否循环播放
        """
        try:
            if name in self.audio_resources and self.audio_resources[name]:
                # 创建一个新的播放器来播放音效
                player = pyglet.media.Player()
                player.loop = loop
                player.queue(self.audio_resources[name])
                player.play()
                
                # 保存音效实例，以便后续控制
                if name not in self.sound_instances:
                    self.sound_instances[name] = []
                self.sound_instances[name].append(player)
                
                return player
        except Exception as e:
            self.logger.error(f"播放音效失败: {e}")
        return None
    
    def stop_sound(self, name):
        """停止播放指定名称的所有音效
        
        Args:
            name: 音频资源名称
        """
        if name in self.sound_instances:
            for player in self.sound_instances[name]:
                try:
                    player.pause()
                    player.seek(0)
                except Exception as e:
                    self.logger.error(f"停止音效失败: {e}")
            self.sound_instances[name].clear()
    
    def set_engine_volume(self, volume):
        """设置引擎音量
        
        Args:
            volume: 引擎音量（0-1）
        """
        self.target_engine_volume = max(0.0, min(1.0, volume))
    
    def set_engine_pitch(self, pitch):
        """设置引擎音调
        
        Args:
            pitch: 引擎音调（0.5-2.0）
        """
        self.target_engine_pitch = max(0.5, min(2.0, pitch))
    
    def set_ambient_volume(self, volume):
        """设置环境音量
        
        Args:
            volume: 环境音量（0-1）
        """
        self.ambient_volume = max(0.0, min(1.0, volume))
    
    def set_wind_volume(self, volume):
        """设置风音量
        
        Args:
            volume: 风音量（0-1）
        """
        self.wind_volume = max(0.0, min(1.0, volume))
    
    def set_rain_volume(self, volume):
        """设置雨音量
        
        Args:
            volume: 雨音量（0-1）
        """
        self.rain_volume = max(0.0, min(1.0, volume))
    
    def set_snow_volume(self, volume):
        """设置雪音量
        
        Args:
            volume: 雪音量（0-1）
        """
        self.snow_volume = max(0.0, min(1.0, volume))
    
    def update(self, delta_time):
        """更新音频系统
        
        Args:
            delta_time: 帧间隔时间（秒）
        """
        # 平滑过渡引擎音量
        self.engine_volume += (self.target_engine_volume - self.engine_volume) * delta_time * 5.0
        
        # 平滑过渡引擎音调
        self.engine_pitch += (self.target_engine_pitch - self.engine_pitch) * delta_time * 3.0
        
        # 更新音频播放（实际项目中应该更新播放器的音量和音调）
        if self.audio_player:
            try:
                self.audio_player.volume = self.engine_volume
                # 注意：不是所有音频驱动都支持实时调整音调
                # self.audio_player.pitch = self.engine_pitch
            except Exception as e:
                self.logger.error(f"更新音频播放器失败: {e}")
    
    def update_weather_sounds(self, weather_info):
        """根据天气信息更新环境声音
        
        Args:
            weather_info: 天气信息字典
        """
        # 根据天气类型和强度更新环境声音
        if weather_info['rain_enabled']:
            self.set_rain_volume(weather_info['rain_intensity'])
        else:
            self.set_rain_volume(0.0)
        
        if weather_info['snow_enabled']:
            self.set_snow_volume(weather_info['snow_intensity'])
        else:
            self.set_snow_volume(0.0)
        
        # 根据风速更新风音量
        wind_speed = weather_info.get('wind_speed', 0.0)
        self.set_wind_volume(min(wind_speed / 100.0, 1.0))
    
    def update_engine_sounds(self, speed):
        """根据飞行速度更新引擎声音
        
        Args:
            speed: 飞行速度
        """
        # 根据速度更新引擎音量和音调
        max_speed = 200.0  # 最大速度
        engine_power = min(speed / max_speed, 1.0)
        
        # 更新引擎音量
        self.set_engine_volume(0.2 + engine_power * 0.8)
        
        # 更新引擎音调
        self.set_engine_pitch(0.8 + engine_power * 0.4)
    
    def shutdown(self):
        """关闭音频系统，释放资源"""
        try:
            # 停止所有音频播放
            for sound_name, instances in self.sound_instances.items():
                for player in instances:
                    player.pause()
                    player.seek(0)
            
            if self.audio_player:
                self.audio_player.pause()
                self.audio_player.seek(0)
            
            self.logger.info("音频系统已关闭")
        except Exception as e:
            self.logger.error(f"关闭音频系统失败: {e}")
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.shutdown()