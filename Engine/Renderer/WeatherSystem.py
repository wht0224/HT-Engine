# -*- coding: utf-8 -*-
"""
天气和大气系统
用于实现真实的大气效果和天气变化
"""

from Engine.Math import Vector3, Quaternion, Matrix4x4
import numpy as np
import math

class AtmosphereSystem:
    """大气系统，用于实现大气散射和天空渲染"""
    
    def __init__(self, renderer):
        """初始化大气系统
        
        Args:
            renderer: 渲染器对象
        """
        self.renderer = renderer
        
        # 天空渲染开关
        self.sky_enabled = True  # 是否渲染天空
        
        # 大气散射参数
        self.atmosphere_radius = 6400000.0  # 大气半径
        self.planet_radius = 6360000.0  # 行星半径
        
        # 大气成分参数
        self.rayleigh_scale = 8.0e3  # 瑞利散射尺度高度
        self.mie_scale = 1.2e3  # 米氏散射尺度高度
        self.rayleigh_scattering = Vector3(5.8e-6, 1.35e-5, 3.31e-5)  # 瑞利散射系数
        self.mie_scattering = Vector3(2.0e-5, 2.0e-5, 2.0e-5)  # 米氏散射系数
        self.absorption = Vector3(2.0e-6, 4.0e-6, 8.0e-6)  # 吸收系数
        self.absorption_scale = 6.0e3  # 吸收尺度高度
        
        # 太阳参数
        self.sun_direction = Vector3(0.5, 0.5, 0.5).normalized()  # 太阳方向
        self.sun_intensity = 10.0  # 太阳强度
        self.mie_g = 0.8  # 米氏散射不对称因子
        
        # 天空颜色
        self.sky_color = Vector3(0.5, 0.7, 0.9)  # 默认天空颜色
        
        # 雾参数
        self.fog_enabled = True  # 是否启用雾
        self.fog_density = 0.001  # 雾密度
        self.fog_color = Vector3(0.7, 0.8, 0.9)  # 雾颜色
        self.fog_near = 10.0  # 雾近裁剪
        self.fog_far = 1000.0  # 雾远裁剪
        
        # 时间参数
        self.time_of_day = 12.0  # 当前时间（小时）
        self.day_length = 24.0  # 一天的长度（秒）
        self.time_scale = 1.0  # 时间缩放（1.0 = 真实时间）
        self.is_time_paused = False  # 时间是否暂停
        
        # 更新大气参数
        self._update_atmosphere_parameters()
    
    def set_time_of_day(self, time):
        """设置一天中的时间
        
        Args:
            time: 时间（小时，0-24）
        """
        self.time_of_day = time % 24.0
        self._update_sun_position()
        self._update_atmosphere_parameters()
    
    def update(self, delta_time):
        """更新大气系统
        
        Args:
            delta_time: 帧间隔时间（秒）
        """
        # 动态时间变化
        if not self.is_time_paused:
            self.time_of_day += delta_time * self.time_scale * (24.0 / self.day_length)
            self.time_of_day %= 24.0
            self._update_sun_position()
            self._update_atmosphere_parameters()
    
    def _update_sun_position(self):
        """根据时间更新太阳位置"""
        # 将时间转换为角度（0-2π），让太阳在12点时达到最高点
        # 6点日出，12点正午，18点日落
        angle = ((self.time_of_day - 6.0) / 24.0) * 2 * math.pi
        
        # 计算太阳方向
        # 太阳在天球上运动，y轴为垂直方向
        self.sun_direction.x = math.cos(angle)
        self.sun_direction.y = math.sin(angle)
        self.sun_direction.z = 0.0
        self.sun_direction = self.sun_direction.normalized()
    
    def _update_atmosphere_parameters(self):
        """更新大气参数"""
        # 根据太阳高度角计算天空颜色
        sun_height = self.sun_direction.y
        
        if sun_height > 0.0:
            # 白天
            # 根据太阳高度角调整天空颜色
            if sun_height < 0.2:
                # 日出/日落
                self.sky_color = Vector3(
                    1.0 - 0.5 * sun_height,
                    0.5 - 0.3 * sun_height,
                    0.2 + 0.1 * sun_height
                )
            else:
                # 白天
                self.sky_color = Vector3(
                    0.5 + 0.4 * sun_height,
                    0.6 + 0.3 * sun_height,
                    0.8 + 0.2 * sun_height
                )
            self.fog_color = self.sky_color * 0.8
            self.sun_intensity = 10.0 * max(0.1, sun_height)
        else:
            # 夜晚
            night_intensity = abs(sun_height) / 2.0  # 夜晚亮度随太阳高度变化
            self.sky_color = Vector3(
                0.02 + night_intensity * 0.03,
                0.02 + night_intensity * 0.03,
                0.05 + night_intensity * 0.1
            )
            self.fog_color = self.sky_color * 0.8
            self.sun_intensity = 0.1
    
    def get_sky_color(self):
        """获取当前天空颜色
        
        Returns:
            Vector3: 天空颜色
        """
        return self.sky_color
    
    def get_fog_parameters(self):
        """获取雾参数
        
        Returns:
            tuple: (enabled, density, color, near, far)
        """
        return (
            self.fog_enabled,
            self.fog_density,
            self.fog_color,
            self.fog_near,
            self.fog_far
        )
    
    def apply_fog(self, color, distance, camera_position, world_position):
        """应用雾效果
        
        Args:
            color: 原始颜色
            distance: 到相机的距离
            camera_position: 相机位置
            world_position: 世界位置
            
        Returns:
            Vector3: 应用雾效果后的颜色
        """
        if not self.fog_enabled:
            return color
        
        # 简化的雾计算
        fog_factor = 1.0 - math.exp(-self.fog_density * distance)
        fog_factor = min(1.0, max(0.0, fog_factor))
        
        # 线性雾混合
        return color * (1.0 - fog_factor) + self.fog_color * fog_factor
    
    def render_sky(self, camera):
        """渲染天空
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glBegin, glEnd, glVertex3f, glColor3f, glDepthMask, glLoadIdentity,
            GL_TRIANGLE_FAN, GL_QUADS, glDisable, GL_DEPTH_TEST, glEnable
        )
        
        # 保存当前渲染状态
        glDepthMask(False)
        glDisable(GL_DEPTH_TEST)
        glLoadIdentity()
        
        # 绘制天空半球
        glBegin(GL_TRIANGLE_FAN)
        
        # 设置天空颜色
        sky_color = self.sky_color
        glColor3f(sky_color.x, sky_color.y, sky_color.z)
        
        # 中心顶点（相机位置）
        glVertex3f(0, 0, 0)
        
        # 绘制半球
        segments = 32
        for i in range(segments + 1):
            angle = i * 2.0 * math.pi / segments
            x = math.cos(angle)
            z = math.sin(angle)
            glVertex3f(x * 1000, 1000, z * 1000)  # 半径为1000的半球
        
        glEnd()
        
        # 绘制太阳
        if self.sun_direction.y > 0.0:
            self._render_sun(camera)
        
        # 恢复渲染状态
        glEnable(GL_DEPTH_TEST)
        glDepthMask(True)
    
    def _render_sun(self, camera):
        """渲染太阳
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glPushMatrix, glPopMatrix, glTranslatef, glRotatef, glScalef,
            glBegin, glEnd, glVertex3f, glColor3f, GL_TRIANGLE_FAN
        )
        
        sun_size = 30.0  # 太阳大小
        sun_distance = 500.0  # 太阳距离
        
        # 计算太阳位置
        sun_pos = self.sun_direction * sun_distance
        
        # 保存矩阵
        glPushMatrix()
        
        # 设置太阳位置
        glTranslatef(sun_pos.x, sun_pos.y, sun_pos.z)
        
        # 绘制太阳（黄色圆形）
        glColor3f(1.0, 1.0, 0.0)
        glBegin(GL_TRIANGLE_FAN)
        
        # 中心
        glVertex3f(0, 0, 0)
        
        # 绘制圆形
        segments = 16
        for i in range(segments + 1):
            angle = i * 2.0 * math.pi / segments
            x = math.cos(angle) * sun_size
            y = math.sin(angle) * sun_size
            glVertex3f(x, y, 0)
        
        glEnd()
        
        # 恢复矩阵
        glPopMatrix()

class WeatherSystem:
    """天气系统，用于实现天气变化"""
    
    # 天气类型枚举
    class Type:
        CLEAR = 0  # 晴天
        CLOUDY = 1  # 多云
        RAIN = 2  # 雨天
        SNOW = 3  # 雪天
        FOGGY = 4  # 雾天
    
    def __init__(self, atmosphere_system):
        """初始化天气系统
        
        Args:
            atmosphere_system: 大气系统
        """
        self.atmosphere = atmosphere_system
        self.weather_type = self.Type.CLEAR
        self.target_weather_type = self.Type.CLEAR  # 目标天气类型（用于过渡）
        
        # 云参数
        self.cloud_density = 0.0  # 当前云密度
        self.target_cloud_density = 0.0  # 目标云密度
        self.cloud_coverage = 0.0  # 当前云覆盖率
        self.target_cloud_coverage = 0.0  # 目标云覆盖率
        self.cloud_height = 1000.0  # 云高度
        
        # 雨参数
        self.rain_enabled = False  # 是否下雨
        self.rain_intensity = 0.0  # 当前雨强度
        self.target_rain_intensity = 0.0  # 目标雨强度
        
        # 雪参数
        self.snow_enabled = False  # 是否下雪
        self.snow_intensity = 0.0  # 当前雪强度
        self.target_snow_intensity = 0.0  # 目标雪强度
        
        # 风速
        self.wind_speed = 0.0  # 风速
        self.wind_direction = Vector3(0, 0, 1).normalized()  # 风向
        
        # 动态天气参数
        self.is_dynamic_weather = True  # 是否启用动态天气
        self.weather_change_interval = 60.0  # 天气变化间隔（秒）
        self.weather_change_timer = 0.0  # 天气变化计时器
        self.weather_transition_time = 5.0  # 天气过渡时间（秒）
        self.weather_transition_timer = 0.0  # 天气过渡计时器
        self.is_transitioning = False  # 是否正在进行天气过渡
        
        # 初始化天气
        self.set_weather_type(self.Type.CLEAR)
    
    def set_weather_type(self, weather_type, immediate=False):
        """设置天气类型
        
        Args:
            weather_type: 天气类型
            immediate: 是否立即生效（不使用过渡效果）
        """
        self.target_weather_type = weather_type
        
        # 根据天气类型设置目标参数
        if weather_type == self.Type.CLEAR:
            self._set_target_clear_weather()
        elif weather_type == self.Type.CLOUDY:
            self._set_target_cloudy_weather()
        elif weather_type == self.Type.RAIN:
            self._set_target_rainy_weather()
        elif weather_type == self.Type.SNOW:
            self._set_target_snowy_weather()
        elif weather_type == self.Type.FOGGY:
            self._set_target_foggy_weather()
        
        if immediate:
            # 立即设置为目标参数
            self._apply_target_parameters()
        else:
            # 开始天气过渡
            self.is_transitioning = True
            self.weather_transition_timer = 0.0
    
    def toggle_weather(self):
        """切换天气类型"""
        self.set_weather_type((self.weather_type + 1) % 5)
        return self.weather_type
    
    def _set_target_clear_weather(self):
        """设置晴天目标参数"""
        self.target_cloud_density = 0.1
        self.target_cloud_coverage = 0.2
        self.target_rain_intensity = 0.0
        self.target_snow_intensity = 0.0
    
    def _set_target_cloudy_weather(self):
        """设置多云目标参数"""
        self.target_cloud_density = 0.5
        self.target_cloud_coverage = 0.8
        self.target_rain_intensity = 0.0
        self.target_snow_intensity = 0.0
    
    def _set_target_rainy_weather(self):
        """设置雨天目标参数"""
        self.target_cloud_density = 0.8
        self.target_cloud_coverage = 1.0
        self.target_rain_intensity = 0.5
        self.target_snow_intensity = 0.0
    
    def _set_target_snowy_weather(self):
        """设置雪天目标参数"""
        self.target_cloud_density = 0.7
        self.target_cloud_coverage = 0.9
        self.target_rain_intensity = 0.0
        self.target_snow_intensity = 0.6
    
    def _set_target_foggy_weather(self):
        """设置雾天目标参数"""
        self.target_cloud_density = 0.3
        self.target_cloud_coverage = 0.5
        self.target_rain_intensity = 0.0
        self.target_snow_intensity = 0.0
    
    def _apply_target_parameters(self):
        """应用目标参数（立即生效）"""
        self.weather_type = self.target_weather_type
        self.cloud_density = self.target_cloud_density
        self.cloud_coverage = self.target_cloud_coverage
        self.rain_intensity = self.target_rain_intensity
        self.snow_intensity = self.target_snow_intensity
        
        # 更新雨和雪的启用状态
        self.rain_enabled = self.rain_intensity > 0.0
        self.snow_enabled = self.snow_intensity > 0.0
        
        # 更新雾参数
        if self.weather_type == self.Type.CLEAR:
            self.atmosphere.fog_density = 0.0005
            self.atmosphere.fog_color = Vector3(0.7, 0.8, 0.9)
        elif self.weather_type == self.Type.CLOUDY:
            self.atmosphere.fog_density = 0.001
            self.atmosphere.fog_color = Vector3(0.6, 0.7, 0.8)
        elif self.weather_type == self.Type.RAIN:
            self.atmosphere.fog_density = 0.003
            self.atmosphere.fog_color = Vector3(0.5, 0.6, 0.7)
        elif self.weather_type == self.Type.SNOW:
            self.atmosphere.fog_density = 0.002
            self.atmosphere.fog_color = Vector3(0.8, 0.8, 0.9)
        elif self.weather_type == self.Type.FOGGY:
            self.atmosphere.fog_density = 0.005
            self.atmosphere.fog_color = Vector3(0.7, 0.7, 0.8)
    
    def _update_weather_transition(self, delta_time):
        """更新天气过渡"""
        if not self.is_transitioning:
            return
        
        # 更新过渡计时器
        self.weather_transition_timer += delta_time
        progress = min(1.0, self.weather_transition_timer / self.weather_transition_time)
        
        # 使用平滑的缓动函数
        progress = 1 - math.pow(1 - progress, 3)  # 三次缓动
        
        # 平滑过渡各项参数
        self.cloud_density = self.cloud_density + (self.target_cloud_density - self.cloud_density) * progress
        self.cloud_coverage = self.cloud_coverage + (self.target_cloud_coverage - self.cloud_coverage) * progress
        self.rain_intensity = self.rain_intensity + (self.target_rain_intensity - self.rain_intensity) * progress
        self.snow_intensity = self.snow_intensity + (self.target_snow_intensity - self.snow_intensity) * progress
        
        # 更新雨和雪的启用状态
        self.rain_enabled = self.rain_intensity > 0.1
        self.snow_enabled = self.snow_intensity > 0.1
        
        # 过渡完成
        if progress >= 1.0:
            self.is_transitioning = False
            self.weather_type = self.target_weather_type
            self._apply_target_parameters()
    
    def _update_dynamic_weather(self, delta_time):
        """更新动态天气变化"""
        if not self.is_dynamic_weather:
            return
        
        # 更新天气变化计时器
        self.weather_change_timer += delta_time
        
        # 时间到，切换天气
        if self.weather_change_timer >= self.weather_change_interval:
            # 随机选择新的天气类型（避免连续相同天气）
            new_weather = self.weather_type
            while new_weather == self.weather_type:
                new_weather = int(np.random.randint(0, 5))
            
            # 设置新的天气类型
            self.set_weather_type(new_weather)
            
            # 重置计时器
            self.weather_change_timer = 0.0
            
            # 随机调整下次天气变化间隔
            self.weather_change_interval = 60.0 + np.random.rand() * 120.0  # 60-180秒
    
    def update(self, delta_time):
        """更新天气系统
        
        Args:
            delta_time: 帧间隔时间（秒）
        """
        # 更新天气过渡
        self._update_weather_transition(delta_time)
        
        # 更新动态天气
        self._update_dynamic_weather(delta_time)
    
    def get_weather_info(self):
        """获取当前天气信息
        
        Returns:
            dict: 天气信息
        """
        return {
            "type": self.weather_type,
            "cloud_density": self.cloud_density,
            "cloud_coverage": self.cloud_coverage,
            "rain_enabled": self.rain_enabled,
            "rain_intensity": self.rain_intensity,
            "snow_enabled": self.snow_enabled,
            "snow_intensity": self.snow_intensity,
            "wind_speed": self.wind_speed,
            "wind_direction": self.wind_direction
        }
    
    def get_weather_type_name(self):
        """获取当前天气类型名称
        
        Returns:
            str: 天气类型名称
        """
        weather_names = ["晴天", "多云", "雨天", "雪天", "雾天"]
        return weather_names[self.weather_type]
    
    def render_weather(self, camera):
        """渲染天气效果
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glDepthMask, glDisable, GL_DEPTH_TEST, glEnable
        )
        
        # 保存当前渲染状态
        glDepthMask(False)
        glDisable(GL_DEPTH_TEST)
        
        # 渲染云层
        if self.cloud_coverage > 0.0:
            self._render_clouds(camera)
        
        # 渲染雨
        if self.rain_enabled and self.rain_intensity > 0.0:
            self._render_rain(camera)
        
        # 渲染雪
        if self.snow_enabled and self.snow_intensity > 0.0:
            self._render_snow(camera)
        
        # 恢复渲染状态
        glEnable(GL_DEPTH_TEST)
        glDepthMask(True)
    
    def _render_clouds(self, camera):
        """渲染云层
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glBegin, glEnd, glVertex3f, glColor3f, GL_TRIANGLES
        )
        
        # 简化的云层渲染
        num_clouds = int(self.cloud_coverage * 50)  # 根据云覆盖率渲染不同数量的云
        cloud_color = Vector3(0.8, 0.8, 0.9)  # 云的颜色
        cloud_alpha = 0.6 * self.cloud_density  # 云的透明度
        
        # 设置云的颜色
        glColor3f(cloud_color.x, cloud_color.y, cloud_color.z)
        
        # 绘制简单的云（使用三角形表示）
        for i in range(num_clouds):
            # 随机生成云的位置和大小
            cloud_x = (i * 137.5) % 1000 - 500  # 使用黄金比例分布
            cloud_z = (i * 275.0) % 1000 - 500
            cloud_y = 200 + (i * 53.0) % 300  # 云的高度在200-500之间
            cloud_size = 50 + (i * 7.0) % 50  # 云的大小在50-100之间
            
            # 绘制云（简化为三角形）
            glBegin(GL_TRIANGLES)
            # 云的三个顶点
            glVertex3f(cloud_x, cloud_y, cloud_z)
            glVertex3f(cloud_x + cloud_size, cloud_y + cloud_size/2, cloud_z + cloud_size/2)
            glVertex3f(cloud_x - cloud_size, cloud_y + cloud_size/3, cloud_z + cloud_size/2)
            glEnd()
    
    def _render_rain(self, camera):
        """渲染雨
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glBegin, glEnd, glVertex3f, glColor3f, GL_LINES, glLineWidth
        )
        
        num_drops = int(self.rain_intensity * 1000)  # 根据雨强度渲染不同数量的雨滴
        drop_length = 20.0  # 雨滴长度
        drop_speed = 50.0  # 雨滴速度
        drop_color = Vector3(0.5, 0.6, 0.7)  # 雨滴颜色
        
        # 设置雨滴颜色
        glColor3f(drop_color.x, drop_color.y, drop_color.z)
        glLineWidth(1.0)  # 雨滴线宽
        
        # 绘制雨滴（使用线条表示）
        for i in range(num_drops):
            # 随机生成雨滴的位置
            drop_x = (i * 137.5) % 200 - 100
            drop_z = (i * 275.0) % 200 - 100
            drop_y = (i * 53.0) % 500  # 雨滴的垂直位置
            
            # 绘制雨滴
            glBegin(GL_LINES)
            glVertex3f(drop_x, drop_y, drop_z)
            glVertex3f(drop_x, drop_y - drop_length, drop_z)
            glEnd()
    
    def _render_snow(self, camera):
        """渲染雪
        
        Args:
            camera: 相机对象
        """
        from OpenGL.GL import (
            glBegin, glEnd, glVertex3f, glColor3f, GL_POINTS, glPointSize
        )
        
        num_flakes = int(self.snow_intensity * 800)  # 根据雪强度渲染不同数量的雪花
        flake_size = 2.0  # 雪花大小
        flake_color = Vector3(1.0, 1.0, 1.0)  # 雪花颜色
        
        # 设置雪花颜色和大小
        glColor3f(flake_color.x, flake_color.y, flake_color.z)
        glPointSize(flake_size)
        
        # 绘制雪花（使用点表示）
        glBegin(GL_POINTS)
        for i in range(num_flakes):
            # 随机生成雪花的位置
            flake_x = (i * 137.5) % 200 - 100
            flake_z = (i * 275.0) % 200 - 100
            flake_y = (i * 53.0) % 500  # 雪花的垂直位置
            
            # 绘制雪花
            glVertex3f(flake_x, flake_y, flake_z)
        glEnd()
        
        # 重置点大小
        glPointSize(1.0)