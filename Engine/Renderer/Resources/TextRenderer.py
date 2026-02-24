# -*- coding: utf-8 -*-
"""
文本渲染器类，用于在OpenGL中渲染真实文本
"""

import freetype
import numpy as np
from Engine.Math import Vector2, Vector3

class TextRenderer:
    """文本渲染器类，使用freetype-py在OpenGL中渲染文本"""
    
    def __init__(self, font_path=None, font_size=16):
        """初始化文本渲染器
        
        Args:
            font_path: 字体文件路径，如果为None则使用默认字体
            font_size: 字体大小
        """
        self.font_path = font_path
        self.font_size = font_size
        
        # 字体库
        self.ft_lib = None
        self.face = None
        
        # 字符缓存
        self.characters = {}
        
        # 初始化FreeType库
        self._init_freetype()
        
        # 加载字体
        self.load_font(font_path, font_size)
    
    def _init_freetype(self):
        """初始化FreeType库"""
        try:
            self.ft_lib = freetype.Face.__default_lib
            if not self.ft_lib:
                self.ft_lib = freetype.Library()
        except Exception as e:
            print(f"初始化FreeType库失败: {e}")
            self.ft_lib = None
    
    def load_font(self, font_path=None, font_size=None):
        """加载字体
        
        Args:
            font_path: 字体文件路径
            font_size: 字体大小
        """
        if not self.ft_lib:
            return False
        
        if font_path:
            self.font_path = font_path
        if font_size:
            self.font_size = font_size
        
        try:
            if self.font_path:
                self.face = freetype.Face(self.ft_lib, self.font_path)
            else:
                # 尝试加载默认字体
                self.face = freetype.Face(self.ft_lib, "arial.ttf")
            
            # 设置字体大小
            self.face.set_pixel_sizes(0, self.font_size)
            return True
        except Exception as e:
            print(f"加载字体失败: {e}")
            try:
                # 尝试加载另一种默认字体
                self.face = freetype.Face(self.ft_lib, "DejaVuSans.ttf")
                self.face.set_pixel_sizes(0, self.font_size)
                return True
            except:
                self.face = None
                return False
    
    def render_text(self, text, x, y, color=(1.0, 1.0, 1.0, 1.0)):
        """渲染文本
        
        Args:
            text: 要渲染的文本
            x: 文本X坐标
            y: 文本Y坐标
            color: 文本颜色
        """
        if not self.face:
            return
        
        # 使用OpenGL直接渲染文本
        from OpenGL.GL import (
            glBegin, glEnd, glVertex2f, glColor4f, GL_QUADS,
            glTexCoord2f, glEnable, glDisable, GL_TEXTURE_2D,
            glBlendFunc, GL_BLEND, GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA
        )
        
        # 启用混合
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        current_x = x
        
        for char in text:
            if char not in self.characters:
                # 加载字符
                self._load_character(char)
            
            # 获取字符信息
            char_info = self.characters[char]
            
            # 计算字符位置
            x0 = current_x + char_info['bearing'][0]
            y0 = y - (char_info['size'][1] - char_info['bearing'][1])
            x1 = x0 + char_info['size'][0]
            y1 = y0 + char_info['size'][1]
            
            # 绘制字符（简化实现，使用矩形表示）
            glBegin(GL_QUADS)
            glColor4f(color[0], color[1], color[2], color[3])
            
            glVertex2f(x0, y0)
            glVertex2f(x1, y0)
            glVertex2f(x1, y1)
            glVertex2f(x0, y1)
            
            glEnd()
            
            # 更新X位置
            current_x += char_info['advance']
        
        # 禁用混合
        glDisable(GL_BLEND)
    
    def _load_character(self, char):
        """加载字符
        
        Args:
            char: 要加载的字符
        """
        if not self.face:
            return
        
        # 加载字符
        self.face.load_char(char, freetype.FT_LOAD_RENDER)
        glyph = self.face.glyph
        
        # 获取字符信息
        bitmap = glyph.bitmap
        width = bitmap.width
        height = bitmap.rows
        
        # 计算字符信息
        char_info = {
            'size': (width, height),
            'bearing': (glyph.bitmap_left, glyph.bitmap_top),
            'advance': glyph.advance.x >> 6  # 转换为像素
        }
        
        # 保存字符信息
        self.characters[char] = char_info
    
    def get_text_size(self, text):
        """获取文本大小
        
        Args:
            text: 要测量的文本
            
        Returns:
            tuple: (width, height)
        """
        if not self.face:
            return (0, 0)
        
        width = 0
        max_height = 0
        
        for char in text:
            if char not in self.characters:
                self._load_character(char)
            
            char_info = self.characters[char]
            width += char_info['advance']
            char_height = char_info['size'][1]
            if char_height > max_height:
                max_height = char_height
        
        return (width, max_height)
    
    def set_font_size(self, font_size):
        """设置字体大小
        
        Args:
            font_size: 新的字体大小
        """
        self.font_size = font_size
        if self.face:
            self.face.set_pixel_sizes(0, font_size)
            # 清除字符缓存
            self.characters.clear()
    
    def destroy(self):
        """销毁文本渲染器"""
        # 清除字符缓存
        self.characters.clear()
        
        # 释放FreeType资源
        if self.face:
            del self.face
        
        if self.ft_lib:
            del self.ft_lib