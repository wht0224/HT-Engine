# -*- coding: utf-8 -*-
"""
沉浸式游戏UI
Minimalist Game UI for immersive experience
"""

import tkinter as tk
from tkinter import messagebox
import sys
import os
import time
from OpenGL.GLUT import glutInit

from Engine.Logger import get_logger
from Engine.UI.TkUI.theme.dark_theme import DarkTheme
from Engine.UI.TkUI.opengl_viewport import OpenGLViewport

class GameUI:
    """
    游戏模式UI
    无边框、全屏、无菜单栏，专注于游戏体验
    """
    
    def __init__(self, engine):
        """
        初始化游戏UI
        """
        self.logger = get_logger("GameUI")
        self.engine = engine
        self.running = False
        
        # 创建主窗口
        self.root = tk.Tk()
        self.root.title("Scottish Highlands - Playable Demo")
        
        # 初始化GLUT (解决字体渲染错误)
        try:
            glut_argv = [arg.encode('utf-8') for arg in sys.argv]
            glutInit(glut_argv)
        except Exception as e:
            self.logger.error(f"GLUT initialization failed: {e}")
        
        # 设置全屏
        self.root.attributes('-fullscreen', True)
        self.width = self.root.winfo_screenwidth()
        self.height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.width}x{self.height}")
        
        # 应用暗色主题
        DarkTheme.apply(self.root)
        
        # 创建视口
        self.viewport = OpenGLViewport(self.root, self.engine)
        self.viewport.pack(fill=tk.BOTH, expand=True)
        
        # Input State
        self.input_state = {
            'w': False, 'a': False, 's': False, 'd': False,
            'space': False, 'shift': False,
            'mouse_dx': 0, 'mouse_dy': 0
        }

        # Share input state with engine - 强制设置
        self.engine.input_state = self.input_state
        self.logger.info(f"input_state 已共享到 Engine: {id(self.input_state)}")
        
        # Mouse handling
        self.mouse_locked = True
        self.center_x = self.width // 2
        self.center_y = self.height // 2
        
        # 绑定事件
        self.root.bind("<Escape>", self._on_escape)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        
        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.bind("<Motion>", self._on_mouse_move)
        
        # 隐藏光标
        self.root.config(cursor="none")
        
        # FPS 显示
        self.fps_label = tk.Label(
            self.root, 
            text="FPS: 0", 
            font=("Consolas", 14),
            fg="#00ff00",
            bg="#000000",
            bd=0,
            padx=10,
            pady=5
        )
        self.fps_label.place(x=10, y=10)
        
        # FPS 计算变量
        self.frame_count = 0
        self.last_fps_update = time.time()
        
        self.logger.info("游戏UI初始化完成")
        
    def _on_key_press(self, event):
        key = event.keysym.lower()
        if key in self.input_state:
            self.input_state[key] = True
            
    def _on_key_release(self, event):
        key = event.keysym.lower()
        if key in self.input_state:
            self.input_state[key] = False

    def _on_mouse_move(self, event):
        if not self.mouse_locked:
            return
            
        dx = event.x - self.center_x
        dy = event.y - self.center_y
        
        if dx == 0 and dy == 0:
            return
            
        self.input_state['mouse_dx'] += dx
        self.input_state['mouse_dy'] += dy
        
        try:
            self.root.event_generate('<Motion>', warp=True, x=self.center_x, y=self.center_y)
        except Exception:
            pass

    def _on_escape(self, event):
        """处理ESC按键"""
        # Unlock mouse to allow interaction with dialog
        self.mouse_locked = False
        self.root.config(cursor="")
        
        if messagebox.askyesno("退出", "确定要退出游戏吗？"):
            self._on_close()
        else:
            # Resume
            self.mouse_locked = True
            self.root.config(cursor="none")
            self.root.event_generate('<Motion>', warp=True, x=self.center_x, y=self.center_y)
            
    def _on_close(self):
        """关闭窗口"""
        self.running = False
        self.root.quit()
        self.root.destroy()
        if self.engine:
            self.engine.shutdown()

    def _update_fps(self):
        """更新 FPS 显示"""
        current_time = time.time()
        self.frame_count += 1
        
        if current_time - self.last_fps_update >= 1.0:
            fps = self.frame_count
            self.fps_label.config(text=f"FPS: {fps}")
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def update_loop(self):
        if not self.running:
            return
            
        # 更新 FPS 显示
        self._update_fps()
        
        # Schedule next update (aim for 60 FPS)
        self.root.after(16, self.update_loop)

    def mainloop(self):
        """启动主循环"""
        self.running = True
        self.last_time = time.time()
        
        # Schedule first FPS update
        self.root.after(100, self.update_loop)
        
        # Initial mouse center
        self.root.event_generate('<Motion>', warp=True, x=self.center_x, y=self.center_y)
        
        self.root.mainloop()
