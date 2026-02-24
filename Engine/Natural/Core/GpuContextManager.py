import moderngl
import logging

class GpuContextManager:
    """
    GPU 上下文管理器 (Singleton-like)
    
    负责统一管理 ModernGL 上下文，避免多个规则重复创建上下文导致的性能开销。
    所有 GPU 规则应共享同一个 Context。
    """
    
    _instance = None
    
    def __init__(self, context=None, deferred=False):
        self.logger = logging.getLogger("Natural.GpuContextManager")
        self.ctx = context
        self.textures = {}  # Shared texture registry
        
        if not self.ctx and not deferred:
            self._init_context()
            
    def set_context(self, context):
        """Set or replace the context manually."""
        if self.ctx:
            # Maybe release old one if we own it?
            pass
        self.ctx = context
        self.logger.info("OpenGL Context manually set")
        
    def get_texture(self, name):
        """Retrieve a shared texture by name."""
        return self.textures.get(name)

    def register_texture(self, name, texture):
        """Register a shared texture."""
        self.textures[name] = texture
        
    def _init_context(self):
        try:
            # 尝试创建独立上下文 (Headless)
            self.ctx = moderngl.create_context(standalone=True)
            self.logger.info("OpenGL Context created successfully (standalone)")
            
            # 打印一些 GPU 信息
            self.logger.info(f"Vendor: {self.ctx.info['GL_VENDOR']}")
            self.logger.info(f"Renderer: {self.ctx.info['GL_RENDERER']}")
            self.logger.info(f"Version: {self.ctx.info['GL_VERSION']}")
            
        except Exception as e:
            self.logger.error(f"Failed to create OpenGL context: {e}")
            self.ctx = None

    @property
    def context(self):
        return self.ctx
        
    def release(self):
        if self.ctx:
            self.ctx.release()
            self.ctx = None
            self.logger.info("OpenGL Context released")
