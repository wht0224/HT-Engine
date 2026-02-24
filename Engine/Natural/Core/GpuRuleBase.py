from abc import abstractmethod
from typing import Dict, Set, Optional
from .RuleBase import Rule
from .FactBase import FactBase

class GpuRuleBase(Rule):
    """
    GPU规则基类 (GPU Rule Base)
    
    在Rule基础上添加:
    - 纹理脏标记管理
    - 共享纹理支持
    - readback控制
    """
    
    def __init__(self, name: str, priority: int = 0, 
                 manager=None, readback: bool = False,
                 use_shared_textures: bool = True):
        super().__init__(name, priority)
        self.manager = manager
        self.readback = readback
        self.use_shared_textures = use_shared_textures
        
        self._texture_dirty: Dict[str, bool] = {}
        self._texture_initialized: Dict[str, bool] = {}
        self._last_data_hash: Dict[str, int] = {}
        
        self._frame_count = 0
        self._execution_interval = 1
        
    def set_execution_interval(self, interval: int):
        self._execution_interval = max(1, interval)
        
    def should_execute(self) -> bool:
        if self._execution_interval <= 1:
            return True
        return (self._frame_count % self._execution_interval) == 0
    
    def mark_texture_dirty(self, texture_name: str):
        self._texture_dirty[texture_name] = True
        
    def mark_all_textures_dirty(self):
        for name in self._texture_dirty:
            self._texture_dirty[name] = True
            
    def is_texture_dirty(self, texture_name: str) -> bool:
        return self._texture_dirty.get(texture_name, True)
    
    def mark_texture_clean(self, texture_name: str):
        self._texture_dirty[texture_name] = False
        self._texture_initialized[texture_name] = True
        
    def is_texture_initialized(self, texture_name: str) -> bool:
        return self._texture_initialized.get(texture_name, False)
    
    def compute_data_hash(self, data) -> int:
        if data is None:
            return 0
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                # 更高效的hash计算：只采样部分数据
                if data.nbytes > 4096:
                    # 采样头部、中部、尾部
                    step = max(1, len(data) // 3)
                    sample = bytes(data[0]) + bytes(data[step]) + bytes(data[-1])
                    return hash((data.shape, data.dtype, sample))
                return hash(data.tobytes())
            return hash(data)
        except:
            return 0
    
    def has_data_changed(self, texture_name: str, data) -> bool:
        new_hash = self.compute_data_hash(data)
        old_hash = self._last_data_hash.get(texture_name, None)
        if old_hash != new_hash:
            self._last_data_hash[texture_name] = new_hash
            return True
        return False
    
    def should_upload_texture(self, texture_name: str, data=None) -> bool:
        if not self.is_texture_initialized(texture_name):
            return True
        if self.is_texture_dirty(texture_name):
            return True
        if data is not None and self.has_data_changed(texture_name, data):
            return True
        return False
    
    def get_shared_texture(self, name: str):
        if self.manager and self.use_shared_textures:
            return self.manager.get_texture(name)
        return None
    
    def register_shared_texture(self, name: str, texture):
        if self.manager and self.use_shared_textures:
            self.manager.register_texture(name, texture)
    
    @abstractmethod
    def evaluate(self, facts: FactBase) -> None:
        pass
    
    def _on_evaluate_start(self):
        self._frame_count += 1
        
    def _on_evaluate_end(self):
        pass
