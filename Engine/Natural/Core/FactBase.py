import numpy as np
from typing import Dict, Any, List, Optional
import logging

class FactBase:
    """
    事实库 (FactBase) - Natural 模块的认知核心
    
    采用 SoA (Structure of Arrays) 布局存储所有运行时状态。
    拒绝对象化，追求极致的数据局部性。
    """
    
    def __init__(self):
        self.logger = logging.getLogger("Natural.FactBase")
        
        # 全局事实 (Global Facts)
        # 存储如风向、时间、温度等环境全局变量
        self.globals: Dict[str, Any] = {
            "time": 0.0,
            "wind_direction": np.array([1.0, 0.0, 0.0], dtype=np.float32),
            "wind_speed": 0.0,
            "temperature": 20.0,
        }
        
        # 实体事实 (Entity Facts) - SoA Tables
        # Key: Table Name (e.g., "vegetation", "rocks")
        # Value: Dict of numpy arrays (Columns)
        self.tables: Dict[str, Dict[str, np.ndarray]] = {}
        
        # 实体数量记录
        self.counts: Dict[str, int] = {}
        
        # 脏标记 (Dirty Flags) - 用于优化增量更新
        # Key: Global Key or Table Name
        self.dirty_flags: Dict[str, bool] = {}

    def set_global(self, key: str, value: Any):
        """设置全局事实"""
        if key in self.globals and np.array_equal(self.globals[key], value):
            return # 值未变，无需触发脏标记
            
        self.globals[key] = value
        self.dirty_flags[f"global.{key}"] = True

    def get_global(self, key: str) -> Any:
        """获取全局事实"""
        return self.globals.get(key)

    def create_table(self, name: str, capacity: int, schema: Dict[str, Any]):
        """
        创建一个新的实体表 (SoA)
        
        Args:
            name: 表名 (如 "vegetation")
            capacity: 预分配容量
            schema: 字段定义 {字段名: dtype}
        """
        self.tables[name] = {}
        self.counts[name] = 0
        
        for col_name, dtype in schema.items():
            self.tables[name][col_name] = np.zeros(capacity, dtype=dtype)
            
        self.logger.info(f"Created table '{name}' with capacity {capacity}")

    def add_entity(self, table_name: str, **kwargs) -> int:
        """
        添加一个实体到指定表
        
        Returns:
            entity_index
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
            
        idx = self.counts[table_name]
        table = self.tables[table_name]
        
        # 检查容量 (简化版暂不处理自动扩容，假设容量足够)
        if idx >= len(next(iter(table.values()))):
            raise IndexError(f"Table {table_name} is full")
            
        # 填充数据
        for col, val in kwargs.items():
            if col in table:
                table[col][idx] = val
                
        self.counts[table_name] += 1
        self.dirty_flags[table_name] = True
        return idx

    def get_count(self, table_name: str) -> int:
        """获取实体数量"""
        return self.counts.get(table_name, 0)

    def set_count(self, table_name: str, count: int):
        """
        强制设置实体数量 (用于批量加载数据)
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        # Check capacity
        capacity = len(next(iter(self.tables[table_name].values())))
        if count > capacity:
             raise ValueError(f"Count {count} exceeds capacity {capacity}")
        self.counts[table_name] = count

    def get_column(self, table_name: str, column_name: str) -> np.ndarray:
        """
        获取整列数据 (用于向量化计算)
        
        Returns:
            View of the numpy array (sliced to valid count)
        """
        if table_name not in self.tables:
            raise KeyError(table_name)
        if column_name not in self.tables[table_name]:
            raise KeyError(column_name)

        count = self.counts.get(table_name, 0)
        return self.tables[table_name][column_name][:count]

    def set_column(self, table_name: str, column_name: str, values: np.ndarray):
        """
        设置整列数据 (用于规则回写)
        """
        count = self.counts.get(table_name, 0)
        if len(values) != count:
            raise ValueError(f"Values length {len(values)} mismatch with entity count {count}")
            
        self.tables[table_name][column_name][:count] = values
        self.dirty_flags[table_name] = True

    def add_column(self, table_name: str, column_name: str, values: np.ndarray):
        """
        添加新列到现有表
        
        Args:
            table_name: 表名
            column_name: 新列名
            values: 初始值数组
        """
        if table_name not in self.tables:
            raise ValueError(f"Table {table_name} does not exist")
        
        if column_name in self.tables[table_name]:
            # 列已存在，直接设置值
            self.set_column(table_name, column_name, values)
            return
        
        # 创建新列
        capacity = len(next(iter(self.tables[table_name].values())))
        if len(values) != capacity:
            raise ValueError(f"Values length {len(values)} mismatch with table capacity {capacity}")
        
        self.tables[table_name][column_name] = values.copy()
        self.dirty_flags[table_name] = True
        self.logger.debug(f"Added column '{column_name}' to table '{table_name}'")
