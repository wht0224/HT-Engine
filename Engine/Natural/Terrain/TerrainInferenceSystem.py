"""
地形推理系统
基于符号主义AI的地形特征推理
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class TerrainFact:
    """地形事实"""
    name: str
    data: np.ndarray
    description: str = ""


class TerrainFactBase:
    """地形事实库"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self._facts: Dict[str, TerrainFact] = {}
    
    def set(self, name: str, data: np.ndarray, description: str = ""):
        """设置事实"""
        if data.shape != (self.height, self.width):
            raise ValueError(f"数据形状不匹配: {data.shape} != ({self.height}, {self.width})")
        self._facts[name] = TerrainFact(name, data, description)
    
    def get(self, name: str) -> Optional[np.ndarray]:
        """获取事实数据"""
        fact = self._facts.get(name)
        return fact.data if fact else None
    
    def has(self, name: str) -> bool:
        """检查事实是否存在"""
        return name in self._facts
    
    def list_facts(self) -> List[str]:
        """列出所有事实"""
        return list(self._facts.keys())


class TerrainRuleBase:
    """地形规则基类"""
    
    def __init__(self, name: str, priority: int = 100):
        self.name = name
        self.priority = priority
        self.enabled = True
    
    def evaluate(self, facts: TerrainFactBase):
        """执行规则"""
        raise NotImplementedError("子类必须实现evaluate方法")
    
    def get_dependencies(self) -> List[str]:
        """获取依赖的事实"""
        return []


class TerrainInferenceSystem:
    """
    地形推理系统
    
    基于符号主义AI，通过规则链进行地形特征推理:
    1. 一级推理: 计算基础属性 (坡度、坡向、曲率)
    2. 二级推理: 识别地形特征 (山脊、河谷、陡崖)
    3. 三级推理: 增强细节 (岩石、侵蚀、植被)
    """
    
    def __init__(self, elevation: np.ndarray, cell_size: float = 30.0):
        """
        初始化推理系统
        
        Args:
            elevation: 高程数据
            cell_size: 单元格大小 (米)
        """
        self.height, self.width = elevation.shape
        self.cell_size = cell_size
        
        self.facts = TerrainFactBase(self.width, self.height)
        self.facts.set("elevation", elevation, "原始高程数据")
        
        self.rules: List[TerrainRuleBase] = []
        self._rule_priorities: Dict[str, int] = {}
    
    def add_rule(self, rule: TerrainRuleBase):
        """添加规则"""
        self.rules.append(rule)
        self._rule_priorities[rule.name] = rule.priority
        self.rules.sort(key=lambda r: r.priority)
    
    def remove_rule(self, name: str):
        """移除规则"""
        self.rules = [r for r in self.rules if r.name != name]
        if name in self._rule_priorities:
            del self._rule_priorities[name]
    
    def enable_rule(self, name: str, enabled: bool = True):
        """启用/禁用规则"""
        for rule in self.rules:
            if rule.name == name:
                rule.enabled = enabled
                break
    
    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """
        运行推理
        
        Args:
            verbose: 是否输出详细信息
            
        Returns:
            推理结果统计
        """
        results = {
            'rules_executed': 0,
            'facts_generated': 0,
            'errors': []
        }
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            deps = rule.get_dependencies()
            missing_deps = [d for d in deps if not self.facts.has(d)]
            
            if missing_deps:
                if verbose:
                    print(f"[跳过] {rule.name}: 缺少依赖 {missing_deps}")
                results['errors'].append({
                    'rule': rule.name,
                    'error': f"缺少依赖: {missing_deps}"
                })
                continue
            
            try:
                if verbose:
                    print(f"[执行] {rule.name}...")
                
                rule.evaluate(self.facts)
                results['rules_executed'] += 1
                
                new_facts = len(self.facts.list_facts())
                if verbose:
                    print(f"  → 事实数: {new_facts}")
                    
            except Exception as e:
                if verbose:
                    print(f"[错误] {rule.name}: {e}")
                results['errors'].append({
                    'rule': rule.name,
                    'error': str(e)
                })
        
        results['facts_generated'] = len(self.facts.list_facts())
        return results
    
    def get_result(self, name: str) -> Optional[np.ndarray]:
        """获取推理结果"""
        return self.facts.get(name)
    
    def get_all_results(self) -> Dict[str, np.ndarray]:
        """获取所有推理结果"""
        return {name: self.facts.get(name) for name in self.facts.list_facts()}
    
    def get_enhanced_elevation(self) -> np.ndarray:
        """获取增强后的高程数据"""
        enhanced = self.facts.get("elevation_enhanced")
        if enhanced is not None:
            return enhanced
        return self.facts.get("elevation")
