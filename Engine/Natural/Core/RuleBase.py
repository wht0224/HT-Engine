from abc import ABC, abstractmethod
from typing import List
from .FactBase import FactBase

class Rule(ABC):
    """
    规则基类 (Base Rule)
    
    符号主义的核心：Event-Condition-Action
    所有规则必须是无状态的 (Stateless)，只根据输入的事实 (Facts) 产生输出。
    """
    
    def __init__(self, name: str, priority: int = 0):
        self.name = name
        self.priority = priority # 优先级，高的先执行

    @abstractmethod
    def evaluate(self, facts: FactBase) -> None:
        """
        执行规则推导
        
        Args:
            facts: 事实库引用
        """
        pass
        
class RuleBase:
    """
    规则库 (RuleBase)
    管理所有注册的规则
    """
    def __init__(self):
        self.rules: List[Rule] = []
        
    def register(self, rule: Rule):
        self.rules.append(rule)
        # 按优先级排序 (降序)
        self.rules.sort(key=lambda x: x.priority, reverse=True)
        
    def get_all_rules(self) -> List[Rule]:
        return self.rules
