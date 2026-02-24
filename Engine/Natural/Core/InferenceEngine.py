import time
import logging
from .FactBase import FactBase
from .RuleBase import RuleBase

class InferenceEngine:
    """
    推理机 (Inference Engine)
    
    Natural 模块的"心脏"，负责驱动规则执行。
    支持按需更新 (Dirty Flag Check)。
    """
    
    def __init__(self):
        self.logger = logging.getLogger("Natural.InferenceEngine")
        self.facts = FactBase()
        self.rules = RuleBase()
        self.frame_count = 0
        self.rule_timings = {}
        
    def step(self, dt: float):
        """
        执行一帧的推理
        
        Args:
            dt: Delta time in seconds
        """
        # 1. 更新时间等基础事实
        current_time = self.facts.get_global("time")
        self.facts.set_global("time", current_time + dt)
        self.facts.set_global("dt", dt)
        
        # 2. 遍历执行规则
        # 可以在这里加入更复杂的调度逻辑 (如每N帧执行一次低优先级规则)
        rule_enabled = self.facts.get_global("rule_enabled")
        rule_intervals = self.facts.get_global("rule_intervals")
        frame_index = self.frame_count

        for rule in self.rules.get_all_rules():
            if isinstance(rule_enabled, dict) and rule_enabled:
                enabled = None
                best_len = -1
                for key, value in rule_enabled.items():
                    key_str = str(key)
                    if rule.name == key_str or rule.name.startswith(key_str):
                        if len(key_str) > best_len:
                            best_len = len(key_str)
                            enabled = bool(value)
                if enabled is False:
                    continue

            if isinstance(rule_intervals, dict) and rule_intervals:
                interval = 1
                best_len = -1
                for key, value in rule_intervals.items():
                    key_str = str(key)
                    if rule.name == key_str or rule.name.startswith(key_str):
                        try:
                            iv = int(value)
                        except Exception:
                            continue
                        if iv < 1:
                            iv = 1
                        if len(key_str) > best_len:
                            best_len = len(key_str)
                            interval = iv
                if interval > 1 and (frame_index % interval) != 0:
                    continue

            # TODO: 这里可以加入 Filter，只执行相关的规则
            # 例如 check rule.dependencies against facts.dirty_flags
            
            try:
                t_rule_start = time.perf_counter()
                rule.evaluate(self.facts)
                t_rule_end = time.perf_counter()
                duration_ms = (t_rule_end - t_rule_start) * 1000.0
                self.rule_timings[rule.name] = duration_ms
            except Exception as e:
                self.logger.error(f"Error evaluating rule {rule.name}: {e}", exc_info=True)
                
        # 3. 清理脏标记 (本帧结束)
        # 注意：实际应用中可能需要更复杂的脏标记生命周期管理
        self.facts.dirty_flags.clear()
        
        self.frame_count += 1
