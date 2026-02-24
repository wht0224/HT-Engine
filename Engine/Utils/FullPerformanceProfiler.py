#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全流程性能追踪器
记录从引擎初始化到游戏关闭的所有关键步骤的耗时
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class PerfStep:
    """性能步骤记录"""
    name: str
    description: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    level: int = 0  # 层级：0=顶级，1=次级，2=更细
    
    def finish(self):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class FullPerformanceProfiler:
    """全流程性能追踪器"""
    
    def __init__(self):
        self.steps: List[PerfStep] = []
        self.current_stack: List[PerfStep] = []
        self.frame_stats: Dict[str, List[float]] = defaultdict(list)
        self.enabled = True
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def start_step(self, name: str, description: str = "", level: int = 0):
        """开始一个性能步骤"""
        if not self.enabled:
            return None
        step = PerfStep(
            name=name,
            description=description,
            start_time=time.perf_counter(),
            level=level
        )
        self.steps.append(step)
        self.current_stack.append(step)
        return step
        
    def end_step(self):
        """结束当前性能步骤"""
        if not self.enabled or not self.current_stack:
            return
        step = self.current_stack.pop()
        step.finish()
        
    def record_frame_metric(self, name: str, duration_ms: float):
        """记录每帧的指标"""
        if self.enabled:
            self.frame_stats[name].append(duration_ms)
            
    def generate_report(self) -> str:
        """生成详细的全流程性能报告"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("全流程性能分析报告")
        report_lines.append("=" * 100)
        report_lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # 初始化阶段报告
        report_lines.append("【初始化阶段耗时】")
        report_lines.append("-" * 100)
        report_lines.append(f"{'步骤名称':<40} {'说明':<30} {'耗时(ms)':<10} {'层级':<5}")
        report_lines.append("-" * 100)
        
        total_init_ms = 0.0
        for step in self.steps:
            if step.duration_ms == 0:
                step.finish()
            indent = "  " * step.level
            report_lines.append(
                f"{indent}{step.name:<{40 - step.level*2}} {step.description:<30} {step.duration_ms:>10.2f} {' ' * step.level}{step.level}"
            )
            total_init_ms += step.duration_ms
        
        report_lines.append("-" * 100)
        report_lines.append(f"初始化阶段总耗时: {total_init_ms:.2f}ms ({total_init_ms/1000:.2f}s)")
        report_lines.append("")
        
        # 每帧统计报告
        if self.frame_stats:
            report_lines.append("【每帧统计】")
            report_lines.append("-" * 100)
            report_lines.append(f"{'指标名称':<40} {'平均(ms)':<10} {'最小(ms)':<10} {'最大(ms)':<10} {'次数':<8}")
            report_lines.append("-" * 100)
            
            for name, durations in self.frame_stats.items():
                if durations:
                    avg = sum(durations) / len(durations)
                    min_d = min(durations)
                    max_d = max(durations)
                    count = len(durations)
                    report_lines.append(
                        f"{name:<40} {avg:>10.2f} {min_d:>10.2f} {max_d:>10.2f} {count:>8}"
                    )
            
            report_lines.append("")
        
        report_lines.append("=" * 100)
        return "\n".join(report_lines)
    
    def save_report(self, filename: str = "full_performance_report.txt"):
        """保存报告到文件"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"全流程性能报告已保存: {filename}")
        return filename
    
    def reset(self):
        """重置所有数据"""
        self.steps.clear()
        self.current_stack.clear()
        self.frame_stats.clear()


# 全局性能追踪器实例
_global_profiler: Optional[FullPerformanceProfiler] = None


def get_full_profiler() -> FullPerformanceProfiler:
    """获取全局性能追踪器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = FullPerformanceProfiler()
    return _global_profiler
