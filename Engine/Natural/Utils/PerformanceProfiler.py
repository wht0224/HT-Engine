"""
Natural系统性能分析器

功能：
1. 测量每个规则的执行时间
2. 检测性能热点
3. 自动标记需要优化的模块
4. 生成优化建议报告

使用：
    profiler = PerformanceProfiler()
    profiler.start_profiling()
    natural.update(dt)
    report = profiler.generate_report()
"""

import time
import sys
import os
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from contextlib import contextmanager

# 添加父路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..'))


@dataclass
class PerformanceRecord:
    """性能记录"""
    module_name: str
    function_name: str
    start_time: float
    end_time: float = 0.0
    duration_ms: float = 0.0
    call_count: int = 1
    
    def finish(self):
        """记录结束时间"""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


@dataclass
class HotspotAlert:
    """热点警报"""
    module_name: str
    function_name: str
    avg_duration_ms: float
    total_duration_ms: float
    call_count: int
    severity: str  # 'low', 'medium', 'high', 'critical'
    suggestion: str


class PerformanceProfiler:
    """
    Natural系统性能分析器
    
    自动检测性能瓶颈，标记需要汇编优化的模块
    """
    
    # 性能阈值配置（毫秒）
    THRESHOLD_LOW = 1.0        # >1ms 值得关注
    THRESHOLD_MEDIUM = 5.0     # >5ms 需要优化
    THRESHOLD_HIGH = 16.67     # >16.67ms (1帧@60fps) 必须优化
    THRESHOLD_CRITICAL = 33.33 # >33.33ms (1帧@30fps) 严重问题
    
    def __init__(self, auto_print: bool = True):
        """
        初始化性能分析器
        
        Args:
            auto_print: 是否自动打印警告信息
        """
        self.records: List[PerformanceRecord] = []
        self.current_stack: List[PerformanceRecord] = []
        self.stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_ms': 0.0,
            'call_count': 0,
            'max_ms': 0.0,
            'min_ms': float('inf'),
            'avg_ms': 0.0
        })
        self.hotspots: List[HotspotAlert] = []
        self.auto_print = auto_print
        self.enabled = True
        
    def enable(self):
        """启用性能分析"""
        self.enabled = True
        
    def disable(self):
        """禁用性能分析"""
        self.enabled = False
        
    @contextmanager
    def profile(self, module_name: str, function_name: str = None):
        """
        性能分析上下文管理器
        
        使用：
            with profiler.profile('LightingRule', 'evaluate'):
                # 要测量的代码
                pass
        """
        if not self.enabled:
            yield
            return
            
        if function_name is None:
            function_name = 'unknown'
            
        record = PerformanceRecord(
            module_name=module_name,
            function_name=function_name,
            start_time=time.perf_counter()
        )
        
        self.current_stack.append(record)
        
        try:
            yield
        finally:
            record.finish()
            self.current_stack.pop()
            self.records.append(record)
            self._update_stats(record)
            
            # 自动检测热点
            if record.duration_ms > self.THRESHOLD_MEDIUM:
                self._check_hotspot(record)
    
    def _update_stats(self, record: PerformanceRecord):
        """更新统计信息"""
        key = f"{record.module_name}.{record.function_name}"
        stats = self.stats[key]
        
        stats['total_ms'] += record.duration_ms
        stats['call_count'] += 1
        stats['max_ms'] = max(stats['max_ms'], record.duration_ms)
        stats['min_ms'] = min(stats['min_ms'], record.duration_ms)
        stats['avg_ms'] = stats['total_ms'] / stats['call_count']
    
    def _check_hotspot(self, record: PerformanceRecord):
        """检查是否为热点"""
        duration = record.duration_ms
        
        if duration >= self.THRESHOLD_CRITICAL:
            severity = 'critical'
            suggestion = '必须立即用汇编优化'
        elif duration >= self.THRESHOLD_HIGH:
            severity = 'high'
            suggestion = '强烈建议用汇编优化'
        elif duration >= self.THRESHOLD_MEDIUM:
            severity = 'medium'
            suggestion = '建议用Numba/Cython优化'
        else:
            severity = 'low'
            suggestion = '值得关注'
        
        alert = HotspotAlert(
            module_name=record.module_name,
            function_name=record.function_name,
            avg_duration_ms=duration,
            total_duration_ms=duration,
            call_count=1,
            severity=severity,
            suggestion=suggestion
        )
        
        self.hotspots.append(alert)
        
        if self.auto_print:
            print(f"⚠️  [{severity.upper()}] {record.module_name}.{record.function_name}: "
                  f"{duration:.2f}ms - {suggestion}")
    
    def generate_report(self) -> str:
        """
        生成性能分析报告
        
        Returns:
            格式化的报告字符串
        """
        if not self.stats:
            return "没有性能数据"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("Natural系统性能分析报告")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # 按总耗时排序
        sorted_stats = sorted(
            self.stats.items(),
            key=lambda x: x[1]['total_ms'],
            reverse=True
        )
        
        # 总体统计
        report_lines.append("【总体统计】")
        total_time = sum(s['total_ms'] for _, s in sorted_stats)
        report_lines.append(f"总测量时间: {total_time:.2f}ms")
        report_lines.append(f"测量模块数: {len(sorted_stats)}")
        report_lines.append("")
        
        # 详细统计
        report_lines.append("【模块耗时详情】")
        report_lines.append(f"{'模块名':<40} {'总耗时':<10} {'平均':<10} {'调用':<8} {'最大':<10}")
        report_lines.append("-" * 80)
        
        for name, stats in sorted_stats:
            total = stats['total_ms']
            avg = stats['avg_ms']
            calls = stats['call_count']
            max_t = stats['max_ms']
            
            # 标记热点
            marker = ""
            if max_t >= self.THRESHOLD_CRITICAL:
                marker = " 🔴"
            elif max_t >= self.THRESHOLD_HIGH:
                marker = " 🟠"
            elif max_t >= self.THRESHOLD_MEDIUM:
                marker = " 🟡"
            
            report_lines.append(
                f"{name:<40} {total:>8.2f}ms {avg:>8.2f}ms {calls:>6} {max_t:>8.2f}ms{marker}"
            )
        
        report_lines.append("")
        
        # 热点警报
        if self.hotspots:
            report_lines.append("【热点警报】")
            report_lines.append("-" * 80)
            
            # 去重并合并相同热点
            merged_hotspots = {}
            for alert in self.hotspots:
                key = f"{alert.module_name}.{alert.function_name}"
                if key in merged_hotspots:
                    merged_hotspots[key].total_duration_ms += alert.total_duration_ms
                    merged_hotspots[key].call_count += alert.call_count
                else:
                    merged_hotspots[key] = alert
            
            # 按严重程度排序
            severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
            sorted_hotspots = sorted(
                merged_hotspots.values(),
                key=lambda x: severity_order.get(x.severity, 4)
            )
            
            for alert in sorted_hotspots:
                severity_emoji = {
                    'critical': '🔴',
                    'high': '🟠',
                    'medium': '🟡',
                    'low': '⚪'
                }.get(alert.severity, '⚪')
                
                report_lines.append(
                    f"{severity_emoji} [{alert.severity.upper()}] "
                    f"{alert.module_name}.{alert.function_name}"
                )
                report_lines.append(
                    f"   平均: {alert.avg_duration_ms:.2f}ms, "
                    f"总计: {alert.total_duration_ms:.2f}ms, "
                    f"调用: {alert.call_count}次"
                )
                report_lines.append(f"   建议: {alert.suggestion}")
                report_lines.append("")
        
        # 优化建议
        report_lines.append("【优化建议】")
        report_lines.append("-" * 80)
        
        critical_count = sum(1 for h in self.hotspots if h.severity == 'critical')
        high_count = sum(1 for h in self.hotspots if h.severity == 'high')
        medium_count = sum(1 for h in self.hotspots if h.severity == 'medium')
        
        if critical_count > 0:
            report_lines.append(f"🔴 发现 {critical_count} 个严重性能问题，必须立即用汇编优化")
        if high_count > 0:
            report_lines.append(f"🟠 发现 {high_count} 个高性能问题，强烈建议用汇编优化")
        if medium_count > 0:
            report_lines.append(f"🟡 发现 {medium_count} 个中等性能问题，建议用Numba/Cython优化")
        
        if not any([critical_count, high_count, medium_count]):
            report_lines.append("✅ 没有明显的性能问题")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def get_optimization_targets(self) -> List[Tuple[str, str, str]]:
        """
        获取需要优化的目标列表
        
        Returns:
            [(模块名, 函数名, 优化建议), ...]
        """
        targets = []
        seen = set()
        
        for alert in self.hotspots:
            key = f"{alert.module_name}.{alert.function_name}"
            if key not in seen and alert.severity in ['critical', 'high']:
                seen.add(key)
                targets.append((
                    alert.module_name,
                    alert.function_name,
                    alert.suggestion
                ))
        
        return targets
    
    def save_report(self, filename: str = "performance_report.txt"):
        """保存报告到文件"""
        report = self.generate_report()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"性能报告已保存: {filename}")
    
    def reset(self):
        """重置所有数据"""
        self.records.clear()
        self.current_stack.clear()
        self.stats.clear()
        self.hotspots.clear()


# 全局性能分析器实例
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """获取全局性能分析器"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_function(module_name: str, function_name: str = None):
    """
    函数装饰器，自动测量函数性能
    
    使用：
        @profile_function('LightingRule')
        def evaluate(self, facts):
            # 函数体
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            profiler = get_profiler()
            func_name = function_name or func.__name__
            
            with profiler.profile(module_name, func_name):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


# 便捷函数
@contextmanager
def profile_scope(module_name: str, function_name: str = None):
    """便捷上下文管理器"""
    profiler = get_profiler()
    with profiler.profile(module_name, function_name):
        yield


if __name__ == "__main__":
    # 测试代码
    profiler = PerformanceProfiler()
    
    # 模拟一些性能数据
    with profiler.profile('TestModule', 'fast_function'):
        time.sleep(0.001)  # 1ms
    
    with profiler.profile('TestModule', 'slow_function'):
        time.sleep(0.02)   # 20ms - 应该触发警告
    
    with profiler.profile('TestModule', 'very_slow_function'):
        time.sleep(0.05)   # 50ms - 应该触发严重警告
    
    # 生成报告
    print(profiler.generate_report())
