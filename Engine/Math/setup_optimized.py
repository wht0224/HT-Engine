# -*- coding: utf-8 -*-
"""
Cython 数学库编译配置文件 - 优化版本
用于将 CythonMath.pyx 编译为 C 扩展模块
"""

import os
import sys
from setuptools import setup
from Cython.Build import cythonize
from setuptools.extension import Extension

# 获取当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 检测 CPU 特性
import platform
machine = platform.machine()
print(f"检测到 CPU 架构：{machine}")

# 定义 Cython 扩展 - 优化版本
cython_extensions = [
    Extension(
        name="CythonMath",  # 生成的扩展模块名称
        sources=[
            "CythonMath.pyx",  # Cython 源文件
            "asm_math.c",  # 汇编优化的 C 包装文件
        ],
        include_dirs=[current_dir],  # 包含目录
        # 编译优化选项
        extra_compile_args=[
            "/O2",  # 最高优化级别
            "/fp:fast",  # 快速浮点运算
            "/arch:AVX2",  # 使用 AVX2 指令集
            "/DNDEBUG",  # 禁用调试信息
            "/MT",  # 静态链接 MSVC 运行时
            "/GL",  # 启用整程序优化
        ],
        # 链接优化选项
        extra_link_args=[
            "/OPT:REF",  # 移除未引用的函数和数据
            "/OPT:ICF",  # 合并相同的函数和数据
            "/LTCG",  # 启用链接时生成代码
        ],
        # 仅在 Windows 上使用的选项
        define_macros=[
            ("MS_WIN64", "1"),  # 定义 64 位 Windows 宏
            ("USE_AVX2", "1"),  # 启用 AVX2 优化
        ]
    )
]

# 编译配置
setup(
    name="CythonMath",
    version="2.0",
    description="高性能数学库，使用 Cython 实现 - 优化版本",
    ext_modules=cythonize(
        cython_extensions,
        compiler_directives={
            "language_level": "3",  # 使用 Python 3 语法
            "boundscheck": False,  # 禁用边界检查
            "wraparound": False,  # 禁用负索引环绕
            "nonecheck": False,  # 禁用 None 检查
            "cdivision": True,  # 启用 C 风格除法
            "profile": False,  # 禁用性能分析
            "linetrace": False,  # 禁用行追踪
            "optimize.use_switch": True,  # 使用 switch 优化
            "optimize.unpack_method_calls": True,  # 优化方法调用
            "initializedcheck": False,  # 禁用初始化检查
            "overflowcheck": False,  # 禁用溢出检查
            "always_allow_keywords": False,  # 禁用关键字参数
            "embedsignature": False,  # 不嵌入签名
            "binding": False,  # 禁用绑定
        },
        annotate=False,  # 不生成注释 HTML 文件
        quiet=False,  # 显示编译输出
        gdb_debug=False,  # 禁用 GDB 调试
        language_level=3,  # Python 3
    ),
    zip_safe=False,  # 禁用 zip 安全，提高加载性能
)

# 如果直接运行该脚本，执行编译
if __name__ == "__main__":
    print("=" * 60)
    print("开始编译优化版 Cython 数学库...")
    print("=" * 60)
    
    # 编译 Cython 扩展
    setup(
        name="CythonMath",
        version="2.0",
        description="高性能数学库，使用 Cython 实现 - 优化版本",
        ext_modules=cythonize(
            cython_extensions,
            compiler_directives={
                "language_level": "3",
                "boundscheck": False,
                "wraparound": False,
                "nonecheck": False,
                "cdivision": True,
                "profile": False,
                "linetrace": False,
                "optimize.use_switch": True,
                "optimize.unpack_method_calls": True,
                "initializedcheck": False,
                "overflowcheck": False,
                "always_allow_keywords": False,
                "embedsignature": False,
                "binding": False,
            },
            annotate=False,
            quiet=False,
            gdb_debug=False,
            language_level=3,
        ),
        zip_safe=False,
    )
    
    print("=" * 60)
    print("编译完成！")
    print("=" * 60)
    print("\n优化说明:")
    print("1. 启用 AVX2 指令集加速")
    print("2. 禁用所有运行时检查")
    print("3. 启用整程序优化 (IPO/LTCG)")
    print("4. 使用快速浮点运算模式")
    print("5. 静态链接运行时库")
    print("\n预期性能提升:")
    print("- 向量运算：5-10 倍")
    print("- 四元数运算：3-5 倍")
    print("- 矩阵运算：4-8 倍")
    print("=" * 60)
