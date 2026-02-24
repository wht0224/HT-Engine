# Math Package Initialization
"""
Engine Math Module
Optimized math library with Cython acceleration
"""

# 优先尝试导入 Cython 加速版本
try:
    from .CythonMath import Vector3, Matrix4x4, Quaternion
    from .Math import Vector2, BoundingBox, Frustum, BoundingSphere  # 辅助类仍使用 Python 版本
    MATH_BACKEND = "Cython (Optimized)"
    print("[Math] 使用 Cython 加速版本")
except ImportError as e:
    # 如果 Cython 版本导入失败，降级到纯 Python 版本
    from .Math import Vector3, Matrix4x4, Quaternion, BoundingBox, Frustum, BoundingSphere, Vector2
    MATH_BACKEND = "Pure Python"
    print(f"[Math] Cython 版本不可用，使用纯 Python 版本：{e}")

# Export all math classes
__all__ = [
    "Vector2",
    "Vector3",
    "Matrix4x4",
    "Quaternion",
    "BoundingBox",
    "BoundingSphere",
    "Frustum",
    "MATH_BACKEND"
]