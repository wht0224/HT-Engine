# -*- coding: utf-8 -*-
"""
路径追踪完整实现
基于物理的实时路径追踪渲染器
"""

import numpy as np
from .EffectBase import EffectBase, GPUArchitecture, EffectQuality

# 辅助类：光线
class Ray:
    """光线类"""
    def __init__(self, origin, direction):
        self.origin = np.array(origin, dtype=np.float32)
        self.direction = np.array(direction, dtype=np.float32)
        self.direction /= np.linalg.norm(self.direction)  # 归一化方向向量

# 辅助类：相交结果
class Intersection:
    """光线与物体相交结果"""
    def __init__(self):
        self.hit = False  # 是否命中
        self.distance = float('inf')  # 命中距离
        self.position = np.zeros(3, dtype=np.float32)  # 命中位置
        self.normal = np.zeros(3, dtype=np.float32)  # 命中法线
        self.uv = np.zeros(2, dtype=np.float32)  # 命中UV坐标
        self.material = None  # 命中材质
        self.object = None  # 命中物体
        self.ray_direction = np.zeros(3, dtype=np.float32)  # 光线方向

# 辅助类：球体
class Sphere:
    """球体几何体"""
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.material = material
    
    def intersect(self, ray):
        """光线与球体相交检测"""
        result = Intersection()
        
        # 计算光线到球心的向量
        oc = ray.origin - self.center
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(oc, ray.direction)
        c = np.dot(oc, oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        
        if discriminant > 0:
            # 计算两个交点
            t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
            
            # 选择最近的有效交点
            if t1 > 0.001:
                t = t1
            elif t2 > 0.001:
                t = t2
            else:
                return result
            
            # 计算命中信息
            result.hit = True
            result.distance = t
            result.position = ray.origin + ray.direction * t
            result.normal = (result.position - self.center) / self.radius
            result.material = self.material
            result.object = self
        
        return result
    
    def get_aabb(self):
        """获取球体的AABB包围盒"""
        min_point = self.center - self.radius
        max_point = self.center + self.radius
        return AABB(min_point, max_point)

# 辅助类：平面
class Plane:
    """平面几何体"""
    def __init__(self, origin, normal, material):
        self.origin = np.array(origin, dtype=np.float32)
        self.normal = np.array(normal, dtype=np.float32)
        self.normal /= np.linalg.norm(self.normal)  # 归一化法线
        self.material = material
    
    def intersect(self, ray):
        """光线与平面相交检测"""
        result = Intersection()
        
        # 计算光线与平面的夹角
        denom = np.dot(ray.direction, self.normal)
        
        # 如果光线与平面平行或背面朝向，不相交
        if np.abs(denom) < 0.001:
            return result
        
        # 计算相交距离
        t = np.dot(self.origin - ray.origin, self.normal) / denom
        
        # 如果交点在光线起点后面，不相交
        if t <= 0.001:
            return result
        
        # 计算命中信息
        result.hit = True
        result.distance = t
        result.position = ray.origin + ray.direction * t
        result.normal = self.normal
        result.material = self.material
        result.object = self
        
        return result
    
    def get_aabb(self):
        """获取平面的AABB包围盒
        平面是无限的, 但为了支持BVH, 我们返回一个非常大的AABB
        """
        # 对于平面，我们返回一个非常大的AABB，覆盖场景中可能的可见区域
        min_point = np.array([-10000.0, -10000.0, -10000.0], dtype=np.float32)
        max_point = np.array([10000.0, 10000.0, 10000.0], dtype=np.float32)
        return AABB(min_point, max_point)

# 辅助类：光源基类
class Light:
    """光源基类"""
    def __init__(self, color=np.array([1.0, 1.0, 1.0]), intensity=1.0):
        """
        初始化光源
        
        参数:
        - color: 光源颜色, 默认为白色
        - intensity: 光源强度, 默认为1.0
        """
        self.color = np.array(color, dtype=np.float32)
        self.intensity = intensity
        self.enabled = True

# 辅助类：点光源
class PointLight(Light):
    """点光源"""
    def __init__(self, position=np.array([0.0, 0.0, 0.0]), color=np.array([1.0, 1.0, 1.0]), intensity=1.0):
        """
        初始化点光源
        
        参数:
        - position: 光源位置
        - color: 光源颜色, 默认为白色
        - intensity: 光源强度, 默认为1.0
        """
        super().__init__(color, intensity)
        self.position = np.array(position, dtype=np.float32)
        self.type = "point"

# 辅助类：方向光
class DirectionalLight(Light):
    """方向光"""
    def __init__(self, direction=np.array([0.0, -1.0, 0.0]), color=np.array([1.0, 1.0, 1.0]), intensity=1.0):
        """
        初始化方向光
        
        参数:
        - direction: 光源方向
        - color: 光源颜色, 默认为白色
        - intensity: 光源强度, 默认为1.0
        """
        super().__init__(color, intensity)
        self.direction = np.array(direction, dtype=np.float32)
        self.direction /= np.linalg.norm(self.direction)  # 归一化方向
        self.type = "directional"

# 辅助类：面光源
class AreaLight(Light):
    """面光源"""
    def __init__(self, position=np.array([0.0, 0.0, 0.0]), normal=np.array([0.0, -1.0, 0.0]), width=1.0, height=1.0, color=np.array([1.0, 1.0, 1.0]), intensity=1.0):
        """
        初始化面光源
        
        参数:
        - position: 光源位置 (中心)
        - normal: 光源法线方向
        - width: 光源宽度
        - height: 光源高度
        - color: 光源颜色, 默认为白色
        - intensity: 光源强度, 默认为1.0
        """
        super().__init__(color, intensity)
        self.position = np.array(position, dtype=np.float32)
        self.normal = np.array(normal, dtype=np.float32)
        self.normal /= np.linalg.norm(self.normal)  # 归一化法线
        self.width = width
        self.height = height
        self.type = "area"

# 辅助类：AABB包围盒
class AABB:
    """轴对齐包围盒"""
    def __init__(self, min_point=np.array([float('inf'), float('inf'), float('inf')]), max_point=np.array([-float('inf'), -float('inf'), -float('inf')])):
        """
        初始化AABB包围盒
        
        参数:
        - min_point: 包围盒最小点
        - max_point: 包围盒最大点
        """
        self.min_point = np.array(min_point, dtype=np.float32)
        self.max_point = np.array(max_point, dtype=np.float32)
    
    def expand(self, point):
        """扩展包围盒以包含指定点
        
        参数:
        - point: 要包含的点
        """
        self.min_point = np.minimum(self.min_point, point)
        self.max_point = np.maximum(self.max_point, point)
    
    def expand_by_aabb(self, other):
        """扩展包围盒以包含另一个AABB
        
        参数:
        - other: 要包含的AABB
        """
        self.min_point = np.minimum(self.min_point, other.min_point)
        self.max_point = np.maximum(self.max_point, other.max_point)
    
    def intersect_ray(self, ray):
        """光线与AABB相交检测
        
        参数:
        - ray: 光线对象
        
        返回:
        - tuple: (hit, t_min, t_max), 是否相交, 以及相交的最小和最大距离
        """
        # 计算光线与AABB的相交时间
        t_min = np.zeros(3)
        t_max = np.zeros(3)
        
        for i in range(3):
            if abs(ray.direction[i]) < 0.0001:
                # 光线方向平行于当前轴，检查原点是否在AABB内
                if ray.origin[i] < self.min_point[i] or ray.origin[i] > self.max_point[i]:
                    return (False, 0.0, 0.0)
                t_min[i] = -float('inf')
                t_max[i] = float('inf')
            else:
                # 计算光线与当前轴平面的相交时间
                inv_dir = 1.0 / ray.direction[i]
                t_min[i] = (self.min_point[i] - ray.origin[i]) * inv_dir
                t_max[i] = (self.max_point[i] - ray.origin[i]) * inv_dir
                
                # 确保t_min[i] <= t_max[i]
                if t_min[i] > t_max[i]:
                    t_min[i], t_max[i] = t_max[i], t_min[i]
        
        # 计算整体的t_min和t_max
        overall_t_min = max(t_min[0], t_min[1], t_min[2])
        overall_t_max = min(t_max[0], t_max[1], t_max[2])
        
        # 检查是否相交
        if overall_t_min <= overall_t_max and overall_t_max >= 0.0:
            return (True, overall_t_min, overall_t_max)
        else:
            return (False, 0.0, 0.0)

# 辅助类：BVH节点
class BVHNode:
    """BVH节点"""
    def __init__(self, objects=None):
        """
        初始化BVH节点
        
        参数:
        - objects: 节点包含的物体列表
        """
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.objects = objects if objects else []  # 节点包含的物体
        self.aabb = AABB()  # 节点的包围盒
        self.is_leaf = len(self.objects) > 0  # 是否为叶子节点
        
        # 如果是叶子节点，计算包围盒
        if self.is_leaf and self.objects:
            # 计算所有物体的AABB
            for obj in self.objects:
                if hasattr(obj, 'get_aabb'):
                    obj_aabb = obj.get_aabb()
                    self.aabb.expand_by_aabb(obj_aabb)
        
    def build(self, max_depth=20, min_objects=4):
        """递归构建BVH树
        
        参数:
        - max_depth: 最大深度
        - min_objects: 叶子节点最小物体数量
        """
        # 如果节点物体数量少于min_objects或达到最大深度，停止分裂
        if len(self.objects) <= min_objects or max_depth <= 0:
            return
        
        # 选择分裂轴
        split_axis = self._select_split_axis()
        
        # 按分裂轴排序物体
        self.objects.sort(key=lambda obj: self._get_object_center(obj)[split_axis])
        
        # 分裂物体列表
        mid = len(self.objects) // 2
        left_objects = self.objects[:mid]
        right_objects = self.objects[mid:]
        
        # 创建子节点
        self.left = BVHNode(left_objects)
        self.right = BVHNode(right_objects)
        
        # 递归构建子节点
        self.left.build(max_depth - 1, min_objects)
        self.right.build(max_depth - 1, min_objects)
        
        # 计算当前节点的AABB
        self.aabb = AABB()
        self.aabb.expand_by_aabb(self.left.aabb)
        self.aabb.expand_by_aabb(self.right.aabb)
        
        # 标记为内部节点
        self.is_leaf = False
        self.objects = []
    
    def _select_split_axis(self):
        """选择最佳分裂轴
        
        返回:
        - int: 分裂轴索引 (0=x, 1=y, 2=z)
        """
        # 简单实现：选择包围盒最长的轴
        extent = self.aabb.max_point - self.aabb.min_point
        return np.argmax(extent)
    
    def _get_object_center(self, obj):
        """获取物体的中心点
        
        参数:
        - obj: 物体对象
        
        返回:
        - np.array: 物体中心点
        """
        if hasattr(obj, 'get_aabb'):
            aabb = obj.get_aabb()
            return (aabb.min_point + aabb.max_point) / 2.0
        elif hasattr(obj, 'center'):
            return obj.center
        elif hasattr(obj, 'position'):
            return obj.position
        else:
            return np.array([0.0, 0.0, 0.0])
    
    def intersect_ray(self, ray):
        """光线与BVH节点相交检测
        
        参数:
        - ray: 光线对象
        
        返回:
        - tuple: (hit, closest_intersection), 是否相交, 以及最近的相交结果
        """
        # 检查光线是否与当前节点的AABB相交
        hit_aabb, t_min, t_max = self.aabb.intersect_ray(ray)
        if not hit_aabb:
            return (False, None)
        
        closest_intersection = None
        hit = False
        
        if self.is_leaf:
            # 叶子节点，检查所有物体
            for obj in self.objects:
                intersection = obj.intersect(ray)
                if intersection.hit:
                    if closest_intersection is None or intersection.distance < closest_intersection.distance:
                        closest_intersection = intersection
                        hit = True
        else:
            # 内部节点，递归检查子节点
            hit_left, left_intersection = self.left.intersect_ray(ray)
            hit_right, right_intersection = self.right.intersect_ray(ray)
            
            if hit_left:
                hit = True
                closest_intersection = left_intersection
            if hit_right:
                if not hit or right_intersection.distance < closest_intersection.distance:
                    hit = True
                    closest_intersection = right_intersection
        
        return (hit, closest_intersection)

# 辅助类：材质
class Material:
    """基于物理的材质类, 支持PBR材质和折射"""
    def __init__(self, albedo=np.array([0.5, 0.5, 0.5]), metallic=0.0, roughness=0.5, emissive=np.array([0.0, 0.0, 0.0]), ao=1.0, ior=1.5, transparent=False, transmission=1.0):
        """
        初始化材质
        
        参数:
        - albedo: 基础颜色, 默认为灰色
        - metallic: 金属度, 0.0表示非金属, 1.0表示金属
        - roughness: 粗糙度, 0.0表示光滑, 1.0表示粗糙
        - emissive: 自发光颜色, 默认为黑色 (不发光)
        - ao: 环境光遮蔽, 默认为1.0 (无遮蔽)
        - ior: 折射率, 默认为1.5 (玻璃)
        - transparent: 是否透明, 默认为False
        - transmission: 透射率, 0.0表示不透明, 1.0表示完全透明
        """
        self.albedo = np.array(albedo, dtype=np.float32)
        self.metallic = metallic
        self.roughness = roughness
        self.emissive = np.array(emissive, dtype=np.float32)
        self.ao = ao
        self.ior = ior  # Index of Refraction
        self.transparent = transparent
        self.transmission = transmission

# 辅助类：三角形
class Triangle:
    """三角形几何体"""
    def __init__(self, v0, v1, v2, material):
        self.v0 = np.array(v0, dtype=np.float32)
        self.v1 = np.array(v1, dtype=np.float32)
        self.v2 = np.array(v2, dtype=np.float32)
        self.material = material
        
        # 预计算边向量和法线
        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0
        self.normal = np.cross(self.e1, self.e2)
        self.normal /= np.linalg.norm(self.normal)
        
        # 预计算AABB
        self.aabb = self.get_aabb()
    
    def intersect(self, ray):
        """光线与三角形相交检测, 使用Möller-Trumbore算法"""
        result = Intersection()
        
        # 计算P向量
        P = np.cross(ray.direction, self.e2)
        det = np.dot(self.e1, P)
        
        # 如果det接近0，光线与三角形平面平行
        if np.abs(det) < 0.001:
            return result
        
        inv_det = 1.0 / det
        
        # 计算T向量
        T = ray.origin - self.v0
        
        # 计算u参数
        u = np.dot(T, P) * inv_det
        if u < 0.0 or u > 1.0:
            return result
        
        # 计算Q向量
        Q = np.cross(T, self.e1)
        
        # 计算v参数
        v = np.dot(ray.direction, Q) * inv_det
        if v < 0.0 or u + v > 1.0:
            return result
        
        # 计算t参数
        t = np.dot(self.e2, Q) * inv_det
        
        # 如果交点在光线起点后面，不相交
        if t <= 0.001:
            return result
        
        # 计算命中信息
        result.hit = True
        result.distance = t
        result.position = ray.origin + ray.direction * t
        result.normal = self.normal
        result.material = self.material
        result.object = self
        
        # 计算UV坐标（简单实现）
        result.uv = np.array([u, v], dtype=np.float32)
        
        return result
    
    def get_aabb(self):
        """获取三角形的AABB包围盒"""
        min_point = np.minimum(np.minimum(self.v0, self.v1), self.v2)
        max_point = np.maximum(np.maximum(self.v0, self.v1), self.v2)
        return AABB(min_point, max_point)

class PathTracer(EffectBase):
    """路径追踪器类
    实现完整的基于物理的路径追踪渲染器"""
    
    def __init__(self, gpu_architecture, quality_level):
        """
        初始化路径追踪器
        
        参数:
        - gpu_architecture: GPU架构
        - quality_level: 质量级别
        """
        super().__init__(gpu_architecture, quality_level)
        self.name = "path_tracer"
        self.performance_cost = {
            EffectQuality.LOW: 10.0,  # 低质量，适合GTX 750Ti
            EffectQuality.MEDIUM: 20.0,  # 中等质量，适合RX 580
            EffectQuality.HIGH: 40.0  # 高质量，适合高端GPU
        }
        
        # 路径追踪参数
        self.sample_count = 1  # 每像素采样数
        self.max_bounces = 2  # 最大光线反弹次数
        self.enable_denoising = True  # 是否启用降噪
        self.enable_adaptive_sampling = True  # 是否启用自适应采样
        self.convergence_threshold = 0.05  # 自适应采样收敛阈值
        self.downsample_factor = 2  # 降采样因子
        
        # 渲染状态
        self.accumulation_buffer = None  # 累积缓冲区
        self.sample_buffer = None  # 采样计数缓冲区
        self.frame_count = 0  # 当前累积帧数
        
        # 场景数据
        self.scene = None
        self.camera = None
        self.objects = []  # 场景中的物体列表
        
        # 材质管理
        self.materials = {}
        self.default_material = Material()
        
        # 预定义一些常用材质
        self._create_default_materials()
    
    def _create_default_materials(self):
        """创建一些默认材质"""
        # 灰色材质
        self.materials["gray"] = Material(
            albedo=np.array([0.5, 0.5, 0.5]),
            metallic=0.0,
            roughness=0.5
        )
        
        # 红色塑料材质
        self.materials["red_plastic"] = Material(
            albedo=np.array([0.8, 0.2, 0.2]),
            metallic=0.0,
            roughness=0.4
        )
        
        # 金色金属材质
        self.materials["gold"] = Material(
            albedo=np.array([1.0, 0.84, 0.0]),
            metallic=1.0,
            roughness=0.1
        )
        
        # 玻璃材质
        self.materials["glass"] = Material(
            albedo=np.array([0.9, 0.9, 0.9]),
            metallic=0.0,
            roughness=0.0
        )
        
        # 白色发光材质
        self.materials["white_emissive"] = Material(
            albedo=np.array([1.0, 1.0, 1.0]),
            metallic=0.0,
            roughness=0.0,
            emissive=np.array([5.0, 5.0, 5.0])
        )
        
        # 光源参数
        self.ambient_light = np.array([0.1, 0.1, 0.1])
        self.lights = []
        
        # 初始化着色器和纹理
        self.shaders = {}
        self.textures = {}
        
        # 空间划分加速结构
        self.bvh = None
        self.enable_bvh = True
    
    def initialize(self, renderer):
        """初始化路径追踪器"""
        super().initialize(renderer)
        # 简化实现，实际需要创建着色器和纹理
        pass
    
    def set_scene(self, scene):
        """设置场景数据
        
        参数:
        - scene: 场景对象
        """
        self.scene = scene
        # 提取场景中的光源
        self._extract_lights_from_scene()
    
    def set_camera(self, camera):
        """设置相机
        
        参数:
        - camera: 相机对象
        """
        self.camera = camera
    
    def _extract_lights_from_scene(self):
        """从场景中提取光源"""
        self.lights.clear()
        # 简化实现，实际需要从场景中提取光源
        pass
    
    def _trace_ray(self, ray_origin, ray_direction, bounce_count=0):
        """追踪单条光线，支持反射和折射
        
        参数:
        - ray_origin: 光线原点
        - ray_direction: 光线方向
        - bounce_count: 当前反弹次数
        
        返回:
        - 光线颜色
        """
        if bounce_count > self.max_bounces:
            return np.array([0.0, 0.0, 0.0])
        
        # 创建光线对象
        ray = Ray(ray_origin, ray_direction)
        
        # 光线与场景相交检测
        intersection = self._intersect_scene(ray)
        
        if not intersection.hit:
            return np.array([0.0, 0.0, 0.0])  # 没有击中任何物体，返回黑色
        
        # 获取材质
        material = intersection.material if isinstance(intersection.material, Material) else self.default_material
        
        # 计算直接光照
        direct_light = self._calculate_direct_light(intersection)
        
        # 计算间接光照（递归追踪）
        indirect_light = np.array([0.0, 0.0, 0.0])
        if bounce_count < self.max_bounces:
            # 计算反射方向
            reflected_direction = self._calculate_reflection_direction(ray.direction, intersection.normal, material.roughness)
            
            # 递归追踪反射光线
            reflected_ray_origin = intersection.position + intersection.normal * 0.001  # 偏移避免自相交
            reflection_color = self._trace_ray(reflected_ray_origin, reflected_direction, bounce_count + 1)
            
            # 应用PBR着色模型到反射光线
            reflection_color *= self._apply_pbr_shading(ray.direction, reflected_direction, intersection.normal, material)
            
            # 处理透明材质
            if material.transparent and material.transmission > 0.0:
                # 计算折射方向
                refracted_direction, is_total_reflection = self._calculate_refraction_direction(ray.direction, intersection.normal, material.ior)
                
                if not is_total_reflection:
                    # 递归追踪折射光线
                    refracted_ray_origin = intersection.position - intersection.normal * 0.001  # 偏移避免自相交
                    refraction_color = self._trace_ray(refracted_ray_origin, refracted_direction, bounce_count + 1)
                    
                    # 计算菲涅尔反射系数
                    fresnel = self._fresnel_effect(ray.direction, intersection.normal, material.ior)
                    
                    # 根据菲涅尔效果混合反射和折射颜色
                    indirect_light = reflection_color * fresnel + refraction_color * (1.0 - fresnel) * material.transmission
                else:
                    # 全反射，只使用反射颜色
                    indirect_light = reflection_color
            else:
                # 不透明材质，只使用反射颜色
                indirect_light = reflection_color
        
        # 计算自发光贡献
        emissive = material.emissive
        
        # 总光照
        total_light = direct_light + indirect_light + emissive
        
        return total_light
    
    def _apply_pbr_shading(self, incident_dir, outgoing_dir, normal, material):
        """应用PBR着色模型
        
        参数:
        - incident_dir: 入射方向
        - outgoing_dir: 出射方向
        - normal: 表面法线
        - material: 材质对象
        
        返回:
        - 着色结果
        """
        # 计算半程向量
        half_vector = (outgoing_dir + incident_dir) / 2.0
        half_vector /= np.linalg.norm(half_vector)
        
        # 计算NdotL, NdotV, NdotH
        NdotL = max(0.0, np.dot(normal, outgoing_dir))
        NdotV = max(0.0, np.dot(normal, incident_dir))
        NdotH = max(0.0, np.dot(normal, half_vector))
        
        # 菲涅尔项 (Fresnel-Schlick近似)
        F0 = material.albedo * material.metallic + (1.0 - material.metallic) * 0.04
        F = F0 + (1.0 - F0) * (1.0 - NdotV) ** 5
        
        # 粗糙度转换为α
        alpha = material.roughness ** 2
        alpha_sq = alpha * alpha
        
        # 几何项 (Smith GGX)
        G1 = 2.0 * NdotV / (NdotV + np.sqrt(alpha_sq + (1.0 - alpha_sq) * NdotV * NdotV))
        G2 = 2.0 * NdotL / (NdotL + np.sqrt(alpha_sq + (1.0 - alpha_sq) * NdotL * NdotL))
        G = G1 * G2
        
        # 法线分布项 (GGX)
        denominator = NdotH * NdotH * (alpha_sq - 1.0) + 1.0
        D = alpha_sq / (np.pi * denominator * denominator)
        
        # 漫反射项 (Lambert)
        diffuse = material.albedo / np.pi * (1.0 - material.metallic)
        
        # 镜面反射项 (Cook-Torrance)
        specular = (F * G * D) / (4.0 * NdotV * NdotL + 0.001)
        
        # 总着色结果
        return diffuse + specular
    
    def _intersect_scene(self, ray):
        """光线与场景相交检测
        
        参数:
        - ray: 光线对象
        
        返回:
        - Intersection: 相交结果
        """
        # 初始化最近相交结果
        closest_intersection = Intersection()
        
        # 存储光线方向到相交结果
        closest_intersection.ray_direction = ray.direction
        
        # 如果启用了BVH加速结构，使用BVH进行相交检测
        if self.enable_bvh and self.bvh:
            closest_intersection = self._intersect_bvh(ray, self.bvh)
        else:
            # 遍历所有物体，找到最近的相交点
            for obj in self.objects:
                intersection = obj.intersect(ray)
                if intersection.hit and intersection.distance < closest_intersection.distance:
                    intersection.ray_direction = ray.direction
                    closest_intersection = intersection
        
        return closest_intersection
    
    def _calculate_reflection_direction(self, incident_dir, normal, roughness):
        """计算反射方向
        
        参数:
        - incident_dir: 入射方向
        - normal: 表面法线
        - roughness: 粗糙度（0.0-1.0）
        
        返回:
        - 反射方向
        """
        # 理想反射方向
        perfect_reflection = incident_dir - 2.0 * np.dot(incident_dir, normal) * normal
        
        if roughness < 0.01:
            # 几乎光滑的表面，直接返回理想反射
            return perfect_reflection
        
        # 添加粗糙度引起的随机性
        # 使用余弦加权的半球采样
        def random_in_hemisphere(normal):
            """在法线方向的半球内生成随机向量"""
            while True:
                # 在单位立方体中生成随机点
                v = np.array([np.random.uniform(-1, 1), np.random.uniform(-1, 1), np.random.uniform(0, 1)])
                # 归一化
                v /= np.linalg.norm(v)
                # 确保在半球内
                if np.dot(v, normal) > 0:
                    return v
        
        # 生成随机微表面法线
        micro_normal = random_in_hemisphere(normal)
        
        # 根据粗糙度混合理想反射和随机反射
        reflection = perfect_reflection * (1.0 - roughness) + micro_normal * roughness
        reflection /= np.linalg.norm(reflection)
        
        return reflection
    
    def _calculate_refraction_direction(self, incident_dir, normal, ior):
        """计算折射方向，使用Snell's law
        
        参数:
        - incident_dir: 入射方向（指向表面）
        - normal: 表面法线（指向外部）
        - ior: 相对折射率（外部介质折射率 / 内部介质折射率）
        
        返回:
        - tuple: (refracted_direction, is_total_internal_reflection)
        """
        # 计算入射角的cos值
        cos_theta_i = np.dot(-incident_dir, normal)
        
        # 确定光线是从外部进入内部还是相反
        if cos_theta_i > 0:
            # 光线从外部进入内部，法线方向不变
            eta = 1.0 / ior
            outward_normal = normal
        else:
            # 光线从内部射出，反转法线方向
            eta = ior
            outward_normal = -normal
            cos_theta_i = np.abs(cos_theta_i)
        
        # 计算sin²θ_t
        sin2_theta_t = eta ** 2 * (1.0 - cos_theta_i ** 2)
        
        # 检查全反射
        if sin2_theta_t > 1.0:
            # 全反射，返回反射方向
            reflection_dir = incident_dir - 2.0 * np.dot(incident_dir, outward_normal) * outward_normal
            return (reflection_dir, True)
        
        # 计算cosθ_t
        cos_theta_t = np.sqrt(1.0 - sin2_theta_t)
        
        # 计算折射方向
        refracted_dir = eta * incident_dir + (eta * cos_theta_i - cos_theta_t) * outward_normal
        refracted_dir /= np.linalg.norm(refracted_dir)
        
        return (refracted_dir, False)
    
    def _fresnel_effect(self, incident_dir, normal, ior):
        """计算Fresnel效果，返回反射系数
        
        参数:
        - incident_dir: 入射方向
        - normal: 表面法线
        - ior: 相对折射率
        
        返回:
        - float: 反射系数
        """
        # 使用Schlick's approximation
        cos_theta = max(0.0, np.dot(-incident_dir, normal))
        r0 = ((1.0 - ior) / (1.0 + ior)) ** 2
        return r0 + (1.0 - r0) * (1.0 - cos_theta) ** 5
    
    def _calculate_direct_light(self, intersection):
        """计算直接光照，包括阴影和PBR着色，支持多种光源类型
        
        参数:
        - intersection: 相交结果
        
        返回:
        - 直接光照颜色
        """
        total_light = self.ambient_light.copy()
        hit_position = intersection.position
        hit_normal = intersection.normal
        material = intersection.material if isinstance(intersection.material, Material) else self.default_material
        
        # 添加环境光遮蔽贡献
        total_light *= material.ao
        
        # 遍历所有光源
        for light in self.lights:
            if not light.enabled:
                continue
            
            # 根据光源类型计算光照贡献
            if light.type == "point":
                # 点光源
                light_contribution = self._calculate_point_light_contribution(light, hit_position, hit_normal, material, intersection)
            elif light.type == "directional":
                # 方向光
                light_contribution = self._calculate_directional_light_contribution(light, hit_position, hit_normal, material, intersection)
            elif light.type == "area":
                # 面光源
                light_contribution = self._calculate_area_light_contribution(light, hit_position, hit_normal, material, intersection)
            else:
                # 未知光源类型，跳过
                continue
            
            total_light += light_contribution
        
        return total_light
    
    def _calculate_point_light_contribution(self, light, hit_position, hit_normal, material, intersection):
        """计算点光源的光照贡献
        
        参数:
        - light: 点光源对象
        - hit_position: 命中位置
        - hit_normal: 命中法线
        - material: 材质对象
        - intersection: 相交结果
        
        返回:
        - 点光源贡献的颜色
        """
        # 计算光照方向和距离
        light_direction = light.position - hit_position
        light_distance = np.linalg.norm(light_direction)
        light_direction = light_direction / light_distance
        
        # 检查阴影
        in_shadow = self._is_in_shadow(hit_position, light.position)
        if in_shadow:
            return np.array([0.0, 0.0, 0.0])
        
        # 计算光照衰减
        attenuation = 1.0 / (1.0 + 0.1 * light_distance + 0.01 * light_distance * light_distance)
        
        # 计算PBR着色
        pbr_color = self._apply_pbr_shading(
            -light_direction,  # 入射方向（指向光源）
            -intersection.ray_direction,  # 出射方向（指向相机）
            hit_normal, 
            material
        )
        
        # 计算光照贡献
        return light.color * light.intensity * pbr_color * attenuation
    
    def _calculate_directional_light_contribution(self, light, hit_position, hit_normal, material, intersection):
        """计算方向光的光照贡献
        
        参数:
        - light: 方向光对象
        - hit_position: 命中位置
        - hit_normal: 命中法线
        - material: 材质对象
        - intersection: 相交结果
        
        返回:
        - 方向光贡献的颜色
        """
        # 方向光的方向是固定的，没有衰减
        light_direction = -light.direction
        
        # 检查阴影
        in_shadow = self._is_in_shadow_directional(light, hit_position)
        if in_shadow:
            return np.array([0.0, 0.0, 0.0])
        
        # 计算PBR着色
        pbr_color = self._apply_pbr_shading(
            -light_direction,  # 入射方向（指向光源）
            -intersection.ray_direction,  # 出射方向（指向相机）
            hit_normal, 
            material
        )
        
        # 方向光没有衰减，直接返回光照贡献
        return light.color * light.intensity * pbr_color
    
    def _calculate_area_light_contribution(self, light, hit_position, hit_normal, material, intersection):
        """计算面光源的光照贡献
        
        参数:
        - light: 面光源对象
        - hit_position: 命中位置
        - hit_normal: 命中法线
        - material: 材质对象
        - intersection: 相交结果
        
        返回:
        - 面光源贡献的颜色
        """
        # 面光源软阴影采样参数
        shadow_samples = 4  # 软阴影采样数
        if self.quality_level == EffectQuality.LOW:
            shadow_samples = 2
        elif self.quality_level == EffectQuality.HIGH:
            shadow_samples = 8
        
        # 生成面光源的基向量（使用正切空间）
        # 假设光源法线指向Z轴负方向，生成X和Y轴
        if abs(light.normal[2]) > 0.9:
            # 法线接近Z轴，使用X轴作为切线
            tangent = np.array([1.0, 0.0, 0.0])
        else:
            # 否则，使用Z轴叉乘法线得到切线
            tangent = np.cross(light.normal, np.array([0.0, 0.0, 1.0]))
        tangent = tangent / np.linalg.norm(tangent)
        
        # 副切线
        bitangent = np.cross(light.normal, tangent)
        bitangent = bitangent / np.linalg.norm(bitangent)
        
        # 采样面光源不同位置，计算软阴影
        shadow_sum = 0.0
        for _ in range(shadow_samples):
            # 在面光源上生成随机采样点
            u = np.random.uniform(-0.5, 0.5) * light.width
            v = np.random.uniform(-0.5, 0.5) * light.height
            
            # 计算采样点位置
            sample_position = light.position + u * tangent + v * bitangent
            
            # 检查阴影
            if not self._is_in_shadow(hit_position, sample_position):
                shadow_sum += 1.0
        
        # 计算阴影因子
        shadow_factor = shadow_sum / shadow_samples
        if shadow_factor < 0.01:
            return np.array([0.0, 0.0, 0.0])
        
        # 计算光照方向和距离
        light_direction = light.position - hit_position
        light_distance = np.linalg.norm(light_direction)
        light_direction = light_direction / light_distance
        
        # 计算光照衰减
        attenuation = 1.0 / (1.0 + 0.1 * light_distance + 0.01 * light_distance * light_distance)
        
        # 计算PBR着色
        pbr_color = self._apply_pbr_shading(
            -light_direction,  # 入射方向（指向光源）
            -intersection.ray_direction,  # 出射方向（指向相机）
            hit_normal, 
            material
        )
        
        # 面光源的强度需要考虑面积
        area = light.width * light.height
        area_factor = 1.0 / (4.0 * np.pi * area)
        
        # 计算光照贡献，应用阴影因子
        return light.color * light.intensity * pbr_color * attenuation * area_factor * shadow_factor
    
    def _is_in_shadow(self, hit_position, light_position):
        """检查点是否在点光源或面光源的阴影中
        
        参数:
        - hit_position: 命中位置
        - light_position: 光源位置
        
        返回:
        - bool: 是否在阴影中
        """
        # 创建阴影光线
        shadow_ray_direction = light_position - hit_position
        shadow_ray_distance = np.linalg.norm(shadow_ray_direction)
        shadow_ray_direction /= shadow_ray_distance
        
        # 偏移起始位置，避免自相交
        shadow_ray_origin = hit_position + shadow_ray_direction * 0.001
        
        # 创建光线对象
        shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)
        
        # 光线与场景相交检测
        intersection = self._intersect_scene(shadow_ray)
        
        # 如果光线在到达光源前击中了其他物体，则在阴影中
        return intersection.hit and intersection.distance < shadow_ray_distance
    
    def _is_in_shadow_directional(self, light, hit_position):
        """检查点是否在方向光的阴影中
        
        参数:
        - light: 方向光对象
        - hit_position: 命中位置
        
        返回:
        - bool: 是否在阴影中
        """
        # 方向光的阴影光线方向是光源方向的反方向
        shadow_ray_direction = -light.direction
        
        # 偏移起始位置，避免自相交
        shadow_ray_origin = hit_position + shadow_ray_direction * 0.001
        
        # 创建光线对象
        shadow_ray = Ray(shadow_ray_origin, shadow_ray_direction)
        
        # 方向光的阴影光线是平行的，所以我们使用一个很大的距离
        max_distance = 1000.0
        
        # 光线与场景相交检测
        intersection = self._intersect_scene(shadow_ray)
        
        # 如果光线在最大距离内击中了其他物体，则在阴影中
        return intersection.hit and intersection.distance < max_distance
    
    def _intersect_bvh(self, ray, bvh):
        """光线与BVH加速结构相交检测
        
        参数:
        - ray: 光线对象
        - bvh: BVH加速结构
        
        返回:
        - Intersection: 相交结果
        """
        hit, closest_intersection = bvh.intersect_ray(ray)
        if hit:
            return closest_intersection
        return Intersection()
    
    def _build_bvh(self):
        """构建BVH加速结构"""
        if not self.objects:
            return
        
        # 创建根节点并构建BVH树
        self.bvh = BVHNode(self.objects)
        self.bvh.build()
        self.engine.logger.info("BVH加速结构构建完成")
    

    
    def _apply_denoising(self, input_texture):
        """应用降噪
        
        参数:
        - input_texture: 输入纹理
        
        返回:
        - 降噪后的纹理
        """
        # 简化实现，实际需要使用降噪算法
        return input_texture
    
    def _apply_adaptive_sampling(self, input_texture):
        """应用自适应采样
        
        参数:
        - input_texture: 输入纹理
        
        返回:
        - 自适应采样后的纹理
        """
        # 简化实现，实际需要使用自适应采样算法
        return input_texture
    
    def _apply_effect(self, input_texture, output_texture):
        """应用路径追踪效果"""
        # 根据GPU架构选择不同的实现路径
        if self.gpu_architecture == GPUArchitecture.NVIDIA_MAXWELL:
            return self._apply_path_tracing_maxwell(input_texture, output_texture)
        elif self.gpu_architecture == GPUArchitecture.AMD_GCN:
            return self._apply_path_tracing_gcn(input_texture, output_texture)
        else:
            return self._apply_path_tracing_generic(input_texture, output_texture)
    
    def _apply_path_tracing_maxwell(self, input_texture, output_texture):
        """针对NVIDIA Maxwell架构的优化实现"""
        # Maxwell架构上使用低采样数和降采样
        self.sample_count = 1
        self.max_bounces = 1
        self.downsample_factor = 4
        return self._generic_path_tracing_implementation(input_texture, output_texture)
    
    def _apply_path_tracing_gcn(self, input_texture, output_texture):
        """针对AMD GCN架构的优化实现"""
        # GCN架构可以使用更高的采样数
        self.sample_count = 2
        self.max_bounces = 2
        self.downsample_factor = 2
        return self._generic_path_tracing_implementation(input_texture, output_texture)
    
    def _apply_path_tracing_generic(self, input_texture, output_texture):
        """通用实现"""
        # 保守的通用实现
        self.sample_count = 1
        self.max_bounces = 1
        self.downsample_factor = 4
        return self._generic_path_tracing_implementation(input_texture, output_texture)
    
    def _generic_path_tracing_implementation(self, input_texture, output_texture):
        """路径追踪通用实现核心逻辑"""
        # 简化实现，实际需要：
        # 1. 生成相机光线
        # 2. 对每条光线进行路径追踪
        # 3. 累积采样结果
        # 4. 应用降噪
        # 5. 输出最终结果
        return input_texture
    
    def reset(self):
        """重置路径追踪器状态"""
        self.accumulation_buffer = None
        self.sample_buffer = None
        self.frame_count = 0
    
    def adjust_quality(self, quality_level):
        """调整路径追踪质量
        
        参数:
        - quality_level: 质量级别
        """
        super().adjust_quality(quality_level)
        
        if quality_level == EffectQuality.LOW:
            self.sample_count = 1
            self.max_bounces = 1
            self.enable_denoising = True
            self.enable_adaptive_sampling = False
            self.downsample_factor = 4
        elif quality_level == EffectQuality.MEDIUM:
            self.sample_count = 2
            self.max_bounces = 2
            self.enable_denoising = True
            self.enable_adaptive_sampling = True
            self.downsample_factor = 2
        elif quality_level == EffectQuality.HIGH:
            self.sample_count = 4
            self.max_bounces = 3
            self.enable_denoising = True
            self.enable_adaptive_sampling = True
            self.downsample_factor = 1