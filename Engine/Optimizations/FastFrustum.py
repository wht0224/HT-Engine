"""
优化的视锥体类
针对性能进行优化，特别适用于GTX 1650 Max-Q等移动GPU
"""

import math
from Engine.Math.Math import Vector3


class OptimizedFrustum:
    """
    针对性能优化的视锥体类
    使用更高效的算法进行包围盒和点的剔除判断
    """
    
    def __init__(self):
        # 视锥体的6个平面
        # 每个平面用4个值表示：Ax + By + Cz + D = 0
        self.planes = [
            [0.0, 0.0, 0.0, 0.0],  # 左平面
            [0.0, 0.0, 0.0, 0.0],  # 右平面
            [0.0, 0.0, 0.0, 0.0],  # 下平面
            [0.0, 0.0, 0.0, 0.0],  # 上平面
            [0.0, 0.0, 0.0, 0.0],  # 近平面
            [0.0, 0.0, 0.0, 0.0]   # 远平面
        ]
        # 添加相机位置用于距离计算
        self.camera_position = Vector3(0, 0, 0)
    
    def extract_from_matrix(self, view_projection_matrix):
        """
        从视图投影矩阵提取视锥体平面（优化版本）
        """
        # 直接访问矩阵数据
        rows = view_projection_matrix.data

        # 提取6个平面 - 使用直接索引访问，避免重复计算
        # 左平面：行3 + 行0
        self.planes[0][0] = rows[12] + rows[0]  # 3*4 + 0 + 0 = 12
        self.planes[0][1] = rows[13] + rows[1]  # 3*4 + 0 + 1 = 13
        self.planes[0][2] = rows[14] + rows[2]  # 3*4 + 0 + 2 = 14
        self.planes[0][3] = rows[15] + rows[3]  # 3*4 + 0 + 3 = 15

        # 右平面：行3 - 行0
        self.planes[1][0] = rows[12] - rows[0]
        self.planes[1][1] = rows[13] - rows[1]
        self.planes[1][2] = rows[14] - rows[2]
        self.planes[1][3] = rows[15] - rows[3]

        # 下平面：行3 + 行1
        self.planes[2][0] = rows[12] + rows[4]  # 3*4 + 1 + 0 = 12 + 4
        self.planes[2][1] = rows[13] + rows[5]  # 3*4 + 1 + 1 = 13 + 4
        self.planes[2][2] = rows[14] + rows[6]  # 3*4 + 1 + 2 = 14 + 4
        self.planes[2][3] = rows[15] + rows[7]  # 3*4 + 1 + 3 = 15 + 4

        # 上平面：行3 - 行1
        self.planes[3][0] = rows[12] - rows[4]
        self.planes[3][1] = rows[13] - rows[5]
        self.planes[3][2] = rows[14] - rows[6]
        self.planes[3][3] = rows[15] - rows[7]

        # 近平面：行3 + 行2
        self.planes[4][0] = rows[12] + rows[8]  # 3*4 + 2 + 0 = 12 + 8
        self.planes[4][1] = rows[13] + rows[9]  # 3*4 + 2 + 1 = 13 + 8
        self.planes[4][2] = rows[14] + rows[10] # 3*4 + 2 + 2 = 14 + 8
        self.planes[4][3] = rows[15] + rows[11] # 3*4 + 2 + 3 = 15 + 8

        # 远平面：行3 - 行2
        self.planes[5][0] = rows[12] - rows[8]
        self.planes[5][1] = rows[13] - rows[9]
        self.planes[5][2] = rows[14] - rows[10]
        self.planes[5][3] = rows[15] - rows[11]

        # 归一化平面 - 优化版本，只在必要时归一化
        for i in range(6):
            self._normalize_plane_optimized(i)
    
    def _normalize_plane_optimized(self, plane_index):
        """
        优化的平面归一化
        """
        plane = self.planes[plane_index]
        # 计算平面的模长的平方
        length_sq = plane[0]*plane[0] + plane[1]*plane[1] + plane[2]*plane[2]
        
        # 使用快速平方根倒数近似（如果可用）
        if length_sq > 1e-6:
            # 使用math.sqrt而不是手动实现
            magnitude = math.sqrt(length_sq)
            inv_magnitude = 1.0 / magnitude
            plane[0] *= inv_magnitude
            plane[1] *= inv_magnitude
            plane[2] *= inv_magnitude
            plane[3] *= inv_magnitude

    def contains_bounding_box_fast(self, bounding_box):
        """
        快速包围盒剔除测试
        使用更高效的算法减少计算量
        """
        # 获取包围盒的8个顶点
        min_pt = bounding_box.min
        max_pt = bounding_box.max
        
        # 对于每个平面，检查包围盒的所有顶点
        for plane in self.planes:
            # 计算包围盒在该平面法线方向上的投影
            # 如果包围盒的所有顶点都在平面的背面，则整个包围盒在视锥体外
            # 使用分离轴定理的简化版本
            plane_normal = Vector3(plane[0], plane[1], plane[2])
            
            # 计算包围盒在平面法线方向上的最小和最大投影
            box_center = Vector3(
                (min_pt.x + max_pt.x) * 0.5,
                (min_pt.y + max_pt.y) * 0.5,
                (min_pt.z + max_pt.z) * 0.5
            )
            
            box_extents = Vector3(
                (max_pt.x - min_pt.x) * 0.5,
                (max_pt.y - min_pt.y) * 0.5,
                (max_pt.z - min_pt.z) * 0.5
            )
            
            # 计算包围盒中心到平面的距离
            center_distance = (
                plane[0] * box_center.x +
                plane[1] * box_center.y +
                plane[2] * box_center.z +
                plane[3]
            )
            
            # 计算包围盒在平面法线方向上的投影半径
            projected_radius = (
                abs(plane[0] * box_extents.x) +
                abs(plane[1] * box_extents.y) +
                abs(plane[2] * box_extents.z)
            )
            
            # 如果包围盒完全在平面背面，则不在视锥体内
            if center_distance + projected_radius < 0:
                return False
        
        return True

    def contains_bounding_box(self, bounding_box):
        """
        检查包围盒是否在视锥体内
        与原始Frustum类保持接口兼容
        """
        # 获取包围盒的8个顶点
        vertices = [
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.min.z),
            Vector3(bounding_box.min.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.min.y, bounding_box.max.z),
            Vector3(bounding_box.min.x, bounding_box.max.y, bounding_box.max.z),
            Vector3(bounding_box.max.x, bounding_box.max.y, bounding_box.max.z)
        ]

        # 检查所有顶点是否在所有平面的正面
        for plane in self.planes:
            # 检查是否有顶点在平面正面
            has_vertex_in_front = False
            for vertex in vertices:
                # 计算点到平面的距离（Ax + By + Cz + D）
                distance = plane[0] * vertex.x + plane[1] * vertex.y + plane[2] * vertex.z + plane[3]
                if distance >= 0:
                    has_vertex_in_front = True
                    break

            # 如果所有顶点都在平面背面，返回False
            if not has_vertex_in_front:
                return False

        return True

    def contains_sphere_fast(self, center, radius):
        """
        快速球体剔除测试
        """
        # 检查球体是否在所有平面的正面
        for plane in self.planes:
            # 计算球心到平面的距离
            distance = (
                plane[0] * center.x +
                plane[1] * center.y +
                plane[2] * center.z +
                plane[3]
            )

            # 如果球心到平面的距离小于负半径，则球体完全在平面背面
            if distance < -radius:
                return False

        return True

    def contains_point(self, point):
        """
        检查点是否在视锥体内
        与原始Frustum类保持接口兼容
        """
        # 检查点是否在所有平面的正面
        for plane in self.planes:
            # 计算点到平面的距离（Ax + By + Cz + D）
            distance = (
                plane[0] * point.x +
                plane[1] * point.y +
                plane[2] * point.z +
                plane[3]
            )

            # 如果点到平面的距离小于0，则点在平面背面
            if distance < 0:
                return False

        return True

    def contains_point_fast(self, point):
        """
        快速点剔除测试
        """
        # 检查点是否在所有平面的正面
        for plane in self.planes:
            # 计算点到平面的距离（Ax + By + Cz + D）
            distance = (
                plane[0] * point.x +
                plane[1] * point.y +
                plane[2] * point.z +
                plane[3]
            )

            # 如果点到平面的距离小于0，则点在平面背面
            if distance < 0:
                return False

        return True

    def contains_sphere(self, center, radius):
        """
        检查球体是否在视锥体内
        与原始Frustum类保持接口兼容
        """
        # 检查球体是否在所有平面的正面
        for plane in self.planes:
            # 计算球心到平面的距离
            distance = (
                plane[0] * center.x +
                plane[1] * center.y +
                plane[2] * center.z +
                plane[3]
            )

            # 如果球心到平面的距离小于负半径，则球体完全在平面背面
            if distance < -radius:
                return False

        return True

    def compute_corners(self):
        """
        计算视锥体的8个角点
        用于更高级的剔除算法
        """
        corners = []
        
        # 视锥体的8个角点，用位掩码表示近/远、左/右、上/下
        # 0=近左下, 1=近左上, 2=近右下, 3=近右上, 
        # 4=远左下, 5=远左上, 6=远右下, 7=远右上
        for i in range(8):
            # 解析位掩码
            near_far = i & 1  # 0=近, 1=远
            left_right = (i >> 1) & 1  # 0=左, 1=右
            bottom_top = (i >> 2) & 1  # 0=下, 1=上
            
            # 这里我们不实际计算交点，而是返回占位符
            # 实际的角点计算在需要时才进行
            corners.append(None)
        
        return corners


class FastFrustumCulling:
    """
    快速视锥体剔除辅助类
    提供常用的剔除函数
    """
    
    @staticmethod
    def sphere_in_frustum(center, radius, frustum_planes):
        """
        快速球体视锥体测试
        """
        for plane in frustum_planes:
            distance = (
                plane[0] * center.x +
                plane[1] * center.y +
                plane[2] * center.z +
                plane[3]
            )
            
            if distance < -radius:
                return False
        
        return True
    
    @staticmethod
    def aabb_in_frustum(min_pt, max_pt, frustum_planes):
        """
        快速AABB视锥体测试
        """
        center = Vector3(
            (min_pt.x + max_pt.x) * 0.5,
            (min_pt.y + max_pt.y) * 0.5,
            (min_pt.z + max_pt.z) * 0.5
        )
        
        extents = Vector3(
            (max_pt.x - min_pt.x) * 0.5,
            (max_pt.y - min_pt.y) * 0.5,
            (max_pt.z - min_pt.z) * 0.5
        )
        
        for plane in frustum_planes:
            # 计算包围盒中心到平面的距离
            center_distance = (
                plane[0] * center.x +
                plane[1] * center.y +
                plane[2] * center.z +
                plane[3]
            )
            
            # 计算包围盒在平面法线方向上的投影半径
            projected_radius = (
                abs(plane[0]) * extents.x +
                abs(plane[1]) * extents.y +
                abs(plane[2]) * extents.z
            )
            
            # 如果包围盒完全在平面背面，则不在视锥体内
            if center_distance + projected_radius < 0:
                return False
        
        return True