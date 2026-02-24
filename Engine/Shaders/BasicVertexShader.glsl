#version 330 core

// 默认实例化宏
#ifndef INSTANCING
#define INSTANCING 0
#endif

// 输入属性 - 显式指定位置以匹配Mesh.py
layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

// 实例化数据 (如果启用)
#if INSTANCING
layout(location = 4) in mat4 a_instanceModel; // 占用 4,5,6,7
#endif

// 输出到片段着色器的变量
out vec2 v_texcoord;      // 纹理坐标
out vec3 v_normal;        // 世界空间法线
out vec3 v_worldPos;      // 世界空间位置

// 统一变量
uniform mat4 u_model;         // 模型矩阵
uniform mat4 u_viewProj;      // 视图投影矩阵
uniform mat3 u_normalMatrix;   // 法线变换矩阵

// 主函数 - 高质量光照版本
void main() {
    mat4 modelMatrix = u_model;
    
    // 如果启用实例化，这里应该使用实例矩阵
    #if INSTANCING
        modelMatrix = a_instanceModel;
    #endif

    // 计算世界空间位置
    vec4 worldPos = modelMatrix * vec4(a_position, 1.0);
    v_worldPos = worldPos.xyz;
    
    // 计算最终裁剪空间位置
    gl_Position = u_viewProj * worldPos;
    
    // 变换法线到世界空间
    v_normal = normalize(u_normalMatrix * a_normal);
    
    // 传递纹理坐标
    v_texcoord = a_texcoord;
}
