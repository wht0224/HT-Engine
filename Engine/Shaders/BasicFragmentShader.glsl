#version 330 core

// 从顶点着色器接收的输入
in vec2 v_texcoord;    // 纹理坐标
in vec3 v_normal;      // 世界空间法线
in vec3 v_worldPos;    // 世界空间位置

// 输出
out vec4 FragColor;

// 统一变量 - 使用材质系统的命名
uniform sampler2D u_baseColorTexture; // 基础颜色纹理（地面岩石）
uniform sampler2D u_rockTexture;       // 岩石纹理（地面）
uniform sampler2D u_snowTexture;       // 雪地纹理（山顶）
uniform sampler2D u_normalMap;         // 法线贴图（可选）
uniform sampler2D u_specularMap;       // 高光贴图（可选）
uniform sampler2D u_aoMap;             // 环境光遮蔽贴图（可选）
uniform sampler2D u_shadowMap;         // 阴影贴图
uniform sampler2D uShadowTexture;      // Natural系统的阴影纹理

// 地形参数
uniform float u_terrainMinHeight;      // 地形最低高度
uniform float u_terrainMaxHeight;      // 地形最高高度
uniform float u_snowLine;              // 雪线高度（0-1）

uniform vec3 u_baseColor;    // 基础颜色
uniform bool u_hasTexture;   // 是否绑定了基础颜色纹理
uniform vec3 u_specularColor; // 高光颜色
uniform float u_shininess;    // 光泽度
uniform float u_alpha;         // 全局透明度

// Emissive（自发光）参数
uniform vec3 u_emissiveColor;  // 自发光颜色
uniform float u_emissiveStrength;  // 自发光强度
uniform float u_emissiveEnabled;   // 是否启用自发光

// 光照统一变量
uniform vec3 u_ambientLight;     // 环境光
uniform vec3 u_directionalLightDir; // 方向光方向
uniform vec3 u_directionalLightColor; // 方向光颜色
uniform float u_directionalLightIntensity; // 方向光强度

// 相机位置
uniform vec3 u_cameraPos;        // 相机世界空间位置

// 阴影统一变量
uniform mat4 u_lightSpaceMatrix; // 光源空间矩阵
uniform float u_shadowBias;      // 阴影偏移

// 雾效统一变量
uniform bool u_fogEnabled;       // 是否启用雾
uniform vec3 u_fogColor;         // 雾颜色
uniform float u_fogDensity;      // 雾密度

// 主函数 - 简化版：优先用 u_baseColor
void main() {
    vec3 finalAlbedo = u_baseColor;
    
    // 如果有基础纹理，就用纹理乘以基础色
    if (u_hasTexture) {
        vec4 texColor = texture(u_baseColorTexture, v_texcoord);
        finalAlbedo = texColor.rgb * u_baseColor;
    }
    
    vec3 albedo = finalAlbedo;
    
    // 简单的漫反射光照 + 伪阴影
    vec3 N = normalize(v_normal);
    vec3 L = normalize(-u_directionalLightDir);
    float NdotL = max(dot(N, L), 0.0);
    
    // 简单的伪阴影：背光面更暗
    float shadowFactor = smoothstep(-0.2, 0.8, NdotL);
    shadowFactor = mix(0.3, 1.0, shadowFactor); // 阴影区域亮度30%-100%
    
    vec3 ambient = u_ambientLight;
    vec3 diffuse = u_directionalLightColor * u_directionalLightIntensity * NdotL * shadowFactor;
    
    vec3 lighting = ambient + diffuse;
    
    vec3 finalColor = albedo * lighting;
    
    // 添加自发光效果
    if (u_emissiveEnabled > 0.5) {
        finalColor = mix(finalColor, u_emissiveColor, u_emissiveStrength);
    }
    
    // 简单的雾效（可选）
    if (u_fogEnabled) {
        float distance = length(v_worldPos - u_cameraPos);
        float fogFactor = exp(-u_fogDensity * distance * 0.1);
        fogFactor = clamp(fogFactor, 0.0, 1.0);
        finalColor = mix(u_fogColor, finalColor, fogFactor);
    }
    
    FragColor = vec4(finalColor, u_alpha);
}
