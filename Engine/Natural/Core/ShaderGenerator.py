"""
着色器生成器

根据Natural系统的规则配置自动生成GLSL着色器。
实现"Data-Driven"的终极形态 - Natural不仅计算数据，还生成渲染代码。
"""

from typing import Dict, List, Any
import textwrap


class ShaderGenerator:
    """着色器生成器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_features = self._analyze_features()
    
    def _analyze_features(self) -> List[str]:
        """分析启用的特性"""
        features = []
        
        # 光照特性
        if self.config.get('enable_lighting', False):
            features.append('LIGHTING')
        
        # 大气特性
        if self.config.get('enable_atmosphere', False):
            features.append('ATMOSPHERE')
        
        # 水文特性
        if self.config.get('enable_hydro_visual', False):
            features.append('HYDRO_VISUAL')
        
        # 风化特性
        if self.config.get('enable_thermal_weathering', False):
            features.append('THERMAL_WEATHERING')
        
        return features
    
    def generate_vertex_shader(self) -> str:
        """生成顶点着色器"""
        features = self.enabled_features
        
        # 顶点着色器基础
        shader = textwrap.dedent(f"""#version 330 core
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec2 aTexCoord;
layout (location = 2) in vec3 aNormal;
layout (location = 3) in vec3 aTangent;
layout (location = 4) in vec3 aBitangent;
layout (location = 5) in vec4 aColor;
layout (location = 6) in mat4 aModelMatrix;
layout (location = 7) in mat4 aViewMatrix;
layout (location = 8) in mat4 aProjectionMatrix;
layout (location = 9) in mat4 aNormalMatrix;
layout (location = 10) in mat4 aModelViewProjectionMatrix;

out vec3 FragPos;
out vec3 FragNormal;
out vec2 FragTexCoord;
out vec4 FragColor;
out float FragAO;
out float FragShadow;
out float FragWetness;
out float FragRoughness;
out float FragGodRay;
out float FogFactor;
""")
        
        # 特性相关输出
        if 'LIGHTING' in features:
            shader += textwrap.dedent("""
// 光照相关
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform float uTime;
""")
        
        if 'ATMOSPHERE' in features:
            shader += textwrap.dedent("""
// 大气相关
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uCameraPos;
uniform sampler2D uGodRayTexture;
""")
        
        if 'HYDRO_VISUAL' in features:
            shader += textwrap.dedent("""
// 水文相关
uniform sampler2D uWetnessMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uReflectionMap;
""")
        
        # 主函数
        shader += textwrap.dedent("""
void main() {
    // 变换到世界空间
    vec4 worldPos = aModelMatrix * vec4(aPos, 1.0);
    FragPos = worldPos.xyz;
    FragNormal = mat3(aNormalMatrix) * aNormal;
    
    // 视差切线
    vec3 T = normalize(FragNormal);
    vec3 B = cross(T, vec3(0.0, 1.0, 0.0));
    vec3 N = cross(B, T);
    FragTangent = normalize(vec3(aTangent.xyz));
    FragBitangent = normalize(vec3(aBitangent.xyz));
    
    // UV坐标
    FragTexCoord = aTexCoord;
    
    // 基础颜色
    FragColor = aColor;
""")
        
        # 光照计算
        if 'LIGHTING' in features:
            shader += textwrap.dedent("""
    // 漫反射
    vec3 N = normalize(FragNormal);
    float NdotL = max(dot(N, -uLightDir), 0.0);
    vec3 diffuse = FragColor * (uAmbientColor + uLightColor * NdotL);
    
    // 简单的日落色调调整
    float sunsetFactor = max(0.0, dot(-uLightDir, vec3(0.0, -1.0, 0.0)));
    vec3 sunsetColor = vec3(1.0, 0.6, 0.3);
    diffuse = mix(diffuse, diffuse * sunsetColor, sunsetFactor * 0.3);
    
    FragColor = diffuse;
""")
        
        # 雾效计算
        if 'ATMOSPHERE' in features:
            shader += textwrap.dedent("""
    // 雾效
    float distance = length(FragPos - uCameraPos);
    float fogFactor = exp(-uFogDensity * distance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    // 日落雾色
    vec3 sunsetFogColor = mix(uFogColor, vec3(1.0, 0.5, 0.3), 0.3);
    vec3 fogColor = mix(sunsetFogColor, FragColor, fogFactor);
    
    FragColor = fogColor;
""")
        
        # 水文效果
        if 'HYDRO_VISUAL' in features:
            shader += textwrap.dedent("""
    // 湿润度
    float wetness = texture(uWetnessMap, FragTexCoord).r;
    
    // 粗糙度
    float roughness = texture(uRoughnessMap, FragTexCoord).r;
    
    // 湿润表面更光滑
    float smoothness = 1.0 - wetness * 0.8;
    roughness = mix(roughness, 0.1, smoothness);
    
    // 反射强度
    float reflectivity = wetness * 0.5;
""")
        
        # 结束
        shader += textwrap.dedent("""
    gl_Position = aProjectionMatrix * aViewMatrix * aModelMatrix * vec4(aPos, 1.0);
}
""")
        
        return shader
    
    def generate_fragment_shader(self) -> str:
        """生成片段着色器"""
        features = self.enabled_features
        
        # 片段着色器基础
        shader = textwrap.dedent(f"""#version 330 core
precision highp float;

in vec3 FragPos;
in vec3 FragNormal;
in vec2 FragTexCoord;
in vec4 FragColor;
in float FragAO;
in float FragShadow;
in float FragWetness;
in float FragRoughness;
in float FragGodRay;
in float FogFactor;
""")
        
        # 光照特性
        if 'LIGHTING' in features:
            shader += textwrap.dedent("""
// 光照uniform
uniform vec3 uLightDir;
uniform vec3 uLightColor;
uniform vec3 uAmbientColor;
uniform sampler2D uAOTexture;
uniform sampler2D uShadowTexture;
""")
        
        # 大气特性
        if 'ATMOSPHERE' in features:
            shader += textwrap.dedent("""
// 大气uniform
uniform vec3 uFogColor;
uniform float uFogDensity;
uniform vec3 uCameraPos;
uniform sampler2D uGodRayTexture;
""")
        
        # 水文特性
        if 'HYDRO_VISUAL' in features:
            shader += textwrap.dedent("""
// 水文uniform
uniform sampler2D uWetnessMap;
uniform sampler2D uRoughnessMap;
uniform sampler2D uReflectionMap;
""")
        
        # 主函数
        shader += textwrap.dedent("""
out vec4 FinalColor;

void main() {
    // 基础颜色
    vec3 color = FragColor;
""")
        
        # 光照效果
        if 'LIGHTING' in features:
            shader += textwrap.dedent("""
    // 采样AO
    float ao = texture(uAOTexture, FragTexCoord).r;
    
    // 采样阴影
    float shadow = texture(uShadowTexture, FragTexCoord).r;
    
    // 漫反射
    vec3 N = normalize(FragNormal);
    // 强制双面光照：如果是背面，翻转法线
    if (!gl_FrontFacing) {
        N = -N;
    }
    float NdotL = max(dot(N, -uLightDir), 0.0);
    vec3 diffuse = color * (uAmbientColor + uLightColor * NdotL);
    
    // 应用AO
    diffuse *= ao;
    
    // 应用阴影
    diffuse *= shadow;
    
    // 日落色调
    float sunsetFactor = max(0.0, dot(-uLightDir, vec3(0.0, -1.0, 0.0)));
    vec3 sunsetColor = vec3(1.0, 0.6, 0.3);
    diffuse = mix(diffuse, diffuse * sunsetColor, sunsetFactor * 0.3);
    
    color = diffuse;
""")
        
        # 雾效
        if 'ATMOSPHERE' in features:
            shader += textwrap.dedent("""
    // 雾效
    float distance = length(FragPos - uCameraPos);
    float fogFactor = exp(-uFogDensity * distance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    
    // 日落雾色
    vec3 sunsetFogColor = mix(uFogColor, vec3(1.0, 0.5, 0.3), 0.3);
    vec3 fogColor = mix(sunsetFogColor, color, fogFactor);
    
    color = fogColor;
""")
        
        # 水文效果
        if 'HYDRO_VISUAL' in features:
            shader += textwrap.dedent("""
    // 湿润度
    float wetness = texture(uWetnessMap, FragTexCoord).r;
    
    // 粗糙度
    float roughness = texture(uRoughnessMap, FragTexCoord).r;
    
    // 湿润表面更光滑
    float smoothness = 1.0 - wetness * 0.8;
    roughness = mix(roughness, 0.1, smoothness);
    
    // 反射强度
    float reflectivity = wetness * 0.5;
    
    // 简单的菲涅尔反射
    vec3 viewDir = normalize(-FragPos);
    vec3 halfDir = normalize(viewDir + vec3(0.0, 1.0, 0.0));
    vec3 halfVector = normalize(halfDir + N);
    float NdotH = max(dot(N, halfVector), 0.0);
    float reflectionFactor = pow(1.0 - NdotH, 5.0 * roughness);
    vec3 reflectionColor = uLightColor * reflectionFactor * reflectivity;
    
    // 混合反射
    color = mix(color, reflectionColor, reflectivity);
""")
        
        # 结束
        shader += textwrap.dedent("""
    FinalColor = vec4(color, 1.0);
}
""")
        
        return shader
    
    def generate_uniform_declarations(self) -> str:
        """生成uniform声明代码"""
        features = self.enabled_features
        declarations = []
        
        if 'LIGHTING' in features:
            declarations.extend([
                "// 光照uniform",
                "uniform vec3 uLightDir;",
                "uniform vec3 uLightColor;",
                "uniform vec3 uAmbientColor;",
                "uniform sampler2D uAOTexture;",
                "uniform sampler2D uShadowTexture;",
            ])
        
        if 'ATMOSPHERE' in features:
            declarations.extend([
                "// 大气uniform",
                "uniform vec3 uFogColor;",
                "uniform float uFogDensity;",
                "uniform vec3 uCameraPos;",
            ])
        
        if 'HYDRO_VISUAL' in features:
            declarations.extend([
                "// 水文uniform",
                "uniform sampler2D uWetnessMap;",
                "uniform sampler2D uRoughnessMap;",
                "uniform sampler2D uReflectionMap;",
            ])
        
        return "\n".join(declarations)
    
    def get_uniform_locations(self) -> Dict[str, str]:
        """获取uniform位置映射"""
        locations = {
            'uModelMatrix': 'model',
            'uViewMatrix': 'view',
            'uProjectionMatrix': 'projection',
            'uNormalMatrix': 'normalMatrix',
            'uModelViewProjectionMatrix': 'modelViewProjection',
        }
        
        features = self.enabled_features
        
        if 'LIGHTING' in features:
            locations.update({
                'uLightDir': 'lightDir',
                'uLightColor': 'lightColor',
                'uAmbientColor': 'ambientColor',
                'uAOTexture': 'aoTexture',
                'uShadowTexture': 'shadowTexture',
            })
        
        if 'ATMOSPHERE' in features:
            locations.update({
                'uFogColor': 'fogColor',
                'uFogDensity': 'fogDensity',
                'uCameraPos': 'cameraPos',
            })
        
        if 'HYDRO_VISUAL' in features:
            locations.update({
                'uWetnessMap': 'wetnessMap',
                'uRoughnessMap': 'roughnessMap',
                'uReflectionMap': 'reflectionMap',
            })
        
        return locations
    
    def generate_shader_program(self) -> str:
        """生成完整的着色器程序（Python代码）"""
        features = self.enabled_features
        
        code = f'''"""
# Natural自动生成的着色器
# 启用的特性: {', '.join(self.enabled_features)}

from OpenGL.GL import shaders
from OpenGL.GL import GL

class NaturalShader:
    """Natural着色器包装类"""
    
    def __init__(self, config: dict):
        self.config = config
        self.generator = ShaderGenerator(config)
        
        # 编译着色器
        self.vertex_shader = self.generator.generate_vertex_shader()
        self.fragment_shader = self.generator.generate_fragment_shader()
        self.program = self._compile_program()
        
        # 获取uniform位置
        self.uniform_locations = self.generator.get_uniform_locations()
        
        # 缓存uniform位置
        self._uniform_cache = {{}}
    
    def _compile_program(self):
        """编译着色器程序"""
        vertex = shaders.compileShader(self.vertex_shader, GL_VERTEX_SHADER)
        fragment = shaders.compileShader(self.fragment_shader, GL_FRAGMENT_SHADER)
        
        program = shaders.compileProgram(vertex, fragment)
        
        if not program:
            raise RuntimeError("着色器编译失败")
        
        return program
    
    def use(self):
        """使用着色器"""
        glUseProgram(self.program)
    
    def set_uniform(self, name: str, value: Any):
        """设置uniform值"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的uniform: {{name}}")
        
        self._uniform_cache[name] = value
        
        if isinstance(value, (int, float)):
            glUniform1f(glGetUniformLocation(self.program, location), value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 3:
                glUniform3f(glGetUniformLocation(self.program, location), value[0], value[1], value[2])
            elif len(value) == 4:
                glUniform4f(glGetUniformLocation(self.program, location), value[0], value[1], value[2], value[3])
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                glUniform1f(glGetUniformLocation(self.program, location), value[0])
            elif value.ndim == 2:
                glUniform2f(glGetUniformLocation(self.program, location), value[0], value[1])
            elif value.ndim == 3:
                glUniform3f(glGetUniformLocation(self.program, location), value[0], value[1], value[2])
    
    def set_matrix4(self, name: str, matrix):
        """设置4x4矩阵"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的uniform: {{name}}")
        
        import glm
        glUniformMatrix4fv(glGetUniformLocation(self.program, location), 1, GL_FALSE, glm.value_ptr(matrix))
    
    def set_texture(self, name: str, texture_id: int):
        """设置纹理"""
        location = self.uniform_locations.get(name)
        if location is None:
            raise ValueError(f"未知的纹理uniform: {{name}}")
        
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(glGetUniformLocation(self.program, location), 0)
    
    def cleanup(self):
        """清理资源"""
        glDeleteProgram(self.program)
    
    def get_uniform_locations(self) -> Dict[str, str]:
        """获取uniform位置"""
        return self.uniform_locations
'''
        
        return code
