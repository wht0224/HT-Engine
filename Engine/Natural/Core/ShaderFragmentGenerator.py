"""
着色器片段生成器

让Natural规则生成GLSL着色器片段，实现真正的"Data-Driven"渲染。
"""

from typing import Dict, List, Optional, Any
import textwrap


class ShaderFragmentGenerator:
    """着色器片段生成器"""
    
    def __init__(self, rule_name: str, rule_config: Dict[str, Any]):
        self.rule_name = rule_name
        self.rule_config = rule_config
    
    def generate_vertex_input(self) -> str:
        """生成顶点输入声明"""
        return textwrap.dedent("""
    layout (location = 0) in vec3 a_position;
    layout (location = 1) in vec3 a_normal;
    layout (location = 2) in vec2 a_texcoord;
    
    uniform mat4 u_model;
    uniform mat4 u_viewProj;
    uniform mat3 u_normalMatrix;
    
    out vec3 FragPos;
    out vec3 FragNormal;
    out vec2 FragTexCoord;
""")
    
    def generate_vertex_main(self, features: List[str]) -> str:
        """生成顶点主函数"""
        code = """
void main() {
    vec4 worldPos = u_model * vec4(a_position, 1.0);
    FragPos = worldPos.xyz;
    FragNormal = normalize(u_normalMatrix * a_normal);
    FragTexCoord = a_texcoord;
    gl_Position = u_viewProj * worldPos;
}
"""
        return code
    
    def generate_fragment_uniforms(self, features: List[str]) -> str:
        """生成片段着色器uniform声明"""
        uniforms = []
        
        if 'LIGHTING' in features:
            uniforms.extend([
                "uniform vec3 uLightDir;",
                "uniform vec3 uLightColor;",
                "uniform vec3 uAmbientColor;",
            ])
        
        if 'ATMOSPHERE' in features:
            uniforms.extend([
                "uniform vec3 uFogColor;",
                "uniform float uFogDensity;",
                "uniform vec3 uCameraPos;",
            ])
        
        if 'HYDRO' in features:
            uniforms.extend([
                "uniform sampler2D uWetnessMap;",
                "uniform sampler2D uRoughnessMap;",
            ])
        
        return "\n".join(uniforms)
    
    def generate_fragment_functions(self, features: List[str]) -> str:
        """生成片段着色器辅助函数"""
        functions = []
        
        if 'LIGHTING' in features:
            functions.append("""
// 光照计算函数
vec3 calculateLighting(vec3 normal, vec3 color) {
    vec3 N = normalize(normal);
    // 强制双面光照：如果是背面，翻转法线
    if (!gl_FrontFacing) {
        N = -N;
    }
    float NdotL = max(dot(N, -uLightDir), 0.0);
    vec3 diffuse = color * (uAmbientColor + uLightColor * NdotL);
    return diffuse;
}
""")
        
        if 'ATMOSPHERE' in features:
            functions.append("""
// 雾效计算函数
vec3 applyFog(vec3 color, vec3 worldPos) {
    float distance = length(worldPos - uCameraPos);
    float fogFactor = exp(-uFogDensity * distance);
    fogFactor = clamp(fogFactor, 0.0, 1.0);
    vec3 fogColor = mix(uFogColor, color, fogFactor);
    return fogColor;
}
""")
        
        if 'HYDRO' in features:
            functions.append("""
// 水文效果函数
vec3 applyHydroEffects(vec3 color, vec2 uv) {
    float wetness = texture(uWetnessMap, uv).r;
    float roughness = texture(uRoughnessMap, uv).r;
    
    // 湿润表面更光滑
    float smoothness = 1.0 - wetness * 0.8;
    float finalRoughness = mix(roughness, 0.1, smoothness);
    
    // 简单的反射增强
    float reflectivity = wetness * 0.3;
    
    return color;
}
""")
        
        return "\n".join(functions)
    
    def generate_fragment_main(self, features: List[str]) -> str:
        """生成片段着色器主函数"""
        code = """
void main() {
    vec3 color = u_diffuseColor;
"""
        
        # 应用光照
        if 'LIGHTING' in features:
            code += """
    // 应用光照
    color = calculateLighting(FragNormal, color);
"""
        
        # 应用水文效果
        if 'HYDRO' in features:
            code += """
    // 应用水文效果
    color = applyHydroEffects(color, FragTexCoord);
"""
        
        # 应用雾效
        if 'ATMOSPHERE' in features:
            code += """
    // 应用雾效
    color = applyFog(color, FragPos);
"""
        
        # 结束
        code += """
    FinalColor = vec4(color, 1.0);
}
"""
        return code
    
    def generate_full_shader(self, features: List[str]) -> str:
        """生成完整的着色器代码"""
        shader = f"""// {self.rule_name} 自动生成的着色器
// 启用的特性: {', '.join(features)}

"""
        
        # 顶点着色器
        shader += self.generate_vertex_input()
        shader += self.generate_vertex_main(features)
        
        # 片段着色器
        shader += "uniform vec3 u_diffuseColor;\n"
        shader += self.generate_fragment_uniforms(features)
        shader += self.generate_fragment_functions(features)
        shader += self.generate_fragment_main(features)
        
        return shader


class DynamicShaderSystem:
    """动态着色器系统
    
    根据Natural规则配置动态生成着色器
    """
    
    def __init__(self):
        self.registered_fragments = {}  # 规则名 -> 着色器片段
        self.enabled_features = set()
        self.current_shader = None
        self.current_program = None
    
    def register_fragment(self, rule_name: str, fragment_generator: ShaderFragmentGenerator):
        """注册规则生成的着色器片段"""
        self.registered_fragments[rule_name] = fragment_generator
    
    def set_enabled_features(self, features: List[str]):
        """设置启用的特性"""
        self.enabled_features = set(features)
    
    def compile_shader(self) -> bool:
        """编译动态着色器"""
        # 收集所有注册的片段
        fragments = []
        for rule_name, generator in self.registered_fragments.items():
            # 检查规则是否启用
            if any(feature in self.enabled_features for feature in self._get_rule_features(rule_name)):
                fragments.append(generator)
        
        # 生成着色器代码
        shader_code = self._assemble_shader(fragments)
        
        from OpenGL.GL import shaders, GL_VERTEX_SHADER, GL_FRAGMENT_SHADER, glDeleteProgram
        vertex = shaders.compileShader(shader_code['vertex'], GL_VERTEX_SHADER)
        fragment = shaders.compileShader(shader_code['fragment'], GL_FRAGMENT_SHADER)
        
        program = shaders.compileProgram(vertex, fragment)
        
        if not program:
            print(f"着色器编译失败")
            return False
        
        # 清理旧的着色器
        if self.current_program:
            glDeleteProgram(self.current_program)
        
        self.current_program = program
        self.current_shader = shader_code
        return True
    
    def _get_rule_features(self, rule_name: str) -> List[str]:
        """获取规则对应的特性列表"""
        feature_map = {
            'LightingRule': ['LIGHTING'],
            'AtmosphereRule': ['ATMOSPHERE'],
            'HydroVisualRule': ['HYDRO'],
            'WindRule': [],
            'VegetationGrowthRule': [],
            'ThermalWeatheringRule': [],
            'HydraulicErosionRule': [],
        }
        return feature_map.get(rule_name, [])
    
    def _assemble_shader(self, fragment_generators: List[ShaderFragmentGenerator]) -> Dict[str, str]:
        """组装着色器代码"""
        # 收集所有特性
        all_features = set()
        for generator in fragment_generators:
            rule_name = generator.rule_name
            features = self._get_rule_features(rule_name)
            all_features.update(features)
        
        # 生成顶点着色器
        vertex_shader = f"""#version 330 core
layout (location = 0) in vec3 a_position;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in vec2 a_texcoord;

uniform mat4 u_model;
uniform mat4 u_viewProj;
uniform mat3 u_normalMatrix;

out vec3 FragPos;
out vec3 FragNormal;
out vec2 FragTexCoord;

void main() {{
    vec4 worldPos = u_model * vec4(a_position, 1.0);
    FragPos = worldPos.xyz;
    FragNormal = normalize(u_normalMatrix * a_normal);
    FragTexCoord = a_texcoord;
    gl_Position = u_viewProj * worldPos;
}}
"""
        
        # 生成片段着色器
        fragment_shader = f"""#version 330 core
in vec3 FragPos;
in vec3 FragNormal;
in vec2 FragTexCoord;

out vec4 FinalColor;

uniform vec3 u_diffuseColor;
uniform sampler2D u_baseColorTexture;
uniform bool u_hasTexture;

"""
        
        # 添加uniform声明
        all_uniforms = set()
        for generator in fragment_generators:
            features = self._get_rule_features(generator.rule_name)
            if any(feature in self.enabled_features for feature in features):
                fragment_shader += generator.generate_fragment_uniforms(features)
                all_uniforms.update(self._extract_uniforms(generator.generate_fragment_uniforms(features)))
        
        # 添加辅助函数
        all_functions = set()
        for generator in fragment_generators:
            features = self._get_rule_features(generator.rule_name)
            if any(feature in self.enabled_features for feature in features):
                fragment_shader += generator.generate_fragment_functions(features)
                all_functions.update(self._extract_functions(generator.generate_fragment_functions(features)))
        
        # 主函数
        fragment_shader += """
void main() {
    vec3 color = u_diffuseColor;
    
    // 采样纹理
    if (u_hasTexture) {
        vec3 texColor = texture(u_baseColorTexture, FragTexCoord).rgb;
        color = texColor * color;
    }
"""
        
        # 调用辅助函数
        for generator in fragment_generators:
            features = self._get_rule_features(generator.rule_name)
            if any(feature in self.enabled_features for feature in features):
                if 'LIGHTING' in features:
                    fragment_shader += "    color = calculateLighting(FragNormal, color);\n"
                if 'HYDRO' in features:
                    fragment_shader += "    color = applyHydroEffects(color, FragTexCoord);\n"
                if 'ATMOSPHERE' in features:
                    fragment_shader += "    color = applyFog(color, FragPos);\n"
        
        fragment_shader += """
    FinalColor = vec4(color, 1.0);
}
"""
        
        return {'vertex': vertex_shader, 'fragment': fragment_shader}
    
    def _extract_uniforms(self, text: str) -> List[str]:
        """从文本中提取uniform声明"""
        import re
        pattern = r'uniform\s+\w+\w+\w+\w+\w+;'
        matches = re.findall(pattern, text)
        return [m.strip().rstrip(';') for m in matches]
    
    def _extract_functions(self, text: str) -> List[str]:
        """从文本中提取函数声明"""
        import re
        pattern = r'\w+\w+\s*\(\s*\w+\s*\)'
        matches = re.findall(pattern, text)
        return [m.strip() for m in matches]
    
    def use(self):
        """使用当前着色器"""
        if self.current_program:
            glUseProgram(self.current_program)
    
    def set_uniform(self, name: str, value: Any):
        """设置uniform值"""
        if not self.current_program:
            return
        
        location = glGetUniformLocation(self.current_program, name)
        if location == -1:
            print(f"警告: uniform {name} 不存在")
            return
        
        if isinstance(value, (int, float)):
            glUniform1f(location, value)
        elif isinstance(value, (list, tuple)):
            if len(value) == 3:
                glUniform3f(location, value[0], value[1], value[2])
            elif len(value) == 4:
                glUniform4f(location, value[0], value[1], value[2], value[3])
        elif isinstance(value, np.ndarray):
            if value.ndim == 1:
                glUniform1f(location, value[0])
            elif value.ndim == 2:
                glUniform2f(location, value[0], value[1])
            elif value.ndim == 3:
                glUniform3f(location, value[0], value[1], value[2])
    
    def cleanup(self):
        """清理资源"""
        if self.current_program:
            glDeleteProgram(self.current_program)
            self.current_program = None
            self.current_shader = None


# 预定义的规则着色器片段
LIGHTING_FRAGMENT = ShaderFragmentGenerator("LightingRule", {
    'ao_strength': 1.0,
    'shadow_strength': 1.0,
})

ATMOSPHERE_FRAGMENT = ShaderFragmentGenerator("AtmosphereRule", {
    'fog_density': 0.015,
})

HYDRO_FRAGMENT = ShaderFragmentGenerator("HydroVisualRule", {
    'wetness_intensity': 0.5,
    'roughness_base': 0.5,
})
