#version 330 core

in vec2 v_texcoord;
in vec3 v_normal;
in vec3 v_worldPos;

out vec4 FragColor;

uniform vec3 u_baseColor;
uniform vec3 u_ambientLight;
uniform vec3 u_directionalLightDir;
uniform vec3 u_directionalLightColor;
uniform float u_directionalLightIntensity;
uniform vec3 u_cameraPos;

void main() {
    vec3 normal = normalize(v_normal);
    vec3 lightDir = normalize(-u_directionalLightDir);
    
    float diff = max(dot(normal, lightDir), 0.0);
    
    vec3 viewDir = normalize(u_cameraPos - v_worldPos);
    vec3 reflectDir = reflect(-lightDir, normal);
    float spec = pow(max(dot(viewDir, reflectDir), 0.0), 32.0);
    
    vec3 ambient = u_ambientLight * u_baseColor;
    vec3 diffuse = u_directionalLightColor * u_directionalLightIntensity * diff * u_baseColor;
    vec3 specular = u_directionalLightColor * 0.3 * spec;
    
    vec3 result = ambient + diffuse + specular;
    
    result = result * 1.2;
    
    FragColor = vec4(result, 1.0);
}
