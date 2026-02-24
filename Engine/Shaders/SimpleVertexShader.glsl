#version 330 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_texcoord;

out vec2 v_texcoord;
out vec3 v_normal;
out vec3 v_worldPos;

uniform mat4 u_model;
uniform mat4 u_viewProj;
uniform mat3 u_normalMatrix;

void main() {
    vec4 worldPos = u_model * vec4(a_position, 1.0);
    v_worldPos = worldPos.xyz;
    gl_Position = u_viewProj * worldPos;
    v_normal = normalize(u_normalMatrix * a_normal);
    v_texcoord = a_texcoord;
}
