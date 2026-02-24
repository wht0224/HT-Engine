#version 330 core
out vec4 FragColor;

in vec3 TexCoords;

uniform vec3 topColor = vec3(0.2, 0.4, 0.8);
uniform vec3 bottomColor = vec3(0.9, 0.9, 1.0);

void main()
{
    vec3 dir = normalize(TexCoords);
    float t = 0.5 * (dir.y + 1.0);
    FragColor = vec4(mix(bottomColor, topColor, t), 1.0);
}
