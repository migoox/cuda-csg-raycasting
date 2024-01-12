#version 330 core
layout (location = 0) in vec3 a_pos;
layout (location = 1) in vec2 a_uvs;

out vec2 uvs;
void main() {
    uvs = a_uvs;
    gl_Position = vec4(a_pos, 1.0);
}
