#version 330 core
uniform sampler2D u_texture;
in vec2 uvs;

out vec4 frag_color;
void main() {
     frag_color = texture(u_texture, uvs);
}
