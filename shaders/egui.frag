#version 450

layout (location = 0) in vec4 color;
layout (location = 1) in vec2 uv;
layout (location = 2) in flat uint texId;
layout (location = 0) out vec4 outFragColor;

layout(set = 2, binding = 0) uniform sampler2D textures[];

void main() {
	outFragColor = color;
}
