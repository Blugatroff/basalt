#version 460
#include <common.glsl>

layout (location = 0) in vec3 color;
layout (location = 0) out vec4 outFragColor;

void main() {
	if (mod(floor(gl_FragCoord.x + gl_FragCoord.y), 4) != 0) {
		discard;
	}
	outFragColor = vec4(color, 1.0);
}
