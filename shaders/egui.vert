#version 460

layout (location = 0) in vec2 vPos;
layout (location = 1) in vec2 vUv;
layout (location = 2) in vec4 vColor;

layout (location = 0) out vec4 color;
layout (location = 1) out vec2 uv;
layout (location = 2) out uint texId;

layout(set = 0, binding = 0) uniform GlobalUniform {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  float time;
  uint renderablesCount;
  float screenWidth;
  float screenHeight;
} globalUniform;

void main() {
    gl_Position = vec4(
        2.0 * vPos.x / globalUniform.screenWidth - 1.0,
        2.0 * vPos.y / globalUniform.screenHeight - 1.0, 
        0.0, 
        1.0
    );
    uv = vUv;
    color = vColor;
}
