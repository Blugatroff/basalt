#version 460
#include <common.glsl>

layout (location = 0) in vec3 vPosition;
layout (location = 1) in uint vColor;

layout(location = 0) out vec3 color;

void main() {
  color = unpackColor(vColor).xyz;
  uint index = objectBuffer.objects[gl_InstanceIndex].redirect;
  Object object = objectBuffer.objects[index];

  vec4 worldPos = object.transform * vec4(vPosition, 1.0);
  gl_Position = globalUniform.viewProj * worldPos;
}
