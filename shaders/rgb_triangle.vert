#version 460

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUv;

layout(location = 0) out vec3 color;

layout(set = 0, binding = 0) uniform GlobalUniform {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  float time;
} globalUniform;
 
struct Object {
	mat4 model;
  uint texture;
  uint mesh;
  uint batch;
  uint redirect;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	Object objects[];
} objectBuffer;

void main() {
  color = vNormal * 0.5 + 0.5;
  Object object = objectBuffer.objects[objectBuffer.objects[gl_InstanceIndex].redirect];
  vec4 worldPos = object.model * vec4(vPosition, 1.0);
  gl_Position = globalUniform.viewProj * worldPos;
}
