#version 460

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUv;

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUv;
layout (location = 3) out uint texId;

layout(set = 0, binding = 0) uniform GlobalUniform {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  float time;
  uint renderablesCount;
  float screenWidth;
  float screenHeight;
} globalUniform;

struct Object {
  mat4 transform;
  uint batch;
  uint draw;
  uint firstInstance;
  uint uncullable;
  uint unused_3;
  uint texture;
  uint mesh;
  uint redirect;
};

layout(std140, set = 1, binding = 0) readonly buffer ObjectBuffer {
	Object objects[];
} objectBuffer;

void main()
{
  Object object = objectBuffer.objects[objectBuffer.objects[gl_InstanceIndex].redirect];
  outPosition = (object.transform * vec4(vPosition, 1.0)).xyz;
  gl_Position = globalUniform.viewProj * object.transform * vec4(vPosition, 1.0f);
  outNormal = mat3(object.transform) * vNormal;
  outUv = vUv;
  texId = object.texture & 0xFFFF;
}
