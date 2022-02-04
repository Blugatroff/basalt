struct Object {
  mat4 transform;
  uint batch;
  uint draw;
  uint firstInstance;
  uint uncullable;
  uint unused_3;
  uint custom_set;
  uint mesh;
  uint redirect;
};

layout(set = 0, binding = 0) uniform GlobalUniform {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  mat4 cameraTransform;
  
  vec4 frustumTopNormal;
  vec4 frustumBottomNormal;
  vec4 frustumRightNormal;
  vec4 frustumLeftNormal;
  vec4 frustumFarNormal;
  vec4 frustumNearNormal;

  float time;
  uint renderablesCount;
  float screenWidth;
  float screenHeight;
  float near;
  float far;
} globalUniform;

layout(std140, set = 0, binding = 1) /* readonly */ buffer ObjectBuffer {
	Object objects[];
} objectBuffer;

vec4 unpackColor(uint c) {
  uint r = (c & 0xFF000000) >> 24;
  uint g = (c & 0x00FF0000) >> 16;
  uint b = (c & 0x0000FF00) >> 8;
  uint a = (c & 0x000000FF);
  return vec4(r, g, b, a)  / 256.0;
}
