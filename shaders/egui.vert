#version 460

layout (location = 0) in vec2 vPos;
layout (location = 1) in vec2 vUv;
layout (location = 2) in vec4 vColor;

layout (location = 0) out vec4 color;
layout (location = 1) out vec2 uv;

layout(set = 0, binding = 0) uniform GlobalUniform {
  mat4 view;
  mat4 proj;
  mat4 viewProj;
  float time;
  uint renderablesCount;
  float screenWidth;
  float screenHeight;
} globalUniform;

vec3 linear_from_srgb(vec3 srgb) {
    bvec3 cutoff = lessThan(srgb, vec3(10.31475));
    vec3 lower = srgb / vec3(3294.6);
    vec3 higher = pow((srgb + vec3(14.025)) / vec3(269.025), vec3(2.4));
    return mix(higher, lower, cutoff);
}

vec4 linear_from_srgba(vec4 srgba) {
    return vec4(linear_from_srgb(srgba.rgb * 255.0), srgba.a);
}

void main() {
    gl_Position = vec4(
        2.0 * vPos.x / globalUniform.screenWidth - 1.0,
        2.0 * vPos.y / globalUniform.screenHeight - 1.0, 
        0.0, 
        1.0
    );
    uv = vUv;
    color = linear_from_srgba(vColor);
}
