#version 460
#include <common.glsl>

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUv;

layout (location = 0) out vec3 outNormal;
layout (location = 1) out vec2 outUv;
//layout (location = 2) out uint custom_set;

void main() {
    uint index = objectBuffer.objects[gl_InstanceIndex].redirect;
    Object object = objectBuffer.objects[index];
    gl_Position = globalUniform.viewProj * object.transform * vec4(vPosition, 1.0f);
    outNormal = mat3(object.transform) * vNormal;
    outUv = vUv;
    //custom_set = object.custom_set;
}
