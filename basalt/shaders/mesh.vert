layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUv;

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUv;
layout (location = 3) out uint custom_set;

void main() {
    uint index = objectBuffer.objects[gl_InstanceIndex].redirect;
    index = gl_InstanceIndex;
    Object object = objectBuffer.objects[index];
    outPosition = (object.transform * vec4(vPosition, 1.0)).xyz;
    gl_Position = globalUniform.viewProj * object.transform * vec4(vPosition, 1.0f);
    outNormal = mat3(object.transform) * vNormal;
    outUv = vUv;
    custom_set = object.custom_set;
}
