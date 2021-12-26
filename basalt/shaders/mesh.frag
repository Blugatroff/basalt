layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUv;
layout (location = 3) in flat uint texId;

layout (location = 0) out vec4 outFragColor;
layout(set = 1, binding = 0) uniform sampler2D tex;

void main() {
  vec3 lightDir = vec3(0, 1, 0);
  float diffuse = dot(vNormal, lightDir) * 0.5 + 0.5;
  diffuse = diffuse * 0.5 + 0.5;
  vec4 color = texture(tex, vUv);
  //color = vec4(1.0);
  if (color.w == 0.0) {
    discard;
  }
  outFragColor = vec4(color.xyz * diffuse, 1.0);
}
