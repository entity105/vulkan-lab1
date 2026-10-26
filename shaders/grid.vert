#version 450
layout(location = 0) in vec3 in_position;

layout(set = 0, binding = 0) uniform GridUBO {
    mat4 viewProj;
} ubo;

void main() {
    gl_Position = ubo.viewProj * vec4(in_position, 1.0);
}