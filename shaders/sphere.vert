#version 450

layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;

layout(location = 0) out vec3 frag_pos;
layout(location = 1) out vec3 frag_normal;
layout(location = 2) out vec3 frag_color;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 counts; // x = кол-во точечных, y = кол-во прожекторов
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normal_matrix;
    vec4 base_color;
    vec4 specular_color;
    vec4 material_params;
} model_data;

void main() {
    vec4 world_pos = model_data.model * vec4(in_position, 1.0);
    gl_Position = scene.proj * scene.view * world_pos;
    
    frag_pos = world_pos.xyz;
    frag_normal = mat3(model_data.normal_matrix) * in_normal;
    frag_color = in_color * model_data.base_color.rgb;
}