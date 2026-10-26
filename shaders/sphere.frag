#version 450

layout(location = 0) in vec3 frag_pos;
layout(location = 1) in vec3 frag_normal;
layout(location = 2) in vec3 frag_color;

layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform SceneUBO {
    mat4 view;
    mat4 proj;
    vec4 camera_pos;
    vec4 light_dir;
    vec4 light_color;
    vec4 ambient;
    vec4 counts;
} scene;

layout(set = 1, binding = 0) uniform ModelUBO {
    mat4 model;
    mat4 normal_matrix;
    vec4 base_color;
    vec4 specular_color;
    vec4 material_params;
} model_data;

struct PointLight {
    vec4 position; // xyz = pos, w = radius
    vec4 color;    // rgb = color, a = intensity
};

struct SpotLight {
    vec4 position;   // xyz = pos, w = radius
    vec4 direction;  // xyz = dir, w = inner_cos
    vec4 color;      // rgb = color, a = intensity
    vec4 angles;     // x = outer_cos
};

layout(std430, set = 0, binding = 1) readonly buffer PointLights {
    PointLight p_lights[];
} point_lights;

layout(std430, set = 0, binding = 2) readonly buffer SpotLights {
    SpotLight s_lights[];
} spot_lights;

void main() {
    vec3 N = normalize(frag_normal);
    vec3 V = normalize(scene.camera_pos.xyz - frag_pos);
    
    // 1. Directional Light
    vec3 L_dir = normalize(-scene.light_dir.xyz);
    vec3 H_dir = normalize(L_dir + V);
    float diff_dir = max(dot(N, L_dir), 0.0);
    float spec_dir = pow(max(dot(N, H_dir), 0.0), model_data.material_params.x);
    
    vec3 result = scene.ambient.rgb * frag_color;
    result += diff_dir * frag_color * scene.light_color.rgb;
    result += spec_dir * model_data.specular_color.rgb * scene.light_color.rgb;

    // 2. Point Lights
    int num_points = int(scene.counts.x);
    for (int i = 0; i < num_points; ++i) {
        vec3 light_pos = point_lights.p_lights[i].position.xyz;
        float radius = point_lights.p_lights[i].position.w;
        vec3 light_col = point_lights.p_lights[i].color.rgb;
        float intensity = point_lights.p_lights[i].color.a;

        vec3 L = light_pos - frag_pos;
        float distance = length(L);
        L = normalize(L);
        
        float attenuation = 1.0 / (1.0 + (distance * distance) / (radius * radius));
        
        vec3 H = normalize(L + V);
        float diff = max(dot(N, L), 0.0);
        float spec = pow(max(dot(N, H), 0.0), model_data.material_params.x);
        
        result += diff * frag_color * light_col * intensity * attenuation;
        result += spec * model_data.specular_color.rgb * light_col * intensity * attenuation;
    }

    // 3. Spot Lights
    int num_spots = int(scene.counts.y);
    for (int i = 0; i < num_spots; ++i) {
        vec3 light_pos = spot_lights.s_lights[i].position.xyz;
        float radius = spot_lights.s_lights[i].position.w;
        vec3 spot_dir = normalize(spot_lights.s_lights[i].direction.xyz);
        float inner_cos = spot_lights.s_lights[i].direction.w;
        float outer_cos = spot_lights.s_lights[i].angles.x;
        
        vec3 light_col = spot_lights.s_lights[i].color.rgb;
        float intensity = spot_lights.s_lights[i].color.a;

        vec3 L = light_pos - frag_pos;
        float distance = length(L);
        L = normalize(L);

        // Проверяем, попадает ли фрагмент в конус прожектора
        float theta = dot(spot_dir, -L); // -L: вектор от света к фрагменту
        
        // smoothstep делает мягкие края (от outer к inner)
        float spot_effect = smoothstep(outer_cos, inner_cos, theta);

        if (spot_effect > 0.0) {
            float attenuation = 1.0 / (1.0 + (distance * distance) / (radius * radius));
            
            vec3 H = normalize(L + V);
            float diff = max(dot(N, L), 0.0);
            float spec = pow(max(dot(N, H), 0.0), model_data.material_params.x);
            
            result += diff * frag_color * light_col * intensity * attenuation * spot_effect;
            result += spec * model_data.specular_color.rgb * light_col * intensity * attenuation * spot_effect;
        }
    }

    out_color = vec4(result, 1.0);
}