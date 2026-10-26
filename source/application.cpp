#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

    namespace {

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec3 color;
        };

        struct GridVertex {
            glm::vec3 pos;
        };

        struct SceneUBO {
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec4 camera_pos;
            glm::vec4 light_dir;
            glm::vec4 light_color;
            glm::vec4 ambient;
            glm::vec4 counts;
        };

        struct ModelUBO {
            glm::mat4 model;
            glm::mat4 normal_matrix;
            glm::vec4 base_color;
            glm::vec4 specular_color;
            glm::vec4 material_params;
        };

        struct GridUBO {
            glm::mat4 viewProj;
        };

        struct PointLight {
            glm::vec4 position;
            glm::vec4 color;
        };

        struct SpotLight {
            glm::vec4 position;
            glm::vec4 direction;
            glm::vec4 color;
            glm::vec4 angles;
        };

        const int MAX_POINT_LIGHTS = 10;
        const int MAX_SPOT_LIGHTS = 10;

        // —ÙÂ‡
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;
        VkBuffer index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_buffer_alloc = VK_NULL_HANDLE;
        uint32_t index_count = 0;

        // —ÂÚÍ‡
        VkBuffer grid_vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation grid_vertex_buffer_alloc = VK_NULL_HANDLE;
        uint32_t grid_vertex_count = 0;

        // UBO
        VkBuffer scene_buffer = VK_NULL_HANDLE;
        VmaAllocation scene_buffer_alloc = VK_NULL_HANDLE;
        void* scene_buffer_mapped = nullptr;

        VkBuffer model_buffer = VK_NULL_HANDLE;
        VmaAllocation model_buffer_alloc = VK_NULL_HANDLE;
        void* model_buffer_mapped = nullptr;

        VkBuffer grid_ubo_buffer = VK_NULL_HANDLE;
        VmaAllocation grid_ubo_buffer_alloc = VK_NULL_HANDLE;
        void* grid_ubo_buffer_mapped = nullptr;

        // SSBO
        VkBuffer point_lights_buffer = VK_NULL_HANDLE;
        VmaAllocation point_lights_buffer_alloc = VK_NULL_HANDLE;
        void* point_lights_buffer_mapped = nullptr;

        VkBuffer spot_lights_buffer = VK_NULL_HANDLE;
        VmaAllocation spot_lights_buffer_alloc = VK_NULL_HANDLE;
        void* spot_lights_buffer_mapped = nullptr;

        // ƒÂÒÍËÔÚÓ˚
        VkDescriptorSetLayout scene_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout model_set_layout = VK_NULL_HANDLE;
        VkDescriptorSetLayout grid_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet scene_set = VK_NULL_HANDLE;
        VkDescriptorSet model_set = VK_NULL_HANDLE;
        VkDescriptorSet grid_set = VK_NULL_HANDLE;

        // Pipeline
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;
        VkPipelineLayout grid_pipeline_layout = VK_NULL_HANDLE;
        VkPipeline grid_pipeline = VK_NULL_HANDLE;

        // UI
        float cam_yaw = 45.0f;
        float cam_pitch = 20.0f;
        float cam_dist = 4.0f;
        float cam_fov = 60.0f;

        glm::vec3 light_dir_ui = glm::vec3(-0.5f, -1.0f, -0.5f);
        glm::vec3 light_color_ui = glm::vec3(1.0f, 1.0f, 1.0f);
        glm::vec3 ambient_ui = glm::vec3(0.1f, 0.1f, 0.1f);

        glm::vec3 base_color_ui = glm::vec3(0.8f, 0.8f, 0.8f);
        glm::vec3 specular_color_ui = glm::vec3(1.0f, 1.0f, 1.0f);
        float shininess_ui = 32.0f;

        glm::vec3 obj_position = glm::vec3(0.0f);
        glm::vec3 obj_rotation = glm::vec3(0.0f);
        glm::vec3 obj_scale = glm::vec3(1.0f);

        std::vector<PointLight> point_lights;
        std::vector<SpotLight> spot_lights;

        std::vector<Vertex> generateSphereVertices(int segments, int rings, float radius) {
            std::vector<Vertex> vertices;
            for (int y = 0; y <= rings; ++y) {
                float v = float(y) / float(rings);
                float phi = v * glm::pi<float>();
                for (int x = 0; x <= segments; ++x) {
                    float u = float(x) / float(segments);
                    float theta = u * 2.0f * glm::pi<float>();
                    float px = radius * std::sin(phi) * std::cos(theta);
                    float py = radius * std::cos(phi);
                    float pz = radius * std::sin(phi) * std::sin(theta);
                    vertices.push_back({ glm::vec3(px, py, pz), glm::normalize(glm::vec3(px, py, pz)), glm::vec3(1.0f) });
                }
            }
            return vertices;
        }

        std::vector<uint32_t> generateSphereIndices(int segments, int rings) {
            std::vector<uint32_t> indices;
            for (int y = 0; y < rings; ++y) {
                for (int x = 0; x < segments; ++x) {
                    uint32_t i0 = y * (segments + 1) + x;
                    uint32_t i1 = i0 + 1;
                    uint32_t i2 = i0 + (segments + 1);
                    uint32_t i3 = i2 + 1;
                    indices.push_back(i0); indices.push_back(i2); indices.push_back(i1);
                    indices.push_back(i1); indices.push_back(i2); indices.push_back(i3);
                }
            }
            return indices;
        }

        std::vector<GridVertex> generateGrid(int size, float step) {
            std::vector<GridVertex> vertices;
            float half_size = size * step * 0.5f;
            for (int i = -size / 2; i <= size / 2; ++i) {
                float pos = i * step;
                vertices.push_back({ glm::vec3(pos, 0.0f, -half_size) });
                vertices.push_back({ glm::vec3(pos, 0.0f, half_size) });
                vertices.push_back({ glm::vec3(-half_size, 0.0f, pos) });
                vertices.push_back({ glm::vec3(half_size, 0.0f, pos) });
            }
            return vertices;
        }

        std::vector<char> readFile(const std::string& path) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) return {};
            size_t size = size_t(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo info{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            info.codeSize = code.size();
            info.pCode = reinterpret_cast<const uint32_t*>(code.data());
            VkShaderModule module;
            vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module);
            return module;
        }

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkBuffer& out_buffer, VmaAllocation& out_alloc, VmaMemoryUsage mem_usage, const void* data = nullptr, void** out_mapped = nullptr) {
            VkBufferCreateInfo buffer_info{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo alloc_info{};
            alloc_info.usage = mem_usage;
            alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

            vmaCreateBuffer(graphics::internal::context.allocator, &buffer_info, &alloc_info, &out_buffer, &out_alloc, nullptr);

            void* mapped = nullptr;
            vmaMapMemory(graphics::internal::context.allocator, out_alloc, &mapped);
            if (data != nullptr) std::memcpy(mapped, data, size_t(size));
            if (out_mapped != nullptr) *out_mapped = mapped;
            else vmaUnmapMemory(graphics::internal::context.allocator, out_alloc);

            return true;
        }

        bool createPipelines() {
            auto& ctx = graphics::internal::context;

            // --- 1. Œ—ÕŒ¬ÕŒ… œ¿…œÀ¿…Õ (—‘≈–¿) ---
            VkDescriptorSetLayoutBinding scene_bindings[3];
            scene_bindings[0] = { 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            scene_bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            scene_bindings[2] = { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };

            VkDescriptorSetLayoutCreateInfo scene_layout_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            scene_layout_info.bindingCount = 3;
            scene_layout_info.pBindings = scene_bindings;
            vkCreateDescriptorSetLayout(ctx.device, &scene_layout_info, nullptr, &scene_set_layout);

            VkDescriptorSetLayoutBinding model_binding{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo model_layout_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            model_layout_info.bindingCount = 1;
            model_layout_info.pBindings = &model_binding;
            vkCreateDescriptorSetLayout(ctx.device, &model_layout_info, nullptr, &model_set_layout);

            VkDescriptorSetLayout set_layouts[] = { scene_set_layout, model_set_layout };
            VkPipelineLayoutCreateInfo pl_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            pl_info.setLayoutCount = 2;
            pl_info.pSetLayouts = set_layouts;
            vkCreatePipelineLayout(ctx.device, &pl_info, nullptr, &pipeline_layout);

            auto vert_code = readFile("shaders/sphere.vert.spv");
            auto frag_code = readFile("shaders/sphere.frag.spv");
            VkShaderModule vert_module = createShaderModule(vert_code);
            VkShaderModule frag_module = createShaderModule(frag_code);

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vert_module, "main" };
            stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, frag_module, "main" };

            VkVertexInputBindingDescription binding_desc{ 0, sizeof(Vertex), VK_VERTEX_INPUT_RATE_VERTEX };
            VkVertexInputAttributeDescription attr_desc[3] = {
                { 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, pos) },
                { 1, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, normal) },
                { 2, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(Vertex, color) }
            };

            VkPipelineVertexInputStateCreateInfo vertex_input{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding_desc;
            vertex_input.vertexAttributeDescriptionCount = 3;
            vertex_input.pVertexAttributeDescriptions = attr_desc;

            VkPipelineInputAssemblyStateCreateInfo input_assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{ VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO };
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR };
            VkPipelineDynamicStateCreateInfo dynamic_state{ VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO };
            dynamic_state.dynamicStateCount = 2;
            dynamic_state.pDynamicStates = dynamic_states;

            VkPipelineRasterizationStateCreateInfo raster{ VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO };
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_BACK_BIT;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{ VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO };
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth_stencil{ VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO };
            depth_stencil.depthTestEnable = VK_TRUE;
            depth_stencil.depthWriteEnable = VK_TRUE;
            depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState color_attach{};
            color_attach.colorWriteMask = 0xF;

            VkPipelineColorBlendStateCreateInfo color_blend{ VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO };
            color_blend.attachmentCount = 1;
            color_blend.pAttachments = &color_attach;

            VkGraphicsPipelineCreateInfo pipeline_info{ VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO };
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = stages;
            pipeline_info.pVertexInputState = &vertex_input;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &raster;
            pipeline_info.pMultisampleState = &multisample;
            pipeline_info.pDepthStencilState = &depth_stencil;
            pipeline_info.pColorBlendState = &color_blend;
            pipeline_info.pDynamicState = &dynamic_state;
            pipeline_info.layout = pipeline_layout;
            pipeline_info.renderPass = ctx.render_pass;
            pipeline_info.subpass = 0;

            vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline);
            vkDestroyShaderModule(ctx.device, vert_module, nullptr);
            vkDestroyShaderModule(ctx.device, frag_module, nullptr);

            // --- 2. œ¿…œÀ¿…Õ —≈“ » ---
            VkDescriptorSetLayoutBinding grid_binding{ 0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_VERTEX_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo grid_layout_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            grid_layout_info.bindingCount = 1;
            grid_layout_info.pBindings = &grid_binding;
            vkCreateDescriptorSetLayout(ctx.device, &grid_layout_info, nullptr, &grid_set_layout);

            VkPipelineLayoutCreateInfo grid_pl_info{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            grid_pl_info.setLayoutCount = 1;
            grid_pl_info.pSetLayouts = &grid_set_layout;
            vkCreatePipelineLayout(ctx.device, &grid_pl_info, nullptr, &grid_pipeline_layout);

            auto grid_vert_code = readFile("shaders/grid.vert.spv");
            auto grid_frag_code = readFile("shaders/grid.frag.spv");
            VkShaderModule grid_vert = createShaderModule(grid_vert_code);
            VkShaderModule grid_frag = createShaderModule(grid_frag_code);

            VkPipelineShaderStageCreateInfo grid_stages[2]{};
            grid_stages[0] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, grid_vert, "main" };
            grid_stages[1] = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, grid_frag, "main" };

            VkVertexInputBindingDescription grid_binding_desc{ 0, sizeof(GridVertex), VK_VERTEX_INPUT_RATE_VERTEX };
            VkVertexInputAttributeDescription grid_attr_desc{ 0, 0, VK_FORMAT_R32G32B32_SFLOAT, offsetof(GridVertex, pos) };

            VkPipelineVertexInputStateCreateInfo grid_vertex_input{ VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO };
            grid_vertex_input.vertexBindingDescriptionCount = 1;
            grid_vertex_input.pVertexBindingDescriptions = &grid_binding_desc;
            grid_vertex_input.vertexAttributeDescriptionCount = 1;
            grid_vertex_input.pVertexAttributeDescriptions = &grid_attr_desc;

            VkPipelineInputAssemblyStateCreateInfo grid_input_assembly{ VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO };
            grid_input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_LINE_LIST; // À»Õ»»

            VkPipelineRasterizationStateCreateInfo grid_raster = raster;
            grid_raster.cullMode = VK_CULL_MODE_NONE; // ÀËÌËË ÌÂ ËÏÂ˛Ú ÎËˆ‡

            VkGraphicsPipelineCreateInfo grid_pipeline_info = pipeline_info;
            grid_pipeline_info.pStages = grid_stages;
            grid_pipeline_info.pVertexInputState = &grid_vertex_input;
            grid_pipeline_info.pInputAssemblyState = &grid_input_assembly;
            grid_pipeline_info.pRasterizationState = &grid_raster;
            grid_pipeline_info.layout = grid_pipeline_layout;

            vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &grid_pipeline_info, nullptr, &grid_pipeline);
            vkDestroyShaderModule(ctx.device, grid_vert, nullptr);
            vkDestroyShaderModule(ctx.device, grid_frag, nullptr);

            return true;
        }

        bool createDescriptorSets() {
            auto& ctx = graphics::internal::context;

            VkDescriptorPoolSize pool_sizes[2];
            pool_sizes[0] = { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 }; // Scene, Model, Grid
            pool_sizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };

            VkDescriptorPoolCreateInfo pool_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            pool_info.poolSizeCount = 2;
            pool_info.pPoolSizes = pool_sizes;
            pool_info.maxSets = 3;
            vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool);

            VkDescriptorSetAllocateInfo alloc_info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            alloc_info.descriptorPool = descriptor_pool;
            alloc_info.descriptorSetCount = 1;

            // Scene Set
            alloc_info.pSetLayouts = &scene_set_layout;
            vkAllocateDescriptorSets(ctx.device, &alloc_info, &scene_set);

            VkDescriptorBufferInfo scene_buffer_info{ scene_buffer, 0, sizeof(SceneUBO) };
            VkDescriptorBufferInfo point_buffer_info{ point_lights_buffer, 0, VK_WHOLE_SIZE };
            VkDescriptorBufferInfo spot_buffer_info{ spot_lights_buffer, 0, VK_WHOLE_SIZE };

            VkWriteDescriptorSet scene_writes[3];
            scene_writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &scene_buffer_info, nullptr };
            scene_writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &point_buffer_info, nullptr };
            scene_writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, scene_set, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spot_buffer_info, nullptr };
            vkUpdateDescriptorSets(ctx.device, 3, scene_writes, 0, nullptr);

            // Model Set
            alloc_info.pSetLayouts = &model_set_layout;
            vkAllocateDescriptorSets(ctx.device, &alloc_info, &model_set);

            VkDescriptorBufferInfo model_buffer_info{ model_buffer, 0, sizeof(ModelUBO) };
            VkWriteDescriptorSet model_write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, model_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &model_buffer_info, nullptr };
            vkUpdateDescriptorSets(ctx.device, 1, &model_write, 0, nullptr);

            // Grid Set
            alloc_info.pSetLayouts = &grid_set_layout;
            vkAllocateDescriptorSets(ctx.device, &alloc_info, &grid_set);

            VkDescriptorBufferInfo grid_buffer_info{ grid_ubo_buffer, 0, sizeof(GridUBO) };
            VkWriteDescriptorSet grid_write{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, grid_set, 0, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &grid_buffer_info, nullptr };
            vkUpdateDescriptorSets(ctx.device, 1, &grid_write, 0, nullptr);

            return true;
        }

    } // namespace

    bool initialize() {
        auto vertices = generateSphereVertices(32, 16, 1.0f);
        auto indices = generateSphereIndices(32, 16);
        index_count = uint32_t(indices.size());

        auto grid_vertices = generateGrid(20, 0.5f);
        grid_vertex_count = uint32_t(grid_vertices.size());

        createBuffer(vertices.size() * sizeof(Vertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vertex_buffer, vertex_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, vertices.data());
        createBuffer(indices.size() * sizeof(uint32_t), VK_BUFFER_USAGE_INDEX_BUFFER_BIT, index_buffer, index_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, indices.data());

        createBuffer(grid_vertices.size() * sizeof(GridVertex), VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, grid_vertex_buffer, grid_vertex_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, grid_vertices.data());

        createBuffer(sizeof(SceneUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, scene_buffer, scene_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, nullptr, &scene_buffer_mapped);
        createBuffer(sizeof(ModelUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, model_buffer, model_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, nullptr, &model_buffer_mapped);
        createBuffer(sizeof(GridUBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, grid_ubo_buffer, grid_ubo_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, nullptr, &grid_ubo_buffer_mapped);

        createBuffer(MAX_POINT_LIGHTS * sizeof(PointLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, point_lights_buffer, point_lights_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, nullptr, &point_lights_buffer_mapped);
        createBuffer(MAX_SPOT_LIGHTS * sizeof(SpotLight), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, spot_lights_buffer, spot_lights_buffer_alloc, VMA_MEMORY_USAGE_AUTO_PREFER_HOST, nullptr, &spot_lights_buffer_mapped);

        if (!createPipelines()) return false;
        if (!createDescriptorSets()) return false;
        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);
        if (descriptor_pool) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        if (scene_set_layout) vkDestroyDescriptorSetLayout(ctx.device, scene_set_layout, nullptr);
        if (model_set_layout) vkDestroyDescriptorSetLayout(ctx.device, model_set_layout, nullptr);
        if (grid_set_layout) vkDestroyDescriptorSetLayout(ctx.device, grid_set_layout, nullptr);
        if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);
        if (grid_pipeline) vkDestroyPipeline(ctx.device, grid_pipeline, nullptr);
        if (grid_pipeline_layout) vkDestroyPipelineLayout(ctx.device, grid_pipeline_layout, nullptr);

        if (scene_buffer_mapped) vmaUnmapMemory(ctx.allocator, scene_buffer_alloc);
        if (scene_buffer) vmaDestroyBuffer(ctx.allocator, scene_buffer, scene_buffer_alloc);
        if (model_buffer_mapped) vmaUnmapMemory(ctx.allocator, model_buffer_alloc);
        if (model_buffer) vmaDestroyBuffer(ctx.allocator, model_buffer, model_buffer_alloc);
        if (grid_ubo_buffer_mapped) vmaUnmapMemory(ctx.allocator, grid_ubo_buffer_alloc);
        if (grid_ubo_buffer) vmaDestroyBuffer(ctx.allocator, grid_ubo_buffer, grid_ubo_buffer_alloc);

        if (point_lights_buffer_mapped) vmaUnmapMemory(ctx.allocator, point_lights_buffer_alloc);
        if (point_lights_buffer) vmaDestroyBuffer(ctx.allocator, point_lights_buffer, point_lights_buffer_alloc);
        if (spot_lights_buffer_mapped) vmaUnmapMemory(ctx.allocator, spot_lights_buffer_alloc);
        if (spot_lights_buffer) vmaDestroyBuffer(ctx.allocator, spot_lights_buffer, spot_lights_buffer_alloc);

        if (index_buffer) vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_alloc);
        if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_alloc);
        if (grid_vertex_buffer) vmaDestroyBuffer(ctx.allocator, grid_vertex_buffer, grid_vertex_buffer_alloc);
    }

    void update([[maybe_unused]] double time) {
        ImGui::Begin("Controls");

        ImGui::Text("Camera");
        ImGui::SliderFloat("Yaw", &cam_yaw, -180.0f, 180.0f);
        ImGui::SliderFloat("Pitch", &cam_pitch, -89.0f, 89.0f);
        ImGui::SliderFloat("Distance", &cam_dist, 1.0f, 20.0f);
        ImGui::SliderFloat("FOV", &cam_fov, 10.0f, 120.0f);
        ImGui::Separator();

        ImGui::Text("Directional Light");
        ImGui::SliderFloat3("Dir", &light_dir_ui.x, -1.0f, 1.0f);
        ImGui::ColorEdit3("Col", &light_color_ui.x);
        ImGui::ColorEdit3("Amb", &ambient_ui.x);
        ImGui::Separator();

        ImGui::Text("Material");
        ImGui::ColorEdit3("Base Color", &base_color_ui.x);
        ImGui::ColorEdit3("Specular Color", &specular_color_ui.x);
        ImGui::SliderFloat("Shininess", &shininess_ui, 2.0f, 256.0f);
        ImGui::Separator();

        ImGui::Text("Transformations");
        ImGui::SliderFloat3("Position", &obj_position.x, -3.0f, 3.0f);
        ImGui::SliderFloat3("Rotation", &obj_rotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", &obj_scale.x, 0.1f, 3.0f);
        ImGui::Separator();

        ImGui::Text("Point Lights");
        if (ImGui::Button("Add Point Light") && point_lights.size() < MAX_POINT_LIGHTS) {
            PointLight p;
            p.position = glm::vec4(1.5f, 1.0f, 1.5f, 2.0f);
            p.color = glm::vec4(1.0f, 0.5f, 0.5f, 1.0f);
            point_lights.push_back(p);
        }
        for (int i = 0; i < point_lights.size(); ++i) {
            ImGui::PushID(i + 1000);
            ImGui::Text("Point %d", i);
            ImGui::SliderFloat3("Pos##p", &point_lights[i].position.x, -5.0f, 5.0f);
            ImGui::SliderFloat("Rad##p", &point_lights[i].position.w, 0.1f, 10.0f);
            ImGui::ColorEdit3("Col##p", &point_lights[i].color.x);
            ImGui::SliderFloat("Int##p", &point_lights[i].color.w, 0.0f, 5.0f);
            if (ImGui::Button("Remove##p")) {
                point_lights.erase(point_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }

        ImGui::Text("Spot Lights");
        if (ImGui::Button("Add Spot Light") && spot_lights.size() < MAX_SPOT_LIGHTS) {
            SpotLight s;
            s.position = glm::vec4(0.0f, 2.0f, 2.0f, 5.0f);
            s.direction = glm::vec4(0.0f, -1.0f, -1.0f, glm::cos(glm::radians(15.0f)));
            s.color = glm::vec4(0.5f, 0.5f, 1.0f, 1.0f);
            s.angles = glm::vec4(glm::cos(glm::radians(25.0f)), 0, 0, 0);
            spot_lights.push_back(s);
        }
        for (int i = 0; i < spot_lights.size(); ++i) {
            ImGui::PushID(i + 2000);
            ImGui::Text("Spot %d", i);
            ImGui::SliderFloat3("Pos##s", &spot_lights[i].position.x, -5.0f, 5.0f);
            ImGui::SliderFloat("Rad##s", &spot_lights[i].position.w, 0.1f, 10.0f);
            ImGui::SliderFloat3("Dir##s", &spot_lights[i].direction.x, -1.0f, 1.0f);
            ImGui::ColorEdit3("Col##s", &spot_lights[i].color.x);
            ImGui::SliderFloat("Int##s", &spot_lights[i].color.w, 0.0f, 5.0f);

            float inner_deg = glm::degrees(glm::acos(spot_lights[i].direction.w));
            float outer_deg = glm::degrees(glm::acos(spot_lights[i].angles.x));

            if (ImGui::SliderFloat("Inner Angle##s", &inner_deg, 1.0f, 45.0f)) {
                spot_lights[i].direction.w = glm::cos(glm::radians(inner_deg));
            }
            if (ImGui::SliderFloat("Outer Angle##s", &outer_deg, 2.0f, 60.0f)) {
                spot_lights[i].angles.x = glm::cos(glm::radians(outer_deg));
            }

            if (ImGui::Button("Remove##s")) {
                spot_lights.erase(spot_lights.begin() + i);
                ImGui::PopID();
                break;
            }
            ImGui::Separator();
            ImGui::PopID();
        }

        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        glm::mat4 model = glm::mat4(1.0f);
        model = glm::translate(model, obj_position);
        model = glm::rotate(model, glm::radians(obj_rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(obj_rotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(obj_rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, obj_scale);

        float yaw_rad = glm::radians(cam_yaw);
        float pitch_rad = glm::radians(cam_pitch);
        glm::vec3 cam_pos;
        cam_pos.x = cam_dist * std::cos(pitch_rad) * std::sin(yaw_rad);
        cam_pos.y = cam_dist * std::sin(pitch_rad);
        cam_pos.z = cam_dist * std::cos(pitch_rad) * std::cos(yaw_rad);
        glm::mat4 view = glm::lookAt(cam_pos, glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));

        float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
        glm::mat4 proj = glm::perspective(glm::radians(cam_fov), aspect, 0.1f, 100.0f);
        proj[1][1] *= -1.0f;

        glm::mat4 normal_matrix = glm::transpose(glm::inverse(model));

        // Œ·ÌÓ‚ÎˇÂÏ Scene UBO
        SceneUBO scene_ubo{};
        scene_ubo.view = view;
        scene_ubo.proj = proj;
        scene_ubo.camera_pos = glm::vec4(cam_pos, 1.0f);
        scene_ubo.light_dir = glm::vec4(glm::normalize(light_dir_ui), 0.0f);
        scene_ubo.light_color = glm::vec4(light_color_ui, 1.0f);
        scene_ubo.ambient = glm::vec4(ambient_ui, 1.0f);
        scene_ubo.counts = glm::vec4(float(point_lights.size()), float(spot_lights.size()), 0, 0);
        std::memcpy(scene_buffer_mapped, &scene_ubo, sizeof(scene_ubo));

        // Œ·ÌÓ‚ÎˇÂÏ Model UBO
        ModelUBO model_ubo{};
        model_ubo.model = model;
        model_ubo.normal_matrix = normal_matrix;
        model_ubo.base_color = glm::vec4(base_color_ui, 1.0f);
        model_ubo.specular_color = glm::vec4(specular_color_ui, 1.0f);
        model_ubo.material_params = glm::vec4(shininess_ui, 0, 0, 0);
        std::memcpy(model_buffer_mapped, &model_ubo, sizeof(model_ubo));

        // Œ·ÌÓ‚ÎˇÂÏ Grid UBO
        GridUBO grid_ubo{};
        grid_ubo.viewProj = proj * view;
        std::memcpy(grid_ubo_buffer_mapped, &grid_ubo, sizeof(grid_ubo));

        if (!point_lights.empty()) {
            std::memcpy(point_lights_buffer_mapped, point_lights.data(), point_lights.size() * sizeof(PointLight));
        }
        if (!spot_lights.empty()) {
            std::memcpy(spot_lights_buffer_mapped, spot_lights.data(), spot_lights.size() * sizeof(SpotLight));
        }

        VkCommandBuffer cmd = fd.command_buffer;
        VkCommandBufferBeginInfo begin_info{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clear_values[2]{};
        clear_values[0].color = { {0.1f, 0.1f, 0.15f, 1.0f} };
        clear_values[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rp_info{ VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO };
        rp_info.renderPass = ctx.render_pass;
        rp_info.framebuffer = fd.framebuffer;
        rp_info.renderArea.extent = ctx.swapchain_extent;
        rp_info.clearValueCount = 2;
        rp_info.pClearValues = clear_values;

        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport{ 0.0f, 0.0f, float(ctx.swapchain_extent.width), float(ctx.swapchain_extent.height), 0.0f, 1.0f };
        vkCmdSetViewport(cmd, 0, 1, &viewport);
        VkRect2D scissor{ {0, 0}, ctx.swapchain_extent };
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        // 1. –»—”≈Ã —≈“ ”
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline);
        VkDeviceSize grid_offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &grid_vertex_buffer, grid_offsets);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, grid_pipeline_layout, 0, 1, &grid_set, 0, nullptr);
        vkCmdDraw(cmd, grid_vertex_count, 1, 0, 0);

        // 2. –»—”≈Ã —‘≈–”
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);
        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, offsets);
        vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
        VkDescriptorSet sets[] = { scene_set, model_set };
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout, 0, 2, sets, 0, nullptr);
        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application