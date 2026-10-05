#include "application.hpp"

#include <imgui.h>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

namespace application {

    // ------------------------------------------------------------------
    // Состояние приложения
    // ------------------------------------------------------------------
    namespace {

        struct Vertex {
            glm::vec3 pos;
            glm::vec3 color;
        };

        struct UniformData {
            glm::mat4 model;
            glm::mat4 view;
            glm::mat4 proj;
            glm::vec4 base_color;
        };

        // Буферы
        VkBuffer vertex_buffer = VK_NULL_HANDLE;
        VmaAllocation vertex_buffer_alloc = VK_NULL_HANDLE;

        VkBuffer index_buffer = VK_NULL_HANDLE;
        VmaAllocation index_buffer_alloc = VK_NULL_HANDLE;

        VkBuffer uniform_buffer = VK_NULL_HANDLE;
        VmaAllocation uniform_buffer_alloc = VK_NULL_HANDLE;
        void* uniform_buffer_mapped = nullptr;

        uint32_t index_count = 0;

        // Descriptor set
        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;

        // Pipeline
        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VkPipeline pipeline = VK_NULL_HANDLE;

        // --- Доп. 1: проекция ---
        bool use_perspective = true;

        // --- Доп. 2: трансформации ---
        glm::vec3 obj_position = glm::vec3(0.0f, 0.0f, 0.0f);
        glm::vec3 obj_rotation = glm::vec3(0.0f, 0.0f, 0.0f); // в градусах
        glm::vec3 obj_scale = glm::vec3(1.0f, 1.0f, 1.0f);

        // --- Доп. 3: анимация ---
        bool  anim_playing = false;    // по умолчанию выключена
        float anim_speed = 1.0f;
        float anim_radius = 1.5f;
        float anim_time = 0.0f;

        // --- Доп. 4: цвет ---
        glm::vec3 base_color_ui = glm::vec3(1.0f, 1.0f, 1.0f);

        // ------------------------------------------------------------------
        // Генератор сферы (UV-сфера)
        // ------------------------------------------------------------------
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

                    Vertex vtx;
                    vtx.pos = glm::vec3(px, py, pz);
                    // Доп. 5: процедурный цвет вершины по её позиции
                    vtx.color = glm::vec3(
                        (px / radius + 1.0f) * 0.5f,
                        (py / radius + 1.0f) * 0.5f,
                        (pz / radius + 1.0f) * 0.5f
                    );
                    vertices.push_back(vtx);
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

                    indices.push_back(i0);
                    indices.push_back(i2);
                    indices.push_back(i1);

                    indices.push_back(i1);
                    indices.push_back(i2);
                    indices.push_back(i3);
                }
            }
            return indices;
        }

        // ------------------------------------------------------------------
        // Утилиты загрузки шейдеров и создания буферов
        // ------------------------------------------------------------------
        std::vector<char> readFile(const std::string& path) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file: " << path << '\n';
                return {};
            }
            size_t size = size_t(file.tellg());
            std::vector<char> buffer(size);
            file.seekg(0);
            file.read(buffer.data(), size);
            return buffer;
        }

        VkShaderModule createShaderModule(const std::vector<char>& code) {
            VkShaderModuleCreateInfo info{};
            info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            info.codeSize = code.size();
            info.pCode = reinterpret_cast<const uint32_t*>(code.data());

            VkShaderModule module;
            if (vkCreateShaderModule(graphics::internal::context.device, &info, nullptr, &module) != VK_SUCCESS) {
                std::cerr << "Failed to create shader module\n";
                return VK_NULL_HANDLE;
            }
            return module;
        }

        bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
            VkBuffer& out_buffer, VmaAllocation& out_alloc,
            VmaMemoryUsage mem_usage,
            const void* data = nullptr,
            void** out_mapped = nullptr) {
            VkBufferCreateInfo buffer_info{};
            buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
            buffer_info.size = size;
            buffer_info.usage = usage;
            buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

            VmaAllocationCreateInfo alloc_info{};
            alloc_info.usage = mem_usage;
            alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;

            VmaAllocationInfo alloc_result{};
            if (vmaCreateBuffer(graphics::internal::context.allocator,
                &buffer_info, &alloc_info,
                &out_buffer, &out_alloc, &alloc_result) != VK_SUCCESS) {
                std::cerr << "Failed to create buffer\n";
                return false;
            }

            void* mapped = nullptr;
            if (vmaMapMemory(graphics::internal::context.allocator, out_alloc, &mapped) != VK_SUCCESS) {
                std::cerr << "Failed to map buffer\n";
                return false;
            }

            if (data != nullptr) {
                std::memcpy(mapped, data, size_t(size));
            }

            if (out_mapped != nullptr) {
                *out_mapped = mapped;
            }
            else {
                vmaUnmapMemory(graphics::internal::context.allocator, out_alloc);
            }

            return true;
        }

        // ------------------------------------------------------------------
        // Создание пайплайна
        // ------------------------------------------------------------------
        bool createPipeline() {
            auto& ctx = graphics::internal::context;

            VkDescriptorSetLayoutBinding binding{};
            binding.binding = 0;
            binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            binding.descriptorCount = 1;
            binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;

            VkDescriptorSetLayoutCreateInfo layout_info{};
            layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
            layout_info.bindingCount = 1;
            layout_info.pBindings = &binding;

            if (vkCreateDescriptorSetLayout(ctx.device, &layout_info, nullptr, &descriptor_set_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor set layout\n";
                return false;
            }

            VkPipelineLayoutCreateInfo pl_info{};
            pl_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pl_info.setLayoutCount = 1;
            pl_info.pSetLayouts = &descriptor_set_layout;

            if (vkCreatePipelineLayout(ctx.device, &pl_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                std::cerr << "Failed to create pipeline layout\n";
                return false;
            }

            auto vert_code = readFile("shaders/sphere.vert.spv");
            auto frag_code = readFile("shaders/sphere.frag.spv");
            if (vert_code.empty() || frag_code.empty()) {
                std::cerr << "Shader files not found. Run from project root!\n";
                return false;
            }

            VkShaderModule vert_module = createShaderModule(vert_code);
            VkShaderModule frag_module = createShaderModule(frag_code);
            if (vert_module == VK_NULL_HANDLE || frag_module == VK_NULL_HANDLE) {
                return false;
            }

            VkPipelineShaderStageCreateInfo stages[2]{};
            stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[0].stage = VK_SHADER_STAGE_VERTEX_BIT;
            stages[0].module = vert_module;
            stages[0].pName = "main";

            stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            stages[1].stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            stages[1].module = frag_module;
            stages[1].pName = "main";

            VkVertexInputBindingDescription binding_desc{};
            binding_desc.binding = 0;
            binding_desc.stride = sizeof(Vertex);
            binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

            VkVertexInputAttributeDescription attr_desc[2]{};
            attr_desc[0].location = 0;
            attr_desc[0].binding = 0;
            attr_desc[0].format = VK_FORMAT_R32G32B32_SFLOAT;
            attr_desc[0].offset = offsetof(Vertex, pos);

            attr_desc[1].location = 1;
            attr_desc[1].binding = 0;
            attr_desc[1].format = VK_FORMAT_R32G32B32_SFLOAT;
            attr_desc[1].offset = offsetof(Vertex, color);

            VkPipelineVertexInputStateCreateInfo vertex_input{};
            vertex_input.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input.vertexBindingDescriptionCount = 1;
            vertex_input.pVertexBindingDescriptions = &binding_desc;
            vertex_input.vertexAttributeDescriptionCount = 2;
            vertex_input.pVertexAttributeDescriptions = attr_desc;

            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            VkDynamicState dynamic_states[] = {
                VK_DYNAMIC_STATE_VIEWPORT,
                VK_DYNAMIC_STATE_SCISSOR,
            };
            VkPipelineDynamicStateCreateInfo dynamic_state{};
            dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic_state.dynamicStateCount = 2;
            dynamic_state.pDynamicStates = dynamic_states;

            VkPipelineRasterizationStateCreateInfo raster{};
            raster.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            raster.polygonMode = VK_POLYGON_MODE_FILL;
            raster.cullMode = VK_CULL_MODE_BACK_BIT;
            raster.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
            raster.lineWidth = 1.0f;

            VkPipelineMultisampleStateCreateInfo multisample{};
            multisample.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

            VkPipelineDepthStencilStateCreateInfo depth_stencil{};
            depth_stencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
            depth_stencil.depthTestEnable = VK_TRUE;
            depth_stencil.depthWriteEnable = VK_TRUE;
            depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;

            VkPipelineColorBlendAttachmentState color_attach{};
            color_attach.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
                VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            color_attach.blendEnable = VK_FALSE;

            VkPipelineColorBlendStateCreateInfo color_blend{};
            color_blend.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            color_blend.attachmentCount = 1;
            color_blend.pAttachments = &color_attach;

            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
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

            if (vkCreateGraphicsPipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &pipeline) != VK_SUCCESS) {
                std::cerr << "Failed to create graphics pipeline\n";
                return false;
            }

            vkDestroyShaderModule(ctx.device, vert_module, nullptr);
            vkDestroyShaderModule(ctx.device, frag_module, nullptr);

            return true;
        }

        // ------------------------------------------------------------------
        // Descriptor set
        // ------------------------------------------------------------------
        bool createDescriptorSet() {
            auto& ctx = graphics::internal::context;

            VkDescriptorPoolSize pool_size{};
            pool_size.type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            pool_size.descriptorCount = 1;

            VkDescriptorPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
            pool_info.poolSizeCount = 1;
            pool_info.pPoolSizes = &pool_size;
            pool_info.maxSets = 1;

            if (vkCreateDescriptorPool(ctx.device, &pool_info, nullptr, &descriptor_pool) != VK_SUCCESS) {
                std::cerr << "Failed to create descriptor pool\n";
                return false;
            }

            VkDescriptorSetAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
            alloc_info.descriptorPool = descriptor_pool;
            alloc_info.descriptorSetCount = 1;
            alloc_info.pSetLayouts = &descriptor_set_layout;

            if (vkAllocateDescriptorSets(ctx.device, &alloc_info, &descriptor_set) != VK_SUCCESS) {
                std::cerr << "Failed to allocate descriptor set\n";
                return false;
            }

            VkDescriptorBufferInfo buffer_info{};
            buffer_info.buffer = uniform_buffer;
            buffer_info.offset = 0;
            buffer_info.range = sizeof(UniformData);

            VkWriteDescriptorSet write{};
            write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            write.dstSet = descriptor_set;
            write.dstBinding = 0;
            write.descriptorCount = 1;
            write.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            write.pBufferInfo = &buffer_info;

            vkUpdateDescriptorSets(ctx.device, 1, &write, 0, nullptr);

            return true;
        }

    } // namespace

    // ------------------------------------------------------------------
    // Публичные функции
    // ------------------------------------------------------------------
    bool initialize() {
        auto& ctx = graphics::internal::context;

        // segments * rings: 16 * 8 -> (16+1)*(8+1) = 153 вершины
        auto vertices = generateSphereVertices(32, 16, 1.0f);
        auto indices = generateSphereIndices(32, 16);
        index_count = uint32_t(indices.size());

        std::cout << "Sphere: " << vertices.size() << " vertices, " << indices.size() << " indices\n";

        if (!createBuffer(vertices.size() * sizeof(Vertex),
            VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
            vertex_buffer, vertex_buffer_alloc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            vertices.data())) {
            return false;
        }

        if (!createBuffer(indices.size() * sizeof(uint32_t),
            VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
            index_buffer, index_buffer_alloc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            indices.data())) {
            return false;
        }

        if (!createBuffer(sizeof(UniformData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            uniform_buffer, uniform_buffer_alloc,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            nullptr,
            &uniform_buffer_mapped)) {
            return false;
        }

        if (!createPipeline()) return false;
        if (!createDescriptorSet()) return false;

        return true;
    }

    void shutdown() {
        auto& ctx = graphics::internal::context;
        vkDeviceWaitIdle(ctx.device);

        if (descriptor_pool) vkDestroyDescriptorPool(ctx.device, descriptor_pool, nullptr);
        if (descriptor_set_layout) vkDestroyDescriptorSetLayout(ctx.device, descriptor_set_layout, nullptr);

        if (pipeline) vkDestroyPipeline(ctx.device, pipeline, nullptr);
        if (pipeline_layout) vkDestroyPipelineLayout(ctx.device, pipeline_layout, nullptr);

        if (uniform_buffer_mapped) vmaUnmapMemory(ctx.allocator, uniform_buffer_alloc);
        if (uniform_buffer) vmaDestroyBuffer(ctx.allocator, uniform_buffer, uniform_buffer_alloc);
        if (index_buffer) vmaDestroyBuffer(ctx.allocator, index_buffer, index_buffer_alloc);
        if (vertex_buffer) vmaDestroyBuffer(ctx.allocator, vertex_buffer, vertex_buffer_alloc);
    }

    void update([[maybe_unused]] double time) {
        static double last_time = 0.0;
        float dt = float(time - last_time);
        last_time = time;
        if (dt < 0.0f) dt = 0.0f;
        if (dt > 0.1f) dt = 0.1f;

        // Доп. 3: анимация по круговой траектории
        if (anim_playing) {
            anim_time += dt * anim_speed;
        }

        ImGui::Begin("Controls");

        // Доп. 1: проекция
        ImGui::Text("Projection");
        int proj_mode = use_perspective ? 1 : 0;
        ImGui::RadioButton("Perspective", &proj_mode, 1); ImGui::SameLine();
        ImGui::RadioButton("Orthographic", &proj_mode, 0);
        use_perspective = (proj_mode == 1);
        ImGui::Separator();

        // Доп. 2: трансформации
        ImGui::Text("Transformations");
        ImGui::SliderFloat3("Position", &obj_position.x, -3.0f, 3.0f);
        ImGui::SliderFloat3("Rotation", &obj_rotation.x, -180.0f, 180.0f);
        ImGui::SliderFloat3("Scale", &obj_scale.x, 0.1f, 3.0f);
        ImGui::Separator();

        // Доп. 3: анимация
        ImGui::Text("Animation");
        ImGui::Checkbox("Playing", &anim_playing);
        ImGui::SliderFloat("Speed", &anim_speed, 0.0f, 5.0f);
        ImGui::SliderFloat("Radius", &anim_radius, 0.0f, 5.0f);
        ImGui::Separator();

        // Доп. 4: цвет
        ImGui::Text("Color");
        ImGui::ColorEdit3("Base color", &base_color_ui.x);

        ImGui::End();
    }

    void render(const graphics::internal::FrameData& fd) {
        auto& ctx = graphics::internal::context;

        // -------- model --------
        glm::mat4 model = glm::mat4(1.0f);

        // Доп. 3: смещение по траектории (круг в плоскости XZ)
        glm::vec3 anim_offset(
            anim_radius * std::cos(anim_time),
            0.0f,
            anim_radius * std::sin(anim_time)
        );
        glm::vec3 final_pos = obj_position + anim_offset;

        model = glm::translate(model, final_pos);
        model = glm::rotate(model, glm::radians(obj_rotation.x), glm::vec3(1, 0, 0));
        model = glm::rotate(model, glm::radians(obj_rotation.y), glm::vec3(0, 1, 0));
        model = glm::rotate(model, glm::radians(obj_rotation.z), glm::vec3(0, 0, 1));
        model = glm::scale(model, obj_scale);

        // -------- view --------
        glm::mat4 view = glm::lookAt(
            glm::vec3(0.0f, 0.0f, 5.0f),
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f)
        );

        // -------- projection --------
        float aspect = float(ctx.swapchain_extent.width) / float(ctx.swapchain_extent.height);
        glm::mat4 proj;
        if (use_perspective) {
            proj = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 100.0f);
        }
        else {
            float half_h = 2.0f;
            float half_w = half_h * aspect;
            proj = glm::ortho(-half_w, half_w, -half_h, half_h, -100.0f, 100.0f);
        }
        // Vulkan Y-flip
        proj[1][1] *= -1.0f;

        // -------- uniform --------
        UniformData ubo{};
        ubo.model = model;
        ubo.view = view;
        ubo.proj = proj;
        ubo.base_color = glm::vec4(base_color_ui, 1.0f);

        std::memcpy(uniform_buffer_mapped, &ubo, sizeof(ubo));

        // -------- рендер --------
        VkCommandBuffer cmd = fd.command_buffer;

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &begin_info);

        VkClearValue clear_values[2]{};
        clear_values[0].color = { {0.1f, 0.1f, 0.15f, 1.0f} };
        clear_values[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo rp_info{};
        rp_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        rp_info.renderPass = ctx.render_pass;
        rp_info.framebuffer = fd.framebuffer;
        rp_info.renderArea.offset = { 0, 0 };
        rp_info.renderArea.extent = ctx.swapchain_extent;
        rp_info.clearValueCount = 2;
        rp_info.pClearValues = clear_values;

        vkCmdBeginRenderPass(cmd, &rp_info, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = float(ctx.swapchain_extent.width);
        viewport.height = float(ctx.swapchain_extent.height);
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        vkCmdSetViewport(cmd, 0, 1, &viewport);

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = ctx.swapchain_extent;
        vkCmdSetScissor(cmd, 0, 1, &scissor);

        VkDeviceSize offsets[] = { 0 };
        vkCmdBindVertexBuffers(cmd, 0, 1, &vertex_buffer, offsets);
        vkCmdBindIndexBuffer(cmd, index_buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout,
            0, 1, &descriptor_set, 0, nullptr);

        vkCmdDrawIndexed(cmd, index_count, 1, 0, 0, 0);

        vkCmdEndRenderPass(cmd);
        vkEndCommandBuffer(cmd);
    }

} // namespace application