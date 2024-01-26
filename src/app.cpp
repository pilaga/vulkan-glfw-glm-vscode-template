#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "../include/config.hpp"
#include "../include/utils.hpp"

struct Vertex {
        glm::vec2 pos;
        glm::vec3 color;
};

/**
 * Template class implementing Vulkan, GLFW for window creation & GLM for algebraic functions.
 */
class VulkanTemplateApp {
    public:
        void run() {
            initGlfwWindow();
            initVulkan();
            renderLoop();
            cleanup();
        }

    private:
        GLFWwindow *window;
        VkInstance vk_instance;
        VkDebugUtilsMessengerEXT vk_debug_messenger;
        VkSurfaceKHR surface;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;  // Physical device
        VkDevice device;                                    // Logical device
        VkQueue graphics_queue;
        VkQueue present_queue;
        VkSwapchainKHR swapchain;
        std::vector<VkImage> swapchain_images;
        VkFormat swapchain_format;
        VkExtent2D swapchain_extent;
        std::vector<VkImageView> swapchain_img_views;
        VkRenderPass render_pass;
        VkPipelineLayout pipeline_layout;
        VkPipeline graphics_pipeline;
        std::vector<VkFramebuffer> swapchain_framebuffers;
        VkCommandPool command_pool;
        std::vector<VkCommandBuffer> command_buffers;
        std::vector<VkSemaphore> img_available_semaphores;
        std::vector<VkSemaphore> render_finished_semaphores;
        std::vector<VkFence> inflight_fences;
        uint32_t frame_index = 0;
        bool framebuffer_resized = false;

        const std::vector<Vertex> vertices = {{{0.0f, -0.5f}, {1.0f, 0.0f, 0.0f}}, {{0.5f, 0.5f}, {0.0f, 1.0f, 0.0f}}, {{-0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}}};

        /**
         * Initializes the GLFW window.
         */
        void initGlfwWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Tell GLFW not to create a GL context
            glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);     // Enable window resizing

            window = glfwCreateWindow(Config::WIDTH, Config::HEIGHT, "Vulkan Template", nullptr, nullptr);
            glfwSetWindowUserPointer(window, this);                             // Set pointer to be retrieved in the call back
            glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);  // Set call back for window resize
        }

        /**
         * Initializes Vulkan, creates VK instance.
         */
        void initVulkan() {
            createVkInstance();
            createVkDebugMessenger();
            createSurface();
            pickGPU();
            createLogicalDevice();
            createSwapChain();
            createImageViews();
            createRenderPass();
            createGraphicsPipeline();
            createFramebuffers();
            createCommandPool();
            createCommandBuffers();
            createSynchronisationObjects();
        }

        /**
         * Set-up framebuffer resize callback.
         * Use a static function because GLFW doesn't know how to call a member function with the right <this> pointer to our app instance.
         * @param window Pointer to the GLFW window.
         * @param width The window width.
         * @param height The window height.
         */
        static void framebufferResizeCallback(GLFWwindow *window, int width, int height) {
            auto app = reinterpret_cast<VulkanTemplateApp *>(glfwGetWindowUserPointer(window));
            app->framebuffer_resized = true;
        }

        /**
         * Creates the command pool to manage the momory used to store the buffers and associated command buffers.
         */
        void createCommandPool() {
            QueueFamilyIndices queue_family_indices = findQueueFamilies(physical_device);

            VkCommandPoolCreateInfo pool_info{};
            pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
            pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;         // Allows command buffers to be rerecorded individually
            pool_info.queueFamilyIndex = queue_family_indices.graphicsFamily.value();  // We're going to record drawing commands so we pick the graphics queue family

            if (vkCreateCommandPool(device, &pool_info, nullptr, &command_pool) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create command pool!");
            }
        }

        /**
         * Allocate a command buffer for each frame.
         */
        void createCommandBuffers() {
            command_buffers.resize(Config::MAX_FRAMES_IN_FLIGHT);

            VkCommandBufferAllocateInfo alloc_info{};
            alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
            alloc_info.commandPool = command_pool;
            alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;  // Can be submitted to a queue for execution
            alloc_info.commandBufferCount = (uint32_t)command_buffers.size();

            if (vkAllocateCommandBuffers(device, &alloc_info, command_buffers.data()) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to allocate command buffers!");
            }
        }

        /**
         * Writes the command we want to execute into the command buffer.
         * @param command_buffer The command buffer.
         * @param image_index The index of the swapchain image to write into.
         */
        void recordCommandBuffer(VkCommandBuffer command_buffer, uint32_t image_index) {
            // Recording a command buffer always starts by calling vkBeginCommandBuffer with an info struct specifying the usage of the command buffer
            VkCommandBufferBeginInfo begin_info{};
            begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            begin_info.flags = 0;
            begin_info.pInheritanceInfo = nullptr;

            if (vkBeginCommandBuffer(command_buffer, &begin_info) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to begin recording command buffer!");
            }

            // Drawing starts by beginning the render pass. Info struct is used to configure the render pass
            VkRenderPassBeginInfo render_pass_info{};
            render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
            render_pass_info.renderPass = render_pass;
            render_pass_info.framebuffer = swapchain_framebuffers[image_index];

            // Define the size of the render area
            render_pass_info.renderArea.offset = {0, 0};
            render_pass_info.renderArea.extent = swapchain_extent;

            // Define the clear values used by VK_ATTACHMENT_LOAD_OP_CLEAR which we use as load operation for the color attachment
            VkClearValue clear_color = {{{0.0f, 0.0f, 0.0f, 1.0f}}};
            render_pass_info.clearValueCount = 1;
            render_pass_info.pClearValues = &clear_color;

            // Begin the render pass
            vkCmdBeginRenderPass(command_buffer, &render_pass_info, VK_SUBPASS_CONTENTS_INLINE);

            // Bind the graphics pipeline
            vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphics_pipeline);

            // Viewport state was specified as dynamic and needs to be set
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = static_cast<float>(swapchain_extent.width);
            viewport.height = static_cast<float>(swapchain_extent.height);
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;
            vkCmdSetViewport(command_buffer, 0, 1, &viewport);

            // Scissor state was specified as dynamic and needs to be set
            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapchain_extent;
            vkCmdSetScissor(command_buffer, 0, 1, &scissor);

            // Issue the draw command
            vkCmdDraw(command_buffer, 3, 1, 0, 0);

            // End the render pass
            vkCmdEndRenderPass(command_buffer);

            // Finish recording the command buffer
            if (vkEndCommandBuffer(command_buffer) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to record command buffer!");
            }
        }

        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Rendering_and_presentation

        /**
         * Creates the graphics pipeline.
         */
        void createGraphicsPipeline() {
            auto vert_shader_code = readFile(Config::SHADERS_PATH + "vert.spv");
            auto frag_shader_code = readFile(Config::SHADERS_PATH + "frag.spv");

            VkShaderModule vert_shader_module = createShaderModule(vert_shader_code);
            VkShaderModule frag_shader_module = createShaderModule(frag_shader_code);

            // Define vertex shader stage info
            VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
            vert_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
            vert_shader_stage_info.module = vert_shader_module;
            vert_shader_stage_info.pName = "main";  // vertex shader entrypoint

            // Define fragment shader stage info
            VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
            frag_shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
            frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
            frag_shader_stage_info.module = frag_shader_module;
            frag_shader_stage_info.pName = "main";  // fragment shader entrypoint

            // Create array containing both shader stages info
            VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info, frag_shader_stage_info};

            // Specify the states which we want to change dynamically after pipeline creation (ie. exclude scissor and viewport state)
            std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};

            VkPipelineDynamicStateCreateInfo dynamic_state{};
            dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
            dynamic_state.dynamicStateCount = static_cast<uint32_t>(dynamic_states.size());
            dynamic_state.pDynamicStates = dynamic_states.data();

            // Specify vertex input - fill with nothing with now
            VkPipelineVertexInputStateCreateInfo vertex_input_info{};
            vertex_input_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
            vertex_input_info.vertexBindingDescriptionCount = 0;
            vertex_input_info.pVertexBindingDescriptions = nullptr;
            vertex_input_info.vertexAttributeDescriptionCount = 0;
            vertex_input_info.pVertexAttributeDescriptions = nullptr;

            // Define input assembly
            // Specify triangle list geometry to be drawn from the vertices
            VkPipelineInputAssemblyStateCreateInfo input_assembly{};
            input_assembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
            input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
            input_assembly.primitiveRestartEnable = VK_FALSE;

            // Define a viewport that is the same resolution as our window
            VkViewport viewport{};
            viewport.x = 0.0f;
            viewport.y = 0.0f;
            viewport.width = (float)swapchain_extent.width;
            viewport.height = (float)swapchain_extent.height;
            viewport.minDepth = 0.0f;
            viewport.maxDepth = 1.0f;

            VkRect2D scissor{};
            scissor.offset = {0, 0};
            scissor.extent = swapchain_extent;

            // Create the viewport state structure
            VkPipelineViewportStateCreateInfo viewport_state{};
            viewport_state.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
            viewport_state.viewportCount = 1;
            viewport_state.scissorCount = 1;

            // Create rasterizer info struct
            VkPipelineRasterizationStateCreateInfo rasterizer{};
            rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
            rasterizer.depthClampEnable = VK_FALSE;
            rasterizer.rasterizerDiscardEnable = VK_FALSE;  // If true all geometry is discarded
            rasterizer.polygonMode = VK_POLYGON_MODE_FILL;  // Fills the polygon
            rasterizer.lineWidth = 1.0f;
            rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
            rasterizer.frontFace = VK_FRONT_FACE_CLOCKWISE;
            rasterizer.depthBiasEnable = VK_FALSE;
            rasterizer.depthBiasConstantFactor = 0.0f;
            rasterizer.depthBiasClamp = 0.0f;
            rasterizer.depthBiasSlopeFactor = 0.0f;

            // Create multisampler info struct
            VkPipelineMultisampleStateCreateInfo multisampling{};
            multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
            multisampling.sampleShadingEnable = VK_FALSE;
            multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
            multisampling.minSampleShading = 1.0f;
            multisampling.pSampleMask = nullptr;
            multisampling.alphaToCoverageEnable = VK_FALSE;
            multisampling.alphaToOneEnable = VK_FALSE;

            // Define color blending (no alpha blending)
            VkPipelineColorBlendAttachmentState color_blend_attachment{};
            color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
            color_blend_attachment.blendEnable = VK_FALSE;
            color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
            color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
            color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
            color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
            color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
            color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;

            VkPipelineColorBlendStateCreateInfo color_blending{};
            color_blending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
            color_blending.logicOpEnable = VK_FALSE;
            color_blending.logicOp = VK_LOGIC_OP_COPY;
            color_blending.attachmentCount = 1;
            color_blending.pAttachments = &color_blend_attachment;
            color_blending.blendConstants[0] = 0.0f;
            color_blending.blendConstants[1] = 0.0f;
            color_blending.blendConstants[2] = 0.0f;
            color_blending.blendConstants[3] = 0.0f;

            // Create pipeline layout info, used to store shader uniforms
            VkPipelineLayoutCreateInfo pipeline_layout_info{};
            pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            pipeline_layout_info.setLayoutCount = 0;
            pipeline_layout_info.pSetLayouts = nullptr;
            pipeline_layout_info.pushConstantRangeCount = 0;
            pipeline_layout_info.pPushConstantRanges = nullptr;

            if (vkCreatePipelineLayout(device, &pipeline_layout_info, nullptr, &pipeline_layout) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create pipeline layout!");
            }

            // Create pipeline info
            VkGraphicsPipelineCreateInfo pipeline_info{};
            pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
            // 1. Reference the array of shader info struct
            pipeline_info.stageCount = 2;
            pipeline_info.pStages = shader_stages;
            // 2. Reference the structs describing the fixed function stage
            pipeline_info.pVertexInputState = &vertex_input_info;
            pipeline_info.pInputAssemblyState = &input_assembly;
            pipeline_info.pViewportState = &viewport_state;
            pipeline_info.pRasterizationState = &rasterizer;
            pipeline_info.pMultisampleState = &multisampling;
            pipeline_info.pDepthStencilState = nullptr;
            pipeline_info.pColorBlendState = &color_blending;
            pipeline_info.pDynamicState = &dynamic_state;
            // 3. Reference the pipeline layout
            pipeline_info.layout = pipeline_layout;
            // 4. Reference the render pass
            pipeline_info.renderPass = render_pass;
            pipeline_info.subpass = 0;
            // 5. Create a new graphics pipeline by deriving from an existing pipeline (not used here)
            pipeline_info.basePipelineHandle = VK_NULL_HANDLE;  // Optional
            pipeline_info.basePipelineIndex = -1;               // Optional

            if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipeline_info, nullptr, &graphics_pipeline) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create graphics pipeline!");
            }

            // Destroy the shader modules after the graphics pipeline is created
            vkDestroyShaderModule(device, frag_shader_module, nullptr);
            vkDestroyShaderModule(device, vert_shader_module, nullptr);
        }

        /**
         * Creates the swap chain frame buffers.
         */
        void createFramebuffers() {
            // Resize container to hold all the framebuffers
            swapchain_framebuffers.resize(swapchain_img_views.size());

            // Iterate through the image views and create framebuffers from them
            for (size_t i = 0; i < swapchain_img_views.size(); i++) {
                VkImageView attachments[] = {swapchain_img_views[i]};

                VkFramebufferCreateInfo framebuffer_info{};
                framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
                framebuffer_info.renderPass = render_pass;
                framebuffer_info.attachmentCount = 1;
                framebuffer_info.pAttachments = attachments;
                framebuffer_info.width = swapchain_extent.width;
                framebuffer_info.height = swapchain_extent.height;
                framebuffer_info.layers = 1;  // Our swap chain images are single images

                if (vkCreateFramebuffer(device, &framebuffer_info, nullptr, &swapchain_framebuffers[i]) != VK_SUCCESS) {
                    throw std::runtime_error("error: failed to create framebuffers!");
                }
            }
        }

        /**
         * Creates a shader module from the shader bytecode.
         */
        VkShaderModule createShaderModule(const std::vector<char> &code) {
            VkShaderModuleCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
            create_info.codeSize = code.size();
            create_info.pCode = reinterpret_cast<const uint32_t *>(code.data());

            VkShaderModule shader_module;
            if (vkCreateShaderModule(device, &create_info, nullptr, &shader_module) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create shader module!");
            }

            return shader_module;
        }

        /*
         * Creates the GLFW window surface.
         */
        void createSurface() {
            if (glfwCreateWindowSurface(vk_instance, window, nullptr, &surface) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create window surface!");
            }
        }

        /**
         * Creates an image view for each image in the swap chain.
         */
        void createImageViews() {
            // Resize view list to fit all the images
            swapchain_img_views.resize(swapchain_images.size());

            // Iterate over all the swap chain images
            for (size_t i = 0; i < swapchain_images.size(); i++) {
                VkImageViewCreateInfo create_info{};
                create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
                create_info.image = swapchain_images[i];
                create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
                create_info.format = swapchain_format;

                // Keep default color channel mapping
                create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
                create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
                create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
                create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;

                create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                create_info.subresourceRange.baseMipLevel = 0;
                create_info.subresourceRange.levelCount = 1;
                create_info.subresourceRange.baseArrayLayer = 0;
                create_info.subresourceRange.layerCount = 1;

                if (vkCreateImageView(device, &create_info, nullptr, &swapchain_img_views[i]) != VK_SUCCESS) {
                    throw std::runtime_error("error: failed to create image views!");
                }
            }
        }

        /**
         * Creates the logical device.
         */
        void createLogicalDevice() {
            QueueFamilyIndices indices = findQueueFamilies(physical_device);

            std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
            std::set<uint32_t> unique_queue_families = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            float queue_priority = 1.0f;
            for (uint32_t queue_family : unique_queue_families) {
                VkDeviceQueueCreateInfo queue_create_info{};
                queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
                queue_create_info.queueFamilyIndex = queue_family;
                queue_create_info.queueCount = 1;
                queue_create_info.pQueuePriorities = &queue_priority;
                queue_create_infos.push_back(queue_create_info);
            }

            // Create info for the logical device
            VkDeviceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
            create_info.pQueueCreateInfos = queue_create_infos.data();
            create_info.pEnabledFeatures = &VkPhysicalDeviceFeatures{};  // Leave empty for now
            create_info.enabledExtensionCount = static_cast<uint32_t>(Config::DEVICE_EXTENSIONS.size());
            create_info.ppEnabledExtensionNames = Config::DEVICE_EXTENSIONS.data();

            // Add validation layer if enabled
            if (Config::ENABLE_VALIDATION_LAYERS) {
                create_info.enabledLayerCount = static_cast<uint32_t>(Config::VALIDATION_LAYERS.size());
                create_info.ppEnabledLayerNames = Config::VALIDATION_LAYERS.data();
            } else {
                create_info.enabledLayerCount = 0;
            }

            if (vkCreateDevice(physical_device, &create_info, nullptr, &device) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create logical device!");
            }

            vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphics_queue);
            vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &present_queue);
        }

        /**
         * Creates the swap chain using the selected present mode, surface format and extent.
         */
        void createSwapChain() {
            SwapChainSupportDetails swapchain_support = querySwapChainSupport(physical_device);
            VkSurfaceFormatKHR surface_format = pickSwapSurfaceFormat(swapchain_support.formats);
            VkPresentModeKHR present_mode = pickSwapPresentMode(swapchain_support.presentModes);
            VkExtent2D extent = pickSwapExtent(swapchain_support.capabilities);

            // It is recommended to request at least 1 more image than the minimum
            uint32_t img_count = swapchain_support.capabilities.minImageCount + 1;

            // Make sure we don't go over the maxImageCount for the swap chain
            // 0 is a special value that means there is no max
            if (swapchain_support.capabilities.maxImageCount > 0 && img_count > swapchain_support.capabilities.maxImageCount) {
                img_count = swapchain_support.capabilities.maxImageCount;
            }

            VkSwapchainCreateInfoKHR create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            create_info.surface = surface;
            create_info.minImageCount = img_count;
            create_info.imageFormat = surface_format.format;
            create_info.imageColorSpace = surface_format.colorSpace;
            create_info.imageExtent = extent;
            create_info.imageArrayLayers = 1;  // should be 1 unless stereoscopic display
            create_info.imageUsage = Config::SWAPCHAIN_IMAGE_USAGE;

            QueueFamilyIndices indices = findQueueFamilies(physical_device);
            uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};

            if (indices.graphicsFamily != indices.presentFamily) {
                create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
                create_info.queueFamilyIndexCount = 2;
                create_info.pQueueFamilyIndices = queueFamilyIndices;
            } else {
                create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
                create_info.queueFamilyIndexCount = 0;
                create_info.pQueueFamilyIndices = nullptr;
            }

            create_info.preTransform = swapchain_support.capabilities.currentTransform;
            create_info.compositeAlpha = Config::SWAPCHAIN_COMPOSITE_ALPHA;
            create_info.presentMode = present_mode;
            create_info.clipped = Config::SWAPCHAIN_CLIPPED;
            create_info.oldSwapchain = VK_NULL_HANDLE;

            if (vkCreateSwapchainKHR(device, &create_info, nullptr, &swapchain) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create swap chain!");
            }

            // Retrieve handle to the swap chain images
            vkGetSwapchainImagesKHR(device, swapchain, &img_count, nullptr);
            swapchain_images.resize(img_count);
            vkGetSwapchainImagesKHR(device, swapchain, &img_count, swapchain_images.data());

            // Store format and extent for later use
            swapchain_extent = extent;
            swapchain_format = surface_format.format;
        }

        /**
         * Recreate the swap chain to handle things such as window resizing.
         */
        void recreateSwapChain() {
            int width = 0, height = 0;
            glfwGetFramebufferSize(window, &width, &height);

            // Pause swap chain recreation until window is in the background (height and width != 0)
            while (width == 0 || height == 0) {
                glfwGetFramebufferSize(window, &width, &height);
                glfwWaitEvents();
            }

            // First wait for device to be available
            vkDeviceWaitIdle(device);

            cleanupSwapChain();

            // Recreate the swapchain objects
            createSwapChain();
            createImageViews();
            createFramebuffers();
        }

        /**
         * Cleans-up the swap chain.
         */
        void cleanupSwapChain() {
            for (auto framebuffer : swapchain_framebuffers) {
                vkDestroyFramebuffer(device, framebuffer, nullptr);
            }

            for (auto img_view : swapchain_img_views) {
                vkDestroyImageView(device, img_view, nullptr);
            }

            vkDestroySwapchainKHR(device, swapchain, nullptr);
        }

        /**
         * Picks the first GPU that provides VK support and assigns its handle to <physicalDevice> class member.
         */
        void pickGPU() {
            uint32_t device_count = 0;
            vkEnumeratePhysicalDevices(vk_instance, &device_count, nullptr);

            if (device_count == 0) {
                throw std::runtime_error("error: could not find a GPU with VK support!");
            }

            std::vector<VkPhysicalDevice> devices(device_count);
            vkEnumeratePhysicalDevices(vk_instance, &device_count, devices.data());

            for (const auto &device : devices) {
                if (isGPUSuitable(device)) {
                    // Grab basic device properties
                    VkPhysicalDeviceProperties properties;
                    vkGetPhysicalDeviceProperties(device, &properties);
                    std::cout << "selected GPU: " << properties.deviceName << "\n";

                    physical_device = device;
                    break;
                }
            }

            if (physical_device == VK_NULL_HANDLE) {
                throw std::runtime_error("error: could not find a suitable GPU!");
            }
        }

        /**
         * Checks the provided GPU device is suitable for our application.
         * @param device The physical device handle.
         * @returns True if the device is suitable.
         */
        bool isGPUSuitable(VkPhysicalDevice device) {
            QueueFamilyIndices indices = findQueueFamilies(device);

            // Check required extensions are supported
            bool extensions_supported = checkDeviceExtensionSupport(device);

            // Check swap chain support is adequate
            bool swapchain_suitable = false;
            if (extensions_supported) {
                SwapChainSupportDetails swapchain_support = querySwapChainSupport(device);
                swapchain_suitable = !swapchain_support.formats.empty() && !swapchain_support.presentModes.empty();
            }

            return indices.isComplete() && extensions_supported && swapchain_suitable;
        }

        /**
         * Retrieves the swap chain support details for the specified device.
         * @param device The physical device.
         * @returns The swap chain support details.
         */
        SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) {
            SwapChainSupportDetails details;

            vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities);

            // Query the supported surface formats
            uint32_t format_count;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, nullptr);

            if (format_count != 0) {
                details.formats.resize(format_count);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &format_count, details.formats.data());
            }

            // Query the supported presentation modes
            uint32_t present_mode_count;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, nullptr);

            if (present_mode_count != 0) {
                details.presentModes.resize(present_mode_count);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &present_mode_count, details.presentModes.data());
            }

            return details;
        }

        /**
         * Picks the best available surface format.
         * @param availableFormats The available surface formats.
         * @returns The best surface format.
         */
        VkSurfaceFormatKHR pickSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> &availableFormats) {
            // Format - VK_FORMAT_B8G8R8A8_SRGB: BGRA color stored in 8 bit unsigned integer for a total of 32 bits per pixel
            // Color space - VK_COLOR_SPACE_SRGB_NONLINEAR_KHR: SRGB format for more accurately perceived colors
            for (const auto &format : availableFormats) {
                if (format.format == Config::SURFACE_FORMAT && format.colorSpace == Config::SURFACE_COLOR_SPACE) {
                    return format;
                }
            }

            // If not format matches the above, return the first format
            return availableFormats[0];
        }

        /**
         * Picks the best available present mode.
         * @param availablePresentModes The available surface present modes.
         * @returns The best present mode, or the default FIFO present mode.
         */
        VkPresentModeKHR pickSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
            for (const auto &present_mode : availablePresentModes) {
                if (present_mode == Config::PRESENT_MODE) {
                    return present_mode;
                }
            }

            // Return FIFO mode which is guaranteed to be available if preferred mode is unavailable
            return VK_PRESENT_MODE_FIFO_KHR;
        }

        /**
         * Picks the best available swap extent (resolution of the swap chain images).
         */
        VkExtent2D pickSwapExtent(const VkSurfaceCapabilitiesKHR &capabilities) {
            // If currentExtent dimension is defined, used that
            if (capabilities.currentExtent.width != (std::numeric_limits<uint32_t>::max)()) {
                std::cout << "resolution: (" << capabilities.currentExtent.width << ", " << capabilities.currentExtent.height << ")\n";
                return capabilities.currentExtent;
            } else {
                // Grab the actual resolution from GLFW
                int width, height;
                glfwGetFramebufferSize(window, &width, &height);
                VkExtent2D actual_extent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

                actual_extent.width = std::clamp(actual_extent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                actual_extent.height = std::clamp(actual_extent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

                std::cout << "resolution: (" << actual_extent.width << ", " << actual_extent.height << ")\n";
                return actual_extent;
            }
        }

        /**
         * Checks the device supports the required extensions listed in deviceExtensions variables.
         * @param device The physical device.
         * @returns True if the required extensions are supported.
         */
        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extension_count;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, nullptr);

            std::vector<VkExtensionProperties> exension_list(extension_count);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_count, exension_list.data());

            // Create temp list of required extensions
            std::set<std::string> required_extensions(Config::DEVICE_EXTENSIONS.begin(), Config::DEVICE_EXTENSIONS.end());
            for (const auto &extension : exension_list) {
                required_extensions.erase(extension.extensionName);
            }

            return required_extensions.empty();
        }

        /**
         * Find queue families for the specified GPU.
         * @param device The GPU device.
         * @returns The found queue familiy indices.
         */
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices;

            uint32_t queue_family_count = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, nullptr);

            std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queue_family_count, queue_families.data());

            int i = 0;
            for (const auto &queue_family : queue_families) {
                // Check queue family supports graphics
                // Check queueFamilyCount > 1 so Intel GPU does no get picked
                if (queue_family.queueFlags & Config::QUEUE_FLAGS && queue_family_count > 1) {
                    indices.graphicsFamily = i;
                }

                // Check device supports window presentation
                VkBool32 present_support = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present_support);
                if (present_support) {
                    indices.presentFamily = i;
                }

                if (indices.isComplete()) {
                    break;
                }

                i++;
            }

            // Logic to find queue family indices to populate struct with
            return indices;
        }

        /**
         * Checks the extensions required by GLFW are available.
         * @param requiredExtensions The required extensions.
         * @param requiredCount The number of required extensions.
         * @returns True if the extensions are available, false otherwise.
         */
        bool checkGlfwExtensionsAvailability(const char **requiredExtensions, uint32_t requiredCount) {
            // Retrieve available extensions count and list
            uint32_t extension_count = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);

            std::vector<VkExtensionProperties> extensions(extension_count);
            vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extensions.data());

            uint32_t available_count = 0;
            std::cout << "available VK extensions (" << extension_count << "):\n";

            // Check if required extension exists
            for (const auto &extension : extensions) {
                for (int i = 0; i < requiredCount; i++) {
                    if (strcmp(requiredExtensions[i], extension.extensionName) == 0) available_count++;
                }

                std::cout << '\t' << extension.extensionName << '\n';
            }

            return available_count == requiredCount;
        }

        /**
         * Fetches and returns the list of GLFW required extensions.
         * @returns The list of required extensions.
         */
        std::vector<const char *> getRequiredExtensions() {
            uint32_t glfw_extension_count = 0;
            const char **glfw_extensions;
            glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);

            // Abort if required extensions are unavailable
            if (!checkGlfwExtensionsAvailability(glfw_extensions, glfw_extension_count)) {
                throw std::runtime_error("error: required GLFW extensions are not available!");
            };

            std::vector<const char *> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

            // Manually add the validation layer extension
            if (Config::ENABLE_VALIDATION_LAYERS) {
                extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
            }

            return extensions;
        }

        /**
         * Checks support for VK validation layers.
         * @returns True if validation layers are supported.
         */
        bool checkVKValidationLayerSupport() {
            uint32_t layer_count = 0;
            vkEnumerateInstanceLayerProperties(&layer_count, nullptr);

            std::vector<VkLayerProperties> available_layers(layer_count);
            vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

            std::cout << "available VK validation layers (" << layer_count << "):\n";
            for (const auto &layer : available_layers) {
                std::cout << '\t' << layer.layerName << '\n';
            }

            // Check required layer is available
            for (const char *layer_name : Config::VALIDATION_LAYERS) {
                bool layer_found = false;

                for (const auto &layer_properties : available_layers) {
                    if (strcmp(layer_name, layer_properties.layerName) == 0) {
                        layer_found = true;
                        break;
                    }
                }

                if (!layer_found) return false;
            }

            return true;
        }

        /**
         * Creates the vulkan instance.
         */
        void createVkInstance() {
            // Create VK application info structure
            VkApplicationInfo app_info{};
            app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            app_info.pApplicationName = "Vulkan Template";
            app_info.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
            app_info.pEngineName = "No Engine";
            app_info.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
            app_info.apiVersion = VK_API_VERSION_1_0;

            // Create info structure
            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &app_info;

            // Add validation layer if enabled
            if (Config::ENABLE_VALIDATION_LAYERS) {
                if (!checkVKValidationLayerSupport()) {
                    throw std::runtime_error("error: required validation layer not available!");
                }

                create_info.enabledLayerCount = static_cast<uint32_t>(Config::VALIDATION_LAYERS.size());
                create_info.ppEnabledLayerNames = Config::VALIDATION_LAYERS.data();

                // Add validation debug callback for instanciationg
                VkDebugUtilsMessengerCreateInfoEXT debugcreate_info{};
                populateVkDebugMessengerCreateInfo(debugcreate_info);
                create_info.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugcreate_info;
            } else {
                create_info.enabledLayerCount = 0;
                create_info.pNext = nullptr;
            }

            // Get the extensions required to interface with the window system
            auto extensions = getRequiredExtensions();
            create_info.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
            create_info.ppEnabledExtensionNames = extensions.data();

            // Abort if VK instance cannot be created
            if (vkCreateInstance(&create_info, nullptr, &vk_instance) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create instance!");
            }
        }

        /**
         * Debug callback function for Vulkan.
         */
        static VKAPI_ATTR VkBool32 VKAPI_CALL vkDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                              const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData) {
            std::cerr << "[validation_layer] " << pCallbackData->pMessage << std::endl;
            return VK_FALSE;
        }

        void populateVkDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &create_info) {
            create_info = {};
            create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
            create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
            create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
            create_info.pfnUserCallback = vkDebugCallback;
        }

        /**
         * Creates a debug messenger using the vkDebugCallback function.
         */
        void createVkDebugMessenger() {
            if (!Config::ENABLE_VALIDATION_LAYERS) return;

            VkDebugUtilsMessengerCreateInfoEXT create_info{};
            populateVkDebugMessengerCreateInfo(create_info);

            if (CreateDebugUtilsMessengerEXT(vk_instance, &create_info, nullptr, &vk_debug_messenger) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to set up debug messenger!");
            }
        }

        /**
         * Creates the render pass.
         */
        void createRenderPass() {
            // Single color buffer attachment represented by one of the images in the swap chain
            VkAttachmentDescription color_attachment{};
            color_attachment.format = swapchain_format;
            color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
            color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;    // Clear the buffer before rendering
            color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;  //  Rendered contents will be stored in memory and can be read later
            color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
            color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
            color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;  // We want the image to be ready for presentation after rendering

            VkAttachmentReference color_attachment_ref{};
            color_attachment_ref.attachment = 0;  // We only have a single attachment description so the index is 0
            color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

            // Create render subpass using attachment reference
            VkSubpassDescription subpass{};
            subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
            subpass.colorAttachmentCount = 1;
            subpass.pColorAttachments = &color_attachment_ref;

            // Create subpass dependency
            VkSubpassDependency dependency{};
            dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
            dependency.dstSubpass = 0;
            // Specify the operations to wait on. Waiting on the color attachment output stage means we wait for the swap chain to finish reading from the image before we access it
            dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.srcAccessMask = 0;
            dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
            dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

            VkRenderPassCreateInfo render_pass_info{};
            render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
            render_pass_info.attachmentCount = 1;
            render_pass_info.pAttachments = &color_attachment;
            render_pass_info.subpassCount = 1;
            render_pass_info.pSubpasses = &subpass;
            render_pass_info.dependencyCount = 1;
            render_pass_info.pDependencies = &dependency;

            if (vkCreateRenderPass(device, &render_pass_info, nullptr, &render_pass) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create render pass!");
            }
        }

        /**
         * Main render loop.
         */
        void renderLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();

                drawFrame();
            }

            // Wait for logical device to finish operations before exiting
            vkDeviceWaitIdle(device);
        }

        // https://vulkan-tutorial.com/en/Drawing_a_triangle/Drawing/Frames_in_flight

        /**
         * Creates the synchronisation objects used to synchronise vulkan API calls order.
         * Sephamores are used to wait on swapchain operations because they happen on the GPU.
         * Fences are used to wait for the frame to finish, because we need the CPU to wait.
         */
        void createSynchronisationObjects() {
            img_available_semaphores.resize(Config::MAX_FRAMES_IN_FLIGHT);
            render_finished_semaphores.resize(Config::MAX_FRAMES_IN_FLIGHT);
            inflight_fences.resize(Config::MAX_FRAMES_IN_FLIGHT);

            VkSemaphoreCreateInfo semaphore_info{};
            semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

            VkFenceCreateInfo fence_info{};
            fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;  // Create fence already signaled so the first draw call does not block indefinitely

            for (size_t i = 0; i < Config::MAX_FRAMES_IN_FLIGHT; i++) {
                if (vkCreateSemaphore(device, &semaphore_info, nullptr, &img_available_semaphores[i]) != VK_SUCCESS ||
                    vkCreateSemaphore(device, &semaphore_info, nullptr, &render_finished_semaphores[i]) != VK_SUCCESS ||
                    vkCreateFence(device, &fence_info, nullptr, &inflight_fences[i]) != VK_SUCCESS) {
                    throw std::runtime_error("error: failed to create sync objects!");
                }
            }
        }

        /**
         * Draws a frame. Steps to follow with vulkan:
         * 1) wait for the previous frame to finish
         * 2) acquire an image from the swap chain
         * 3) record a command buffer which draws the scene onto that image
         * 4) submit the recorded command buffer
         * 5) present the swap chain image
         * Note: Vulkan API calls to the GPU are usually asynchronous:
         * - Acquiring an image from the swap chain
         * - executing commands to draw onto the acquired image
         * - presenting that image to the screen, returning it to the swapchain
         * This means we need to control the order of the functions using semaphore, because each call relies on the previous finishing.
         */
        void drawFrame() {
            // At the start of the frame, wait until the previous frame has finished using the fence
            // VK_TRUE to indicate we want to wait for all fences (only one fence here so doesn't matter)
            // UINT64_MAX to disable the timeout
            vkWaitForFences(device, 1, &inflight_fences[frame_index], VK_TRUE, UINT64_MAX);

            // Acquire an image from the swapchain
            uint32_t img_index;
            VkResult result = vkAcquireNextImageKHR(device, swapchain, UINT64_MAX, img_available_semaphores[frame_index], VK_NULL_HANDLE, &img_index);

            // Recreate swapchain if out of date
            if (result == VK_ERROR_OUT_OF_DATE_KHR) {
                recreateSwapChain();
                return;
            } else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR) {  // VK_SUBOPTIMAL_KHR: the swapchain can still be used to present but the surface properties are no longer exactly matched
                throw std::runtime_error("error: failed to acquire swapchain image!");
            }

            // After waiting, we reset the fence to unsignaled state (only if we are submitting a command buffer)
            vkResetFences(device, 1, &inflight_fences[frame_index]);

            // Reset the command buffer to make sure it can be recorded
            vkResetCommandBuffer(command_buffers[frame_index], 0);

            // Record our predefined command into the command buffer
            recordCommandBuffer(command_buffers[frame_index], img_index);

            // Prepare for submitting the command buffer
            VkSubmitInfo submit_info{};
            submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;

            VkSemaphore wait_semaphores[] = {img_available_semaphores[frame_index]};
            VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
            // Specify which semaphores to wait on before execution begins
            submit_info.waitSemaphoreCount = 1;
            submit_info.pWaitSemaphores = wait_semaphores;
            submit_info.pWaitDstStageMask = wait_stages;
            // Specify which command buffer to submit
            submit_info.commandBufferCount = 1;
            submit_info.pCommandBuffers = &command_buffers[frame_index];

            // Specify which semaphores to signal once command buffer execution has finished
            VkSemaphore signal_semaphores[] = {render_finished_semaphores[frame_index]};
            submit_info.signalSemaphoreCount = 1;
            submit_info.pSignalSemaphores = signal_semaphores;

            // Submit command buffer to the graphics queu
            if (vkQueueSubmit(graphics_queue, 1, &submit_info, inflight_fences[frame_index]) != VK_SUCCESS) {
                throw std::runtime_error("failed to submit draw command buffer!");
            }

            // Configure presentation so we can show the image on the screen
            VkPresentInfoKHR present_info{};
            present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

            // Specify which semaphores to wait on before presentation happens
            present_info.waitSemaphoreCount = 1;
            present_info.pWaitSemaphores = signal_semaphores;

            // Specify the swapchains to present images to and the index of the image for each swap chain
            VkSwapchainKHR swap_chains[] = {swapchain};
            present_info.swapchainCount = 1;
            present_info.pSwapchains = swap_chains;
            present_info.pImageIndices = &img_index;
            present_info.pResults = nullptr;

            // Submit request to present an image to the swap chain
            result = vkQueuePresentKHR(present_queue, &present_info);

            if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR || framebuffer_resized) {
                // Reset resized flag
                framebuffer_resized = false;
                recreateSwapChain();
            } else if (result != VK_SUCCESS) {
                throw std::runtime_error("error: failed to present swapchain image!");
            }

            // Update current frame index
            frame_index = (frame_index + 1) % Config::MAX_FRAMES_IN_FLIGHT;
        }

        /**
         * Clean-up: destroy VK instance and GLFW window.
         */
        void cleanup() {
            cleanupSwapChain();

            vkDestroyPipeline(device, graphics_pipeline, nullptr);
            vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
            vkDestroyRenderPass(device, render_pass, nullptr);

            for (size_t i = 0; i < Config::MAX_FRAMES_IN_FLIGHT; i++) {
                vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
                vkDestroySemaphore(device, img_available_semaphores[i], nullptr);
                vkDestroyFence(device, inflight_fences[i], nullptr);
            }

            vkDestroyCommandPool(device, command_pool, nullptr);
            vkDestroyDevice(device, nullptr);

            if (Config::ENABLE_VALIDATION_LAYERS) {
                DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);
            }

            vkDestroySurfaceKHR(vk_instance, surface, nullptr);
            vkDestroyInstance(vk_instance, nullptr);

            glfwDestroyWindow(window);
            glfwTerminate();
        }
};

int main() {
    VulkanTemplateApp app;

    try {
        app.run();
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}