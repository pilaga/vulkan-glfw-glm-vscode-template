#ifndef _VERTEX_H_
#define _VERTEX_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <fstream>
#include <glm/glm.hpp>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

/**
 * Vertex input attribute and binding description.
 */
struct VertexInputDescription {
        glm::vec2 pos;
        glm::vec3 color;

        // Vertex binding describes at which rate to load data from memory through the vertices
        static VkVertexInputBindingDescription getBindingDescription() {
            VkVertexInputBindingDescription binding_desc{};
            binding_desc.binding = 0;
            binding_desc.stride = sizeof(VertexInputDescription);
            binding_desc.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;  // Move to the next data point after each vertex

            return binding_desc;
        }

        // Vertex input attributes describes the format of a vertex within a data chunk
        // A vertex consists of 2 data points (position and color) so we initialize a list
        static std::array<VkVertexInputAttributeDescription, 2> getAttributeDescriptions() {
            std::array<VkVertexInputAttributeDescription, 2> attribute_desc_list{};

            // Attribute description for position
            attribute_desc_list[0].binding = 0;
            attribute_desc_list[0].location = 0;                      // Location directive of the input in the vertex shader
            attribute_desc_list[0].format = VK_FORMAT_R32G32_SFLOAT;  // Format matching vec2
            attribute_desc_list[0].offset = offsetof(VertexInputDescription, pos);

            // Attribute description for color
            attribute_desc_list[1].binding = 0;
            attribute_desc_list[1].location = 1;                         // Location in the vertex shader
            attribute_desc_list[1].format = VK_FORMAT_R32G32B32_SFLOAT;  // Format matching vec3
            attribute_desc_list[1].offset = offsetof(VertexInputDescription, color);

            return attribute_desc_list;
        }
};

#endif  // _VERTEX_H_
