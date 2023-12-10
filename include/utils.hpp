#ifndef _UTILS_H_
#define _UTILS_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

/**
 * Creates the debug messenger extesion for Vulkan.
 * @param instance The VK instance.
 * @param pCreateInfo Pointer to the create info for the debug messenger.
 * @param pAllocator Leave nullptr for now.
 * @param pDebugMessenger The pointer to VK debug messenger.
 */
VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pDebugMessenger);

/**
 * Destroys the debug messenger extesion for Vulkan.
 * @param instance The VK instance.
 * @param debugMessenger The VK debug messenger.
 * @param pAllocator Leave nullptr for now.
 */
void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks *pAllocator);

/**
 * Struct used to query VK queue families.
 */
struct QueueFamilyIndices {
        std::optional<uint32_t> graphicsFamily;
        std::optional<uint32_t> presentFamily;

        bool isComplete() { return graphicsFamily.has_value() && presentFamily.has_value(); }
};

/**
 * Struct used to store swap chain support details.
 */
struct SwapChainSupportDetails {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
};

/**
 * Reads a file from provided filename/path.
 */
std::vector<char> readFile(const std::string &filename);

#endif  // _UTILS_H_
