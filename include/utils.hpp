#ifndef UTILS_H
#define UTILS_H

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include "../include/optional.hpp"

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
    Optional<uint32_t> graphicsFamily;

    bool isComplete() { return graphicsFamily.has_value(); }
};

#endif  // UTILS_H