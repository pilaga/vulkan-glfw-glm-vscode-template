#ifndef _CONFIG_H_
#define _CONFIG_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>
#include <vector>

/**
 * Used for storing the application configuration.
 */
class Config {
    public:
        static uint32_t WIDTH;   // The window width
        static uint32_t HEIGHT;  // The window height

        static std::vector<const char *> DEVICE_EXTENSIONS;  // The required device extension

        static VkFormat SURFACE_FORMAT;              // The surface format
        static VkColorSpaceKHR SURFACE_COLOR_SPACE;  // The surface color space

        static VkPresentModeKHR PRESENT_MODE;  // The present mode
        static VkQueueFlags QUEUE_FLAGS;       // The queue flags

        static std::string SHADERS_PATH;

        static VkBool32 SWAPCHAIN_CLIPPED;                             // True to enable clipping
        static VkImageUsageFlags SWAPCHAIN_IMAGE_USAGE;                // The swap chain image usage
        static VkCompositeAlphaFlagBitsKHR SWAPCHAIN_COMPOSITE_ALPHA;  // The swap chain composite alpha

        static bool ENABLE_VALIDATION_LAYERS;                // True to enable validation layers
        static std::vector<const char *> VALIDATION_LAYERS;  // The required validation layers

        static int MAX_FRAMES_IN_FLIGHT;  // The number of frames to be processed concurrently
};

#endif  // _CONFIG_H_
