#ifndef _CONFIG_H_
#define _CONFIG_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>

/**
 * Used for storing application configuration.
 */
class Config {
    public:
        // The window resolution (width, height)
        static uint32_t WIDTH;
        static uint32_t HEIGHT;

        static VkFormat SURFACE_FORMAT;
        static VkColorSpaceKHR SURFACE_COLOR_SPACE;

        static VkPresentModeKHR PRESENT_MODE;
        static VkQueueFlags QUEUE_FLAGS;

        static VkBool32 SWAPCHAIN_CLIPPED;
        static VkImageUsageFlags SWAPCHAIN_IMAGE_USAGE;
        static VkCompositeAlphaFlagBitsKHR SWAPCHAIN_COMPOSITE_ALPHA;
};

#endif  // _CONFIG_H_
