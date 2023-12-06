#ifndef _APP_CONFIG_H_
#define _APP_CONFIG_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>

/**
 * Used for storing application configuration.
 */
class AppConfig {
    public:
        // The window resolution (width, height)
        static uint32_t WIDTH;
        static uint32_t HEIGHT;
};

#endif  // _APP_CONFIG_H_
