#ifndef _LOGICAL_DEVICE_MANAGER_H_
#define _LOGICAL_DEVICE_MANAGER_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>

/**
 * Manager class for VK logical device.
 */
class LogicalDeviceManager {
   private:
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;  // vk physical device

   public:
    LogicalDeviceManager() { std::cout << "Creating LogicalDeviceManager\n"; }
};

#endif  // _LOGICAL_DEVICE_MANAGER_H_
