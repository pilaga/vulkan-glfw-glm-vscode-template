#ifndef _PHYSICAL_DEVICE_MANAGER_H_
#define _PHYSICAL_DEVICE_MANAGER_H_

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <iostream>

/**
 * Manager class for VK physical device.
 */
class PhysicalDeviceManager {
   private:
    VkDevice device;

   public:
    PhysicalDeviceManager() { std::cout << "Creating PhysicalDeviceManager\n"; }
};

#endif  // _PHYSICAL_DEVICE_MANAGER_H_
