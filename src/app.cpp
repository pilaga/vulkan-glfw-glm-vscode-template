#define VK_USE_PLATFORM_WIN32_KHR
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3native.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#include "../include/config.hpp"
#include "../include/utils.hpp"

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

        /**
         * Initializes the GLFW window.
         */
        void initGlfwWindow() {
            glfwInit();

            glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Tell GLFW not to create a GL context
            glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Disable window resizing

            window = glfwCreateWindow(Config::WIDTH, Config::HEIGHT, "Vulkan Template", nullptr, nullptr);
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
        }

        void createSurface() {
            if (glfwCreateWindowSurface(vk_instance, window, nullptr, &surface) != VK_SUCCESS) {
                throw std::runtime_error("error: failed to create window surface!");
            }
        }

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

            // Specify device features, leave empty for now as we don't need anything specific
            VkPhysicalDeviceFeatures deviceFeatures{};

            // Create info for the logical device
            VkDeviceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
            create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
            create_info.pQueueCreateInfos = queue_create_infos.data();
            create_info.pEnabledFeatures = &deviceFeatures;
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
            uint32_t imageCount = swapchain_support.capabilities.minImageCount + 1;

            // Make sure we don't go over the maxImageCount for the swap chain
            // 0 is a special value that means there is no max
            if (swapchain_support.capabilities.maxImageCount > 0 && imageCount > swapchain_support.capabilities.maxImageCount) {
                imageCount = swapchain_support.capabilities.maxImageCount;
            }

            VkSwapchainCreateInfoKHR create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
            create_info.surface = surface;
            create_info.minImageCount = imageCount;
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
            vkGetSwapchainImagesKHR(device, swapchain, &imageCount, nullptr);
            swapchain_images.resize(imageCount);
            vkGetSwapchainImagesKHR(device, swapchain, &imageCount, swapchain_images.data());

            // Store format and extent for later use
            swapchain_extent = extent;
            swapchain_format = surface_format.format;
        }

        /**
         * Picks the first GPU that provides VK support and assigns its handle to <physicalDevice> class member.
         */
        void pickGPU() {
            uint32_t deviceCount = 0;
            vkEnumeratePhysicalDevices(vk_instance, &deviceCount, nullptr);

            if (deviceCount == 0) {
                throw std::runtime_error("error: could not find a GPU with VK support!");
            }

            std::vector<VkPhysicalDevice> devices(deviceCount);
            vkEnumeratePhysicalDevices(vk_instance, &deviceCount, devices.data());

            for (const auto &device : devices) {
                if (isGPUSuitable(device)) {
                    // Grab basic device properties
                    VkPhysicalDeviceProperties deviceProperties;
                    vkGetPhysicalDeviceProperties(device, &deviceProperties);
                    std::cout << "selected GPU: " << deviceProperties.deviceName << "\n";

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
            bool extensionsSupported = checkDeviceExtensionSupport(device);

            // Check swap chain support is adequate
            bool swapChainAdequate = false;
            if (extensionsSupported) {
                SwapChainSupportDetails swapChainSupport = querySwapChainSupport(device);
                swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
            }

            return indices.isComplete() && extensionsSupported && swapChainAdequate;
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
            uint32_t formatCount;
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

            if (formatCount != 0) {
                details.formats.resize(formatCount);
                vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
            }

            // Query the supported presentation modes
            uint32_t presentModeCount;
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

            if (presentModeCount != 0) {
                details.presentModes.resize(presentModeCount);
                vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
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
            for (const auto &availableFormat : availableFormats) {
                if (availableFormat.format == Config::SURFACE_FORMAT && availableFormat.colorSpace == Config::SURFACE_COLOR_SPACE) {
                    return availableFormat;
                }
            }

            // If not format matches the above, return the first format
            return availableFormats[0];
        }

        /**
         * Picks the best available present mode.
         * @param availablePresentModes The available surface present modes.
         * @returns The best present mode.
         */
        VkPresentModeKHR pickSwapPresentMode(const std::vector<VkPresentModeKHR> &availablePresentModes) {
            for (const auto &availablePresentMode : availablePresentModes) {
                if (availablePresentMode == Config::PRESENT_MODE) {
                    return availablePresentMode;
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
                VkExtent2D actualExtent = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};

                actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
                actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);

                std::cout << "resolution: (" << actualExtent.width << ", " << actualExtent.height << ")\n";
                return actualExtent;
            }
        }

        /**
         * Checks the device supports the required extensions listed in deviceExtensions variables.
         * @param device The physical device.
         * @returns True if the required extensions are supported.
         */
        bool checkDeviceExtensionSupport(VkPhysicalDevice device) {
            uint32_t extensionCount;
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

            // Create temp list of required extensions
            std::set<std::string> requiredExtensions(Config::DEVICE_EXTENSIONS.begin(), Config::DEVICE_EXTENSIONS.end());

            for (const auto &extension : availableExtensions) {
                requiredExtensions.erase(extension.extensionName);
            }

            return requiredExtensions.empty();
        }

        /**
         * Find queue families for the specified GPU.
         * @param device The GPU device.
         * @returns The found queue familiy indices.
         */
        QueueFamilyIndices findQueueFamilies(VkPhysicalDevice device) {
            QueueFamilyIndices indices;

            uint32_t queueFamilyCount = 0;
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, nullptr);

            std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
            vkGetPhysicalDeviceQueueFamilyProperties(device, &queueFamilyCount, queueFamilies.data());

            int i = 0;
            for (const auto &queueFamily : queueFamilies) {
                // Check queue family supports graphics
                // Check queueFamilyCount > 1 so Intel GPU does no get picked
                if (queueFamily.queueFlags & Config::QUEUE_FLAGS && queueFamilyCount > 1) {
                    indices.graphicsFamily = i;
                }

                // Check device supports window presentation
                VkBool32 presentSupport = false;
                vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &presentSupport);
                if (presentSupport) {
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
            uint32_t extensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

            std::vector<VkExtensionProperties> extensions(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

            uint32_t availableCount = 0;
            std::cout << "available VK extensions (" << extensionCount << "):\n";

            // Check if required extension exists
            for (const auto &extension : extensions) {
                for (int i = 0; i < requiredCount; i++) {
                    if (strcmp(requiredExtensions[i], extension.extensionName) == 0) availableCount++;
                }

                std::cout << '\t' << extension.extensionName << '\n';
            }

            return availableCount == requiredCount;
        }

        /**
         * Fetches and returns the list of GLFW required extensions.
         * @returns The list of required extensions.
         */
        std::vector<const char *> getRequiredExtensions() {
            uint32_t glfwExtensionCount = 0;
            const char **glfwExtensions;
            glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

            // Abort if required extensions are unavailable
            if (!checkGlfwExtensionsAvailability(glfwExtensions, glfwExtensionCount)) {
                throw std::runtime_error("error: required GLFW extensions are not available!");
            };

            std::vector<const char *> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

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
            uint32_t layerCount = 0;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

            std::cout << "available VK validation layers (" << layerCount << "):\n";
            for (const auto &availableLayer : availableLayers) {
                std::cout << '\t' << availableLayer.layerName << '\n';
            }

            // Check required layer is available
            for (const char *layerName : Config::VALIDATION_LAYERS) {
                bool layerFound = false;

                for (const auto &layerProperties : availableLayers) {
                    if (strcmp(layerName, layerProperties.layerName) == 0) {
                        layerFound = true;
                        break;
                    }
                }

                if (!layerFound) return false;
            }

            return true;
        }

        /**
         * Creates the vulkan instance.
         */
        void createVkInstance() {
            // Create VK application info structure
            VkApplicationInfo appInfo{};
            appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
            appInfo.pApplicationName = "Vulkan Template";
            appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
            appInfo.pEngineName = "No Engine";
            appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
            appInfo.apiVersion = VK_API_VERSION_1_0;

            // Create info structure
            VkInstanceCreateInfo create_info{};
            create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
            create_info.pApplicationInfo = &appInfo;

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
                throw std::runtime_error("failed to set up debug messenger!");
            }
        }

        /**
         * Main render loop.
         */
        void renderLoop() {
            while (!glfwWindowShouldClose(window)) {
                glfwPollEvents();
            }
        }

        /**
         * Clean-up: destroy VK instance and GLFW window.
         */
        void cleanup() {
            if (Config::ENABLE_VALIDATION_LAYERS) {
                DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);
            }

            for (auto img_view : swapchain_img_views) {
                vkDestroyImageView(device, img_view, nullptr);
            }

            vkDestroySwapchainKHR(device, swapchain, nullptr);
            vkDestroyDevice(device, nullptr);
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