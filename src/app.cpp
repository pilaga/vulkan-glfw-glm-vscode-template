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
            createGraphicsPipeline();
        }

        /**
         * Creates the graphics pipeline.
         */
        void createGraphicsPipeline() {}

        /**
         * Reads a file from provided filename/path.
         */
        static std::vector<char> readFile(const std::string &filename) {
            // Start reading at the end of file so we can use the read position to determine the size of the file
            std::ifstream file(filename, std::ios::ate | std::ios::binary);

            if (!file.is_open()) {
                throw std::runtime_error("failed to open file!");
            }

            size_t file_size = (size_t)file.tellg();

            // Create an populate buffer with file content
            std::vector<char> buffer(file_size);
            file.seekg(0);
            file.read(buffer.data(), file_size);

            file.close();
            return buffer;
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