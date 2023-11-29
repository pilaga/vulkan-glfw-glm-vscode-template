#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "../include/utils.hpp"

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

// Only enable validation layers in debug mode
#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

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

    /**
     * Initializes the GLFW window.
     */
    void initGlfwWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Tell GLFW not to create a GL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Disable window resizing

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Template", nullptr, nullptr);
    }

    /**
     * Initializes Vulkan, creates VK instance.
     */
    void initVulkan() {
        createVulkanInstance();
        createVkDebugMessenger();
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
        if (enableValidationLayers) {
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
        for (const char *layerName : validationLayers) {
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
    void createVulkanInstance() {
        // Create VK application info structure
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Template";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        // Create info structure
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Add validation layer if enabled
        if (enableValidationLayers) {
            if (!checkVKValidationLayerSupport()) {
                throw std::runtime_error("error: required validation layer not available!");
            }

            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();

            // Add validation debug callback for instanciationg
            VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
            populateVkDebugMessengerCreateInfo(debugCreateInfo);
            createInfo.pNext = (VkDebugUtilsMessengerCreateInfoEXT *)&debugCreateInfo;
        } else {
            createInfo.enabledLayerCount = 0;
            createInfo.pNext = nullptr;
        }

        // Get the extensions required to interface with the window system
        auto extensions = getRequiredExtensions();
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();

        // Abort if VK instance cannot be created
        if (vkCreateInstance(&createInfo, nullptr, &vk_instance) != VK_SUCCESS) {
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

    void populateVkDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT &createInfo) {
        createInfo = {};
        createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        createInfo.pfnUserCallback = vkDebugCallback;
    }

    /**
     * Creates a debug messenger using the vkDebugCallback function.
     */
    void createVkDebugMessenger() {
        if (!enableValidationLayers) return;

        VkDebugUtilsMessengerCreateInfoEXT createInfo{};
        populateVkDebugMessengerCreateInfo(createInfo);

        if (CreateDebugUtilsMessengerEXT(vk_instance, &createInfo, nullptr, &vk_debug_messenger) != VK_SUCCESS) {
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
        if (enableValidationLayers) {
            DestroyDebugUtilsMessengerEXT(vk_instance, vk_debug_messenger, nullptr);
        }

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