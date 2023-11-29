#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

class VulkanTemplateApp {
   public:
    void run() {
        initWindow();
        initVulkan();
        renderLoop();
        cleanup();
    }

   private:
    GLFWwindow *window;
    VkInstance instance;

    void initWindow() {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);  // Tell GLFW not to create a GL context
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);    // Disable window resizing

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan Template", nullptr, nullptr);
    }

    void initVulkan() { createVulkanInstance(); }

    bool checkGlfwExtensionsAvailability(const char **requiredExtensions, uint32_t requiredCount) {
        // Retrieve available extensions count
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        // Retrieve available extensions list
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "available VK extensions (" << extensionCount << "):\n";

        uint32_t availableCount = 0;

        // Check whether required extension is in available extension list
        for (const auto &extension : extensions) {
            for (int i = 0; i < requiredCount; i++) {
                std::string s1 = requiredExtensions[i];
                std::string s2 = extension.extensionName;
                if (s1.compare(s2) == 0) availableCount++;
            }

            std::cout << '\t' << extension.extensionName << '\n';
        }

        return availableCount == requiredCount;
    }

    void createVulkanInstance() {
        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Vulkan Template";
        appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // Get the extensions required to interface with the window system
        uint32_t glfwExtensionCount = 0;
        const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Abort if required extensions are unavailable
        if (!checkGlfwExtensionsAvailability(glfwExtensions, glfwExtensionCount)) {
            throw std::runtime_error("error: required GLFW extensions are not available!");
        };

        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0;

        // Abort if VK instance cannot be created
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("error: failed to create instance!");
        }
    }

    void renderLoop() {
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();
        }
    }

    void cleanup() {
        vkDestroyInstance(instance, nullptr);

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