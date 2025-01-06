#include "..\\include\\config.hpp"

uint32_t Config::WIDTH = 800;
uint32_t Config::HEIGHT = 600;

int Config::MAX_FRAMES_IN_FLIGHT = 2;

std::string Config::SHADERS_PATH = "../shaders/";
std::string Config::TEXTURES_PATH = "../textures/";

std::vector<const char *> Config::DEVICE_EXTENSIONS = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VkFormat Config::SURFACE_FORMAT = VK_FORMAT_B8G8R8A8_SRGB;
VkColorSpaceKHR Config::SURFACE_COLOR_SPACE = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;

VkPresentModeKHR Config::PRESENT_MODE = VK_PRESENT_MODE_MAILBOX_KHR;  // VK_PRESENT_MODE_MAILBOX_KHR is a variation of the FIFO mode where images in the queue get replaced if the queue is already full

VkQueueFlags Config::QUEUE_FLAGS = VK_QUEUE_GRAPHICS_BIT;

VkBool32 Config::SWAPCHAIN_CLIPPED = VK_TRUE;
VkImageUsageFlags Config::SWAPCHAIN_IMAGE_USAGE = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
VkCompositeAlphaFlagBitsKHR Config::SWAPCHAIN_COMPOSITE_ALPHA = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

std::vector<const char *> Config::VALIDATION_LAYERS = {"VK_LAYER_KHRONOS_validation"};

// Only enable validation layers in debug mode
#ifdef NDEBUG
bool Config::ENABLE_VALIDATION_LAYERS = false;
#else
bool Config::ENABLE_VALIDATION_LAYERS = true;
#endif
