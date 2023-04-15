#include "graphics.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <Vulkan/vulkan.h>
#include <cstdint>
#include <iostream>
#include <vector>
#include <vulkan/vulkan_core.h>

namespace {
#define ASSERT(assertion, errMsg)                                              \
    {                                                                          \
        if (!(assertion)) {                                                    \
            std::cerr << errMsg << std::endl;                                  \
            abort();                                                           \
        }                                                                      \
    }

#define SDL_CHECK(x)                                                           \
    {                                                                          \
        SDL_bool err = x;                                                      \
        if (err == SDL_FALSE) {                                                \
            std::cerr << "Detected SDL error: " << SDL_GetError()              \
                      << std::endl;                                            \
            abort();                                                           \
        }                                                                      \
    }

#define VK_CHECK(x)                                                            \
    {                                                                          \
        VkResult err = x;                                                      \
        if (err) {                                                             \
            std::cerr << "Detected Vulkan error: " << err << std::endl;        \
            abort();                                                           \
        }                                                                      \
    }

class VulkanResources {
  public:
    SDL_Window* window;
    VkInstance instance;
    VkSurfaceKHR surface;
    VkDevice device;

    ~VulkanResources() {
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

VulkanResources vulkan_data{};

namespace Engine_VK {
VkExtent2D windowExtent{1280, 720};
auto app_name = "Vulkan Game";
auto engine_name = "Andrei Game Engine";

VkPhysicalDevice physical_device;
uint32_t graphics_family;
VkQueue graphics_queue;
} // namespace Engine_VK

void init_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    // TODO load name from a game project
    app_info.pApplicationName = Engine_VK::app_name;
    // TODO load version from a game project
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = Engine_VK::engine_name;
    // TODO load version at compile time
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pNext = nullptr;
    // create_info.flags;
    create_info.pApplicationInfo = &app_info;
    // create_info.enabledLayerCount;
    // create_info.ppEnabledLayerNames;

    uint32_t sdl_extension_count;
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(vulkan_data.window,
                                               &sdl_extension_count, nullptr));
    std::vector<const char*> sdl_extensions(sdl_extension_count);
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(
        vulkan_data.window, &sdl_extension_count, sdl_extensions.data()));

    create_info.enabledExtensionCount = sdl_extension_count;
    create_info.ppEnabledExtensionNames = sdl_extensions.data();

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &vulkan_data.instance));
}

void create_surface() {
    SDL_CHECK(SDL_Vulkan_CreateSurface(vulkan_data.window, vulkan_data.instance,
                                       &vulkan_data.surface));
}

int32_t physical_device_score(VkPhysicalDevice device) {
    int32_t score = 0;

    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(device, &physical_device_properties);

    if (physical_device_properties.deviceType ==
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 50;
    }
    return score;
}

void init_device() {
    uint32_t physical_device_count;
    VK_CHECK(vkEnumeratePhysicalDevices(vulkan_data.instance,
                                        &physical_device_count, nullptr));
    ASSERT(physical_device_count > 0, "No graphics device found!")
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(
        vulkan_data.instance, &physical_device_count, physical_devices.data()));

    auto best_device_score = 0;
    auto best_device = 0;
    for (auto i = 0; i < physical_device_count; ++i) {
        auto score = physical_device_score(physical_devices[i]);

        if (score > best_device_score) {
            best_device = i;
            best_device_score = score;
        }
    }
    Engine_VK::physical_device = physical_devices[best_device];

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(Engine_VK::physical_device,
                                             &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);

    vkGetPhysicalDeviceQueueFamilyProperties(
        Engine_VK::physical_device, &queue_family_count, queue_families.data());

    auto has_graphics = false;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        VkBool32 present_suport{};

        VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(
            Engine_VK::physical_device, i, vulkan_data.surface,
            &present_suport));

        if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
            present_suport == VK_TRUE) {
            Engine_VK::graphics_family = i;
            has_graphics = true;
            break;
        }
    }
    ASSERT(has_graphics, "No graphics queue found!");

    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    // queue_create_info.pNext;
    // queue_create_info.flags;
    queue_create_info.queueFamilyIndex = Engine_VK::graphics_family;
    queue_create_info.queueCount = 1;
    float queue_priorities = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priorities;

    VkPhysicalDeviceFeatures device_features{};

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // create_info.pNext;
    // create_info.flags;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    // create_info.enabledLayerCount;
    // create_info.ppEnabledLayerNames;
    // create_info.enabledExtensionCount;
    // create_info.ppEnabledExtensionNames;
    create_info.pEnabledFeatures = &device_features;

    VK_CHECK(vkCreateDevice(Engine_VK::physical_device, &create_info, nullptr,
                            &vulkan_data.device));

    vkGetDeviceQueue(vulkan_data.device, Engine_VK::graphics_family, 0,
                     &Engine_VK::graphics_queue);
}

} // namespace

namespace graphics {
void init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    vulkan_data.window =
        SDL_CreateWindow("Vulkan Game Engine", SDL_WINDOWPOS_UNDEFINED,
                         SDL_WINDOWPOS_UNDEFINED, Engine_VK::windowExtent.width,
                         Engine_VK::windowExtent.height, SDL_WINDOW_VULKAN);

    init_instance();
    create_surface();
    init_device();
}

void draw() {}

} // namespace graphics
