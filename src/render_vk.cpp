#include "asset_loader.hpp"
#include "graphics.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <Vulkan/vulkan.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
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
    VkSwapchainKHR swapchain;
    std::vector<VkImageView> image_views{};

    ~VulkanResources() {
        for (auto image_view : image_views) {
            vkDestroyImageView(device, image_view, nullptr);
        }
        vkDestroySwapchainKHR(device, swapchain, nullptr);
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
const std::vector<const char*> required_device_extensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VkPhysicalDevice physical_device = VK_NULL_HANDLE;
uint32_t graphics_family;
VkQueue graphics_queue = VK_NULL_HANDLE;
VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;
VkColorSpaceKHR swapchain_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
VkPresentModeKHR present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
VkExtent2D swapchain_extend{};
uint32_t swapchain_min_image_count{};
std::vector<VkImage> images{};
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

    uint32_t extension_count;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device, nullptr,
                                                  &extension_count, nullptr));
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(
        device, nullptr, &extension_count, available_extensions.data()));
    uint32_t required_extensions_found = 0;
    for (auto extension : available_extensions) {
        for (auto required : Engine_VK::required_device_extensions) {
            if (std::strcmp(extension.extensionName, required) == 0) {
                ++required_extensions_found;
            }
        }
    }
    if (required_extensions_found <
        Engine_VK::required_device_extensions.size()) {
        return -1;
    }
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
    // for (auto i = 0; i < physical_device_count; ++i) {
    for (auto physical_device : physical_devices) {
        auto score = physical_device_score(physical_device);
        if (score <= 0 || score < best_device_score) {
            continue;
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count, nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(
            physical_device, &queue_family_count, queue_families.data());

        auto has_graphics = false;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            VkBool32 present_suport{};

            VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(
                physical_device, i, vulkan_data.surface, &present_suport));

            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
                present_suport == VK_TRUE) {
                Engine_VK::graphics_family = i;
                has_graphics = true;
                break;
            }
        }
        if (!has_graphics) {
            continue;
        }
        if (score > best_device_score) {
            best_device_score = score;
            Engine_VK::physical_device = physical_device;
        }
    }
    ASSERT(Engine_VK::physical_device, "No graphics device selected!");

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
    create_info.enabledExtensionCount =
        Engine_VK::required_device_extensions.size();
    create_info.ppEnabledExtensionNames =
        Engine_VK::required_device_extensions.data();
    create_info.pEnabledFeatures = &device_features;

    VK_CHECK(vkCreateDevice(Engine_VK::physical_device, &create_info, nullptr,
                            &vulkan_data.device));

    vkGetDeviceQueue(vulkan_data.device, Engine_VK::graphics_family, 0,
                     &Engine_VK::graphics_queue);
}

void create_swapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(
        Engine_VK::physical_device, vulkan_data.surface, &capabilities));
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        Engine_VK::swapchain_extend = capabilities.currentExtent;
    } else {
        int width, height;
        SDL_Vulkan_GetDrawableSize(vulkan_data.window, &width, &height);

        Engine_VK::swapchain_extend.width = std::clamp(
            static_cast<uint32_t>(width), capabilities.minImageExtent.width,
            capabilities.maxImageExtent.width);
        Engine_VK::swapchain_extend.width = std::clamp(
            static_cast<uint32_t>(height), capabilities.minImageExtent.height,
            capabilities.maxImageExtent.height);
    }

    Engine_VK::swapchain_min_image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        Engine_VK::swapchain_min_image_count > capabilities.maxImageCount) {
        Engine_VK::swapchain_min_image_count = capabilities.maxImageCount;
    }

    uint32_t format_count;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(Engine_VK::physical_device,
                                                  vulkan_data.surface,
                                                  &format_count, nullptr));
    std::vector<VkSurfaceFormatKHR> surface_formats(format_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(
        Engine_VK::physical_device, vulkan_data.surface, &format_count,
        surface_formats.data()));
    bool found_format = false;
    for (auto surface_format : surface_formats) {
        if (surface_format.format == Engine_VK::swapchain_format &&
            surface_format.colorSpace == Engine_VK::swapchain_color_space) {
            found_format = true;
            break;
        }
    }
    if (!found_format) {
        Engine_VK::swapchain_format = surface_formats[0].format;
        Engine_VK::swapchain_color_space = surface_formats[0].colorSpace;
    }

    uint32_t present_mode_count;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
        Engine_VK::physical_device, vulkan_data.surface, &present_mode_count,
        nullptr));
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(
        Engine_VK::physical_device, vulkan_data.surface, &present_mode_count,
        present_modes.data()));
    bool present_found = false;
    for (auto present_mode : present_modes) {
        if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_found = true;
            break;
        }
    }
    if (!present_found) {
        Engine_VK::present_mode = VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    // create_info.pNext;
    // create_info.flags;
    create_info.surface = vulkan_data.surface;
    create_info.minImageCount = Engine_VK::swapchain_min_image_count;
    create_info.imageFormat = Engine_VK::swapchain_format;
    create_info.imageColorSpace = Engine_VK::swapchain_color_space;
    create_info.imageExtent = Engine_VK::swapchain_extend;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = Engine_VK::present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(vulkan_data.device, &create_info, nullptr,
                                  &vulkan_data.swapchain));

    uint32_t image_count;
    vkGetSwapchainImagesKHR(vulkan_data.device, vulkan_data.swapchain,
                            &image_count, nullptr);
    Engine_VK::images.resize(image_count);
    vkGetSwapchainImagesKHR(vulkan_data.device, vulkan_data.swapchain,
                            &image_count, Engine_VK::images.data());

    vulkan_data.image_views.resize(image_count);
    for (auto i = 0; i < image_count; ++i) {
        VkImageViewCreateInfo create_info{};
        create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        // create_info.pNext;
        // create_info.flags;
        create_info.image = Engine_VK::images[i];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = Engine_VK::swapchain_format;
        create_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0;
        create_info.subresourceRange.levelCount = 1;
        create_info.subresourceRange.baseArrayLayer = 0;
        create_info.subresourceRange.layerCount = 1;
        VK_CHECK(vkCreateImageView(vulkan_data.device, &create_info, nullptr,
                                   &vulkan_data.image_views[i]));
    }
}

VkShaderModule create_shader_module(const std::vector<std::byte>& spv_code) {
    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    // create_info.pNext;
    // create_info.flags;
    create_info.codeSize = spv_code.size();
    create_info.pCode = reinterpret_cast<const uint32_t*>(spv_code.data());

    VkShaderModule shader_module;
    VK_CHECK(vkCreateShaderModule(vulkan_data.device, &create_info, nullptr,
                                  &shader_module));
    return shader_module;
}

void create_graphics_pipeline() {
    auto vert_shader_code = read_file("shaders/shader.vert.spv");
    auto frag_shader_code = read_file("shaders/shader.frag.spv");

    VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
    VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

    vkDestroyShaderModule(vulkan_data.device, vert_shader_module, nullptr);
    vkDestroyShaderModule(vulkan_data.device, frag_shader_module, nullptr);

    VkPipelineShaderStageCreateInfo vert_shader_stage_info;
    vert_shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // vert_shader_stage_info.pNext;
    // vert_shader_stage_info.flags;
    vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module;
    vert_shader_stage_info.pName = "main";
    // vert_shader_stage_info.pSpecializationInfo;

    VkPipelineShaderStageCreateInfo frag_shader_stage_info;
    frag_shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // frag_shader_stage_info.pNext;
    // frag_shader_stage_info.flags;
    frag_shader_stage_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_shader_stage_info.module = frag_shader_module;
    frag_shader_stage_info.pName = "main";
    // frag_shader_stage_info.pSpecializationInfo;

    VkPipelineShaderStageCreateInfo shader_stages[] = {vert_shader_stage_info,
                                                       frag_shader_stage_info};
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
    create_swapchain();
    create_graphics_pipeline();
}

void draw() {}

} // namespace graphics