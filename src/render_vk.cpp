#include "render_vk.hpp"

#include "asset_loader.hpp"
#include "graphics.hpp"
#include "mathlib.hpp"

#include <SDL.h>
#include <SDL_video.h>
#include <SDL_vulkan.h>
#include <Vulkan/vulkan.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <unordered_map>
#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>
#include <vector>
#include <vulkan/vk_platform.h>
#include <vulkan/vulkan_core.h>

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
            std::cerr << "Detected Vulkan error: " << vk_result_string(err)    \
                      << std::endl;                                            \
            abort();                                                           \
        }                                                                      \
    }

struct Vertex {
    math::vec3 pos;
    math::vec3 color;
    math::vec2 tex_coord;

    static VkVertexInputBindingDescription get_binding_description() {
        VkVertexInputBindingDescription binding_description{};
        binding_description.binding = 0;
        binding_description.stride = sizeof(Vertex);
        binding_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return binding_description;
    }

    static std::array<VkVertexInputAttributeDescription, 3>
    get_attribute_descriptions() {
        std::array<VkVertexInputAttributeDescription, 3>
            attribute_descriptions{};
        attribute_descriptions[0].location = 0;
        attribute_descriptions[0].binding = 0;
        attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[0].offset = offsetof(Vertex, pos);

        attribute_descriptions[1].location = 1;
        attribute_descriptions[1].binding = 0;
        attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attribute_descriptions[1].offset = offsetof(Vertex, color);

        attribute_descriptions[2].location = 2;
        attribute_descriptions[2].binding = 0;
        attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attribute_descriptions[2].offset = offsetof(Vertex, tex_coord);

        return attribute_descriptions;
    }

    bool operator==(const Vertex& other) const {
        return pos == other.pos && color == other.color &&
               tex_coord == other.tex_coord;
    }
};
namespace std {
template <> struct hash<Vertex> {
    size_t operator()(Vertex const& vertex) const {
        return ((hash<math::vec3>()(vertex.pos) ^
                 (hash<math::vec3>()(vertex.color) << 1)) >>
                1) ^
               (hash<math::vec2>()(vertex.tex_coord) << 1);
    }
};
} // namespace std
namespace {

struct UniformBufferObject {
    alignas(16) math::mat4 model;
    alignas(16) math::mat4 view;
    alignas(16) math::mat4 proj;
};

void cleanup_swapchain();
void recreate_swapchain();
void create_color_resources();

class VulkanGlobals {
  public:
    VkExtent2D windowExtent{1280, 720};
    const char* app_name = "Vulkan Game";
    const char* engine_name = "Andrei Game Engine";
    const std::string MODEL_PATH = "models/viking_room.obj";
    const std::string TEXTURE_PATH = "textures/viking_room.png";
    const std::vector<const char*> required_device_extensions = {
        VK_KHR_SWAPCHAIN_EXTENSION_NAME};
    const std::vector<const char*> validation_layers = {
        "VK_LAYER_KHRONOS_validation"};

#if _DEBUG
    const bool validation_layer = true;
#endif

    const uint32_t double_buffered = 2;
    uint32_t current_frame = 0;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    uint32_t graphics_family;
    VkQueue graphics_queue = VK_NULL_HANDLE;
    VkFormat swapchain_format = VK_FORMAT_B8G8R8A8_SRGB;
    VkColorSpaceKHR swapchain_color_space = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkPresentModeKHR present_mode = VK_PRESENT_MODE_MAILBOX_KHR;
    VkExtent2D swapchain_extend{};
    uint32_t swapchain_min_image_count{};
    std::vector<VkImage> images{};
    VkSampleCountFlagBits msaa_samples = VK_SAMPLE_COUNT_1_BIT;

    // Variables that need cleanup
  public:
    SDL_Window* window;
    VkInstance instance;
#ifdef _DEBUG
    VkDebugUtilsMessengerEXT debug_messenger;
#endif
    VkSurfaceKHR surface;
    VkDevice device;
    VkSwapchainKHR swapchain;
    std::vector<VkImageView> image_views{};
    VkRenderPass render_pass;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorPool descriptor_pool;
    std::vector<VkDescriptorSet> descriptor_sets;
    VkPipelineLayout pipeline_layout;
    VkPipeline graphics_pipeline;
    std::vector<VkFramebuffer> framebuffers;
    VkCommandPool command_pool;
    uint32_t mip_levels;
    VkImage texture_image;
    VkImageView texture_image_view;
    VkDeviceMemory texture_image_memory;
    VkSampler texture_sampler;
    VkImage depth_image;
    VkDeviceMemory depth_image_memory;
    VkImageView depth_image_view;
    VkImage color_image;
    VkDeviceMemory color_image_memory;
    VkImageView color_image_view;
    std::vector<Vertex> vertices;
    std::vector<uint32_t> indices;
    VkBuffer vertex_buffer;
    VkDeviceMemory vertex_buffer_memory;
    VkBuffer index_buffer;
    VkDeviceMemory index_buffer_memory;
    std::vector<VkBuffer> uniform_buffers;
    std::vector<VkDeviceMemory> uniform_buffers_memory;
    std::vector<void*> uniform_buffers_mapped;
    std::vector<VkCommandBuffer> command_buffers;
    std::vector<VkSemaphore> image_available_semaphores;
    std::vector<VkSemaphore> render_finished_semaphores;
    std::vector<VkFence> in_flight_fences;

    ~VulkanGlobals() {
        vkDeviceWaitIdle(device);
        cleanup_swapchain();
        for (auto i = 0; i < double_buffered; ++i) {
            vkDestroyFence(device, in_flight_fences[i], nullptr);
            vkDestroySemaphore(device, render_finished_semaphores[i], nullptr);
            vkDestroySemaphore(device, image_available_semaphores[i], nullptr);
        }
        vkDestroyBuffer(device, index_buffer, nullptr);
        vkFreeMemory(device, index_buffer_memory, nullptr);
        vkDestroyBuffer(device, vertex_buffer, nullptr);
        vkFreeMemory(device, vertex_buffer_memory, nullptr);
        vkDestroySampler(device, texture_sampler, nullptr);
        vkDestroyImageView(device, texture_image_view, nullptr);
        vkDestroyImage(device, texture_image, nullptr);
        vkFreeMemory(device, texture_image_memory, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyPipeline(device, graphics_pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        for (auto i = 0; i < double_buffered; ++i) {
            vkDestroyBuffer(device, uniform_buffers[i], nullptr);
            vkFreeMemory(device, uniform_buffers_memory[i], nullptr);
        }
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyRenderPass(device, render_pass, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroySurfaceKHR(instance, surface, nullptr);
#ifdef _DEBUG
        if (debug_messenger) {
            auto vkDestroyDebugUtilsMessengerEXT =
                reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                    vkGetInstanceProcAddr(instance,
                                          "vkDestroyDebugUtilsMessengerEXT"));
            vkDestroyDebugUtilsMessengerEXT(instance, debug_messenger, nullptr);
        }
#endif
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

VulkanGlobals vkg{};

#ifdef _DEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL
debug_callback(VkDebugUtilsMessageSeverityFlagBitsEXT message_severity,
               VkDebugUtilsMessageTypeFlagsEXT message_type,
               const VkDebugUtilsMessengerCallbackDataEXT* p_callback_data,
               void* p_user_data) {
    switch (message_severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
        std::cerr << "[ERROR]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
        std::cerr << "[VERBOSE]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
        std::cerr << "[INFO]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
        std::cerr << "[WARNING]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_FLAG_BITS_MAX_ENUM_EXT:
        break;
    }

    switch (message_type) {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
        std::cerr << "[GENERAL]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
        std::cerr << "[VALIDATION]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
        std::cerr << "[PERFORMANCE]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
        std::cerr << "[DEVICE_ADDRESS_BINDING]";
        break;
    case VK_DEBUG_UTILS_MESSAGE_TYPE_FLAG_BITS_MAX_ENUM_EXT:
        break;
    }

    std::cerr << p_callback_data->pMessage << std::endl;
    return VK_FALSE;
}

bool check_validation_layer_support() {
    uint32_t layer_count;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> available_layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, available_layers.data());

    for (const char* layerName : vkg.validation_layers) {
        bool layer_found = false;
        for (const auto& layerProperties : available_layers) {
            if (strcmp(layerName, layerProperties.layerName) == 0) {
                layer_found = true;
                break;
            }
        }
        if (!layer_found) {
            return false;
        }
    }

    return true;
}

VkDebugUtilsMessengerCreateInfoEXT get_debug_messenger() {
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    // create_info.pNext;
    // create_info.flags;
    create_info.messageSeverity =
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
        // VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType =
        VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT;
    create_info.pfnUserCallback = debug_callback;
    // create_info.pUserData;
    return create_info;
}

void init_debug_messenger() {
    ASSERT(check_validation_layer_support(),
           "Requested validation not supported!");
    auto create_info = get_debug_messenger();
    auto vkCreateDebugUtilsMessengerEXT =
        reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(vkg.instance,
                                  "vkCreateDebugUtilsMessengerEXT"));

    VK_CHECK(vkCreateDebugUtilsMessengerEXT(vkg.instance,
                                            &create_info,
                                            nullptr,
                                            &vkg.debug_messenger));
}
#endif

void init_instance() {
#ifdef _DEBUG
    auto messenger_create_info = get_debug_messenger();
#endif
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    // TODO load name from a game project
    app_info.pApplicationName = vkg.app_name;
    // TODO load version from a game project
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = vkg.engine_name;
    // TODO load version at compile time
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion = VK_MAKE_API_VERSION(0, 1, 3, 0);

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pNext = nullptr;
    // create_info.flags;
    create_info.pApplicationInfo = &app_info;
#ifdef _DEBUG
    if (vkg.validation_layer) {
        create_info.enabledLayerCount = vkg.validation_layers.size();
        create_info.ppEnabledLayerNames = vkg.validation_layers.data();
        create_info.pNext = &messenger_create_info;
    }
#endif
    uint32_t sdl_extension_count;
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(vkg.window,
                                               &sdl_extension_count,
                                               nullptr));
    std::vector<const char*> sdl_extensions(sdl_extension_count);
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(vkg.window,
                                               &sdl_extension_count,
                                               sdl_extensions.data()));

    std::vector<const char*> extensions(sdl_extensions);
#ifdef _DEBUG
    if (vkg.validation_layer) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
#endif
    create_info.enabledExtensionCount = extensions.size();
    create_info.ppEnabledExtensionNames = extensions.data();

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &vkg.instance));
}

void create_surface() {
    SDL_CHECK(SDL_Vulkan_CreateSurface(vkg.window, vkg.instance, &vkg.surface));
}

int32_t physical_device_score(VkPhysicalDevice device) {
    int32_t score = 0;

    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(device, &physical_device_properties);

    uint32_t extension_count;
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device,
                                                  nullptr,
                                                  &extension_count,
                                                  nullptr));
    std::vector<VkExtensionProperties> available_extensions(extension_count);
    VK_CHECK(vkEnumerateDeviceExtensionProperties(device,
                                                  nullptr,
                                                  &extension_count,
                                                  available_extensions.data()));
    uint32_t required_extensions_found = 0;
    for (auto extension : available_extensions) {
        for (auto required : vkg.required_device_extensions) {
            if (std::strcmp(extension.extensionName, required) == 0) {
                ++required_extensions_found;
            }
        }
    }
    VkPhysicalDeviceFeatures supported_features;
    vkGetPhysicalDeviceFeatures(device, &supported_features);
    if (!(supported_features.samplerAnisotropy == VK_TRUE)) {
        return -1;
    }
    if (required_extensions_found < vkg.required_device_extensions.size()) {
        return -1;
    }
    if (physical_device_properties.deviceType ==
        VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
        score += 50;
    }
    return score;
}

VkSampleCountFlagBits get_max_usable_sample_count() {
    VkPhysicalDeviceProperties physical_device_properties;
    vkGetPhysicalDeviceProperties(vkg.physical_device,
                                  &physical_device_properties);

    VkSampleCountFlags counts =
        physical_device_properties.limits.framebufferColorSampleCounts &
        physical_device_properties.limits.framebufferDepthSampleCounts;

    if (counts & VK_SAMPLE_COUNT_64_BIT) {
        return VK_SAMPLE_COUNT_64_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_32_BIT) {
        return VK_SAMPLE_COUNT_32_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_16_BIT) {
        return VK_SAMPLE_COUNT_16_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_8_BIT) {
        return VK_SAMPLE_COUNT_8_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_4_BIT) {
        return VK_SAMPLE_COUNT_4_BIT;
    }
    if (counts & VK_SAMPLE_COUNT_2_BIT) {
        return VK_SAMPLE_COUNT_2_BIT;
    }

    return VK_SAMPLE_COUNT_1_BIT;
}

void init_device() {
    uint32_t physical_device_count;
    VK_CHECK(vkEnumeratePhysicalDevices(vkg.instance,
                                        &physical_device_count,
                                        nullptr));
    ASSERT(physical_device_count > 0, "No graphics device found!")
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(vkg.instance,
                                        &physical_device_count,
                                        physical_devices.data()));

    auto best_device_score = 0;
    // for (auto i = 0; i < physical_device_count; ++i) {
    for (auto physical_device : physical_devices) {
        auto score = physical_device_score(physical_device);
        if (score <= 0 || score < best_device_score) {
            continue;
        }

        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count,
                                                 nullptr);
        std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
        vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                                 &queue_family_count,
                                                 queue_families.data());

        auto has_graphics = false;
        for (uint32_t i = 0; i < queue_family_count; ++i) {
            VkBool32 present_suport{};

            VK_CHECK(vkGetPhysicalDeviceSurfaceSupportKHR(physical_device,
                                                          i,
                                                          vkg.surface,
                                                          &present_suport));

            if (queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT &&
                present_suport == VK_TRUE) {
                vkg.graphics_family = i;
                has_graphics = true;
                break;
            }
        }
        if (!has_graphics) {
            continue;
        }
        if (score > best_device_score) {
            best_device_score = score;
            vkg.physical_device = physical_device;
        }
    }
    ASSERT(vkg.physical_device, "No graphics device selected!");
    vkg.msaa_samples = get_max_usable_sample_count();

    VkDeviceQueueCreateInfo queue_create_info{};
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    // queue_create_info.pNext;
    // queue_create_info.flags;
    queue_create_info.queueFamilyIndex = vkg.graphics_family;
    queue_create_info.queueCount = 1;
    float queue_priorities = 1.0f;
    queue_create_info.pQueuePriorities = &queue_priorities;

    VkPhysicalDeviceFeatures device_features{};
    device_features.samplerAnisotropy = VK_TRUE;

    VkDeviceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    // create_info.pNext;
    // create_info.flags;
    create_info.queueCreateInfoCount = 1;
    create_info.pQueueCreateInfos = &queue_create_info;
    // create_info.enabledLayerCount;
    // create_info.ppEnabledLayerNames;
    create_info.enabledExtensionCount = vkg.required_device_extensions.size();
    create_info.ppEnabledExtensionNames = vkg.required_device_extensions.data();
    create_info.pEnabledFeatures = &device_features;

    VK_CHECK(vkCreateDevice(vkg.physical_device,
                            &create_info,
                            nullptr,
                            &vkg.device));

    vkGetDeviceQueue(vkg.device, vkg.graphics_family, 0, &vkg.graphics_queue);
}

VkImageView create_image_view(VkImage image,
                              VkFormat format,
                              VkImageAspectFlags aspect_flags,
                              uint32_t mip_levels) {
    VkImageViewCreateInfo view_info{};
    view_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    // view_info.pNext;
    // view_info.flags;
    view_info.image = image;
    view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
    view_info.format = format;
    view_info.components.r = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.g = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.b = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.components.a = VK_COMPONENT_SWIZZLE_IDENTITY;
    view_info.subresourceRange.aspectMask = aspect_flags;
    view_info.subresourceRange.baseMipLevel = 0;
    view_info.subresourceRange.levelCount = mip_levels;
    view_info.subresourceRange.baseArrayLayer = 0;
    view_info.subresourceRange.layerCount = 1;

    VkImageView image_view;

    VK_CHECK(vkCreateImageView(vkg.device, &view_info, nullptr, &image_view));
    return image_view;
}

void create_swapchain() {
    VkSurfaceCapabilitiesKHR capabilities;
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(vkg.physical_device,
                                                       vkg.surface,
                                                       &capabilities));
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
        vkg.swapchain_extend = capabilities.currentExtent;
    } else {
        int width, height;
        SDL_Vulkan_GetDrawableSize(vkg.window, &width, &height);

        vkg.swapchain_extend.width =
            std::clamp(static_cast<uint32_t>(width),
                       capabilities.minImageExtent.width,
                       capabilities.maxImageExtent.width);
        vkg.swapchain_extend.width =
            std::clamp(static_cast<uint32_t>(height),
                       capabilities.minImageExtent.height,
                       capabilities.maxImageExtent.height);
    }

    vkg.swapchain_min_image_count = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 &&
        vkg.swapchain_min_image_count > capabilities.maxImageCount) {
        vkg.swapchain_min_image_count = capabilities.maxImageCount;
    }

    uint32_t format_count;
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkg.physical_device,
                                                  vkg.surface,
                                                  &format_count,
                                                  nullptr));
    std::vector<VkSurfaceFormatKHR> surface_formats(format_count);
    VK_CHECK(vkGetPhysicalDeviceSurfaceFormatsKHR(vkg.physical_device,
                                                  vkg.surface,
                                                  &format_count,
                                                  surface_formats.data()));
    bool found_format = false;
    for (auto surface_format : surface_formats) {
        if (surface_format.format == vkg.swapchain_format &&
            surface_format.colorSpace == vkg.swapchain_color_space) {
            found_format = true;
            break;
        }
    }
    if (!found_format) {
        vkg.swapchain_format = surface_formats[0].format;
        vkg.swapchain_color_space = surface_formats[0].colorSpace;
    }

    uint32_t present_mode_count;
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vkg.physical_device,
                                                       vkg.surface,
                                                       &present_mode_count,
                                                       nullptr));
    std::vector<VkPresentModeKHR> present_modes(present_mode_count);
    VK_CHECK(vkGetPhysicalDeviceSurfacePresentModesKHR(vkg.physical_device,
                                                       vkg.surface,
                                                       &present_mode_count,
                                                       present_modes.data()));
    bool present_found = false;
    for (auto present_mode : present_modes) {
        if (present_mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            present_found = true;
            break;
        }
    }
    if (!present_found) {
        vkg.present_mode = VK_PRESENT_MODE_FIFO_KHR;
    }

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    // create_info.pNext;
    // create_info.flags;
    create_info.surface = vkg.surface;
    create_info.minImageCount = vkg.swapchain_min_image_count;
    create_info.imageFormat = vkg.swapchain_format;
    create_info.imageColorSpace = vkg.swapchain_color_space;
    create_info.imageExtent = vkg.swapchain_extend;
    create_info.imageArrayLayers = 1;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    create_info.queueFamilyIndexCount = 0;
    create_info.pQueueFamilyIndices = nullptr;
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode = vkg.present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;

    VK_CHECK(vkCreateSwapchainKHR(vkg.device,
                                  &create_info,
                                  nullptr,
                                  &vkg.swapchain));

    uint32_t image_count;
    VK_CHECK(vkGetSwapchainImagesKHR(vkg.device,
                                     vkg.swapchain,
                                     &image_count,
                                     nullptr));
    vkg.images.resize(image_count);
    VK_CHECK(vkGetSwapchainImagesKHR(vkg.device,
                                     vkg.swapchain,
                                     &image_count,
                                     vkg.images.data()));

    vkg.image_views.resize(image_count);
    for (auto i = 0; i < image_count; ++i) {
        vkg.image_views[i] = create_image_view(vkg.images[i],
                                               vkg.swapchain_format,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               1);
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
    VK_CHECK(vkCreateShaderModule(vkg.device,
                                  &create_info,
                                  nullptr,
                                  &shader_module));
    return shader_module;
}

void create_graphics_pipeline() {
    auto vert_shader_code = read_file("shaders/shader.vert.spv");
    auto frag_shader_code = read_file("shaders/shader.frag.spv");

    VkShaderModule vert_shader_module = create_shader_module(vert_shader_code);
    VkShaderModule frag_shader_module = create_shader_module(frag_shader_code);

    VkPipelineShaderStageCreateInfo vert_shader_stage_info{};
    vert_shader_stage_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    // vert_shader_stage_info.pNext;
    // vert_shader_stage_info.flags;
    vert_shader_stage_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
    vert_shader_stage_info.module = vert_shader_module;
    vert_shader_stage_info.pName = "main";
    // vert_shader_stage_info.pSpecializationInfo;

    VkPipelineShaderStageCreateInfo frag_shader_stage_info{};
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

    std::vector<VkDynamicState> dynamic_states = {VK_DYNAMIC_STATE_VIEWPORT,
                                                  VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic_state{};
    dynamic_state.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    // dynamic_state.pNext;
    // dynamic_state.flags;
    dynamic_state.dynamicStateCount =
        static_cast<uint32_t>(dynamic_states.size());
    dynamic_state.pDynamicStates = dynamic_states.data();

    auto binding_description = Vertex::get_binding_description();
    auto attribute_descriptions = Vertex::get_attribute_descriptions();

    VkPipelineVertexInputStateCreateInfo vertex_input_info{};
    vertex_input_info.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    // vertex_input_info.pNext;
    // vertex_input_info.flags;
    vertex_input_info.vertexBindingDescriptionCount = 1;
    vertex_input_info.pVertexBindingDescriptions = &binding_description;
    vertex_input_info.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attribute_descriptions.size());
    vertex_input_info.pVertexAttributeDescriptions =
        attribute_descriptions.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType =
        VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    // input_assembly.pNext;
    // input_assembly.flags;
    input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = vkg.swapchain_extend.width;
    viewport.height = vkg.swapchain_extend.height;
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = vkg.swapchain_extend;

    VkPipelineViewportStateCreateInfo viewport_state{};
    viewport_state.sType =
        VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    // viewport_state.pNext;
    // viewport_state.flags;
    viewport_state.viewportCount = 1;
    viewport_state.pViewports = &viewport;
    viewport_state.scissorCount = 1;
    viewport_state.pScissors = &scissor;

    // Rasterizer
    VkPipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.sType =
        VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    // rasterizer.pNext;
    // rasterizer.flags;
    rasterizer.depthClampEnable = VK_FALSE;
    rasterizer.rasterizerDiscardEnable = VK_FALSE;
    rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
    rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
    rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
    rasterizer.depthBiasEnable = VK_FALSE;
    rasterizer.depthBiasConstantFactor = 0.0f;
    rasterizer.depthBiasClamp = 0.0f;
    rasterizer.depthBiasSlopeFactor = 0.0f;
    rasterizer.lineWidth = 1.0f;

    VkPipelineMultisampleStateCreateInfo multisampling{};
    multisampling.sType =
        VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    // multisampling.pNext;
    // multisampling.flags;
    multisampling.rasterizationSamples = vkg.msaa_samples;
    multisampling.sampleShadingEnable = VK_FALSE;
    multisampling.minSampleShading = 1.0f;
    multisampling.pSampleMask = nullptr;
    multisampling.alphaToCoverageEnable = VK_FALSE;
    multisampling.alphaToOneEnable = VK_FALSE;

    VkPipelineDepthStencilStateCreateInfo depth_stencil{};
    depth_stencil.sType =
        VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    // depth_stencil.pNext;
    // depth_stencil.flags;
    depth_stencil.depthTestEnable = VK_TRUE;
    depth_stencil.depthWriteEnable = VK_TRUE;
    depth_stencil.depthCompareOp = VK_COMPARE_OP_LESS;
    depth_stencil.depthBoundsTestEnable = VK_FALSE;
    depth_stencil.stencilTestEnable = VK_FALSE;
    depth_stencil.front = {};
    depth_stencil.back = {};
    depth_stencil.minDepthBounds = 0.0f;
    depth_stencil.maxDepthBounds = 1.0f;

    VkPipelineColorBlendAttachmentState color_blend_attachment{};
    color_blend_attachment.blendEnable = VK_FALSE;
    color_blend_attachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.colorBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE;
    color_blend_attachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO;
    color_blend_attachment.alphaBlendOp = VK_BLEND_OP_ADD;
    color_blend_attachment.colorWriteMask =
        VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT |
        VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;

    VkPipelineColorBlendStateCreateInfo color_blending{};
    color_blending.sType =
        VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    // color_blending.pNext;
    // color_blending.flags;
    color_blending.logicOpEnable = VK_FALSE;
    color_blending.logicOp = VK_LOGIC_OP_COPY;
    color_blending.attachmentCount = 1;
    color_blending.pAttachments = &color_blend_attachment;
    color_blending.blendConstants[0] = 0.0f;
    color_blending.blendConstants[1] = 0.0f;
    color_blending.blendConstants[2] = 0.0f;
    color_blending.blendConstants[3] = 0.0f;

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    // pipeline_layout_info.pNext;
    // pipeline_layout_info.flags;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &vkg.descriptor_set_layout;
    pipeline_layout_info.pushConstantRangeCount = 0;
    pipeline_layout_info.pPushConstantRanges = nullptr;

    VK_CHECK(vkCreatePipelineLayout(vkg.device,
                                    &pipeline_layout_info,
                                    nullptr,
                                    &vkg.pipeline_layout));

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    // pipeline_info.pNext;
    // pipeline_info.flags;
    pipeline_info.stageCount = 2;
    pipeline_info.pStages = shader_stages;
    pipeline_info.pVertexInputState = &vertex_input_info;
    pipeline_info.pInputAssemblyState = &input_assembly;
    // pipeline_info.pTessellationState;
    pipeline_info.pViewportState = &viewport_state;
    pipeline_info.pRasterizationState = &rasterizer;
    pipeline_info.pMultisampleState = &multisampling;
    pipeline_info.pDepthStencilState = &depth_stencil;
    pipeline_info.pColorBlendState = &color_blending;
    pipeline_info.pDynamicState = &dynamic_state;
    pipeline_info.layout = vkg.pipeline_layout;
    pipeline_info.renderPass = vkg.render_pass;
    pipeline_info.subpass = 0;
    pipeline_info.basePipelineHandle = VK_NULL_HANDLE;
    pipeline_info.basePipelineIndex = -1;

    VK_CHECK(vkCreateGraphicsPipelines(vkg.device,
                                       VK_NULL_HANDLE,
                                       1,
                                       &pipeline_info,
                                       nullptr,
                                       &vkg.graphics_pipeline));

    vkDestroyShaderModule(vkg.device, vert_shader_module, nullptr);
    vkDestroyShaderModule(vkg.device, frag_shader_module, nullptr);
}

VkFormat find_supported_format(const std::vector<VkFormat>& candidates,
                               VkImageTiling tiling,
                               VkFormatFeatureFlags features) {
    for (VkFormat format : candidates) {
        VkFormatProperties props;
        vkGetPhysicalDeviceFormatProperties(vkg.physical_device,
                                            format,
                                            &props);

        if (tiling == VK_IMAGE_TILING_LINEAR &&
            (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == VK_IMAGE_TILING_OPTIMAL &&
                   (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }
    ASSERT(false, "No format supported");
}

bool has_stencial_component(VkFormat format) {
    return format == VK_FORMAT_D32_SFLOAT_S8_UINT ||
           format == VK_FORMAT_D24_UNORM_S8_UINT;
}

VkFormat find_depth_format() {
    return find_supported_format(
        {VK_FORMAT_D32_SFLOAT,
         VK_FORMAT_D32_SFLOAT_S8_UINT,
         VK_FORMAT_D24_UNORM_S8_UINT},
        VK_IMAGE_TILING_OPTIMAL,
        VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
}

void create_render_pass() {
    VkAttachmentDescription color_attachment{};
    // color_attachment.flags;
    color_attachment.format = vkg.swapchain_format;
    color_attachment.samples = vkg.msaa_samples;
    color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_attachment_ref{};
    color_attachment_ref.attachment = 0;
    color_attachment_ref.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription depth_attachment{};
    // depth_attachment.flags;
    depth_attachment.format = find_depth_format();
    depth_attachment.samples = vkg.msaa_samples;
    depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    depth_attachment.finalLayout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_attachment_ref{};
    depth_attachment_ref.attachment = 1;
    depth_attachment_ref.layout =
        VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentDescription color_attachment_resolve{};
    color_attachment_resolve.format = vkg.swapchain_format;
    color_attachment_resolve.samples = VK_SAMPLE_COUNT_1_BIT;
    color_attachment_resolve.loadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment_resolve.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    color_attachment_resolve.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    color_attachment_resolve.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    color_attachment_resolve.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    color_attachment_resolve.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_attachment_resolve_ref{};
    color_attachment_resolve_ref.attachment = 2;
    color_attachment_resolve_ref.layout =
        VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    // subpass.flags;
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    // subpass.inputAttachmentCount;
    // subpass.pInputAttachments;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &color_attachment_ref;
    subpass.pResolveAttachments = &color_attachment_resolve_ref;
    subpass.pDepthStencilAttachment = &depth_attachment_ref;
    // subpass.preserveAttachmentCount;
    // subpass.pPreserveAttachments;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT |
                              VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.srcAccessMask = 0;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT |
                               VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
    // dependency.dependencyFlags;

    std::array<VkAttachmentDescription, 3> attachments = {
        color_attachment,
        depth_attachment,
        color_attachment_resolve};
    VkRenderPassCreateInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    // render_pass_info.pNext;
    // render_pass_info.flags;
    render_pass_info.attachmentCount =
        static_cast<uint32_t>(attachments.size());
    render_pass_info.pAttachments = attachments.data();
    render_pass_info.subpassCount = 1;
    render_pass_info.pSubpasses = &subpass;
    render_pass_info.dependencyCount = 1;
    render_pass_info.pDependencies = &dependency;

    VK_CHECK(vkCreateRenderPass(vkg.device,
                                &render_pass_info,
                                nullptr,
                                &vkg.render_pass));
}

void create_framebuffers() {
    vkg.framebuffers.resize(vkg.image_views.size());

    for (auto i = 0; i < vkg.image_views.size(); ++i) {
        std::array<VkImageView, 3> attachments = {vkg.color_image_view,
                                                  vkg.depth_image_view,
                                                  vkg.image_views[i]};

        VkFramebufferCreateInfo framebuffer_info{};
        framebuffer_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        // framebuffer_info.pNext;
        // framebuffer_info.flags;
        framebuffer_info.renderPass = vkg.render_pass;
        framebuffer_info.attachmentCount =
            static_cast<uint32_t>(attachments.size());
        framebuffer_info.pAttachments = attachments.data();
        framebuffer_info.width = vkg.swapchain_extend.width;
        framebuffer_info.height = vkg.swapchain_extend.height;
        framebuffer_info.layers = 1;

        VK_CHECK(vkCreateFramebuffer(vkg.device,
                                     &framebuffer_info,
                                     nullptr,
                                     &vkg.framebuffers[i]));
    }
}

void create_command_pool() {
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    // pool_info.pNext;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = vkg.graphics_family;

    VK_CHECK(vkCreateCommandPool(vkg.device,
                                 &pool_info,
                                 nullptr,
                                 &vkg.command_pool));
}

void create_command_buffer() {
    vkg.command_buffers.resize(vkg.double_buffered);

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // alloc_info.pNext;
    alloc_info.commandPool = vkg.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount =
        static_cast<uint32_t>(vkg.command_buffers.size());

    VK_CHECK(vkAllocateCommandBuffers(vkg.device,
                                      &alloc_info,
                                      vkg.command_buffers.data()));
}

uint32_t find_memory_type(uint32_t type_filter,
                          VkMemoryPropertyFlags properties) {

    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(vkg.physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; ++i) {
        if (type_filter & (1 << i) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) ==
                properties) {
            return i;
        }
    }
    ASSERT(false, "Failed to find memory type");
}

void create_buffer(VkDeviceSize size,
                   VkBufferUsageFlags usage,
                   VkMemoryPropertyFlags properties,
                   VkBuffer& buffer,
                   VkDeviceMemory& buffer_memory) {
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    // buffer_info.pNext;
    // buffer_info.flags;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // buffer_info.queueFamilyIndexCount;
    // buffer_info.pQueueFamilyIndices;

    VK_CHECK(vkCreateBuffer(vkg.device, &buffer_info, nullptr, &buffer));

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(vkg.device, buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // alloc_info.pNext;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(mem_requirements.memoryTypeBits, properties);

    VK_CHECK(
        vkAllocateMemory(vkg.device, &alloc_info, nullptr, &buffer_memory));

    VK_CHECK(vkBindBufferMemory(vkg.device, buffer, buffer_memory, 0));
}

VkCommandBuffer begin_single_time_commands() {
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    // alloc_info.pNext;
    alloc_info.commandPool = vkg.command_pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer;
    VK_CHECK(
        vkAllocateCommandBuffers(vkg.device, &alloc_info, &command_buffer));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // begin_info.pNext;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    // begin_info.pInheritanceInfo;
    vkBeginCommandBuffer(command_buffer, &begin_info);
    return command_buffer;
}

void end_single_time_commands(VkCommandBuffer command_buffer) {
    vkEndCommandBuffer(command_buffer);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    vkQueueSubmit(vkg.graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
    vkQueueWaitIdle(vkg.graphics_queue);

    vkFreeCommandBuffers(vkg.device, vkg.command_pool, 1, &command_buffer);
}

void copy_buffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
    VkCommandBuffer command_buffer = begin_single_time_commands();

    VkBufferCopy copy_region{};
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = size;
    vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &copy_region);

    end_single_time_commands(command_buffer);
}

void create_vertex_buffer() {
    VkDeviceSize buffer_size = sizeof(vkg.vertices[0]) * vkg.vertices.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    create_buffer(buffer_size,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging_buffer,
                  staging_buffer_memory);

    void* data;
    VK_CHECK(vkMapMemory(vkg.device,
                         staging_buffer_memory,
                         0,
                         buffer_size,
                         0,
                         &data));
    std::memcpy(data, vkg.vertices.data(), static_cast<size_t>(buffer_size));
    vkUnmapMemory(vkg.device, staging_buffer_memory);

    create_buffer(buffer_size,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  vkg.vertex_buffer,
                  vkg.vertex_buffer_memory);

    copy_buffer(staging_buffer, vkg.vertex_buffer, buffer_size);

    vkDestroyBuffer(vkg.device, staging_buffer, nullptr);
    vkFreeMemory(vkg.device, staging_buffer_memory, nullptr);
}

void create_index_buffer() {
    VkDeviceSize buffer_size = sizeof(vkg.indices[0]) * vkg.indices.size();

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    create_buffer(buffer_size,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging_buffer,
                  staging_buffer_memory);

    void* data;
    vkMapMemory(vkg.device, staging_buffer_memory, 0, buffer_size, 0, &data);
    std::memcpy(data, vkg.indices.data(), static_cast<size_t>(buffer_size));
    vkUnmapMemory(vkg.device, staging_buffer_memory);

    create_buffer(buffer_size,
                  VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                      VK_BUFFER_USAGE_INDEX_BUFFER_BIT,
                  VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                  vkg.index_buffer,
                  vkg.index_buffer_memory);

    copy_buffer(staging_buffer, vkg.index_buffer, buffer_size);

    vkDestroyBuffer(vkg.device, staging_buffer, nullptr);
    vkFreeMemory(vkg.device, staging_buffer_memory, nullptr);
}

void record_command_buffer(VkCommandBuffer command_buffer,
                           uint32_t image_index) {
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    // begin_info.pNext;
    begin_info.flags = 0;
    begin_info.pInheritanceInfo = nullptr;
    VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));

    VkRenderPassBeginInfo render_pass_info{};
    render_pass_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    // render_pass_info.pNext;
    render_pass_info.renderPass = vkg.render_pass;
    render_pass_info.framebuffer = vkg.framebuffers[image_index];
    render_pass_info.renderArea.offset = {0, 0};
    render_pass_info.renderArea.extent = vkg.swapchain_extend;
    std::array<VkClearValue, 2> clear_values{};
    clear_values[0].color = {
        {0.0f, 0.0f, 0.0f, 1.0f}
    };
    clear_values[1].depthStencil = {1.0f, 0};
    render_pass_info.clearValueCount =
        static_cast<uint32_t>(clear_values.size());
    render_pass_info.pClearValues = clear_values.data();

    vkCmdBeginRenderPass(command_buffer,
                         &render_pass_info,
                         VK_SUBPASS_CONTENTS_INLINE);

    // Begin Render Pass

    vkCmdBindPipeline(command_buffer,
                      VK_PIPELINE_BIND_POINT_GRAPHICS,
                      vkg.graphics_pipeline);

    VkViewport viewport{};
    viewport.x = 0.0f;
    viewport.y = 0.0f;
    viewport.width = static_cast<float>(vkg.swapchain_extend.width);
    viewport.height = static_cast<float>(vkg.swapchain_extend.height);
    viewport.minDepth = 0.0f;
    viewport.maxDepth = 1.0f;
    vkCmdSetViewport(command_buffer, 0, 1, &viewport);

    VkRect2D scissor{};
    scissor.offset = {0, 0};
    scissor.extent = vkg.swapchain_extend;
    vkCmdSetScissor(command_buffer, 0, 1, &scissor);

    VkBuffer vertex_buffers[] = {vkg.vertex_buffer};
    VkDeviceSize offsets[] = {0};
    vkCmdBindVertexBuffers(command_buffer, 0, 1, vertex_buffers, offsets);

    vkCmdBindIndexBuffer(command_buffer,
                         vkg.index_buffer,
                         0,
                         VK_INDEX_TYPE_UINT32);

    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_GRAPHICS,
                            vkg.pipeline_layout,
                            0,
                            1,
                            &vkg.descriptor_sets[vkg.current_frame],
                            0,
                            nullptr);

    vkCmdDrawIndexed(command_buffer,
                     static_cast<uint32_t>(vkg.indices.size()),
                     1,
                     0,
                     0,
                     0);

    // End Render Pass

    vkCmdEndRenderPass(command_buffer);
    VK_CHECK(vkEndCommandBuffer(command_buffer));
}

void create_sync_objects() {
    vkg.image_available_semaphores.resize(vkg.double_buffered);
    vkg.render_finished_semaphores.resize(vkg.double_buffered);
    vkg.in_flight_fences.resize(vkg.double_buffered);

    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    // semaphore_info.pNext;
    // semaphore_info.flags;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    // fence_info.pNext;
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    for (auto i = 0; i < vkg.double_buffered; ++i) {
        VK_CHECK(vkCreateSemaphore(vkg.device,
                                   &semaphore_info,
                                   nullptr,
                                   &vkg.image_available_semaphores[i]));

        VK_CHECK(vkCreateSemaphore(vkg.device,
                                   &semaphore_info,
                                   nullptr,
                                   &vkg.render_finished_semaphores[i]));

        VK_CHECK(vkCreateFence(vkg.device,
                               &fence_info,
                               nullptr,
                               &vkg.in_flight_fences[i]));
    }
}

void cleanup_swapchain() {
    vkDestroyImageView(vkg.device, vkg.color_image_view, nullptr);
    vkDestroyImage(vkg.device, vkg.color_image, nullptr);
    vkFreeMemory(vkg.device, vkg.color_image_memory, nullptr);

    vkDestroyImageView(vkg.device, vkg.depth_image_view, nullptr);
    vkDestroyImage(vkg.device, vkg.depth_image, nullptr);
    vkFreeMemory(vkg.device, vkg.depth_image_memory, nullptr);
    for (auto framebuffer : vkg.framebuffers) {
        vkDestroyFramebuffer(vkg.device, framebuffer, nullptr);
    }

    for (auto image_view : vkg.image_views) {
        vkDestroyImageView(vkg.device, image_view, nullptr);
    }
    vkDestroySwapchainKHR(vkg.device, vkg.swapchain, nullptr);
}

void recreate_swapchain() {
    vkDeviceWaitIdle(vkg.device);

    cleanup_swapchain();

    create_swapchain();
    create_color_resources();
    create_depth_resources();
    create_framebuffers();
}

void create_descriptor_set_layout() {
    VkDescriptorSetLayoutBinding ubo_layout_binding{};
    ubo_layout_binding.binding = 0;
    ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    ubo_layout_binding.descriptorCount = 1;
    ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;
    ubo_layout_binding.pImmutableSamplers = nullptr;

    VkDescriptorSetLayoutBinding sampler_layout_binding{};
    sampler_layout_binding.binding = 1;
    sampler_layout_binding.descriptorCount = 1;
    sampler_layout_binding.descriptorType =
        VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    sampler_layout_binding.pImmutableSamplers = nullptr;
    sampler_layout_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

    std::array<VkDescriptorSetLayoutBinding, 2> bindings = {
        ubo_layout_binding,
        sampler_layout_binding};
    VkDescriptorSetLayoutCreateInfo layout_info{};
    layout_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    // layout_info.pNext;
    // layout_info.flags;
    layout_info.bindingCount = static_cast<uint32_t>(bindings.size());
    layout_info.pBindings = bindings.data();

    VK_CHECK(vkCreateDescriptorSetLayout(vkg.device,
                                         &layout_info,
                                         nullptr,
                                         &vkg.descriptor_set_layout));
}

void create_uniform_buffers() {
    VkDeviceSize buffer_size = sizeof(UniformBufferObject);

    vkg.uniform_buffers.resize(vkg.double_buffered);
    vkg.uniform_buffers_memory.resize(vkg.double_buffered);
    vkg.uniform_buffers_mapped.resize(vkg.double_buffered);

    for (size_t i = 0; i < vkg.double_buffered; ++i) {
        create_buffer(buffer_size,
                      VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                          VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                      vkg.uniform_buffers[i],
                      vkg.uniform_buffers_memory[i]);

        vkMapMemory(vkg.device,
                    vkg.uniform_buffers_memory[i],
                    0,
                    buffer_size,
                    0,
                    &vkg.uniform_buffers_mapped[i]);
    }
}

void update_uniform_buffer(uint32_t current_image) {
    static auto start_time = std::chrono::high_resolution_clock::now();

    auto current_time = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     current_time - start_time)
                     .count();

    UniformBufferObject ubo{};
    ubo.model = math::rotate(math::mat4::identity(),
                             time * math::radians(90.0f),
                             math::vec3(0.0f, 0.0f, 1.0f));

    ubo.view = math::look_at(math::vec3(2.0f, 2.0f, 2.0f),
                             math::vec3(0.0f, 0.0f, 0.0f),
                             math::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj =
        math::perspesctive(math::radians(45.0f),
                           vkg.swapchain_extend.width /
                               static_cast<float>(vkg.swapchain_extend.height),
                           0.1f,
                           10.f);

    ubo.proj[1][1] *= -1;

    std::memcpy(vkg.uniform_buffers_mapped[current_image], &ubo, sizeof(ubo));
}

void create_descriptor_pool() {
    std::array<VkDescriptorPoolSize, 2> pool_sizes{};
    pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool_sizes[0].descriptorCount = static_cast<uint32_t>(vkg.double_buffered);
    pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    pool_sizes[1].descriptorCount = static_cast<uint32_t>(vkg.double_buffered);

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    // pool_info.pNext;
    // pool_info.flags;
    pool_info.maxSets = static_cast<uint32_t>(vkg.double_buffered);
    pool_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    VK_CHECK(vkCreateDescriptorPool(vkg.device,
                                    &pool_info,
                                    nullptr,
                                    &vkg.descriptor_pool));
}

void create_descriptor_sets() {
    std::vector<VkDescriptorSetLayout> layouts(vkg.double_buffered,
                                               vkg.descriptor_set_layout);
    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    // alloc_info.pNext;
    alloc_info.descriptorPool = vkg.descriptor_pool;
    alloc_info.descriptorSetCount = static_cast<uint32_t>(vkg.double_buffered);
    alloc_info.pSetLayouts = layouts.data();

    vkg.descriptor_sets.resize(vkg.double_buffered);
    VK_CHECK(vkAllocateDescriptorSets(vkg.device,
                                      &alloc_info,
                                      vkg.descriptor_sets.data()));

    for (size_t i = 0; i < vkg.double_buffered; ++i) {
        VkDescriptorBufferInfo buffer_info{};
        buffer_info.buffer = vkg.uniform_buffers[i];
        buffer_info.offset = 0;
        buffer_info.range = sizeof(UniformBufferObject);

        VkDescriptorImageInfo image_info{};
        image_info.sampler = vkg.texture_sampler;
        image_info.imageView = vkg.texture_image_view;
        image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        std::array<VkWriteDescriptorSet, 2> descriptor_writes{};
        descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // descriptor_writes[0].pNext;
        descriptor_writes[0].dstSet = vkg.descriptor_sets[i];
        descriptor_writes[0].dstBinding = 0;
        descriptor_writes[0].dstArrayElement = 0;
        descriptor_writes[0].descriptorCount = 1;
        descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        descriptor_writes[0].pImageInfo = nullptr;
        descriptor_writes[0].pBufferInfo = &buffer_info;
        descriptor_writes[0].pTexelBufferView = nullptr;

        descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        // descriptor_writes[1].pNext;
        descriptor_writes[1].dstSet = vkg.descriptor_sets[i];
        descriptor_writes[1].dstBinding = 1;
        descriptor_writes[1].dstArrayElement = 0;
        descriptor_writes[1].descriptorCount = 1;
        descriptor_writes[1].descriptorType =
            VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        descriptor_writes[1].pImageInfo = &image_info;
        descriptor_writes[1].pBufferInfo = nullptr;
        descriptor_writes[1].pTexelBufferView = nullptr;

        vkUpdateDescriptorSets(vkg.device,
                               static_cast<uint32_t>(descriptor_writes.size()),
                               descriptor_writes.data(),
                               0,
                               nullptr);
    }
}

void generate_mipmaps(VkImage image,
                      VkFormat image_format,
                      int32_t tex_width,
                      int32_t tex_height,
                      uint32_t mip_levels) {

    // There are two alternatives in this case. You could implement a function
    // that searches common texture image formats for one that does support
    // linear blitting, or you could implement the mipmap generation in software
    // with a library like stb_image_resize. Each mip level can then be loaded
    // into the image in the same way that you loaded the original image.

    // It should be noted that it is uncommon in practice to generate the mipmap
    // levels at runtime anyway. Usually they are pregenerated and stored in the
    // texture file alongside the base level to improve loading speed.
    // Implementing resizing in software and loading multiple levels from a file
    // is left as an exercise to the reader.
    VkFormatProperties format_properties;
    vkGetPhysicalDeviceFormatProperties(vkg.physical_device,
                                        image_format,
                                        &format_properties);
    ASSERT(format_properties.optimalTilingFeatures &
               VK_FORMAT_FEATURE_SAMPLED_IMAGE_FILTER_LINEAR_BIT,
           "texture image format does not support linear blitting!");

    VkCommandBuffer command_buffer = begin_single_time_commands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barrier.pNext;
    barrier.image = image;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mip_width = tex_width;
    int32_t mip_height = tex_height;

    for (uint32_t i = 1; i < mip_levels; ++i) {
        barrier.subresourceRange.baseMipLevel = i - 1;
        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        VkImageBlit blit{};
        blit.srcOffsets[0] = {0, 0, 0};
        blit.srcOffsets[1] = {mip_width, mip_height, 1};
        blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.srcSubresource.mipLevel = i - 1;
        blit.srcSubresource.baseArrayLayer = 0;
        blit.srcSubresource.layerCount = 1;
        blit.dstOffsets[0] = {0, 0, 0};
        blit.dstOffsets[1] = {mip_width > 1 ? mip_width / 2 : 1,
                              mip_height > 1 ? mip_height / 2 : 1,
                              1};
        blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        blit.dstSubresource.mipLevel = i;
        blit.dstSubresource.baseArrayLayer = 0;
        blit.dstSubresource.layerCount = 1;

        vkCmdBlitImage(command_buffer,
                       image,
                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       image,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                       1,
                       &blit,
                       VK_FILTER_LINEAR);

        barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        vkCmdPipelineBarrier(command_buffer,
                             VK_PIPELINE_STAGE_TRANSFER_BIT,
                             VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                             0,
                             0,
                             nullptr,
                             0,
                             nullptr,
                             1,
                             &barrier);

        if (mip_width > 1) {
            mip_width /= 2;
        }
        if (mip_height > 1) {
            mip_height /= 2;
        }
    }

    barrier.subresourceRange.baseMipLevel = mip_levels - 1;
    barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

    vkCmdPipelineBarrier(command_buffer,
                         VK_PIPELINE_STAGE_TRANSFER_BIT,
                         VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);

    end_single_time_commands(command_buffer);
}

void create_image(uint32_t width,
                  uint32_t height,
                  uint32_t mip_levels,
                  VkSampleCountFlagBits num_samples,
                  VkFormat format,
                  VkImageTiling tiling,
                  VkImageUsageFlags usage,
                  VkMemoryPropertyFlags properties,
                  VkImage& image,
                  VkDeviceMemory& image_memory) {

    VkImageCreateInfo image_info{};
    image_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    // image_info.pNext;
    // image_info.flags;
    image_info.imageType = VK_IMAGE_TYPE_2D;
    image_info.format = format;
    image_info.extent.width = width;
    image_info.extent.height = height;
    image_info.extent.depth = 1;
    image_info.mipLevels = mip_levels;
    image_info.arrayLayers = 1;
    image_info.samples = num_samples;
    image_info.tiling = tiling;
    image_info.usage = usage;
    image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // image_info.queueFamilyIndexCount;
    // image_info.pQueueFamilyIndices;
    image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VK_CHECK(vkCreateImage(vkg.device, &image_info, nullptr, &image));

    VkMemoryRequirements mem_requirements{};
    vkGetImageMemoryRequirements(vkg.device, image, &mem_requirements);
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    // alloc_info.pNext;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(mem_requirements.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory(vkg.device, &alloc_info, nullptr, &image_memory));
    vkBindImageMemory(vkg.device, image, image_memory, 0);
}

void transition_image_layout(VkImage image,
                             VkFormat format,
                             VkImageLayout old_layout,
                             VkImageLayout new_layout,
                             uint32_t mip_levels) {
    VkCommandBuffer command_buffer = begin_single_time_commands();

    VkImageMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
    // barrier.pNext;
    barrier.srcAccessMask = 0;
    barrier.dstAccessMask = 0;
    barrier.oldLayout = old_layout;
    barrier.newLayout = new_layout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = mip_levels;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkPipelineStageFlags source_stage;
    VkPipelineStageFlags destination_stage;

    if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED &&
        new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL) {
        barrier.srcAccessMask = 0;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;

        source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
        destination_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
    } else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL &&
               new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL) {
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;

        source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
        destination_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    } else {
        ASSERT(true, "unsupported layout transition!");
    }

    vkCmdPipelineBarrier(command_buffer,
                         source_stage,
                         destination_stage,
                         0,
                         0,
                         nullptr,
                         0,
                         nullptr,
                         1,
                         &barrier);

    end_single_time_commands(command_buffer);
}

void copy_buffer_to_image(VkBuffer buffer,
                          VkImage image,
                          uint32_t width,
                          uint32_t height) {
    VkCommandBuffer command_buffer = begin_single_time_commands();

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(command_buffer,
                           buffer,
                           image,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1,
                           &region);

    end_single_time_commands(command_buffer);
}

void create_texture_image() {
    auto texture = load_image(vkg.TEXTURE_PATH);

    vkg.mip_levels = static_cast<uint32_t>(std::floor(
                         std::log2(std::max(texture.width, texture.height)))) +
                     1;

    VkDeviceSize image_size = texture.width * texture.height * 4;

    ASSERT(texture.pixels, "failed to load texture image!");

    VkBuffer staging_buffer;
    VkDeviceMemory staging_buffer_memory;

    create_buffer(image_size,
                  VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                  staging_buffer,
                  staging_buffer_memory);

    void* data;
    vkMapMemory(vkg.device, staging_buffer_memory, 0, image_size, 0, &data);
    std::memcpy(data, texture.pixels, static_cast<size_t>(image_size));
    vkUnmapMemory(vkg.device, staging_buffer_memory);

    create_image(texture.width,
                 texture.height,
                 vkg.mip_levels,
                 VK_SAMPLE_COUNT_1_BIT,
                 VK_FORMAT_R8G8B8A8_SRGB,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                     VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                     VK_IMAGE_USAGE_SAMPLED_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vkg.texture_image,
                 vkg.texture_image_memory);

    transition_image_layout(vkg.texture_image,
                            VK_FORMAT_R8G8B8A8_SRGB,
                            VK_IMAGE_LAYOUT_UNDEFINED,
                            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                            vkg.mip_levels);
    copy_buffer_to_image(staging_buffer,
                         vkg.texture_image,
                         static_cast<uint32_t>(texture.width),
                         static_cast<uint32_t>(texture.height));
    // transitioned to VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL while generating
    // mipmaps
    // transition_image_layout(vkg.texture_image,
    //                         VK_FORMAT_R8G8B8A8_SRGB,
    //                         VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
    //                         VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
    //                         vkg.mip_levels);
    generate_mipmaps(vkg.texture_image,
                     VK_FORMAT_R8G8B8A8_SRGB,
                     texture.width,
                     texture.height,
                     vkg.mip_levels);

    vkDestroyBuffer(vkg.device, staging_buffer, nullptr);
    vkFreeMemory(vkg.device, staging_buffer_memory, nullptr);
}

void create_texture_image_view() {
    vkg.texture_image_view = create_image_view(vkg.texture_image,
                                               VK_FORMAT_R8G8B8A8_SRGB,
                                               VK_IMAGE_ASPECT_COLOR_BIT,
                                               vkg.mip_levels);
}

void create_texture_sampler() {
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(vkg.physical_device, &properties);

    VkSamplerCreateInfo sampler_info{};
    sampler_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    // sampler_info.pNext;
    // sampler_info.flags;
    sampler_info.magFilter = VK_FILTER_LINEAR;
    sampler_info.minFilter = VK_FILTER_LINEAR;
    sampler_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sampler_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sampler_info.mipLodBias = 0.0f;
    sampler_info.anisotropyEnable = VK_TRUE;
    sampler_info.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    sampler_info.compareEnable = VK_FALSE;
    sampler_info.compareOp = VK_COMPARE_OP_ALWAYS;
    sampler_info.minLod = 0.0f;
    sampler_info.maxLod = static_cast<float>(vkg.mip_levels);
    sampler_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
    sampler_info.unnormalizedCoordinates = VK_FALSE;

    // Instead of enforcing the availability of anisotropic filtering, it's also
    // possible to simply not use it by conditionally setting:
    // samplerInfo.anisotropyEnable = VK_FALSE;
    // samplerInfo.maxAnisotropy = 1.0f;

    VK_CHECK(vkCreateSampler(vkg.device,
                             &sampler_info,
                             nullptr,
                             &vkg.texture_sampler));
}

void create_depth_resources() {
    VkFormat depth_format = find_depth_format();
    create_image(vkg.swapchain_extend.width,
                 vkg.swapchain_extend.height,
                 1,
                 vkg.msaa_samples,
                 depth_format,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vkg.depth_image,
                 vkg.depth_image_memory);
    vkg.depth_image_view = create_image_view(vkg.depth_image,
                                             depth_format,
                                             VK_IMAGE_ASPECT_DEPTH_BIT,
                                             1);
}

void load_model() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string err;

    auto result = tinyobj::LoadObj(&attrib,
                                   &shapes,
                                   &materials,
                                   &err,
                                   vkg.MODEL_PATH.c_str());

    ASSERT(result, "Loading model" + err);

    std::unordered_map<Vertex, uint32_t> unique_vertices{};
    for (const auto& shape : shapes) {
        for (const auto& index : shape.mesh.indices) {
            Vertex vertex{};

            vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                          attrib.vertices[3 * index.vertex_index + 1],
                          attrib.vertices[3 * index.vertex_index + 2]};
            vertex.tex_coord = {
                attrib.texcoords[2 * index.texcoord_index + 0],
                1.0f - attrib.texcoords[2 * index.texcoord_index + 1]};
            vertex.color = {1.0f, 1.0f, 1.0f};
            vkg.vertices.push_back(vertex);

            if (unique_vertices.count(vertex) == 0) {
                unique_vertices[vertex] =
                    static_cast<uint32_t>(vkg.vertices.size());
                vkg.vertices.push_back(vertex);
            }
            vkg.indices.push_back(unique_vertices[vertex]);
        }
    }
}

void create_color_resources() {
    VkFormat color_format = vkg.swapchain_format;
    create_image(vkg.swapchain_extend.width,
                 vkg.swapchain_extend.height,
                 1,
                 vkg.msaa_samples,
                 color_format,
                 VK_IMAGE_TILING_OPTIMAL,
                 VK_IMAGE_USAGE_TRANSIENT_ATTACHMENT_BIT |
                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                 vkg.color_image,
                 vkg.color_image_memory);
    vkg.color_image_view = create_image_view(vkg.color_image,
                                             color_format,
                                             VK_IMAGE_ASPECT_COLOR_BIT,
                                             1);
}

} // namespace

namespace graphics {
void init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    vkg.window = SDL_CreateWindow("Vulkan Game Engine",
                                  SDL_WINDOWPOS_UNDEFINED,
                                  SDL_WINDOWPOS_UNDEFINED,
                                  vkg.windowExtent.width,
                                  vkg.windowExtent.height,
                                  SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

    init_instance();
    create_surface();
#ifdef _DEBUG
    if (vkg.validation_layer) {
        init_debug_messenger();
    }
#endif
    init_device();
    create_swapchain();
    create_render_pass();
    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_command_pool();
    create_color_resources();
    create_depth_resources();
    create_framebuffers();
    create_texture_image();
    create_texture_image_view();
    create_texture_sampler();
    load_model();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffer();
    create_sync_objects();
}

void draw() {
    VK_CHECK(vkWaitForFences(vkg.device,
                             1,
                             &vkg.in_flight_fences[vkg.current_frame],
                             VK_TRUE,
                             std::numeric_limits<uint64_t>::max()));

    uint32_t image_index;
    auto acquire_result =
        vkAcquireNextImageKHR(vkg.device,
                              vkg.swapchain,
                              std::numeric_limits<uint64_t>::max(),
                              vkg.image_available_semaphores[vkg.current_frame],
                              VK_NULL_HANDLE,
                              &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR ||
        acquire_result == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain();
        return;
    } else {
        VK_CHECK(acquire_result);
    }

    update_uniform_buffer(vkg.current_frame);

    VK_CHECK(
        vkResetFences(vkg.device, 1, &vkg.in_flight_fences[vkg.current_frame]));

    VK_CHECK(vkResetCommandBuffer(vkg.command_buffers[vkg.current_frame], 0));

    record_command_buffer(vkg.command_buffers[vkg.current_frame], image_index);

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    // submit_info.pNext;

    VkSemaphore wait_semaphores[] = {
        vkg.image_available_semaphores[vkg.current_frame]};
    VkPipelineStageFlags wait_stages[] = {
        VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    submit_info.waitSemaphoreCount = 1;
    submit_info.pWaitSemaphores = wait_semaphores;
    submit_info.pWaitDstStageMask = wait_stages;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &vkg.command_buffers[vkg.current_frame];

    VkSemaphore signal_semaphores[] = {
        vkg.render_finished_semaphores[vkg.current_frame]};

    submit_info.signalSemaphoreCount = 1;
    submit_info.pSignalSemaphores = signal_semaphores;

    VK_CHECK(vkQueueSubmit(vkg.graphics_queue,
                           1,
                           &submit_info,
                           vkg.in_flight_fences[vkg.current_frame]));

    VkPresentInfoKHR present_info{};
    present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    // present_info.pNext;
    present_info.waitSemaphoreCount = 1;
    present_info.pWaitSemaphores = signal_semaphores;
    VkSwapchainKHR swapchains[] = {vkg.swapchain};
    present_info.swapchainCount = 1;
    present_info.pSwapchains = swapchains;
    present_info.pImageIndices = &image_index;
    present_info.pResults = nullptr;

    auto present_result = vkQueuePresentKHR(vkg.graphics_queue, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR ||
        present_result == VK_SUBOPTIMAL_KHR) {
        recreate_swapchain();
    } else {
        VK_CHECK(present_result);
    }

    vkg.current_frame = (vkg.current_frame + 1) % vkg.double_buffered;
}

void resize_window() {
    // TODO save custom size in settings
    recreate_swapchain();
}

} // namespace graphics
