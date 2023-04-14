#include "graphics.hpp"

#include <SDL.h>
#include <SDL_vulkan.h>
#include <Vulkan/vulkan.h>
#include <iostream>
#include <vector>

namespace {
#define SDL_CHECK(x)                                                           \
    {                                                                          \
        SDL_bool err = x;                                                      \
        if (err == SDL_FALSE) {                                                \
            std::cout << "Detected SDL error: " << SDL_GetError()              \
                      << std::endl;                                            \
            abort();                                                           \
        }                                                                      \
    }

#define VK_CHECK(x)                                                            \
    {                                                                          \
        VkResult err = x;                                                      \
        if (err) {                                                             \
            std::cout << "Detected Vulkan error: " << err << std::endl;        \
            abort();                                                           \
        }                                                                      \
    }

class graphicsState {
  public:
    SDL_Window* window;
    VkInstance instance;
    VkSurfaceKHR surface;

    ~graphicsState() {
        vkDestroyInstance(instance, nullptr);
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

VkExtent2D windowExtent{1280, 720};
graphicsState context{};
auto app_name = "Vulkan Game";
auto engine_name = "Andrei Game Engine";

void init_instance() {
    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pNext = nullptr;
    // TODO load name from a game project
    app_info.pApplicationName = app_name;
    // TODO load version from a game project
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = engine_name;
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
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(context.window,
                                               &sdl_extension_count, nullptr));
    std::vector<const char*> sdl_extensions(sdl_extension_count);
    SDL_CHECK(SDL_Vulkan_GetInstanceExtensions(
        context.window, &sdl_extension_count, sdl_extensions.data()));

    create_info.enabledExtensionCount = sdl_extension_count;
    create_info.ppEnabledExtensionNames = sdl_extensions.data();

    VK_CHECK(vkCreateInstance(&create_info, nullptr, &context.instance));
}

} // namespace

namespace graphics {
void init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    context.window = SDL_CreateWindow(
        "Vulkan Game Engine", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED,
        windowExtent.width, windowExtent.height, SDL_WINDOW_VULKAN);

    init_instance();
}

void draw() {}

} // namespace graphics
