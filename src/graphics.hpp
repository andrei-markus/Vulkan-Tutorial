#ifndef GRAPHICS_HPP
#define GRAPHICS_HPP

#include <SDL.h>
#include <SDL_video.h>
#include <Vulkan/vulkan.h>

class graphics {
  public:
    VkExtent2D _windowExtent{1280, 720};
    SDL_Window* _window;

    graphics();
    ~graphics();
    void run();
};

#endif