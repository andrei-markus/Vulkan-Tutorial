#include "graphics.hpp"

#include <SDL.h>
#include <Vulkan/vulkan.h>

namespace {

class graphicsState {
  public:
    VkExtent2D windowExtent{1280, 720};
    SDL_Window* window;

    ~graphicsState() {
        SDL_DestroyWindow(window);
        SDL_Quit();
    }
};

graphicsState context;

} // namespace

namespace graphics {
void init() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    context.window =
        SDL_CreateWindow("Vulkan Game Engine", SDL_WINDOWPOS_UNDEFINED,
                         SDL_WINDOWPOS_UNDEFINED, context.windowExtent.width,
                         context.windowExtent.height, SDL_WINDOW_VULKAN);
}

void draw() {}

} // namespace graphics
