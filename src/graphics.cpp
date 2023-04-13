#include "graphics.hpp"

#include <SDL.h>

graphics::graphics() {
    // We initialize SDL and create a window with it.
    SDL_Init(SDL_INIT_VIDEO);

    _window = SDL_CreateWindow("Vulkan Engine", SDL_WINDOWPOS_UNDEFINED,
                               SDL_WINDOWPOS_UNDEFINED, _windowExtent.width,
                               _windowExtent.height, SDL_WINDOW_VULKAN);
}

graphics::~graphics() {
    SDL_DestroyWindow(_window);
    SDL_Quit();
}

void graphics::run() {
    SDL_Event e;
    bool bQuit = false;

    // main loop
    while (!bQuit) {
        // Handle events on queue
        while (SDL_PollEvent(&e) != 0) {
            // close the window
            if (e.type == SDL_QUIT) {
                bQuit = true;
            }
        }

        // draw();
    }
}