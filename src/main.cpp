#include "graphics.hpp"

#include <SDL.h>

int main(int argc, char* argv[]) {
    bool is_window_minimized = false;
    graphics::init();

    SDL_Event sdl_event;
    bool quit_app = false;
    // main loop
    while (!quit_app) {
        // Handle events on queue
        while (SDL_PollEvent(&sdl_event) != 0) {
            switch (sdl_event.type) {
            case SDL_QUIT:
                // close the window
                quit_app = true;
                break;
            case SDL_WINDOWEVENT:
                switch (sdl_event.window.event) {
                case SDL_WINDOWEVENT_MINIMIZED:
                    is_window_minimized = true;
                    break;
                case SDL_WINDOWEVENT_RESTORED:
                    is_window_minimized = false;
                    break;
                case SDL_WINDOWEVENT_SIZE_CHANGED:
                    graphics::resize_window();
                    break;
                }
                break;
            }
        }

        if (!is_window_minimized) { graphics::draw(); }
    }

    return 0;
}
