#include "graphics.hpp"

#include <SDL.h>
#include <SDL_events.h>
#include <SDL_video.h>
#include <iostream>

int main(int argc, char* argv[]) {
    bool is_window_minimized = false;
    graphics::init();

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
            if (e.type == SDL_WINDOWEVENT) {
                if (e.window.event == SDL_WINDOWEVENT_MINIMIZED) {
                    is_window_minimized = true;
                }
                if (e.window.event == SDL_WINDOWEVENT_RESTORED) {
                    is_window_minimized = false;
                }
                if (e.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
                    graphics::resize_window();
                }
            }
        }

        if (!is_window_minimized) {

            graphics::draw();
        }
    }

    return 0;
}
