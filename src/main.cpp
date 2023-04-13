#include "graphics.hpp"

#include <SDL.h>

int main(int argc, char* argv[]) {
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
        }

        graphics::draw();
    }

    return 0;
}