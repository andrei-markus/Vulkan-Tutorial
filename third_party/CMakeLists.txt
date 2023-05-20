add_library(sdl2 INTERFACE)
if (MSVC)
    set(sdl2_DIR "SDL_PATH" CACHE FILEPATH "Path to SDL2")
    target_include_directories(sdl2 INTERFACE ${sdl2_DIR}/include ${sdl2_image_DIR}/include)
    target_link_directories(sdl2 INTERFACE ${sdl2_DIR}/lib/x64 ${sdl2_image_DIR}/lib/x64)
else()
endif()
find_package(SDL2 REQUIRED CONFIG)

target_link_libraries(sdl2 INTERFACE SDL2::SDL2main SDL2::SDL2)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)
