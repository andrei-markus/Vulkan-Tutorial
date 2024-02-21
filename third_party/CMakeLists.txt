include(FetchContent)

FetchContent_Declare(
    SDL2
    GIT_REPOSITORY  https://github.com/libsdl-org/SDL
    GIT_TAG         release-2.30.0
    GIT_PROGRESS    TRUE
)

FetchContent_MakeAvailable(SDL2)

add_library(stb INTERFACE)
target_include_directories(stb INTERFACE stb)
add_library(tinyobj INTERFACE)
target_include_directories(tinyobj INTERFACE tinyobjloader)
