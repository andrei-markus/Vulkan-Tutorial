#include "asset_loader.hpp"

#include <cstddef>
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <string>
#include <vector>

std::vector<std::byte> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "failed to open file! " << filename << std::endl;
        return {};
    }
    auto fileSize = file.tellg();
    std::vector<std::byte> buffer(fileSize);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    return buffer;
}

img_data load_image(const std::string& filename) {
    img_data result;
    result.pixels = reinterpret_cast<std::byte*>(stbi_load(filename.c_str(),
                                                           &result.width,
                                                           &result.height,
                                                           &result.channels,
                                                           STBI_rgb_alpha));
    return result;
}
