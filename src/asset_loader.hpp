#ifndef ASSET_LOADER_HPP
#define ASSET_LOADER_HPP

#include <cstddef>
#include <string>
#include <vector>

struct img_data {
    int width;
    int height;
    int channels;
    std::byte* pixels;
};

std::vector<std::byte> read_file(const std::string& filename);
img_data load_image(const std::string& filename);

#endif
