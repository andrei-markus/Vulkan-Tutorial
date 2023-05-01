#include <cstddef>
#include <fstream>
#include <iostream>
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
