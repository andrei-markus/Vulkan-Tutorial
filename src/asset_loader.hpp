#ifndef ASSET_LOADER_HPP
#define ASSET_LOADER_HPP

#include <cstddef>
#include <string>
#include <vector>

std::vector<std::byte> read_file(const std::string& filename);

#endif
