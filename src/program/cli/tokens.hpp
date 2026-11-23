#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>


namespace granite::program::cli {

bool parse_i64_list(const char* text, std::vector<int64_t>& out, std::string& error);
std::optional<int> load_tokens_file(const char* path, std::vector<int64_t>& tokens);

}  // namespace granite::program::cli
