#include "tokens.hpp"

#include "granite/log.hpp"

#include <charconv>
#include <fstream>
#include <iterator>
#include <sstream>

namespace granite::program::cli {

bool parse_i64_list(const char* s, std::vector<int64_t>& out, std::string& err) {
    out.clear();
    std::string text(s);
    for (char& c : text) if (c == ',') c = ' ';
    std::istringstream input(text);
    std::string token;
    while (input >> token) {
        int32_t id = 0;
        const auto parsed = std::from_chars(token.data(), token.data() + token.size(), id);
        if (parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || id < 0) {
            err = "invalid token id: expected an integer in [0, 2147483647]";
            out.clear();
            return false;
        }
        out.push_back(id);
    }
    if (out.empty()) { err = "token list was empty"; return false; }
    return true;
}


std::optional<int> load_tokens_file(const char* path, std::vector<int64_t>& tokens) {
    std::ifstream input(path);
    if (!input) { granite::log::error("granite", "cannot open token file"); return 2; }
    std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    if (input.bad()) { granite::log::error("granite", "cannot read token file"); return 2; }
    std::string error;
    if (text.find('\0') != std::string::npos || !parse_i64_list(text.c_str(), tokens, error)) {
        granite::log::error("granite", "malformed token file: %s", error.c_str()); return 2;
    }
    return std::nullopt;
}

}  // namespace granite::program::cli
