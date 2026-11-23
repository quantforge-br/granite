#pragma once

#include <optional>
#include <string>

namespace granite::program { struct CommonOptions; }
namespace granite::model { class ModelAdapter; }

namespace granite::program::cli {

// An exit code for help or an invalid argument; otherwise startup can continue.
std::optional<int> parse_options(int argc, char** argv, CommonOptions& o, const model::ModelAdapter* adapter);

}  // namespace granite::program::cli
