#include "serve.hpp"

#include "granite/log.hpp"
#include "granite/platform/file.hpp"
#include "granite/platform/process.hpp"
#include "granite/platform/system.hpp"

#include <cerrno>
#include <cstdlib>
#include <cstring>

namespace granite::program::cli {

/// `granite serve --config FILE` or `granite serve MODEL.granite [--python PY] [server args...]`: runs the Python server
/// over this binary.  The server gets `--engine-bin` and what was typed; a MODEL.granite argument stands for `--model MODEL.granite`.
/// Only returns on failure.
int serve_subcommand(int argc, char** argv) {
    std::string python;
    std::vector<std::string> extra;
    bool have_model = false;
    // the working directory changes below: the paths the server reads must not depend on it
    auto absolute = [](const std::string& path) {
        std::string real;
        if (!granite::platform::absolute_path(path, real)) {
            granite::log::error("granite serve", "cannot open %s", path.c_str());
            std::exit(1);
        }
        return real;
    };
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) { granite::log::error("granite serve", "%s needs a value", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--python") python = value();
        else if (a == "--model") { extra.insert(extra.end(), {a, absolute(value())}); have_model = true; }
        else if (a == "--config" || a == "--drafter") extra.insert(extra.end(), {a, absolute(value())});
        // only the first argument can be MODEL.granite; a later bare word is the value of the option before it
        else if (i == 2 && !have_model && !a.empty() && a[0] != '-') { extra.insert(extra.end(), {"--model", absolute(a)}); have_model = true; }
        else extra.push_back(a);
    }
    if (extra.empty()) {
        granite::log::error("granite", "usage: granite serve --config FILE | MODEL.granite [--python PY] [server args...]");
        return 2;
    }
    const std::string exe = granite::platform::module_path();
    if (exe.empty()) { granite::log::error("granite serve", "cannot find this binary"); return 1; }
    const std::string src_dir = granite::platform::install_root();
    if (src_dir.empty()) { granite::log::error("granite serve", "cannot find the directory above this binary"); return 1; }
    if (python.empty()) python = granite::platform::python_interpreter(src_dir);
    if (granite::platform::set_working_dir(src_dir) != 0) {
        granite::log::error("granite serve", "cannot enter %s", src_dir.c_str());
        return 1;
    }
    std::vector<std::string> args = {python, "-m", "serve.server", "--engine-bin", exe};
    args.insert(args.end(), extra.begin(), extra.end());
    // Does not return when it works. Windows has no exec, so this process waits for the server and ends with the
    // server's exit code. Reaching the line below means the server never started, and errno says why.
    granite::platform::exec_replacing(args);
    granite::log::error("granite serve", "cannot run %s: %s", python.c_str(), std::strerror(errno));
    return 1;
}

}  // namespace granite::program::cli
