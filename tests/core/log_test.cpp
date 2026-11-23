// include/granite/log.hpp: the line each level renders. The writes themselves are one call each to stderr.
#include "granite/log.hpp"

#include <cstdio>
#include <string>

namespace {

int failures = 0;

void is(const std::string& got, const std::string& want, const char* what) {
    const bool ok = got == want;
    std::printf("log %-52s %s\n", what, ok ? "ok" : "FAILED   <-- WRONG");
    if (!ok) {
        std::printf("   got  %s\n   want %s\n", got.c_str(), want.c_str());
        ++failures;
    }
}

}  // namespace

int main() {
    using granite::log::Level;
    using granite::log::render;
    is(render(Level::info, "granite generate", "session is up (engine %s)", "1.2"),
       "granite generate: session is up (engine 1.2)\n", "info keeps the tag and has no marker");
    is(render(Level::warning, "granite serve", "RAM %d MiB low", 512), "granite serve: warning: RAM 512 MiB low\n",
       "warning marker after the tag");
    is(render(Level::error, "granite generate", "cannot open %s", "x.basalt"),
       "granite generate: error: cannot open x.basalt\n", "error marker after the tag");
    is(render(Level::error, "granite", "kernel: %s", "bad"), "granite: error: kernel: bad\n", "a line of no area");
    is(render(Level::warning, "granite-bench", "NVML is unavailable"), "granite-bench: warning: NVML is unavailable\n",
       "a tool's own tag");
    is(render(Level::info, "granite load", "plan %s%s %llu", "copy ", "cuda1", 4096ull),
       "granite load: plan copy cuda1 4096\n", "a parsed progress line is unchanged");
    is(render(Level::info, "granite serve", "%.0f%% of %lld", 12.0, 7ll), "granite serve: 12% of 7\n", "percent signs");
    is(render(Level::info, "granite", "%s", ""), "granite: \n", "an empty message");
    const std::string longer(1500, 'x');
    is(render(Level::error, "granite generate", "%s", longer.c_str()), "granite generate: error: " + longer + "\n",
       "a message longer than the first buffer");
    is(render(Level::info, "granite serve", "two\nlines"), "granite serve: two\nlines\n", "an inner newline is kept");
    is(granite::log::line(Level::warning, "granite mtp", "text"), "granite mtp: warning: text", "line() has no newline");

    // parse() reads back what render() wrote, and the older forms
    const auto parsed = [](const std::string& l) {
        granite::log::Parsed p;
        if (!granite::log::parse(l, p)) return std::string("not a log line");
        return std::string(p.level == Level::error ? "error" : p.level == Level::warning ? "warning" : "info") + "|" +
               p.tag + "|" + p.text;
    };
    is(parsed("granite generate: error: no GPU"), "error|granite generate|no GPU", "parse: error");
    is(parsed("granite serve: warning: low RAM"), "warning|granite serve|low RAM", "parse: warning");
    is(parsed("granite load: cuda0 1 2"), "info|granite load|cuda0 1 2", "parse: info");
    is(parsed("granite: error: kernel: bad"), "error|granite|kernel: bad", "parse: no area");
    is(parsed("granite-bench: warning: NVML"), "warning|granite-bench|NVML", "parse: a tool's tag");
    is(parsed("granite serve: no error here"), "info|granite serve|no error here", "parse: the word error is no marker");
    is(parsed("CUDA error: out of memory"), "not a log line", "parse: a line of another writer");
    is(parsed("granite (ms/window): x"), "not a log line", "parse: odd characters in the tag");
    return failures == 0 ? 0 : 1;
}
