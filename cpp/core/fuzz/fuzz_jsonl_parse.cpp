// Coverage-guided fuzz target for the JSON-lines request parser
// (qp/tools/jsonl.hpp) that the differential-test shims read stdin with.
//
// Any byte string is fed to jsonl::parse. Properties:
//   1. parse returns or throws std::invalid_argument; nothing else, and no
//      sanitizer report (out-of-bounds reads are the classic parser bug).
//   2. On success every typed accessor (num, flag) either returns or
//      throws std::invalid_argument, and str() of a key that has() reports
//      present never throws.
//   3. jsonl::escape output re-parses: {"k":"<escape(v)>"} gives back v
//      for any v without a NUL or newline byte
//      (escape maps a newline to a space by design).
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "qp/tools/jsonl.hpp"

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
    const std::string line(reinterpret_cast<const char*>(data), size);
    try {
        const jsonl::Request r = jsonl::parse(line);
        for (const auto& [k, v] : r.raw) {
            if (!r.has(k)) std::abort();
            (void)r.str(k);
            try {
                (void)r.num(k);
            } catch (const std::invalid_argument&) {
            }
            try {
                (void)r.flag(k);
            } catch (const std::invalid_argument&) {
            }
        }
    } catch (const std::invalid_argument&) {
    }

    if (line.find('\0') == std::string::npos && line.find('\n') == std::string::npos) {
        const std::string doc = "{\"k\":\"" + jsonl::escape(line) + "\"}";
        const jsonl::Request back = jsonl::parse(doc);
        if (!back.has("k") || back.str("k") != line) std::abort();
    }
    return 0;
}
