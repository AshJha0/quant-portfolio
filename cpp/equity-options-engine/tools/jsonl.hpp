// Minimal JSON-lines request/response helpers for the differential-test shim.
//
// A request is one flat JSON object per line with number, string or bool
// values. A response is {"id":..,"ok":true,"out":{..}} or
// {"id":..,"ok":false,"err":".."}. Doubles are written with std::format
// "{}" (shortest round-trip form); non-finite values as "nan"/"inf"/"-inf".
#ifndef OPT_TOOLS_JSONL_HPP
#define OPT_TOOLS_JSONL_HPP

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <format>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace jsonl {

struct Request {
    std::map<std::string, std::string> raw;  // value text, strings unquoted

    [[nodiscard]] bool has(const std::string& k) const { return raw.count(k) != 0; }
    [[nodiscard]] const std::string& str(const std::string& k) const {
        auto it = raw.find(k);
        if (it == raw.end()) throw std::invalid_argument("missing field: " + k);
        return it->second;
    }
    [[nodiscard]] double num(const std::string& k) const {
        const std::string& s = str(k);
        char* end = nullptr;
        const double v = std::strtod(s.c_str(), &end);
        if (end == s.c_str() || *end != '\0') throw std::invalid_argument("not a number: " + k);
        return v;
    }
    [[nodiscard]] bool flag(const std::string& k) const {
        const std::string& s = str(k);
        if (s == "true") return true;
        if (s == "false") return false;
        throw std::invalid_argument("not a bool: " + k);
    }
};

inline Request parse(const std::string& line) {
    Request r;
    std::size_t i = 0;
    auto skip = [&] {
        while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    };
    auto read_string = [&] {
        if (line[i] != '"') throw std::invalid_argument("expected string");
        std::string out;
        for (++i; i < line.size() && line[i] != '"'; ++i) {
            if (line[i] == '\\' && i + 1 < line.size()) ++i;
            out.push_back(line[i]);
        }
        if (i >= line.size()) throw std::invalid_argument("unterminated string");
        ++i;
        return out;
    };
    skip();
    if (i >= line.size() || line[i] != '{') throw std::invalid_argument("expected object");
    ++i;
    for (;;) {
        skip();
        if (i < line.size() && line[i] == '}') break;
        const std::string key = read_string();
        skip();
        if (i >= line.size() || line[i] != ':') throw std::invalid_argument("expected ':'");
        ++i;
        skip();
        std::string value;
        if (i < line.size() && line[i] == '"') {
            value = read_string();
        } else {
            while (i < line.size() && line[i] != ',' && line[i] != '}' &&
                   !std::isspace(static_cast<unsigned char>(line[i]))) {
                value.push_back(line[i++]);
            }
        }
        r.raw[key] = value;
        skip();
        if (i < line.size() && line[i] == ',') {
            ++i;
            continue;
        }
        if (i < line.size() && line[i] == '}') break;
        throw std::invalid_argument("malformed object");
    }
    return r;
}

inline std::string number(double v) {
    if (std::isnan(v)) return "\"nan\"";
    if (std::isinf(v)) return v > 0 ? "\"inf\"" : "\"-inf\"";
    return std::format("{}", v);
}

inline std::string escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        if (c == '"' || c == '\\') out.push_back('\\');
        out.push_back(c == '\n' ? ' ' : c);
    }
    return out;
}

inline std::string ok(const std::string& id, const std::vector<std::pair<std::string, double>>& out) {
    std::string s = "{\"id\":" + id + ",\"ok\":true,\"out\":{";
    for (std::size_t k = 0; k < out.size(); ++k) {
        if (k) s += ",";
        s += "\"" + out[k].first + "\":" + number(out[k].second);
    }
    return s + "}}";
}

inline std::string err(const std::string& id, const std::string& msg) {
    return "{\"id\":" + id + ",\"ok\":false,\"err\":\"" + escape(msg) + "\"}";
}

}  // namespace jsonl

#endif  // OPT_TOOLS_JSONL_HPP
