// Minimal JSON reader and canonical writer for the ULL feed handler and gateway.
//
// Plain idea: exchange websocket messages and the kill-contract golden vectors
// are small JSON documents. We need to read them without a third-party
// dependency, and we need to write the kill events in exactly the byte form
// Python's json.dumps(sort_keys=True, separators=(",", ":")) produces, because
// the W28d contract is checked byte for byte.
//
// Formal statement: parse() accepts RFC 8259 JSON (objects, arrays, strings
// with all escapes including surrogate pairs, numbers, true/false/null) and
// throws json::Error on anything else. Numbers keep their source text so that
// exchange decimal strings and integers are never rounded through a double.
// escape() reproduces CPython's ensure_ascii=True string encoding: '"' and
// '\\' escaped, \b \f \n \r \t short forms, other C0 controls and every
// non-ASCII code point as \uXXXX (UTF-16 surrogate pairs above U+FFFF).
//
// Not a hot-path component: the handler parses one message at a time off the
// socket thread. The QPBF binary path (part 1) is the zero-allocation one.
#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace qp::ull::json {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct Value {
  enum class Kind { Null, Bool, Number, String, Array, Object };
  Kind kind = Kind::Null;
  bool b = false;
  std::string text;  // number source text, or decoded string
  std::vector<Value> arr;
  std::map<std::string, Value> obj;  // keys unique; duplicate key keeps last

  bool is_null() const { return kind == Kind::Null; }
  bool is_string() const { return kind == Kind::String; }
  bool is_array() const { return kind == Kind::Array; }
  bool is_object() const { return kind == Kind::Object; }
  bool has(const std::string& k) const { return kind == Kind::Object && obj.count(k) != 0; }
  const Value& at(const std::string& k) const {
    auto it = obj.find(k);
    if (kind != Kind::Object || it == obj.end()) throw Error("missing key: " + k);
    return it->second;
  }
  const Value& at(std::size_t i) const {
    if (kind != Kind::Array || i >= arr.size()) throw Error("index out of range");
    return arr[i];
  }
  const std::string& str() const {
    if (kind != Kind::String) throw Error("not a string");
    return text;
  }
  std::int64_t i64() const {
    if (kind == Kind::Number) {
      std::size_t pos = 0;
      long long v = std::stoll(text, &pos);
      if (pos != text.size()) throw Error("not an integer: " + text);
      return v;
    }
    if (kind == Kind::String) {  // Binance sends some ids as numbers, Coinbase as strings
      std::size_t pos = 0;
      long long v = std::stoll(text, &pos);
      if (pos != text.size()) throw Error("not an integer: " + text);
      return v;
    }
    throw Error("not a number");
  }
  double f64() const {
    if (kind != Kind::Number && kind != Kind::String) throw Error("not a number");
    return std::stod(text);
  }
};

namespace detail {

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s) {}
  Value run() {
    Value v = value(0);
    ws();
    if (i_ != s_.size()) fail("trailing characters");
    return v;
  }

 private:
  std::string_view s_;
  std::size_t i_ = 0;
  static constexpr int kMaxDepth = 64;

  [[noreturn]] void fail(const char* what) const {
    throw Error(std::string("json: ") + what + " at offset " + std::to_string(i_));
  }
  void ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }
  char peek() {
    ws();
    if (i_ >= s_.size()) fail("unexpected end");
    return s_[i_];
  }
  void expect(char c) {
    if (peek() != c) fail("unexpected character");
    ++i_;
  }
  void lit(std::string_view w) {
    if (s_.substr(i_, w.size()) != w) fail("bad literal");
    i_ += w.size();
  }
  Value value(int depth) {
    if (depth > kMaxDepth) fail("nesting too deep");
    Value v;
    char c = peek();
    if (c == '{') {
      v.kind = Value::Kind::Object;
      ++i_;
      if (peek() == '}') { ++i_; return v; }
      for (;;) {
        if (peek() != '"') fail("expected key");
        std::string k = string();
        expect(':');
        v.obj[k] = value(depth + 1);
        char d = peek();
        ++i_;
        if (d == '}') break;
        if (d != ',') fail("expected , or }");
      }
    } else if (c == '[') {
      v.kind = Value::Kind::Array;
      ++i_;
      if (peek() == ']') { ++i_; return v; }
      for (;;) {
        v.arr.push_back(value(depth + 1));
        char d = peek();
        ++i_;
        if (d == ']') break;
        if (d != ',') fail("expected , or ]");
      }
    } else if (c == '"') {
      v.kind = Value::Kind::String;
      v.text = string();
    } else if (c == 't') {
      lit("true"); v.kind = Value::Kind::Bool; v.b = true;
    } else if (c == 'f') {
      lit("false"); v.kind = Value::Kind::Bool;
    } else if (c == 'n') {
      lit("null");
    } else {
      v.kind = Value::Kind::Number;
      v.text = number();
    }
    return v;
  }
  std::string number() {
    std::size_t b = i_;
    auto digit = [&] { return i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9'; };
    if (i_ < s_.size() && s_[i_] == '-') ++i_;
    if (!digit()) fail("bad number");
    if (s_[i_] == '0') ++i_; else while (digit()) ++i_;
    if (i_ < s_.size() && s_[i_] == '.') { ++i_; if (!digit()) fail("bad fraction"); while (digit()) ++i_; }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      ++i_;
      if (i_ < s_.size() && (s_[i_] == '+' || s_[i_] == '-')) ++i_;
      if (!digit()) fail("bad exponent");
      while (digit()) ++i_;
    }
    return std::string(s_.substr(b, i_ - b));
  }
  unsigned hex4() {
    if (i_ + 4 > s_.size()) fail("short \\u escape");
    unsigned v = 0;
    for (int k = 0; k < 4; ++k) {
      char c = s_[i_++];
      v <<= 4;
      if (c >= '0' && c <= '9') v |= unsigned(c - '0');
      else if (c >= 'a' && c <= 'f') v |= unsigned(c - 'a' + 10);
      else if (c >= 'A' && c <= 'F') v |= unsigned(c - 'A' + 10);
      else fail("bad hex digit");
    }
    return v;
  }
  static void utf8(std::string& o, unsigned cp) {
    if (cp < 0x80) o += char(cp);
    else if (cp < 0x800) { o += char(0xC0 | (cp >> 6)); o += char(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { o += char(0xE0 | (cp >> 12)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
    else { o += char(0xF0 | (cp >> 18)); o += char(0x80 | ((cp >> 12) & 0x3F)); o += char(0x80 | ((cp >> 6) & 0x3F)); o += char(0x80 | (cp & 0x3F)); }
  }
  std::string string() {
    expect('"');
    std::string o;
    for (;;) {
      if (i_ >= s_.size()) fail("unterminated string");
      char c = s_[i_++];
      if (c == '"') return o;
      if (static_cast<unsigned char>(c) < 0x20) fail("control character in string");
      if (c != '\\') { o += c; continue; }
      if (i_ >= s_.size()) fail("bad escape");
      char e = s_[i_++];
      switch (e) {
        case '"': o += '"'; break;
        case '\\': o += '\\'; break;
        case '/': o += '/'; break;
        case 'b': o += '\b'; break;
        case 'f': o += '\f'; break;
        case 'n': o += '\n'; break;
        case 'r': o += '\r'; break;
        case 't': o += '\t'; break;
        case 'u': {
          unsigned cp = hex4();
          if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (i_ + 2 <= s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
              i_ += 2;
              unsigned lo = hex4();
              if (lo < 0xDC00 || lo > 0xDFFF) fail("bad surrogate pair");
              cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
            } else fail("lone surrogate");
          } else if (cp >= 0xDC00 && cp <= 0xDFFF) fail("lone surrogate");
          utf8(o, cp);
          break;
        }
        default: fail("bad escape");
      }
    }
  }
};

}  // namespace detail

inline Value parse(std::string_view s) { return detail::Parser(s).run(); }

// CPython json.dumps(ensure_ascii=True) string encoding, with the quotes.
inline std::string escape(std::string_view s) {
  static const char* hx = "0123456789abcdef";
  std::string o = "\"";
  auto u = [&](unsigned v) { o += "\\u"; for (int sh = 12; sh >= 0; sh -= 4) o += hx[(v >> sh) & 0xF]; };
  for (std::size_t i = 0; i < s.size();) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      ++i;
      switch (c) {
        case '"': o += "\\\""; break;
        case '\\': o += "\\\\"; break;
        case '\b': o += "\\b"; break;
        case '\f': o += "\\f"; break;
        case '\n': o += "\\n"; break;
        case '\r': o += "\\r"; break;
        case '\t': o += "\\t"; break;
        // CPython's ESCAPE_ASCII is [^\ -~]: DEL (0x7F) is escaped too.
        default: if (c < 0x20 || c == 0x7F) u(c); else o += char(c);
      }
      continue;
    }
    int n = c >= 0xF0 ? 4 : c >= 0xE0 ? 3 : c >= 0xC0 ? 2 : 0;
    if (n == 0 || i + std::size_t(n) > s.size()) throw Error("escape: invalid UTF-8");
    unsigned cp = c & (0xFF >> (n + 1));
    for (int k = 1; k < n; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[i + std::size_t(k)]);
      if ((cc & 0xC0) != 0x80) throw Error("escape: invalid UTF-8");
      cp = (cp << 6) | (cc & 0x3F);
    }
    i += std::size_t(n);
    if (cp >= 0x10000) { cp -= 0x10000; u(0xD800 + (cp >> 10)); u(0xDC00 + (cp & 0x3FF)); }
    else u(cp);
  }
  o += '"';
  return o;
}

}  // namespace qp::ull::json
