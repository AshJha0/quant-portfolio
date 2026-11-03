// Dependency-free test harness for cpp/ull/{feed,gateway} tests.
// TEST(name) registers a case; CHECK/CHECK_EQ record failures and continue;
// main() runs every case and returns non-zero if any check failed.
#pragma once

#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qp::ull::testing {

struct Case {
  const char* name;
  std::function<void()> fn;
};
inline std::vector<Case>& registry() {
  static std::vector<Case> r;
  return r;
}
inline int& failures() {
  static int f = 0;
  return f;
}
inline long& checks() {
  static long c = 0;
  return c;
}
struct Reg {
  Reg(const char* n, std::function<void()> f) { registry().push_back({n, std::move(f)}); }
};
inline std::string slurp(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}
inline int run_all() {
  int bad_cases = 0;
  for (auto& c : registry()) {
    int before = failures();
    try {
      c.fn();
    } catch (const std::exception& e) {
      std::printf("  EXCEPTION in %s: %s\n", c.name, e.what());
      ++failures();
    }
    bool ok = failures() == before;
    if (!ok) ++bad_cases;
    std::printf("[%s] %s\n", ok ? " OK " : "FAIL", c.name);
  }
  std::printf("%zu cases, %d failed, %ld checks\n", registry().size(), bad_cases, checks());
  return failures() == 0 ? 0 : 1;
}

}  // namespace qp::ull::testing

#define QPT_CAT2(a, b) a##b
#define QPT_CAT(a, b) QPT_CAT2(a, b)
#define TEST(name)                                                                       \
  static void QPT_CAT(test_, name)();                                                    \
  static ::qp::ull::testing::Reg QPT_CAT(reg_, name)(#name, &QPT_CAT(test_, name));      \
  static void QPT_CAT(test_, name)()
#define CHECK(cond)                                                                      \
  do {                                                                                   \
    ++::qp::ull::testing::checks();                                                      \
    if (!(cond)) {                                                                       \
      std::printf("  %s:%d CHECK(%s) failed\n", __FILE__, __LINE__, #cond);             \
      ++::qp::ull::testing::failures();                                                  \
    }                                                                                    \
  } while (0)
#define CHECK_EQ(a, b)                                                                   \
  do {                                                                                   \
    ++::qp::ull::testing::checks();                                                      \
    if (!((a) == (b))) {                                                                 \
      std::ostringstream qpt_ss;                                                         \
      qpt_ss << (a) << " != " << (b);                                                    \
      std::printf("  %s:%d CHECK_EQ(%s, %s) failed: %s\n", __FILE__, __LINE__, #a, #b,   \
                  qpt_ss.str().c_str());                                                 \
      ++::qp::ull::testing::failures();                                                  \
    }                                                                                    \
  } while (0)
#define CHECK_THROWS(expr)                                                               \
  do {                                                                                   \
    ++::qp::ull::testing::checks();                                                      \
    bool qpt_threw = false;                                                              \
    try { (void)(expr); } catch (...) { qpt_threw = true; }                              \
    if (!qpt_threw) {                                                                    \
      std::printf("  %s:%d CHECK_THROWS(%s) did not throw\n", __FILE__, __LINE__, #expr); \
      ++::qp::ull::testing::failures();                                                  \
    }                                                                                    \
  } while (0)
#define QPT_MAIN() \
  int main() { return ::qp::ull::testing::run_all(); }
