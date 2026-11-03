// Tiny helper shared by the snippets: print a token followed by a space.
#pragma once
#include <cstdio>
#include <string>
inline void out(const std::string& s) { std::printf("%s ", s.c_str()); }
inline void out(long long v) { std::printf("%lld ", v); }
