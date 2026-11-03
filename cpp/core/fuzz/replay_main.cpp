// replay_main.cpp: run a libFuzzer target over files without libFuzzer.
//
// GCC and MSVC have no -fsanitize=fuzzer. Linking a target with this main
// turns it into a plain program that feeds each file named on the command
// line (or every file in each named directory) to LLVMFuzzerTestOneInput
// once. CI uses it to replay the checked-in corpus and every crash
// reproducer under GCC with ASan and UBSan, so a fixed bug stays fixed.
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

namespace {
int run_file(const std::filesystem::path& p) {
    std::ifstream f(p, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(f)),
                                  std::istreambuf_iterator<char>());
    return LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()),
                                  bytes.size());
}
}  // namespace

int main(int argc, char** argv) {
    std::size_t n = 0;
    for (int a = 1; a < argc; ++a) {
        const std::filesystem::path p(argv[a]);
        if (std::filesystem::is_directory(p)) {
            for (const auto& e : std::filesystem::directory_iterator(p)) {
                if (e.is_regular_file()) {
                    run_file(e.path());
                    ++n;
                }
            }
        } else {
            run_file(p);
            ++n;
        }
    }
    std::printf("replayed %zu inputs\n", n);
    return 0;
}
