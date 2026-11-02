#!/usr/bin/env bash
# Install C++ toolchains on an Ubuntu 24.04 runner, plus CMake, Ninja and
# GoogleTest, and export CC / CXX through $GITHUB_ENV.
#
#   gcc    GCC 15 from the ubuntu-toolchain-r/test PPA     (CC=gcc-15)
#   clang  Clang 21 from apt.llvm.org                      (CC=clang-21)
#   both   GCC 15 and Clang 21                             (CC=clang-21)
#   tools  Clang 21 plus clang-format-21 and clang-tidy-21 (CC=clang-21)
#
# The CMake presets (cpp/<engine>/CMakePresets.json) choose the compiler.
# CC / CXX are exported for presets that read them from the environment.
#
# GoogleTest comes from the distribution package libgtest-dev, which ships
# static libraries and the GTestConfig.cmake that find_package(GTest) uses.
set -euo pipefail

kind="${1:?usage: install_cxx.sh gcc|clang|both|tools}"
# shellcheck source=/dev/null
codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"

sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
  ca-certificates gnupg software-properties-common cmake ninja-build libgtest-dev

install_gcc() {
  sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
  sudo apt-get update -qq
  sudo apt-get install -y -qq --no-install-recommends gcc-15 g++-15
  g++-15 --version | head -1
}

install_clang() {
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
    | sudo gpg --dearmor --yes -o /usr/share/keyrings/llvm.gpg
  echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] https://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-21 main" \
    | sudo tee /etc/apt/sources.list.d/llvm-21.list >/dev/null
  sudo apt-get update -qq
  sudo apt-get install -y -qq --no-install-recommends clang-21 libclang-rt-21-dev "$@"
  clang++-21 --version | head -1
}

case "$kind" in
  gcc)   install_gcc;   cc=gcc-15;   cxx=g++-15 ;;
  clang) install_clang; cc=clang-21; cxx=clang++-21 ;;
  both)  install_gcc; install_clang; cc=clang-21; cxx=clang++-21 ;;
  tools)
    install_clang clang-format-21 clang-tidy-21
    clang-format-21 --version
    clang-tidy-21 --version | head -2
    cc=clang-21
    cxx=clang++-21
    ;;
  *)
    echo "unknown toolchain: $kind" >&2
    exit 2
    ;;
esac

cmake --version | head -1
{
  echo "CC=$cc"
  echo "CXX=$cxx"
} >> "${GITHUB_ENV:-/dev/null}"
