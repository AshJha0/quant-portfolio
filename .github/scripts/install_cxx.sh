#!/usr/bin/env bash
# Install one C++ toolchain on an Ubuntu 24.04 runner, plus CMake, Ninja and
# GoogleTest, and export CC / CXX through $GITHUB_ENV.
#
#   gcc    GCC 15 from the ubuntu-toolchain-r/test PPA
#   clang  Clang 21 from apt.llvm.org (llvm-toolchain-noble-21)
#
# GoogleTest comes from the distribution package libgtest-dev, which ships
# static libraries and the GTestConfig.cmake that find_package(GTest) uses.
set -euo pipefail

kind="${1:?usage: install_cxx.sh gcc|clang}"
# shellcheck source=/dev/null
codename="$(. /etc/os-release && echo "$VERSION_CODENAME")"

sudo apt-get update -qq
sudo apt-get install -y -qq --no-install-recommends \
  ca-certificates gnupg software-properties-common cmake ninja-build libgtest-dev

case "$kind" in
  gcc)
    sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends gcc-15 g++-15
    cc=gcc-15
    cxx=g++-15
    ;;
  clang)
    curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
      | sudo gpg --dearmor -o /usr/share/keyrings/llvm.gpg
    echo "deb [signed-by=/usr/share/keyrings/llvm.gpg] https://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-21 main" \
      | sudo tee /etc/apt/sources.list.d/llvm-21.list >/dev/null
    sudo apt-get update -qq
    sudo apt-get install -y -qq --no-install-recommends clang-21 libclang-rt-21-dev
    cc=clang-21
    cxx=clang++-21
    ;;
  *)
    echo "unknown toolchain: $kind" >&2
    exit 2
    ;;
esac

"$cxx" --version
cmake --version | head -1
{
  echo "CC=$cc"
  echo "CXX=$cxx"
} >> "${GITHUB_ENV:-/dev/null}"
