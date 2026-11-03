#!/usr/bin/env bash
# Build the C++ and Rust benches of two commits and run them in alternating
# rounds on this machine, for .github/scripts/bench_ab.py.
#
#   bench_ab_run.sh BASE_SHA OUT_DIR [ROUNDS]
#
# The head is the checked-out working tree; the base is checked out into a
# git worktree next to it. Results land in OUT_DIR/r<k>/<side>/<lang>-<engine>.json.
# A bench that does not build on the base (for example a new engine) is
# skipped there; bench_ab.py then reports that engine without failing.
set -euo pipefail

base_sha="$1"
out="$2"
rounds="${3:-3}"
head_dir="$PWD"
base_dir="$RUNNER_TEMP/bench-base"

engines=(equity-options-engine:eqopt_bench:eq-options-engine
         equity-var-engine:bench_eqvar:eq-var-engine
         fx-options-engine:fxopt_bench:fx-options-engine
         fx-var-engine:fxvar_bench:fx-var-engine)

git worktree add --detach "$base_dir" "$base_sha"

# Both sides are timed with the head's harness, so the gate compares library
# code and never two timing methods. Before this, a base older than the
# shared harness made rust/fx-var-engine portfolio_sigma_50f look 1.49x slower
# with identical library code (2,342 ns on both sides under one harness).
# If the base library does not build under the head's harness, the engine is
# skipped on the base below and bench_ab.py reports it without failing.
cp "$head_dir/cpp/core/include/qp/bench/latency.hpp" \
   "$base_dir/cpp/core/include/qp/bench/latency.hpp" 2> /dev/null ||
    { mkdir -p "$base_dir/cpp/core/include/qp/bench" &&
      cp "$head_dir/cpp/core/include/qp/bench/latency.hpp" "$base_dir/cpp/core/include/qp/bench/"; }
cp "$head_dir/rust/core/src/bench.rs" "$base_dir/rust/core/src/bench.rs"
grep -q '^pub mod bench;' "$base_dir/rust/core/src/lib.rs" ||
    echo 'pub mod bench;' >> "$base_dir/rust/core/src/lib.rs"
for e in "${engines[@]}"; do
    IFS=: read -r engine _ _ <<<"$e"
    if [ -d "$base_dir/cpp/$engine/bench" ]; then
        cp "$head_dir/cpp/$engine/bench/bench_main.cpp" "$base_dir/cpp/$engine/bench/"
    fi
    if [ -d "$base_dir/rust/$engine/src/bin" ]; then
        cp "$head_dir/rust/$engine/src/bin/bench.rs" "$base_dir/rust/$engine/src/bin/"
    fi
done

build() {  # build <tree> <bin dir>
    local tree="$1" bin="$2"
    mkdir -p "$bin"
    for e in "${engines[@]}"; do
        IFS=: read -r engine target package <<<"$e"
        if [ -d "$tree/cpp/$engine" ] &&
           cmake -S "$tree/cpp/$engine" -B "$bin/build-$engine" -G Ninja \
                 -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-15 > /dev/null &&
           cmake --build "$bin/build-$engine" --target "$target" -j 4 > /dev/null; then
            cp "$bin/build-$engine/$target" "$bin/cpp-$engine"
        else
            echo "::warning::C++ bench $engine did not build at $tree"
        fi
        if (cd "$tree/rust" && CARGO_TARGET_DIR="$bin/target" \
                cargo build --release --locked -p "$package" --bin bench > /dev/null); then
            cp "$bin/target/release/bench" "$bin/rust-$engine"
        else
            echo "::warning::Rust bench $engine did not build at $tree"
        fi
    done
}

build "$base_dir" "$RUNNER_TEMP/bin-base"
build "$head_dir" "$RUNNER_TEMP/bin-head"

for k in $(seq 1 "$rounds"); do
    for side in base head; do
        dir="$out/r$k/$side"
        mkdir -p "$dir"
        for e in "${engines[@]}"; do
            IFS=: read -r engine _ _ <<<"$e"
            for lang in cpp rust; do
                exe="$RUNNER_TEMP/bin-$side/$lang-$engine"
                [ -x "$exe" ] || continue
                # Pinned to one core so the two sides share the same CPU.
                taskset -c 1 "$exe" --label "$side" --out "$dir/$lang-$engine.json" > /dev/null
            done
        done
    done
done

git worktree remove --force "$base_dir"
