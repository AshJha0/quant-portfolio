#!/usr/bin/env bash
# Build and test the standalone C++ projects that are not in the engine matrix.
set -uo pipefail
ROOT=${1:-.}
B=${2:-/tmp/qp-cppnew}
rc=0
for d in exec-core rates-curves aad ull/feed ull/gateway ull/bench; do
  echo "=== cpp/$d"
  bd="$B/${d//\//_}"
  if cmake -S "$ROOT/cpp/$d" -B "$bd" -G Ninja -DCMAKE_BUILD_TYPE=Release -DQP_WERROR=ON >"$bd.cfg.log" 2>&1 \
     && cmake --build "$bd" -j 4 >"$bd.build.log" 2>&1; then
    (cd "$bd" && ctest --output-on-failure -LE long -j 4 -E 'ull_qpbf_roundtrip_1e8|fuzz') 2>&1 | tail -4 || rc=1
    [[ ${PIPESTATUS[0]} -eq 0 ]] || rc=1
  else
    echo "BUILD FAILED cpp/$d"; tail -30 "$bd.cfg.log" "$bd.build.log"; rc=1
  fi
done
exit $rc
