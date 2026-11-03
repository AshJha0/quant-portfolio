# qp::aad

This is the header-only C++ port of the adjoint algorithmic differentiation
in `qp_core.aad` (Python) and `rust/aad` (plan workstream W19). It is all in
`include/qp/aad.hpp`.

It has the same parts as the other two ports:
- the tape (at most two parents per node, with `mark()` and `rewind()` for
  per-path Monte Carlo sweeps);
- the Black-Scholes, pathwise Monte Carlo and swap pricers;
- quote Greeks from the implicit function theorem.

`rust/aad/README.md` explains the method, lists the sources and has the
cost-ratio table for all three languages.

**Oracle.** `tests/test_aad.cpp` makes 105 checks:
- closed-form Black-Scholes Greeks, to 1e-12;
- central bumps of every input, to 1e-6. The Monte Carlo bumps use the same
  normals.
- central bumps of every market quote, with the curve rebuilt by
  `qp::rates::bootstrap_dated` from the QuantLib 1.44 USD OIS fixture, to
  1e-6.

The test is a plain executable, so no test framework is downloaded.

    cmake -S cpp/aad -B build/aad -DQP_WERROR=ON
    cmake --build build/aad -j && ctest --test-dir build/aad
    build/aad/qp_aad_cost_ratio            # AAD time / price time

The test is clean under g++ 15.2 with `-Werror` and with
`-fsanitize=address,undefined`. MSVC and clang have not been run locally.
