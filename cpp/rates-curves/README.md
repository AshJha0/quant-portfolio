# qp::rates_curves

This is the header-only C++ port of the dated OIS, multi-curve and
cross-currency curve bootstrap (plan workstream W13b), in
`include/qp/rates_curves.hpp`. It uses the same algorithm, the same loop
order and the same QuantLib fixture as the Python code
(`fi_rates.bootstrap`, `fi_rates.multicurve`, `fx_rates.xccy`) and the Rust
code (`rust/rates-curves`). The formulas are written out at the top of that
README.

**Oracle.** `tests/test_rates_curves.cpp` reads
`python/equity/04-fixed-income/tests/data/dated_bootstrap_usd.txt`, which
was written by QuantLib 1.44. Every pillar must match QuantLib to 1e-12 in
discount factor, and every quote must reprice to 1e-10 in rate.

    cmake -S cpp/rates-curves -B build/rates-curves -DQP_WERROR=ON
    cmake --build build/rates-curves -j
    ctest --test-dir build/rates-curves
