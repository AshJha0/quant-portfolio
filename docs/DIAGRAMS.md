# Architecture & Flow Diagrams

Visual companions to [ARCHITECTURE.md](ARCHITECTURE.md). All diagrams are
Mermaid — GitHub renders them inline. Numbers quoted (test counts, tolerances)
are reproduced from a clean build and match the top-level `README.md`.

---

## 1. The portfolio at a glance — areas, asset classes, languages

```mermaid
flowchart TB
    subgraph AREAS["10 project areas × 2 asset classes = 20 Python projects"]
        direction LR
        A1["01 Options Pricing<br/>& Greeks"]
        A2["02 Volatility<br/>Modeling"]
        A3["03 VaR / ES<br/>Engine"]
        A4["04 Fixed<br/>Income"]
        A5["05 Pairs<br/>Trading"]
        A6["06 Credit<br/>Risk"]
        A7["07 Portfolio<br/>Optimization"]
        A8["08 Algo<br/>Execution"]
        A9["09 Vol<br/>Surface"]
        A10["10 Regime<br/>Switching"]
    end

    subgraph FOUND["3 foundations projects — equity-only, Python-only"]
        direction LR
        F1["F1 Risk Metrics<br/>on Real Data"]
        F2["F2 Trading Signal<br/>Backtest"]
        F3["F3 Black-Scholes<br/>Replication"]
    end

    subgraph TWINS["Compiled twins — the 4 latency-sensitive areas only"]
        direction LR
        CPP["C++20 engines<br/>GoogleTest, -Werror"]
        RUST["Rust 2021 engines<br/>cargo test, zero deps"]
    end

    A1 -.->|golden vectors| CPP & RUST
    A3 -.->|golden vectors| CPP & RUST

    style AREAS fill:#1f2937,color:#fff
    style FOUND fill:#374151,color:#fff
    style TWINS fill:#111827,color:#fff
```

Why areas 1 and 3 specifically get compiled twins, not the other eight: see
[ARCHITECTURE.md](ARCHITECTURE.md) "Why only these four get compiled twins."

---

## 2. Cross-language golden-vector validation pipeline

The technical backbone of the whole portfolio — how a Python-computed number
becomes a pinned test assertion in two other languages.

```mermaid
flowchart LR
    subgraph PY["Python reference project"]
        MODEL["black_scholes.py /<br/>garman_kohlhagen.py /<br/>parametric_var.py / ..."]
        GEN["tests/golden/<br/>generate_golden.py"]
    end

    subgraph GOLD["Golden vector generation"]
        JSON["golden case list<br/>(inputs + expected outputs)"]
        HGEN["tools/gen_golden_header.py"]
        RGEN["tools/gen_golden_rs.py"]
    end

    subgraph CPPE["C++ engine"]
        HPP["tests/golden_vectors.hpp<br/>(constexpr, committed)"]
        GT["GoogleTest<br/>AllGoldenCases suite"]
    end

    subgraph RUSTE["Rust engine"]
        GRS["src/golden.rs<br/>(committed)"]
        CT["cargo test<br/>golden module"]
    end

    MODEL --> GEN --> JSON
    JSON --> HGEN --> HPP --> GT
    JSON --> RGEN --> GRS --> CT
    GT -->|"1e-9 (options)<br/>1e-6..1e-8 (VaR/ES)"| PASS(["identical result,<br/>3 independent codebases"])
    CT -->|same tolerance| PASS

    style PASS fill:#065f46,color:#fff
```

**What this proves**: the C++ and Rust engines compute the *same numbers* as
the Python reference for every pinned case, to floating-point-noise tolerance.
**What it does not prove**: that the Python reference itself is correct — that
burden is carried by each project's own analytic-identity and convergence
tests (put-call parity, Greeks-vs-finite-difference, tree→BS convergence),
not by cross-language agreement. Two wrong implementations that made the same
mistake would still agree with each other.

---

## 3. Per-area data flow — Options Pricing & Greeks (areas 01/01)

```mermaid
flowchart TD
    IN["Spot, strike, rate(s), vol, time-to-expiry<br/>(equity: q dividend yield; FX: r_f foreign rate)"]
    VALID["Input validation<br/>(finite, positive vol/T, rate sanity)"]
    D1D2["d1, d2<br/>(Black-Scholes / Garman-Kohlhagen)"]
    PRICE["call_price / put_price<br/>(closed form)"]
    GREEKS["Greeks: delta, gamma, vega,<br/>theta, rho (analytic + FD cross-check)"]
    IV["implied_volatility(price, ...)<br/>Newton-Raphson, vega-scaled step"]
    STALL{"Newton stalled /<br/>plateau near vega≈0?"}
    BISECT["bracket-bisection<br/>refinement to convergence"]
    OUT["priced option + Greeks + implied vol"]
    TESTS["tests: put-call parity, no-arbitrage bounds,<br/>monotonic-in-vol, tree→BS convergence,<br/>FD Greeks agreement, IV round-trip"]

    IN --> VALID --> D1D2 --> PRICE --> GREEKS --> OUT
    PRICE --> IV --> STALL
    STALL -->|yes| BISECT --> OUT
    STALL -->|no, converged| OUT
    OUT -.->|golden vectors| TESTS

    style STALL fill:#7c2d12,color:#fff
    style BISECT fill:#065f46,color:#fff
```

The `STALL` branch is the fix from the deep-rigor review pass: all six
options engines (Python/C++/Rust × equity/FX) used to exit as soon as the
Newton residual stopped improving, which silently lost precision (or, in the
FX engines, saturated at the no-arbitrage bound and returned a vol that did
not actually reprice the input) near flat-vega regions. See Round 3 in
[LEARN.md](LEARN.md) and diagram 9 below.

---

## 4. Per-area data flow — Market Risk VaR / Expected Shortfall (areas 03/03)

```mermaid
flowchart TD
    subgraph INPUTS["Inputs"]
        HIST["historical P&L /<br/>return series"]
        COV["factor covariance<br/>matrix"]
        BOOK["portfolio exposures<br/>(FX: full Book with<br/>spot/fwd/option legs)"]
    end

    HVAR["historical_var<br/>(plain + age-weighted)"]
    PVAR["parametric_var<br/>(delta-normal + Cornish-Fisher<br/>skew/kurtosis correction)"]
    MVAR["monte_carlo_var<br/>(normal / Student-t / jump-mixture<br/>factor simulation, full revaluation)"]
    ES["expected_shortfall<br/>(tail integral, same rank rule<br/>as the VaR quantile)"]
    SE1["order-statistic / KDE<br/>standard error"]
    SE2["bootstrap standard error<br/>(distribution-free cross-check)"]
    BT["backtesting: Kupiec POF,<br/>Basel traffic-light zones"]
    OUT["VaR, ES, SE, backtest verdict"]

    HIST --> HVAR
    COV --> PVAR
    BOOK --> MVAR
    COV --> MVAR
    HVAR & PVAR & MVAR --> ES
    MVAR --> SE1
    MVAR --> SE2
    HVAR & PVAR & MVAR & ES & SE1 & SE2 --> OUT
    OUT --> BT

    style SE2 fill:#065f46,color:#fff
```

`SE2` (bootstrap standard error) was added to all four Monte Carlo VaR engines
(equity/FX × C++/Rust) in the deep-rigor review pass specifically because
`SE1`'s local density estimate systematically underestimates the true
sampling error by 9-17% at deep tails or modest scenario counts — see
Round 7 in [LEARN.md](LEARN.md).

---

## 5. Per-area data flow — Volatility Surface & Stochastic Vol (areas 09/09)

```mermaid
flowchart LR
    QUOTES["Market vol quotes<br/>(equity: strike/expiry grid;<br/>FX: ATM/RR/BF per tenor)"]
    SURF["Surface construction<br/>(interpolation across<br/>strike × expiry)"]
    HESTON["Heston stochastic-vol<br/>calibration + Monte Carlo<br/>(FX: antithetic variance reduction)"]
    ARB["No-arbitrage checks<br/>(calendar-spread, butterfly)"]
    OUT["Full implied-vol surface<br/>+ MC-consistent pricer"]

    QUOTES --> SURF --> ARB --> OUT
    SURF --> HESTON --> OUT
```

---

## 6. Per-area data flow — Portfolio Optimization & Risk Allocation (areas 07/07)

```mermaid
flowchart LR
    RETURNS["Expected returns +<br/>covariance estimate"]
    CONSTRAINTS["Constraints<br/>(long-only, box, sector,<br/>turnover, leverage)"]
    MVO["Mean-variance /<br/>min-variance / max-Sharpe<br/>optimizer"]
    RP["Risk parity /<br/>risk budgeting"]
    ATTR["Risk attribution<br/>(marginal contribution,<br/>component VaR)"]
    OUT["Optimal weights +<br/>risk decomposition"]

    RETURNS --> MVO
    CONSTRAINTS --> MVO
    RETURNS --> RP
    MVO & RP --> OUT --> ATTR
```

---

## 7. Per-area data flow — Algorithmic Trading & Execution (areas 08/08)

```mermaid
flowchart LR
    ORDER["Parent order<br/>(size, side, horizon)"]
    SCHED["Execution schedule<br/>(TWAP / VWAP / participation-rate)"]
    SIM["Fill simulation against<br/>synthetic/historical volume+price"]
    COST["Transaction cost analysis<br/>(implementation shortfall,<br/>market impact)"]
    OUT["Executed schedule +<br/>realized cost vs. benchmark"]

    ORDER --> SCHED --> SIM --> COST --> OUT
```

---

## 8. Per-area data flow — Regime-Switching Strategy (areas 10/10)

```mermaid
flowchart LR
    RETURNS["Return series"]
    HMM["Regime detection<br/>(e.g. Markov-switching /<br/>volatility-regime classifier)"]
    SIGNAL["Regime-conditional<br/>signal / allocation"]
    BT["backtest.py<br/>(no-look-ahead, costed,<br/>walk-forward)"]
    OUT["Regime-aware equity curve<br/>vs. static benchmark"]

    RETURNS --> HMM --> SIGNAL --> BT --> OUT
```

---

## 9. The NaN-guard defect class — and how it was found and fixed

```mermaid
flowchart TD
    IN["Input reaches a public<br/>entry point (NaN, from an<br/>upstream feed/calc)"]
    GUARD1["Guard written as:<br/>if x &lt;= 0: raise ValueError"]
    IEEE["IEEE 754: NaN &lt;= 0 is False<br/>for every comparison"]
    SILENT["Guard silently passes —<br/>NaN flows into the pricer/risk calc"]
    BAD1["Pricer: NaN price<br/>(visible, at least)"]
    BAD2["max(NaN, 0.0) in Rust/C++<br/>VaR aggregation propagates the<br/>NON-NaN operand"]
    BAD3["Backtest: NaN P&L counted<br/>as zero breaches —<br/>a broken model passes Kupiec green"]
    FIX["Fix: explicit finiteness checks<br/>(isfinite / is_finite) at every<br/>public entry point, portfolio-wide"]
    ZERO["Most dangerous outcome:<br/>portfolio VaR of exactly 0.0 —<br/>MORE dangerous than a visible NaN,<br/>because it looks like a hedged book"]

    IN --> GUARD1 --> IEEE --> SILENT
    SILENT --> BAD1
    SILENT --> BAD2 --> ZERO
    SILENT --> BAD3
    ZERO -.->|found & fixed, review pass 1| FIX
    BAD1 -.-> FIX
    BAD3 -.-> FIX

    style ZERO fill:#7c2d12,color:#fff
    style FIX fill:#065f46,color:#fff
```

---

## 10. Anatomy of the implied-vol solver fix (all six options engines)

```mermaid
flowchart TD
    START["Newton-Raphson step:<br/>sigma -= (price(sigma) - target) / vega(sigma)"]
    CHECK{"|price(sigma) - target|<br/>stopped improving?"}
    OLD["OLD behaviour:<br/>exit immediately<br/>(precision loss, or in FX<br/>saturate at no-arb bound)"]
    NEW["NEW behaviour:<br/>always fall through to<br/>bracket-bisection, run to<br/>full convergence tolerance"]
    RESULT_OLD["Silently imprecise —<br/>or, worst case, a vol that<br/>does not reprice the input"]
    RESULT_NEW["Converged implied vol,<br/>every time, regardless of<br/>local vega flatness"]

    START --> CHECK
    CHECK -->|yes, old code| OLD --> RESULT_OLD
    CHECK -->|yes, new code| NEW --> RESULT_NEW
    CHECK -->|no| START

    style OLD fill:#7c2d12,color:#fff
    style RESULT_OLD fill:#7c2d12,color:#fff
    style NEW fill:#065f46,color:#fff
    style RESULT_NEW fill:#065f46,color:#fff
```

No golden-vector value changed as a result of this fix — it is a robustness
correction to how the solver behaves near flat vega, not to the pricing
formula the golden vectors pin. Full write-up: Round 3 in
[LEARN.md](LEARN.md).

---

## 11. Anatomy of the Cornish-Fisher grid-resolution fix (all six VaR/ES engines)

```mermaid
flowchart LR
    subgraph OLD["OLD: finite-grid domain check"]
        G1(["grid point"]) --- G2(["grid point"]) --- G3(["grid point"])
        MISS(["non-monotone region<br/>hiding BETWEEN grid points —<br/>invisible to the scan"])
        G2 -.-> MISS
    end

    subgraph NEW["NEW: exact closed-form stationary point"]
        FORM["Cornish-Fisher expansion's<br/>stationary point, solved exactly<br/>(quadratic in the standardized quantile)"]
        CATCH["Cannot miss a non-monotone<br/>region — no grid to have gaps in"]
        FORM --> CATCH
    end

    OLD -.->|replaced by| NEW

    style MISS fill:#7c2d12,color:#fff
    style CATCH fill:#065f46,color:#fff
```

---

## 12. Testing & CI pipeline — how "5,963 tests, all passing" is produced

```mermaid
flowchart TD
    subgraph PYCHECK["Each of 23 Python projects"]
        P1["pip install -e ."] --> P2["pytest<br/>(pyproject addopts=-q)"]
    end
    subgraph CPPCHECK["Each of 4 C++ engines"]
        C1["cmake -S . -B build"] --> C2["cmake --build build -j"] --> C3["ctest --test-dir build"]
    end
    subgraph RUSTCHECK["Each of 4 Rust engines"]
        R1["RUSTFLAGS='-D warnings'"] --> R2["cargo test --release"]
    end

    PYCHECK --> SUM["Sum every project's own<br/>reported test count —<br/>never taken from a prior run's<br/>build log or an agent's self-report"]
    CPPCHECK --> SUM
    RUSTCHECK --> SUM
    SUM --> TOTAL(["5,963 passing tests,<br/>reproduced from a clean build"])

    style TOTAL fill:#065f46,color:#fff
```

Sequential, not parallel/backgrounded, for the C++ builds specifically — two
`cmake --build` invocations sharing a build directory (or racing for CPU in a
resource-constrained CI runner) is its own failure mode independent of the
code being tested.

---

## 13. The full portfolio review history, in one line

```mermaid
flowchart LR
    V1["Initial 20+3-project<br/>buildout"] --> V2["Hardening pass 1:<br/>NaN-guard defect class,<br/>found & fixed portfolio-wide"]
    V2 --> V3["Hardening pass 2:<br/>implied-vol plateau +<br/>Cornish-Fisher grid bugs,<br/>found & fixed in all 12<br/>golden engines"]
    V3 --> V4["Follow-up: MC VaR<br/>standard-error bootstrap<br/>cross-check, ported to<br/>all 4 remaining C++/Rust<br/>MC VaR engines"]
    V4 --> V5(["Current state:<br/>5,963 tests, independently<br/>reproduced, zero open<br/>follow-ups"])

    style V5 fill:#065f46,color:#fff
```

Every one of these passes is a teaching case study in
[LEARN.md](LEARN.md) Round 18 — what the bug actually was, why it survived
the first round of tests, and what kind of test would have caught it sooner.
