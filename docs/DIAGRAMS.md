# Architecture & Flow Diagrams

Diagrams for [ARCHITECTURE.md](ARCHITECTURE.md). All diagrams are Mermaid and
render on GitHub. Test counts are not quoted here. The CI `counts` job writes
them to `counts.json`.

---

## 1. The portfolio at a glance: areas, asset classes, languages

```mermaid
flowchart TB
    subgraph AREAS["10 project areas x 2 asset classes = 20 Python projects"]
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

    subgraph FOUND["3 foundations projects: equity-only, Python-only"]
        direction LR
        F1["F1 Risk<br/>Metrics"]
        F2["F2 Trading Signal<br/>Backtest"]
        F3["F3 Black-Scholes<br/>Replication"]
    end

    subgraph PORTS["C++ and Rust ports of areas 01 and 03"]
        direction LR
        CPP["C++23 engines + qp::core<br/>GoogleTest"]
        RUST["Rust 2021 engines<br/>cargo workspace + qp-core"]
    end

    A1 -.->|golden vectors| CPP & RUST
    A3 -.->|"golden vectors<br/>(linear book)"| CPP & RUST

    style AREAS fill:#1f2937,color:#fff
    style FOUND fill:#374151,color:#fff
    style PORTS fill:#111827,color:#fff
```

Why only areas 01 and 03 are ported: see [ARCHITECTURE.md](ARCHITECTURE.md)
"The 10 project areas".

---

## 2. Golden-vector pipeline

How a Python-side expected value becomes a test assertion in C++ and Rust.

```mermaid
flowchart LR
    subgraph PY["Python reference project"]
        ORACLE["options: mpmath oracle<br/>VaR/ES: Python reference engine"]
        GEN["tests/golden/<br/>generate_golden.py"]
        LOCK["tests/test_golden_vectors.py<br/>(Python-side lock)"]
    end

    subgraph GOLD["Generated files"]
        JSON["tests/golden/golden_vectors.json<br/>(committed; inputs, outputs,<br/>per-output tolerances)"]
        HGEN["cpp/*/tools/gen_golden_header.py"]
        RGEN["rust/*/tools/gen_golden_rs.py"]
        CHK["CI job: goldens --check"]
    end

    subgraph CPPE["C++ engine"]
        HPP["tests/golden_vectors.hpp<br/>(constexpr, committed)"]
        GT["GoogleTest"]
    end

    subgraph RUSTE["Rust engine"]
        GRS["src/golden.rs (options) /<br/>tests/golden/mod.rs (VaR)<br/>(committed)"]
        CT["cargo test"]
    end

    ORACLE --> GEN --> JSON
    JSON -.-> LOCK
    JSON --> HGEN --> HPP --> GT
    JSON --> RGEN --> GRS --> CT
    HGEN -.-> CHK
    RGEN -.-> CHK
```

Each options output carries its own tolerance from the JSON's
`tolerance_rules`. VaR/ES values come from the Python engine through the same
generator. Agreement shows the ports compute what Python computes. It does not
show that Python is correct. Each project's analytic-identity and convergence
tests carry that.

---

## 3. Differential-test pipeline (options engines)

```mermaid
flowchart LR
    subgraph PYT["pytest: python/{equity,fx}/01-options-pricing/tests/test_differential.py"]
        GRID["input grid"]
        REF["Python result"]
        CMP["compare per output"]
    end

    subgraph LOC["locate shim"]
        ENV["env var set?<br/>QP_RS_EQ_JSONL, QP_RS_FX_JSONL,<br/>QP_CPP_EQ_JSONL, QP_CPP_FX_JSONL"]
        DEF["default build path<br/>rust/target/release/&lt;bin&gt;<br/>cpp/&lt;engine&gt;/build/&lt;target&gt;<br/>or build/gcc/&lt;target&gt;"]
        SKIP(["binary absent:<br/>pytest.skip"])
    end

    subgraph SHIMS["JSONL shims"]
        RS["Rust bins<br/>eq_jsonl, fx_jsonl"]
        CX["C++ targets<br/>eqopt_jsonl, fxopt_jsonl"]
    end

    GRID --> REF --> CMP
    GRID --> ENV
    ENV -->|no| DEF
    ENV -->|yes| SHIMS
    DEF -->|found| SHIMS
    DEF -->|missing| SKIP
    SHIMS -->|JSON lines out| CMP
```

---

## 4. Data flow: Options Pricing & Greeks (area 01)

```mermaid
flowchart TD
    IN["Spot, strike, rate(s), vol, time-to-expiry<br/>(equity: q dividend yield; FX: r_f foreign rate)"]
    VALID["Input validation<br/>(finite, positive vol/T, rate sanity)"]
    D1D2["d1, d2<br/>(Black-Scholes / Garman-Kohlhagen)"]
    PRICE["call_price / put_price<br/>(closed form)"]
    GREEKS["Greeks: delta, gamma, vega,<br/>theta, rho (analytic + FD cross-check)"]
    IV["implied_volatility(price, ...)<br/>Newton-Raphson, vega-scaled step"]
    STALL{"Newton stalled /<br/>plateau near vega≈0?"}
    BISECT["bracket-bisection<br/>to convergence"]
    OUT["priced option + Greeks + implied vol"]
    TESTS["tests: put-call parity, no-arbitrage bounds,<br/>monotonic-in-vol, tree to BS convergence,<br/>FD Greeks agreement, IV round-trip"]

    IN --> VALID --> D1D2 --> PRICE --> GREEKS --> OUT
    PRICE --> IV --> STALL
    STALL -->|yes| BISECT --> OUT
    STALL -->|no, converged| OUT
    OUT -.->|golden vectors| TESTS
```

When Newton stops improving, the solver falls through to bisection and runs to
the full tolerance.

---

## 5. Data flow: Market Risk VaR / Expected Shortfall (area 03)

```mermaid
flowchart TD
    subgraph INPUTS["Inputs (synthetic unless named)"]
        HIST["historical P&L /<br/>return series"]
        COV["factor covariance<br/>matrix"]
        BOOK["portfolio exposures<br/>(Python FX: full Book with<br/>spot/fwd/option legs)"]
    end

    HVAR["historical_var<br/>order statistic L(ceil(n*alpha))<br/>plain, age-weighted, FHS"]
    PVAR["parametric_var<br/>(delta-normal, Student-t,<br/>Cornish-Fisher)"]
    MVAR["monte_carlo_var<br/>(normal / Student-t / jump<br/>factor simulation)"]
    ES["expected_shortfall<br/>(exact Acerbi-Tasche)"]
    SE1["bootstrap standard error<br/>(primary)"]
    SE2["KDE standard error<br/>(diagnostic)"]
    BT["backtesting: Kupiec POF (keyword-only),<br/>Christoffersen, Basel traffic light<br/>(any window, any alpha)"]
    OUT["VaR, ES, SE, backtest verdict"]

    HIST --> HVAR
    COV --> PVAR
    BOOK --> MVAR
    COV --> MVAR
    HVAR & PVAR & MVAR --> ES
    MVAR --> SE1
    MVAR --> SE2
    HVAR & PVAR & MVAR & ES & SE1 --> OUT
    OUT --> BT
```

`alpha` is the tail probability in (0, 0.5) everywhere. The C++ and Rust VaR
engines are linear-book ports and do not revalue options.

---

## 6. CI jobs

```mermaid
flowchart TD
    DISC["discover<br/>(list Python projects + test extra)"]
    PYC["py core<br/>python/core x Python 3.10-3.13"]
    PY["py &lt;project&gt;<br/>per project x Python 3.10-3.13<br/>pytest, JUnit XML"]
    LINT["lint<br/>uv lock --check, ruff, mypy"]
    GOLD["goldens --check<br/>gen_golden_header.py / gen_golden_rs.py"]
    CPP["cpp &lt;engine&gt; (&lt;preset&gt;)<br/>gcc, clang, asan, tsan,<br/>gcc16-cxx26, msvc"]
    FMT["clang-format, clang-tidy"]
    RUST["rust &lt;crate&gt;, rust fmt<br/>qp-core + 4 engines<br/>cargo test, clippy -D warnings"]
    BIND["bind &lt;package&gt;<br/>qp-cpp, qp-rust"]
    BENCH["bench<br/>2x p50 rule"]
    HYG["hygiene<br/>banned-word check,<br/>voice report (report-only)"]
    CNT["counts<br/>aggregate executed counts<br/>into counts.json,<br/>run tests/test_docs_counts.py"]
    OK(["ci-ok<br/>required check:<br/>every job succeeded"])

    DISC --> PY
    PY --> CNT
    CPP --> CNT
    RUST --> CNT
    DISC & PYC & PY & LINT & GOLD & CPP & FMT & RUST & BIND & BENCH & HYG & CNT --> OK
```

Fix history is in [CHANGELOG.md](../CHANGELOG.md).
