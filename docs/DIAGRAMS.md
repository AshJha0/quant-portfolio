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
        RUST["Rust engines (edition 2024)<br/>cargo workspace + qp-core"]
    end

    DATA["python/data (qp_data)<br/>keyless public data,<br/>content-addressed store,<br/>publication lags"]
    DATA -.->|"live.py fetchers delegate<br/>(deprecated, removal v4.0.0)"| AREAS

    A1 -.->|golden vectors| CPP & RUST
    A3 -.->|"golden vectors<br/>(linear book, options)"| CPP & RUST

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
engines revalue linear books and European options (full repricing per
scenario, vol held fixed).

---

## 6. CI jobs

```mermaid
flowchart TD
    DISC["discover<br/>(list Python projects + test extra)"]
    HASH["hashed_install.sh<br/>uv export from uv.lock,<br/>pip --require-hashes --only-binary,<br/>local packages --no-build-isolation, pip check"]
    PYC["py core<br/>python/core x Python 3.11-3.13"]
    PY["py &lt;project&gt;<br/>per project x Python 3.11-3.13<br/>pytest, JUnit XML"]
    LINT["lint<br/>uv lock --check, constraints = lock,<br/>ruff, mypy"]
    GOLD["goldens --check<br/>gen_golden_header.py / gen_golden_rs.py"]
    CPP["cpp &lt;engine&gt; (&lt;preset&gt;)<br/>gcc, clang, asan, tsan,<br/>gcc16-cxx26, msvc"]
    CORE["cpp core (&lt;preset&gt;)<br/>shared kernels + tests"]
    FMT["clang-format, clang-tidy"]
    RUST["rust &lt;crate&gt;, rust fmt<br/>qp-core + 4 engines<br/>cargo test, clippy -D warnings"]
    BIND["bind &lt;package&gt;<br/>qp-cpp, qp-rust"]
    DIFF["engine differential<br/>both bindings, 10,000-input harness,<br/>no skips allowed"]
    CMI["cmake install (gcc, clang)<br/>install QuantPortfolio,<br/>build find_package consumer<br/>with source tree moved aside"]
    WHL["wheels &lt;package&gt; (&lt;os&gt;)<br/>cibuildwheel cp311-cp313,<br/>manylinux_2_34 + Windows,<br/>artefacts wheels-*"]
    BENCH["bench<br/>base vs head on one runner,<br/>3 alternating rounds, CPU 1;<br/>fail: median +25% AND head p10 &gt; base p90"]
    HYG["hygiene<br/>banned-word check,<br/>voice report (report-only)"]
    RES["research<br/>prereg.py check, research/tests"]
    PYD["py data<br/>python/data x Python 3.11-3.13<br/>recorded responses"]
    LRN["learning<br/>notebook + career-prep tests,<br/>nbmake on solution notebooks"]
    CNT["counts<br/>aggregate executed counts<br/>into counts.json"]
    OK(["ci-ok<br/>required check:<br/>every gated job succeeded"])

    HASH -.->|installs for| PYC & PY & LINT & BIND & DIFF & RES & PYD & LRN
    DISC --> PY
    PY & PYC & CPP & RUST & BIND --> CNT
    DISC & PYC & PY & LINT & GOLD & CPP & CORE & FMT & RUST & BIND & DIFF & CMI & WHL & BENCH & HYG & RES & PYD & LRN & CNT --> OK
```

The `wheels` job is part of `ci-ok`; it joined after its first green run. The bench gate is described in [BENCHMARKS.md](BENCHMARKS.md).

---

## 7. The `engine=` switch

```mermaid
flowchart LR
    CALL["eq_options.engine / fx_options.engine<br/>eq_var.engine / fx_var.engine<br/>f(..., engine=name)"]
    CHK{"qp_core.engines<br/>check_engine(name)"}
    ERR1(["ValueError:<br/>unknown name"])
    PYREF["python (default)<br/>reference function, unchanged"]
    LOAD{"load_engine:<br/>extension imports?"}
    ERR2(["ImportError with<br/>pip install ./bindings/..."])
    CPP["qp_cpp (nanobind)<br/>C++ engines"]
    RS["qp_rust (PyO3)<br/>Rust crates"]
    DIFF["tests/test_engine_differential.py<br/>10,000 random inputs per group"]

    CALL --> CHK
    CHK -->|unknown| ERR1
    CHK -->|python| PYREF
    CHK -->|cpp or rust| LOAD
    LOAD -->|no| ERR2
    LOAD -->|cpp| CPP
    LOAD -->|rust| RS
    CPP & RS -.->|compared with| PYREF
    DIFF -.-> CPP & RS
```

---

## 8. Pre-registration workflow 

```mermaid
flowchart TD
    FETCH["data/fetch/qp_fetch.py<br/>ECB, FRED to data/raw/ + .sha256"]
    DRAFT["1. draft<br/>research/prereg/&lt;name&gt;.md"]
    HASH["prereg.py hash<br/>SHA-256 of the text above ## Signature"]
    SIGN["2. sign<br/>signed_by, signed_on, sha256"]
    TRAIN["training run, 1999-2021<br/>training_only guard"]
    CHECK{"3. prereg.py check<br/>hash matches?<br/>hold-out only if signed"}
    HOLD["4. hold-out run, 2022-2025"]
    REF(["result refused"])
    OK(["result accepted"])

    FETCH --> TRAIN
    DRAFT --> HASH --> SIGN
    DRAFT --> TRAIN
    SIGN --> HOLD
    TRAIN --> CHECK
    HOLD --> CHECK
    CHECK -->|no| REF
    CHECK -->|yes| OK
```

State: `regime-fx-basket` signed and run once on the hold-out (H0 not
rejected); `pairs-fx-g10` signed, then withdrawn before the hold-out
(DR-R4-5); `pairs-fx-g10-v2` and `momentum-fx-g10` signed on 2026-11-02 and
scored once on forward data (2026-11-03 to 2028-11-02, `research/forward/`),
on or after 2028-11-13; the order-book execution study an unsigned draft,
not done because its data needs an account. Equity pairs on a
point-in-time universe are not drafted for the same reason. Diagram 11 shows the whole
life of a study.

---

## 9. Newer model dependencies

```mermaid
flowchart LR
    subgraph RATES["fi_rates (equity/04)"]
        OIS["multicurve<br/>bootstrap_ois"] --> PROJ["multicurve<br/>bootstrap_projection"]
        PROJ --> RO["rate_options<br/>caps, swaptions"]
        OIS --> HW["hull_white<br/>Jamshidian swaptions"]
    end
    subgraph SURF["eq_surface (equity/09)"]
        SSVI["ssvi"] --> DUP["dupire"]
        HES["heston"] --- BAT["bates"]
    end
    subgraph AMER["eq_options: American pricer choice"]
        LR["lr_tree<br/>reference"] -.->|oracle| ALO["alo<br/>fast"]
        LR -.->|oracle| PDE["pde<br/>Crank-Nicolson"]
    end
    subgraph RISK["eq_var / fx_var"]
        REV["revaluation"] --> ATT["risk_attribution"]
        REV --> FRTB["frtb_ima"]
    end
    subgraph CMP["cpp + rust VaR engines"]
        OR["option_reval"] --> OV["historical / MC option VaR"]
    end
    REV -.->|goldens 1e-11| OR
```

---

## 10. Shared kernels and the packages that wrap them

```mermaid
flowchart TB
    subgraph PYK["python/core (qp_core)"]
        PBS["black_scholes<br/>carry form, scalar + NumPy"]
        PIV["iv_core"]
        PN["normal"]
    end
    subgraph CK["cpp/core (qp::core)"]
        CBS["qp/black_scholes.hpp<br/>CDF as template parameter"]
        CVM["qp/black_scholes_vm.hpp<br/>+ qp/vmath.hpp (vectorised)"]
        CIV["qp/iv_core.hpp, qp/normal.hpp"]
    end
    subgraph RK["rust/core (qp_core)"]
        RBS["black_scholes<br/>NormalDist type parameter"]
        RIV["iv, special"]
    end
    subgraph WRAP["asset-class wrappers: names, argument order, validation, boundaries"]
        EQ["eq_options<br/>q = dividend yield"]
        FX["fx_options<br/>r = r_d, q = r_f"]
        EQC["eqopt (C++), eq-options-engine (Rust)"]
        FXC["fxopt (C++), fx-options-engine (Rust)"]
    end
    PBS --> EQ & FX
    PIV --> EQ & FX
    CBS --> EQC & FXC
    CVM -->|_fast batches| EQC & FXC
    RBS --> EQC & FXC
    TW["tests/test_carry_twin.py<br/>equity(q) = FX(r_f)"] -.-> EQ & FX
```

The kernel is Black-Scholes in cost-of-carry form, `b = r - q`: equity
passes its yield, FX passes `r_d` and `r_f`, futures pass `q = r`
(`b = 0`). Moving the maths into the kernel changed no output bit in any
language. The C++ and Rust kernels take the normal CDF as a parameter
because the equity and FX engines use different implementations.

---

## 11. Research lifecycle of one study

```mermaid
flowchart LR
    TR["train<br/>1999-2021, training_only guard;<br/>every run logged to<br/>results/trials.jsonl"]
    PW{"power check<br/>synthetic planted effect,<br/>research/power/*.json"}
    SG["sign<br/>prereg.py sign refuses<br/>without a power file;<br/>sha256 of the text"]
    HO["hold-out 2022-2025<br/>run once (spent)"]
    FL["forward log<br/>ECB fixes from 2026-11-03,<br/>forward-log branch, daily job"]
    IN["INTERIM figures<br/>evaluate.py before 2028-11-13:<br/>decide nothing"]
    SC(["scoring<br/>on or after 2028-11-13,<br/>one look"])
    WD(["withdrawn<br/>before any hold-out read"])

    TR --> PW
    PW -->|signal cannot fire| WD
    PW -->|power stated| SG
    SG -->|regime study| HO
    SG -->|pairs v2, momentum| FL
    FL -.-> IN
    FL --> SC
```

The hold-out was used once, by the regime study, and is spent: later
studies can only be tested on data that did not exist when they were
signed. The forward rule is scored once, so looking at interim numbers
cannot change the outcome.

---

## 12. Desk stacks: from an order to post-trade

```mermaid
flowchart LR
    subgraph CORE["qp_core"]
        INS["instruments, market<br/>Instrument.pv(MarketSnapshot) -> Money"]
        VOL["vol, trees, rng, exotics,<br/>risk, scenarios, aad, xva"]
        EV["eventlog<br/>hash-chained log"]
    end
    BT["qp_bt<br/>fills, costs, ledger"]
    OMS["qp_oms<br/>orders, FIX session, EMS,<br/>pre-trade risk, kill switch"]
    SOR["qp_exec.sor<br/>smart order router"]
    VEN["simulated venues<br/>qp_exec.lob reference book"]
    PT["qp_posttrade<br/>booking, matching,<br/>settlement, best execution"]
    KV[("python/oms/kill/vectors.json")]
    GW["gateways<br/>cpp/ull, rust/ull, java"]

    INS --> BT
    VOL --> INS
    BT -.->|reference book fills| VEN
    OMS -->|parent order| SOR
    SOR -->|child orders| VEN
    VEN -->|execution reports| OMS
    OMS -->|fills| PT
    EV --> OMS
    KV --> OMS
    KV --> GW
```

Nothing in this diagram touches a real venue. The kill-switch vectors are
the contract (`docs/KILL_CONTRACT.md`): Python, C++, Rust and Java each
replay them and must produce the same bytes.

Fix history is in [CHANGELOG.md](../CHANGELOG.md).
