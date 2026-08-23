# quant-portfolio docs

This folder is the portfolio-wide documentation layer — how the 31
sub-projects fit together, how to learn the material, and how to actually
use the code. It complements, and does not repeat, the per-project
documentation every one of those 31 sub-projects carries in its own
`docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`.

| File | What it's for |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | The project map: what lives where, the package/engine tables, the cross-language golden-vector validation pipeline, and the design invariants worth knowing before you change anything. |
| [DIAGRAMS.md](DIAGRAMS.md) | The same architecture as Mermaid diagrams — the portfolio shape, the golden-vector pipeline, per-area data flows, and the numerical-robustness bug case studies as pictures. |
| [LEARN.md](LEARN.md) | The teaching companion: a plain-language tour of all 13 areas (Part I), why the portfolio is engineered the way it is (Part II), a suggested learning path by background (Part III), and a 481-question self-test bank organized into 18 rounds (Part IV). |
| [COOKBOOK.md](COOKBOOK.md) | 61 copy-pasteable, verified-runnable "how do I..." recipes spanning every project area, from pricing an option to rebuilding a Rust engine from a clean checkout. |
| [MARKET_RISK.md](MARKET_RISK.md) | The VaR/ES/backtesting workflow the risk engines are built to support, end to end — the daily cycle, the limit structure, stress testing, and where equity and FX genuinely diverge. |

**Where to start** depends on what you're after:

- New to quant finance entirely → [LEARN.md](LEARN.md) Part III, Path A.
- Know the math, new to this codebase → [LEARN.md](LEARN.md) Part III, Path B.
- Comfortable with code, want the cross-language engineering story →
  [LEARN.md](LEARN.md) Part III, Path C.
- Just want to run something → [COOKBOOK.md](COOKBOOK.md).
- Need the big picture before touching code → [ARCHITECTURE.md](ARCHITECTURE.md)
  and [DIAGRAMS.md](DIAGRAMS.md).

The top-level [`../README.md`](../README.md) is still the source of truth
for what the portfolio contains and its verified test counts (5,963 passing
tests, reproduced from a clean build); this folder is the map and the
teaching material layered on top of it.
