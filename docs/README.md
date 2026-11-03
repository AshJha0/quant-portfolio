# quant-portfolio docs

This folder holds the documentation that spans all 31 sub-projects. Each
sub-project also has its own `docs/{METHODOLOGY,VALIDATION,DESK_GUIDE}.md`.
All results quoted here come from seeded synthetic data unless a page names a
dataset. The research studies in [`../research/`](../research/README.md) use
ECB data and report training-period results only, which are not
out-of-sample evidence.

| File | What it is for |
|---|---|
| [START_HERE.md](START_HERE.md) | The five things worth reading first, and the known limits. |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Layout, package and engine tables, risk conventions, the golden and differential test pipeline, CI jobs. |
| [DIAGRAMS.md](DIAGRAMS.md) | The same structure as Mermaid diagrams. |
| [LEARN.md](LEARN.md) | Study guide: a tour of all 13 areas (Part I), design choices (Part II), learning paths (Part III) and 532 self-test questions in 19 sets (Part IV). |
| [COOKBOOK.md](COOKBOOK.md) | 84 recipes ("how do I ...") across every area. They are written against current main; CI does not run them, so outputs are indicative. |
| [MARKET_RISK.md](MARKET_RISK.md) | How the VaR, ES and backtesting code fits together, and its conventions. |

Where to start:

- Ten minutes to spare: [START_HERE.md](START_HERE.md).
- New to quant finance: [LEARN.md](LEARN.md) Part III, Path A.
- Know the maths, new to this code: [LEARN.md](LEARN.md) Part III, Path B.
- Interested in the C++ and Rust ports: [LEARN.md](LEARN.md) Part III, Path C.
- Want to run something: [COOKBOOK.md](COOKBOOK.md).
- Want the overall layout first: [ARCHITECTURE.md](ARCHITECTURE.md) and
  [DIAGRAMS.md](DIAGRAMS.md).

Test counts are not typed into these pages. The CI `counts` job writes them
to `counts.json` and to the run summary.
