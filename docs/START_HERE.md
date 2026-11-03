# Start here

The repository is large: 23 Python projects, four C++ engines, four Rust
crates and several thousand lines of study notes. If you have ten minutes,
read these five things in this order and skip the rest.

1. **[`../README.md`](../README.md), first screen.** What the code is, what
   it is not (no live book, synthetic data unless a page names a dataset),
   and a three-command quickstart.
2. **[`../research/README.md`](../research/README.md).** How a study is
   registered, hashed, signed and only then run on the hold-out, and the
   dated decision records. DR-R4-5 shows a registration withdrawn before any
   hold-out data was read, because a simulation showed the signal could not
   fire. That record says more about the process than any positive result.
3. **[`python/equity/01-options-pricing/tests/golden/generate_golden.py`](../python/equity/01-options-pricing/tests/golden/generate_golden.py).**
   Reference values come from mpmath at 50 digits, never from the code under
   test. The C++ and Rust engines read the same file.
4. **[ARCHITECTURE.md](ARCHITECTURE.md), "Cross-language
   testing".** How Python, C++ and Rust are compared on random inputs, and
   why a skipped comparison fails CI.
5. **One package's `docs/VALIDATION.md`**, for example
   [`python/equity/01-options-pricing/docs/VALIDATION.md`](../python/equity/01-options-pricing/docs/VALIDATION.md).
   Each model is listed with the independent check it passes and the
   tolerance.

Known limits worth knowing before you read further:

- Equity and FX are separate codebases for conventions (quoting, rates,
  day counts, delta styles). The Black-Scholes maths is shared: one
  cost-of-carry kernel per language, which the two option packages wrap.
  Other models (vol surfaces, VaR revaluation) still have one copy per
  asset class.
- The latency tables in the READMEs are one laptop run with an older
  harness. Current figures come from the CI bench job, which runs the old
  and new commit on the same runner ([BENCHMARKS.md](BENCHMARKS.md)).
- One study has an out-of-sample result, and it is a null (regime
  hold-out, H0 not rejected). The two forward registrations are scored
  once, on or after 2028-11-13; anything printed before then is labelled
  INTERIM and decides nothing.
- CI installs every Python package with its hash checked against
  `uv.lock`; QuantLib cross-checks exist but run only where QuantLib is
  installed.
- The only real-data studies use ECB daily reference rates. There is no
  point-in-time equity or intraday evidence.
- Execution costs come from a schedule-level cost model on a bucket
  simulator, not from an order book. The figures are model-implied, and
  they changed in v2.1.0 when a double-counted bucket width was removed
  (equity 08 `docs/VALIDATION.md` section 6).

After that, [LEARN.md](LEARN.md) is a study guide with self-test questions
(each answer is folded so you can try first; each question is tagged by
topic, so a search for a tag such as `` `execution` `` gives one subject;
Set 19 covers power, optional stopping, false discovery rate, hashed
installs and latency measurement) and [COOKBOOK.md](COOKBOOK.md) has task
recipes, including NumPy array pricing, using the C++ engines through
`find_package(QuantPortfolio)`, and reading the trials log.
[DIAGRAMS.md](DIAGRAMS.md) has the CI pipeline, the shared-kernel layout
and the life of a research study.
