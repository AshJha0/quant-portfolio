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

- Equity and FX are separate codebases with much shared maths; the copies
  have drifted. Treat the FX packages as convention variants.
- Latency tables are from one laptop under WSL2. The C++ and Rust tables use
  different timing methods and are not comparable with each other.
- One study has an out-of-sample result, and it is a null (regime
  hold-out, H0 not rejected). The two forward drafts cannot be scored
  before November 2028.
- The only real-data studies use ECB daily reference rates. There is no
  point-in-time equity or intraday evidence.

After that, [LEARN.md](LEARN.md) is a study guide with self-test questions
and [COOKBOOK.md](COOKBOOK.md) has task recipes.
