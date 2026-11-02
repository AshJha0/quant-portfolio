# research

The Phase 4 research programme: pre-registered studies on real data.

## How a study runs

1. A pre-registration in `prereg/` fixes the hypothesis, data, universe,
   parameters, costs, primary metric and decision rule. Its registration
   hash is the SHA-256 of everything above its `## Signature` heading (line
   endings normalised to LF).
2. The maintainer signs it by filling `signed_by`, `signed_on` and
   `sha256` under that heading. Signing does not change the hash.
3. The study writes a JSON result to `results/` carrying `prereg`,
   `prereg_sha256` and `period`.
4. `python research/tools/prereg.py check` refuses a result when its hash
   does not match the registration, or when its period reaches the
   2022-2025 hold-out and the registration is not signed with that hash.
   CI runs it on every push.

Every study also calls `models.training_only` on its input, which raises if
the data reaches 2022-01-01. Lifting that for a hold-out run is a code
change made only after signing.

To get a hash for signing:

```bash
python research/tools/prereg.py hash research/prereg/pairs-fx-g10.md
```

## Studies

| Registration | Status | Training result |
|---|---|---|
| `prereg/pairs-fx-g10.md` | signed 2026-11-02; hold-out not run | `results/pairs_fx_g10_train.json` |
| `prereg/regime-fx-basket.md` | signed 2026-11-02; hold-out not run | `results/regime_fx_basket_train.json` |
| `prereg/execution-lobster.md` | draft, unsigned, blocked on data access | none |

The two FX registrations were signed by the maintainer under owner decision
9, which delegated pre-registration signing. The signature records the hash
printed by `prereg.py hash`; `prereg.py check` passes on both. The execution
study needs LOBSTER files, which sit behind LOBSTER's terms of use, so it
stays an unsigned draft. Equity pairs on a point-in-time universe need a
Nasdaq Data Link account and are not drafted.

Training results (ECB data, 1999-2021) are numbers from the training
period only. They are not out-of-sample evidence; the hold-out decides.

## Models added here

- `studies/models.py: johansen_trace`, the Johansen (1991) trace test with
  an unrestricted constant. Oracle: statsmodels `coint_johansen`, matched to
  1e-9 relative.
- `studies/models.py: kalman_hedge`, a random-walk hedge ratio (Harvey
  1989, section 3.2). Oracle: with delta = 0 it equals recursive least
  squares.

Tests: `python -m pytest research/tests` (needs qp-core, fx-pairs,
fx-regime; statsmodels for the Johansen oracle).
