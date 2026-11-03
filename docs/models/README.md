# Model inventory and validation

## The idea

A bank does not just write pricing and risk code; it keeps a register of
every model, says who owns each one, how much harm a wrong answer could do,
what the model assumes, where it is known to break, and when it was last
checked by someone other than its author. US and UK supervisors require
this: the Federal Reserve's SR 11-7 (2011) and the PRA's SS1/23 (2023).

This folder is that register for the repository, kept small enough that one
maintainer can keep it true, and tied to the code by a test so it cannot
drift silently.

| File | What it holds |
|---|---|
| `models.toml` | One entry per model: id, owner, tier, kind, purpose, entrypoints, assumptions, limitations, validation status and date, KPIs, tests, independent checks. |
| `VALIDATION_TEMPLATE.md` | The headings every validation write-up uses. |
| `VALIDATION.md` | The validation pack: one section per model, in template order. |

## The formal statement

**Model.** Any function or class whose output is a price, a sensitivity, a
risk measure or a forecast. This is SR 11-7's definition ("a quantitative
method ... that processes input data into quantitative estimates"),
narrowed to the four output kinds this repository produces.

**Tag.** Each model's entrypoints carry `@model(id="...")` from
`qp_core.model`. The decorator returns the function unchanged apart from a
`__qp_model_id__` attribute, so it costs nothing at call time.

**Tier.** How much a wrong answer would matter if the model were used for
real decisions:

| Tier | Meaning | Review interval | Independent check |
|---|---|---|---|
| 1 | Feeds a price or a risk number directly (pricers, VaR, curves) | 6 months | required |
| 2 | Feeds a tier-1 model, or is a calibration or advanced pricer | 12 months | encouraged |
| 3 | Teaching or research only | 12 months | optional |

**Validation status.**

| Status | Evidence required |
|---|---|
| `independently-checked` | At least one test compares the model with an implementation it does not share code with (QuantLib, mpmath, `arch`, scikit-learn, the Rust or C++ engine, a published table) or with an independently derived closed form. |
| `self-tested` | Tests check the model only against its own properties: parity, limits, round trips, repricing of inputs. |
| `not-validated` | Neither. |
| `retired` | Kept for reference; not called by any engine. |

**The check** (`tests/test_model_inventory.py`, also
`python -m qp_core.model check`) fails when:

1. a decorated function's id has no entry (the plan's first acceptance
   criterion);
2. a cited test (`path::name`, with `Class.method` for methods) does not
   exist (the second criterion);
3. an entrypoint does not exist, or its tag disagrees with the entry's
   `decorated` flag in either direction;
4. a tier-1 model, or one marked `independently-checked`, cites no
   independent check;
5. dates are malformed, in the future, out of order, or the review
   interval is longer than the tier allows;
6. an entry is listed twice or has an unknown or missing field.

## Worked example

`EQ-VAR-PARAM` is the parametric VaR. Its entry names the two entrypoints
(`parametric_var`, `cornish_fisher_var`), says it assumes jointly normal
returns and a linear book, and lists "understates fat-tailed losses" as a
limitation. Tier 1, so it must cite an independent check: it cites
`tests/golden/test_oracle_equity.py::test_parametric_closed_forms`, which
compares against an oracle that imports no engine code
(`test_oracle_imports_no_engine_code` in the same file enforces that). Its
next review is due six months after `validated_on`.

If someone renames `test_parametric_closed_forms`, the inventory test fails
with `EQ-VAR-PARAM: test test_parametric_closed_forms not found in ...`. If
someone tags a new function `@model(id="EQ-VAR-NEW")` without adding an
entry, it fails with `... is tagged @model(id='EQ-VAR-NEW') but
docs/models/models.toml has no entry for it`.

## Adding or changing a model

1. Tag the entrypoint: `from qp_core.model import model`, then
   `@model(id="AREA-PKG-NAME")` above the function.
2. Add a `[[model]]` table to `models.toml` (copy a neighbour).
3. Add its section to `VALIDATION.md` using the template headings.
4. Run `python -m qp_core.model check` from the repository root.

## What this is not

A single-maintainer repository cannot provide what SR 11-7 calls
"effective challenge" by an independent validation function. The
`independent_checks` field records the closest substitute available here:
agreement with code the author did not write. The owner field reads
`maintainer` throughout for the same reason.

## Format note

The plan named the file `models.yaml`. It is TOML instead: Python reads it
with the standard-library `tomllib` (3.11+), so the check needs no new
dependency, and TOML has native dates.

## References

- Board of Governors of the Federal Reserve System, *SR 11-7: Guidance on
  Model Risk Management*, 4 April 2011.
- Prudential Regulation Authority, *SS1/23: Model risk management
  principles for banks*, May 2023.
