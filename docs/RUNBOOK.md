# Runbook

## The idea

When something goes wrong, the person on hand should not have to work out
the procedure from scratch. Each entry below names a symptom as it first
shows up, how to confirm it, what to do, and how to know it is fixed. The
objectives these entries defend are in `docs/SLO.md`.

Everything here uses free tools: GitHub Actions on standard hosted
runners, `gh`, and the local toolchains in `CONTRIBUTING.md`. No entry
requires an account, a key or a paid service.

## Entries

### CI is red on main

1. **Confirm.** `gh run list --workflow ci --branch main --limit 5`. Open
   the failing run; the `ci-ok` job lists which jobs failed.
2. **Classify.** A network fetch (apt, crates.io, PyPI) that fails in one
   job and passes on re-run is infrastructure: re-run the failed jobs once
   (`gh run rerun <id> --failed`). Anything that fails twice, or fails a
   test, is code.
3. **Act on code failures.** Find the first red commit
   (`gh run list --workflow ci --branch main`), reproduce locally, and
   either fix forward in one pull request or revert that commit. Write the
   failing test first if the failure exposed a wrong answer.
4. **Done when** `ci-ok` is green on main. Count the red run against S1.

### An independent check is red (QuantLib, goldens, engine differential)

These jobs compare the code with something it does not share code with.
A failure means a number may be wrong, so it outranks every other entry.

1. **Confirm** which comparison failed and by how much (the assertion
   message prints both values and the tolerance).
2. **Decide which side moved.** A change to `uv.lock` or
   `requirements/` that moved QuantLib is a reference change: pin back,
   then investigate. A change to engine code is a suspected wrong answer.
3. **Fix** with a failing test that pins the correct value, cite the
   source of the correct value, and add a CHANGELOG line with the before
   and after numbers.
4. **Re-review** the model entry in `docs/models/VALIDATION.md`.

### The bench gate failed

1. **Read the table** in the `bench` job summary. A kernel fails only if
   the head median is more than 25% above base *and* the p10-p90 bands do
   not overlap (`docs/BENCHMARKS.md`, "The CI gate").
2. **Re-run once.** Shared runners are noisy; a real regression fails
   again.
3. **If it fails again**, profile locally on the kernel named, with the
   bench binary built from both commits. Fix, or justify the slowdown in
   the pull request (for example, a correctness fix that costs time) and
   record the new figure in the CHANGELOG.

### The nightly fuzz job found something

1. **Download** the `cpp-fuzz-findings-<target>` or
   `rust-fuzz-findings-<target>` artefact from the failed `fuzz` run.
2. **Reproduce.** C++: build `cpp/core/fuzz` with
   `-DQP_FUZZ_ENGINE=replay` and run `build/fuzz_<target> <file>`. Rust:
   copy the file into the crate's `corpus/` and run `cargo test` in the
   fuzz crate.
3. **Decide** whether the code or the property is wrong. A property that
   demands something mathematically impossible is fixed in the property,
   with the reasoning in its comment (example: the implied-vol price
   elasticity case in `cpp/core/fuzz/fuzz_implied_vol.cpp`). A code bug
   gets a failing unit test first.
4. **Keep** the input as `cpp/core/fuzz/regressions/<target>/<what-it-found>`
   so the replay job runs it on every change. Done within 7 days (S4).

### The forward log missed a day

1. **Confirm** the date is an ECB business day and that
   `FORWARD_LOG_ENABLED` is `true` (repository variables).
2. **Check the run.** `gh run list --workflow forward-log --limit 5`. If
   the ECB feed was late, a manual dispatch later the same day is safe:
   the job appends nothing for a day already logged.
3. **Do not back-fill** by editing the branch: lines are never rewritten
   (`research/forward/README.md`, "One line per date, never rewritten").
   A day the job could not log stays missing; note it in the next round's
   handback so the evaluation can account for it.

### The model inventory check failed

`tests/test_model_inventory.py` prints one line per problem. The usual
causes and fixes:

| Message contains | Fix |
|---|---|
| `has no entry for it` | Add a `[[model]]` table to `docs/models/models.toml` and a section to `docs/models/VALIDATION.md`. |
| `test ... not found` | A cited test was renamed or removed: update the citation, or restore the test. |
| `is not tagged` / `set decorated = true` | Make the `decorated` flag match the source. |
| `review interval exceeds` / `in the future` | Correct the dates; a tier-1 review is due within 6 months. |

### Cutting a release

Releases run only from `.github/workflows/release.yml` (its header lists
the preconditions: commit on main, `ci-ok` green on that commit, VERSION
and CHANGELOG agree, tag absent). Do a dry run first (`dry_run: true`),
check the artefact, then run for real. Nothing is published to PyPI
(owner decision D7).

### A dependency update arrives

Dependabot opens pull requests. Batch them into one pull request per
round, regenerate the hash-locked files (`uv lock`), and let CI decide. A
QuantLib version change is a reference change: read its release notes
before merging, since the independent checks compare against it.

## Escalation

There is one maintainer. Anything that would need money, an account, a key
or a signature (for example paid runners, a hosted fuzzing service, or a
code-signing certificate) is outside this runbook and goes to the owner.

## References

- Beyer et al., *Site Reliability Engineering* (2016), ch. 4 and ch. 14
  ("Managing Incidents").
- libFuzzer documentation, "Corpus" and "Options"
  (llvm.org/docs/LibFuzzer.html).
