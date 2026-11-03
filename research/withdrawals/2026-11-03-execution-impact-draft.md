# Withdrawal: draft execution-impact registration (2026-11-03)

- **What is withdrawn.** The unsigned draft registration
  `research/prereg/execution-<excluded source>.md`, added in commit `1f1e1cf`
  (Phase 4) and removed in Round 7, lane L1. Its file name and text named a
  limit-order-book sample source that the owner has ruled out (see the
  excluded-sources note in `data/README.md`), so the name is not repeated here.
- **Why.** The owner decided not to sign up for that source (zero spend, no
  accounts). The study cannot be run without its data.
- **State at withdrawal.**
  - The draft was never signed: `signed_by`, `signed_on` and `sha256` were empty.
  - No data was ever downloaded, fitted or scored. No result file and no line in
    `research/results/trials.jsonl` refer to it, so the programme trial count is
    unchanged by this withdrawal.
  - SHA-256 of the file as last committed (git blob at `b2a2652`):
    `5cfc5d36c03982b76a051079a6a67606c2d378e50c9fa7733e4d83fef232f776`.
    `git show 1f1e1cf:research/prereg/` recovers the text.
- **What replaces it.** The square-root versus linear impact question moves to
  the W25/W31 execution work, calibrated on recorded free crypto feeds only.
  That work will need its own registration.
