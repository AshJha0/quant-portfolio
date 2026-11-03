# Maintainer process: rounds, reviews and budgets

The top-level `CONTRIBUTING.md` covers setup, the CI test gate and the
writing rules for any change. This page covers how the maintainer runs a
round of work: lanes, reviews, the docs review, the dependency bump and the
budget record. It exists because one person carries about 200,000 lines,
four languages and more than twenty CI jobs, and the process only holds if
each step is written down and checked rather than remembered.

## A round in one paragraph

A round is about one week of planned work split into three or four lanes.
Each lane owns a fixed set of files and works on its own branch in its own
worktree. When the lanes finish, an integration lane merges them into one
branch with one commit series per lane, runs the full local test suites,
does the docs review and the dependency bump, and opens one pull request.
The pull request merges when every required check is green. Tags are cut
only at the release points in `docs/VERSIONING.md`, after `main` CI is
green on the merge commit.

## Lanes

- **File ownership is fixed before work starts** and written in the round
  plan. A lane edits only its files. If it needs a change elsewhere, it
  writes the request in its handback and the owner of that file (or the
  integration lane) makes it.
- **A package's `__init__.py` belongs to the lane that owns the package**,
  unless the round plan names a different owner.
- **Local tests finish before any commit.** A run that stalls is treated as
  a hang and investigated, not retried until it passes.
- **Every lane writes a handback** at
  `review/<date>/handbacks/<round>/<lane>.md` with: the commits; what
  changed and why; the tests run with exact pass, skip and fail counts;
  requests for other lanes; and the effort estimate against the actual.

## Fresh-context review per lane

Before integration, each lane's diff is read by a reviewer who did not
write it and starts from the diff and the plan, not from the author's
summary. The reviewer writes a short note next to the handback
(`<lane>-review.md`) listing findings as fixed, accepted with a reason, or
deferred with a tracking item. A lane is not merged with an open "fixed"
finding.

## Docs review checklist

Done by the integration lane before the round pull request is opened. Tick
each line in the round record; "not affected" is a valid answer, a blank
is not.

- [ ] `README.md` and `docs/START_HERE.md`: every statement still true;
      links resolve.
- [ ] `docs/LEARN.md` and `docs/learn/`: new or changed behaviour has a
      question or a note; answers still match the code.
- [ ] `docs/COOKBOOK.md`: every recipe touched by the round still runs.
- [ ] `docs/ARCHITECTURE.md` and `docs/DIAGRAMS.md`: modules, jobs and data
      flows match the code.
- [ ] Per-package `README`, `METHODOLOGY` and `VALIDATION` for every
      package the round touched.
- [ ] `CHANGELOG.md` `[Unreleased]` lists every user-visible change.
- [ ] `docs/MIGRATION.md` has an entry for every changed number, changed
      error type, deprecation and removal, with the reference that
      verified a changed number.
- [ ] `docs/VERSIONING.md`: the deprecation table and the platform table
      are current.
- [ ] Counts are generated, not typed: `tests/test_docs_counts.py` passes
      locally, and the CI `counts` job is green.
- [ ] `git grep -niI 'intervie[w]'` finds nothing, and
      `tests/test_repo_policy.py` passes (no excluded data source named
      outside the excluded-sources note).
- [ ] Zero spend: nothing added costs money or needs payment details, and
      every optional free sign-up has a tested no-account fallback.

## Generated counts

No count of tests, files, projects, questions or recipes is typed into the
docs by hand and left unchecked:

- Test counts come only from CI: the `counts` job aggregates every test
  job's JUnit XML and cargo log into `counts.json` and publishes it in the
  run summary. `tests/test_docs_counts.py` fails if a test total appears in
  the top-level docs.
- Structural counts (sub-projects, Python projects, engines) and the
  question and recipe counts may appear in prose, but
  `tests/test_docs_counts.py` derives each one from the repository and
  fails if any doc quotes a different number.

## One dependency bump per round

Dependabot and manual updates are collected into one commit on the
integration branch: update the constraints, run `python -m uv lock`, run
the full local suites, and record the bumped packages in the round record.
Version pins stay exact and every wheel stays hash-checked through
`uv.lock` (`.github/scripts/hashed_install.sh`). New tools join a
dependency group in the root `pyproject.toml` rather than being installed
unpinned in a workflow.

## Budget record

Every round record (`review/<date>/ROUND_<n>.md`) carries this table,
filled in at the end of the round and read at the start of the next one to
re-estimate it:

| Lane | Estimate (dd) | Actual (dd) | Tokens | Notes |
|---|---|---|---|---|
| L1 | | | | |
| L2 | | | | |
| L3 | | | | |
| L4 | | | | |
| Integration | | | | |
| **Total** | | | | |

A dd ("developer-day") is the plan's unit of effort. Estimates are
wall-clock; reviewer time, CI time and anything that only accrues (a
forward test) are stated separately. Usage stays under 90% of the weekly
limit; paid extra usage is never used. If a round runs over, the plan's
cut order applies; nothing is cut silently.

## Round record checklist

The round record is complete when it has:

1. the lane handbacks and their review notes;
2. the docs review checklist above, ticked;
3. the dependency bump list;
4. the budget table;
5. the test counts from the local runs and the CI run on the pull request;
6. the link to the merged pull request and, at a release point, the tag.
