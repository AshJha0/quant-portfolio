# Linux, git and debugging: answer key

Questions are in [`../questions.md`](../questions.md). Each command in
backticks is run by `career-prep/tests/test_tools_pack.py` against the
scenario in the question. Other correct commands exist; these are the ones
tested.

- **T01** `git reset --soft HEAD~1`. `--soft` moves the branch only; `--mixed` (the default) would also unstage; `--hard` would discard the work.
- **T02** `git restore notes.txt`. Restores the file from the index (older form: `git checkout -- notes.txt`).
- **T03** `git revert --no-edit HEAD`. Adds a new commit that inverts the old one, so nobody who pulled has history rewritten under them.
- **T04** `git branch feat HEAD@{1}`. The reflog keeps where `HEAD` was before the switch, which was the tip of `feat` (Pro Git 10.7). Unreachable commits survive until garbage collection, 30 days or more by default.
- **T05** `git bisect start HEAD v1 && git bisect run sh check.sh`. Binary search over the history: about log2(n) test runs.
- **T06** `git blame -L 3,3 --porcelain prices.csv`. The first token of the output is the commit hash.
- **T07** `git stash push -m wip`. `git stash pop` brings it back.
- **T08** `git show --name-only --format= HEAD`. An empty format suppresses the commit header.
- **T09** `cut -d, -f1 trades.csv | sort | uniq -c | sort -rn | head -1`. `uniq -c` counts only adjacent duplicates, so sort first.
- **T10** `awk -F, '{s += $2} END {print s}' trades.csv`.
- **T11** `grep -rl ERROR logs | sort`. `-l` prints matching file names once each.
- **T12** `tail -n 1 trades.csv | cut -d, -f2`.

## Discussion notes

- **TD1** Confirm the regression on a dashboard against the deploy time; roll back if the change is cheap to revert, then investigate. Compare a flame graph or profile before and after; check GC or allocator behaviour, lock contention, new logging on the hot path, and changes in request mix. Look at saturation (CPU, run queue, network) with the USE method (Gregg 2020, ch. 2).
- **TD2** `top -H` to find the thread; `perf top -p PID` or `perf record -g` for where it spins; `gdb -p PID` then `thread apply all bt` for the stacks; `strace -p PID` to see whether it is in a syscall loop.
- **TD3** Build with frame pointers and separate debug symbols; enable core dumps (`ulimit -c unlimited`, `core_pattern` or `coredumpctl`); open the core with `gdb binary core` and `bt full`. Reproduce under AddressSanitizer in a test build.
- **TD4** Rebase only your own unpublished commits; merge (or squash-merge through a PR) into shared branches, because rebasing published history forces everyone else to recover (Pro Git 3.6, "The Perils of Rebasing").
- **TD5** A process still holds a deleted file open, so the blocks are not freed. `lsof +L1` lists open files with no links; restart or signal the process to reopen its log.
