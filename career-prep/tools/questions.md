# Linux, git and debugging: knowledge check (questions only)

Part A: twelve tasks, each answered with one shell command line. Part B:
discussion prompts on debugging and production support. Answers are in
[`keys/answers.md`](keys/answers.md).

How it is checked: `career-prep/tests/test_tools_pack.py` builds the
scenario each task describes in a temporary git repository, runs the key's
command with `bash`, and checks the resulting repository state or output.
Run it yourself to see the scenarios. It needs `git` and `bash` (Git Bash on
Windows) and skips otherwise.

References: Chacon and Straub, *Pro Git* (2nd ed.), chapters 2, 3, 7.3
(stashing), 7.7 (reset), 7.10 (bisect) and 10.7 (recovery); the GNU
coreutils manual; Gregg, *Systems Performance* (2nd ed.) for part B.

## Part A: one command each

Git (repository with `main` checked out unless stated):

- **T01** Undo the last commit but keep its changes staged.
- **T02** Throw away your unstaged edits to `notes.txt` only.
- **T03** The last commit is already pushed and shared. Undo its effect without rewriting history.
- **T04** You ran `git switch main` then `git branch -D feat` by mistake. Recreate `feat` at its old tip.
- **T05** Tag `v1` was good and `HEAD` is bad; `check.sh` exits 0 on a good commit and 1 on a bad one. Find the first bad commit automatically.
- **T06** Show which commit last changed line 3 of `prices.csv`, in machine-readable form.
- **T07** Put your uncommitted work aside with the message `wip`, leaving a clean tree.
- **T08** List the names of the files changed by the last commit, one per line, nothing else.

Shell (`trades.csv` has lines `SYM,QTY` and no header; `logs/` holds text files):

- **T09** Print the most traded symbol by number of lines, as `count symbol`.
- **T10** Print the total quantity in `trades.csv`.
- **T11** List, sorted, the files under `logs/` that contain `ERROR`.
- **T12** Print the second field of the last line of `trades.csv`.

## Part B: discussion (not graded)

- **TD1** A service's p99 latency doubled after a deploy. Walk through the first ten minutes.
- **TD2** A process is using 100% of one core and not making progress. Which tools do you reach for, and in what order?
- **TD3** A C++ binary crashes in production with a segmentation fault once a day. How do you get a usable stack trace?
- **TD4** `git rebase` versus `git merge` for a shared branch: what rule do you follow, and why?
- **TD5** Disk is full on a box and `du` does not account for the space. What is going on?
