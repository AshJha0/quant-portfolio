"""Report-only voice check: em-dash and banned-phrase counts.

Prints a table and appends it to ``$GITHUB_STEP_SUMMARY`` when set. Always
exits 0. The hard ban on the hiring-process word is a separate CI step.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter

BANNED = [
    "flagship", "production-grade", "world-class", "honest", "deliberately",
    "genuinely", "real desk", "same discipline", "not an oversight",
    "verified, not claimed", "bit-identical",
]
GLOBS = ["*.md", "*.html", "*.py", "*.rs", "*.cpp", "*.hpp", "*.h", "*.toml", "*.yml"]


def main() -> int:
    files = subprocess.run(["git", "ls-files", *GLOBS], capture_output=True, text=True,
                           check=True).stdout.split()
    per_file: Counter[str] = Counter()
    per_term: Counter[str] = Counter()
    pat = re.compile("|".join(re.escape(b) for b in BANNED), re.I)
    for f in files:
        if f.startswith(".github/scripts/style_report.py"):
            continue
        try:
            with open(f, encoding="utf-8") as fh:
                txt = fh.read()
        except (UnicodeDecodeError, OSError):
            continue
        n_dash = txt.count(chr(0x2014))  # em-dash
        hits = [m.group(0).lower() for m in pat.finditer(txt)]
        per_term["em-dash"] += n_dash
        per_term.update(hits)
        if n_dash or hits:
            per_file[f] = n_dash + len(hits)
    lines = ["| term | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in per_term.most_common()]
    lines += ["", "| file (top 20) | hits |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in per_file.most_common(20)]
    report = "\n".join(lines)
    print(report)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write("## Voice report (report-only)\n\n" + report + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
