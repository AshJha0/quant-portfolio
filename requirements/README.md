# requirements

CI installs every third-party Python package from `uv.lock`, the lock file at
the repository root, and pip checks each wheel against the sha256 hash the
lock records. The files in this folder are for local installs with plain pip.

| File | Purpose |
|---|---|
| `../uv.lock` | The source of truth: one resolved version per package (with Python-version markers where they differ) and the sha256 of every wheel and sdist. CI installs from it. |
| `ci-requirements.in` | The union of every Python project's runtime dependencies, the test extras, the plotting extra, `pytest` and `hypothesis`, the lint tools, the binding build backends and `setuptools`. Live-data extras (`yfinance`) are left out. |
| `ci-constraints.txt` | The same versions as `uv.lock`, without hashes, for local `pip install -c`. The `lint` job fails if the two disagree. |
| `../.github/scripts/hashed_install.sh` | The install step every Python CI job runs. |

## How a CI job installs

The plain idea: pip is told exactly which file to download for each package
and what its fingerprint must be, and refuses anything else.

Formally, `hashed_install.sh [--extra E] [--group G] [--editable] PATH...`
does four things:

1. `uv export --frozen --no-emit-workspace` writes the locked third-party
   closure of the projects at `PATH...` (with extra `E`) and of the root
   dependency groups (`G`, plus `backend` always) to a requirements file,
   each line carrying its `--hash=sha256:` values.
2. `pip install --require-hashes --only-binary :all: --no-deps -r` that
   file. A wheel whose hash is not listed, a line without a hash, or a
   package with no wheel for the platform each stop the install, and
   `--no-deps` means nothing outside the file is fetched.
3. `pip install --no-deps --no-build-isolation -e PATH...` for the local
   projects, so setuptools (the `backend` group) is the hashed copy rather
   than a fresh download into an isolated build environment.
4. `pip check`, which fails if the hashed set leaves any declared
   dependency unmet.

The root `pyproject.toml` defines the groups:

| Group | Contents | Used by |
|---|---|---|
| `test` | pytest, hypothesis | every test job |
| `lint` | ruff, mypy, pandas-stubs, scipy-stubs | `lint` (through `uv sync`) |
| `backend` | setuptools | every install (added by the script) |
| `build` | nanobind, scikit-build-core, maturin | `bind <package>`, `engine differential` |

The bindings are then built with
`pip install --no-deps --no-build-isolation ./bindings/<package>`, using the
hashed backends from the `build` group.

Worked example, the `py fx/05-pairs-trading (3.13)` job:

```bash
python -m venv "$RUNNER_TEMP/venv" && export PATH="$RUNNER_TEMP/venv/bin:$PATH"
bash .github/scripts/hashed_install.sh --group test --extra dev \
    --editable python/core python/fx/05-pairs-trading
```

`uv export` resolves `qp-core` and `fx-pairs` with the `dev` extra to about
30 locked packages; on Python 3.13 the markers select `numpy==2.5.4` and
`scipy==1.18.1`. Each is downloaded and hash-checked, then the two local
projects go in editable.

Evidence: running the script locally on Ubuntu (WSL) for `python/core`, all
23 projects, the research set and the bindings set on Python 3.11, 3.12 and
3.13 installs cleanly and passes `pip check`; the same install with every
hash in the exported file replaced by zeros is refused by pip with a hash
mismatch.

## Regenerate

From the repository root with [uv](https://docs.astral.sh/uv/) 0.13 or later:

```bash
uv lock
uv pip compile --universal --python-version 3.11 --no-header \
    requirements/ci-requirements.in -o requirements/ci-constraints.txt
python .github/scripts/check_constraints_lock.py
```

`uv lock` refreshes the lock (CI runs `uv lock --check`). The compile keeps
the constraints file for local pip users; the last command confirms it pins
the same versions as the lock. If it reports a difference, run
`uv lock --upgrade-package <name>` or adjust `ci-requirements.in` until the
two agree.

To check that the locked set installs from binary wheels on each supported
version and on the CI platform:

```bash
for v in 3.11 3.12 3.13; do
  uv pip compile -q --no-header --python-version "$v" \
      --python-platform x86_64-manylinux_2_28 --only-binary :all: \
      requirements/ci-requirements.in -c requirements/ci-constraints.txt \
      -o /dev/null && echo "$v ok"
done
```

## When to regenerate

- A project adds or raises a dependency in its `pyproject.toml`. Add it to
  `ci-requirements.in` too.
- Dependabot opens a pull request against this folder or a project.
- A new Python version joins the CI matrix.

The golden vectors depend on SciPy special functions at about 1e-12. When a
regenerated lock moves NumPy or SciPy, run the four golden suites
(`python/{equity,fx}/{01-options-pricing,03-var-es-engine}`) before merging.
