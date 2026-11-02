# Contributing

This file covers setup, the test gate a change must pass, the writing rules
for docs, and how to regenerate golden vectors.

## Layout

| Path | What it is |
|---|---|
| `python/core/` | `qp-core` (import `qp_core`): the shared statistics, execution, calendar, day-count, validation and metrics code. |
| `python/{equity,fx,foundations}/NN-name/` | 23 Python projects. Each has its own `pyproject.toml`, `src/` and `tests/`, and depends on `qp-core`. |
| `pyproject.toml`, `uv.lock` | The uv workspace: `python/core` and the 23 projects, locked together. |
| `cpp/core/` | `qp::core`, a header-only CMake target shared by the engines. |
| `cpp/<engine>/` | 4 C++ engines built with CMake presets. Tests use GoogleTest. |
| `rust/` | Cargo workspace: `core` (`qp-core`) and the 4 engine crates, one `Cargo.lock`. Library crates have no third-party dependencies. |
| `bindings/cpp-nanobind/` | `qp-cpp` (import `qp_cpp`): nanobind bindings over the C++ engines. |
| `bindings/rust-pyo3/` | `qp-rust` (import `qp_rust`): PyO3 bindings over the Rust crates. |
| `requirements/` | The CI constraints file and how to regenerate it. |
| `tests/` | `test_docs_counts.py` checks the counts quoted in the docs; `test_ci_scripts.py` tests the CI helper scripts. |

## Setup

### Python

Python 3.10 to 3.13. The quickest setup is the uv workspace, which installs
`python/core` and all 23 projects into one `.venv` at the root from
`uv.lock`:

```bash
uv sync --all-packages
uv run pytest -p no:cacheprovider -q python/equity/02-volatility-modeling
```

CI installs each project on its own instead, so a missing dependency shows
up. To reproduce one CI job, use a fresh venv outside the repository,
install `python/core` first and then the project with its test extra (`test`
if the project declares it, else `dev`):

```bash
python -m venv ~/.venvs/qp
. ~/.venvs/qp/bin/activate            # Windows: ~/.venvs/qp/Scripts/activate
python -m pip install -c requirements/ci-constraints.txt -e python/core
cd python/equity/02-volatility-modeling
python -m pip install -c ../../../requirements/ci-constraints.txt -e ".[test]" pytest hypothesis
python -m pytest -p no:cacheprovider -q
```

After you change a dependency in any `pyproject.toml`, run `uv lock` and
commit `uv.lock`. CI runs `uv lock --check` and fails on a stale lock.

### C++

GCC 15 or 16, Clang 21 or MSVC 19.44 (Visual Studio 2022), CMake, Ninja and
GoogleTest. Each engine has a `CMakePresets.json` with these presets:

| Preset | Compiler and flags |
|---|---|
| `gcc` | GCC 15, C++23, warnings as errors |
| `clang` | Clang 21, C++23, warnings as errors |
| `gcc16-cxx26` | GCC 16 with `QP_CXX26=ON` (C++26 features behind feature-test macros) |
| `asan` | AddressSanitizer and UBSan |
| `tsan` | ThreadSanitizer |
| `msvc` | MSVC (`cl`), from a Developer Command Prompt or after `vcvars64.bat` |

Run them from the engine directory. Build trees go under `build/<preset>`,
which git ignores:

```bash
cd cpp/fx-var-engine
cmake --preset gcc
cmake --build --preset gcc
ctest --preset gcc
```

If GoogleTest is not installed (for example with MSVC), the engine fetches
it at configure time.

### Rust

`rust/rust-toolchain.toml` pins the toolchain. rustup installs it on first
use. Run every command from `rust/`:

```bash
cd rust
cargo test --release --locked -p fx-var-engine
cargo clippy --release --all-targets --locked -p fx-var-engine -- -D warnings
cargo fmt --all --check
```

### Bindings

Install `python/core` and the four reference packages, then build each
binding package from source. The C++ package needs a C++23 compiler and
CMake. The Rust package needs the pinned Rust toolchain:

```bash
python -m pip install -e python/core     -e python/equity/01-options-pricing -e python/fx/01-options-pricing     -e python/equity/03-var-es-engine -e python/fx/03-var-es-engine pytest hypothesis
python -m pip install ./bindings/cpp-nanobind
python -m pip install ./bindings/rust-pyo3
python -m pytest -p no:cacheprovider -q bindings/cpp-nanobind/tests bindings/rust-pyo3/tests
```

Build directories go outside the repository or under a git-ignored
`build/` or `target/`. Example scripts write to a temp directory or to the
path given by `--out`, never into the tree.

## Lint

The same commands run in the `lint`, `clang-format`, `clang-tidy` and
`rust fmt` CI jobs:

```bash
uv lock --check
uv run ruff check python
uv run mypy                                   # config at the root decides the files
git ls-files 'cpp/*.cpp' 'cpp/*.hpp' | xargs clang-format-21 --dry-run --Werror
cd cpp/fx-var-engine && cmake --preset clang -DCMAKE_EXPORT_COMPILE_COMMANDS=ON     && clang-tidy-21 -p build/clang src/*.cpp
cd rust && cargo fmt --all --check
```

`pre-commit` runs ruff and clang-format on the files you change.

## Test gate

A pull request can merge only when the `ci-ok` check is green. `ci-ok`
passes only if every other CI job passed:

| Job | What it runs |
|---|---|
| `py core (<version>)` | pytest for `python/core` on Python 3.10 to 3.13 |
| `py <project> (<version>)` | pytest for one project on one Python version, in a fresh venv with `python/core` installed first |
| `lint (uv lock, ruff, mypy)` | `uv lock --check`, `ruff check python`, `mypy` |
| `goldens --check` | regenerated C++ headers and Rust modules match the golden JSON |
| `cpp <engine> (<preset>)` | build and ctest with the `gcc`, `clang`, `asan` and `tsan` presets on Ubuntu |
| `cpp <engine> (gcc16-cxx26)` | build and ctest in the `gcc:16` container |
| `cpp <engine> (msvc)` | build and ctest on Windows Server 2025 with MSVC |
| `clang-format` | every C++ file matches `.clang-format` |
| `clang-tidy <engine>` | clang-tidy with the root `.clang-tidy`; any finding fails |
| `rust <crate>` | `cargo test --release` with warnings denied, then clippy, per workspace crate |
| `rust fmt` | `cargo fmt --all --check` |
| `bind <package> (<version>)` | builds `qp-cpp` or `qp-rust` and runs its tests against the Python reference packages |
| `bench` | Release benches; fails on a p50 more than 2.0 times the committed value for a kernel above 1 us |
| `hygiene` | banned-word check; voice report; tests for the CI scripts |
| `counts` | sums executed pass counts into `counts.json`, then runs `tests/test_docs_counts.py` |

CI runner benchmark numbers are not the published numbers. Runners are
shared virtual machines, so the `bench` job only catches large regressions.
Its table is in the run summary and the JSON is in the `bench-results`
artefact.

Before you push:

1. Run the full test suite of every project you changed, to completion.
2. Every bug fix comes with a regression test that fails on the old code.
   Say in the commit message that you checked it fails first.
3. Do not widen a tolerance without writing down why the new value is right
   (ulps, condition number, Monte Carlo standard error).
4. Run `pre-commit run --all-files` (install with `pip install pre-commit`
   and `pre-commit install`).

## Writing rules

These apply to every doc, docstring and commit message.

- Short declarative sentences. No em-dashes: use a full stop, a comma or a
  colon.
- No marketing superlatives and no sincerity or intent markers. The
  banned word and phrase list is in `.github/scripts/style_report.py`.
- The hiring-process word that starts with "intervie" must not appear
  anywhere. CI fails if it does.
- A result from synthetic data says "synthetic" in the same sentence.
- Numbers go in tables, produced by a command you name. Never hand-edit a
  result.
- State a limitation in one sentence.
- Record fixes in `CHANGELOG.md` under `[Unreleased]`, not in prose.

## Golden vectors

The four compiled engines are checked against values computed by their
Python twins. The flow has three steps and runs in this order.

1. Regenerate the JSON from the Python reference:

   ```bash
   cd python/equity/01-options-pricing          # or fx/01, equity/03, fx/03
   python tests/golden/generate_golden.py
   ```

2. Regenerate the C++ header and the Rust module from that JSON:

   ```bash
   python cpp/equity-options-engine/tools/gen_golden_header.py
   python rust/equity-options-engine/tools/gen_golden_rs.py
   ```

3. Check that all three agree. CI runs the same commands:

   ```bash
   python cpp/equity-options-engine/tools/gen_golden_header.py --check
   python rust/equity-options-engine/tools/gen_golden_rs.py --check
   ```

Explain every changed golden value in the commit message.

## Releases

Maintainers release from the Actions tab with the `release` workflow. It
refuses unless `ci-ok` passed on the chosen commit and `CHANGELOG.md` has a
section for the version.
