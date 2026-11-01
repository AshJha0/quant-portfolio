# requirements

| File | Purpose |
|---|---|
| `ci-requirements.in` | The union of every Python project's runtime dependencies, the test extras CI installs, the plotting extra, `pytest` and `hypothesis`. Live-data extras (`yfinance`) are left out. |
| `ci-constraints.txt` | One resolved version per package, with environment markers where the version differs by Python version. CI passes it to every install with `pip install -c`. |

The file is a constraints file, not a requirements file. It pins the version
of anything a project installs and does not add packages a project did not ask
for. Each project still declares its own dependencies in its `pyproject.toml`.

## Regenerate

Use [uv](https://docs.astral.sh/uv/) 0.13 or later from the repository root:

```bash
uv pip compile --universal --python-version 3.10 --no-header \
    requirements/ci-requirements.in -o requirements/ci-constraints.txt
```

`--universal` resolves for every Python version from 3.10 up in one file.
NumPy, SciPy, pandas and a few others need different versions on 3.10, so
the output carries `python_full_version` markers.

Then check that the pinned set installs from binary wheels on each supported
version and on the CI platform:

```bash
for v in 3.10 3.11 3.12 3.13; do
  uv pip compile -q --no-header --python-version "$v" \
      --python-platform x86_64-manylinux_2_28 --only-binary :all: \
      requirements/ci-requirements.in -c requirements/ci-constraints.txt \
      -o /dev/null && echo "$v ok"
done
```

## When to regenerate

- A project adds or raises a dependency in its `pyproject.toml`. Add it to
  `ci-requirements.in` too.
- Dependabot opens a pull request against this folder.
- A new Python version joins the CI matrix.

The golden vectors depend on SciPy special functions at about 1e-12. When a
regenerated file moves NumPy or SciPy, run the four golden suites
(`python/{equity,fx}/{01-options-pricing,03-var-es-engine}`) before merging.
