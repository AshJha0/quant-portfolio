"""Repository-wide pytest setup: Hypothesis settings profiles (W12).

Three profiles, chosen in this order:

1. ``--hypothesis-profile NAME`` on the command line (Hypothesis's own
   option; it is applied after this file loads, so it wins);
2. the ``QP_HYPOTHESIS_PROFILE`` environment variable;
3. ``ci`` when the ``CI`` environment variable is set (GitHub Actions sets
   it), otherwise ``dev``.

=========  ============================================================
``dev``    Hypothesis defaults: random examples, local example database.
``ci``     ``derandomize=True``: examples come from a fixed seed derived
           from each test, so two runs on one commit draw identical
           examples and a red build reproduces exactly. No example
           database (runners are fresh), no deadline (shared runners
           are noisy), and the reproduction blob is printed on failure.
``nightly``  Random examples and ten times the CI budget, to look for
           inputs the fixed seed never reaches. Run by the scheduled CI.
=========  ============================================================

How this file is loaded. pytest loads a ``conftest.py`` from every
directory between the test file and ``--confcutdir`` (default: the
rootdir). Runs from the repository root (``tests/``, ``research/``) pick
it up directly. The per-project CI jobs run inside ``python/<area>/<nn>``
and pass ``--confcutdir="$GITHUB_WORKSPACE"`` so it is loaded there too.
A local ``pytest`` inside a project does not load it and keeps the
Hypothesis defaults, which is what the ``dev`` profile is anyway.

Reference: Hypothesis documentation, "Settings profiles".
"""

from __future__ import annotations

import os

try:
    from hypothesis import HealthCheck, Phase, settings
except ImportError:  # jobs that install pytest only (hygiene, counts)
    settings = None  # type: ignore[assignment,misc]

PROFILES = ("dev", "ci", "nightly")
CI_MAX_EXAMPLES = 100


def _register() -> None:
    assert settings is not None
    settings.register_profile("dev", settings.get_profile("default"))
    settings.register_profile(
        "ci",
        derandomize=True,
        database=None,
        deadline=None,
        print_blob=True,
        max_examples=CI_MAX_EXAMPLES,
        suppress_health_check=[HealthCheck.too_slow],
    )
    settings.register_profile(
        "nightly",
        derandomize=False,
        database=None,
        deadline=None,
        print_blob=True,
        max_examples=10 * CI_MAX_EXAMPLES,
        phases=list(Phase),
        suppress_health_check=[HealthCheck.too_slow],
    )


def chosen_profile(environ: os._Environ[str] | dict[str, str] = os.environ) -> str:
    """The profile name from the environment (the CLI option overrides it later)."""
    name = environ.get("QP_HYPOTHESIS_PROFILE") or ("ci" if environ.get("CI") else "dev")
    if name not in PROFILES:
        raise ValueError(f"QP_HYPOTHESIS_PROFILE={name!r}; expected one of {PROFILES}")
    return name


if settings is not None:
    _register()
    settings.load_profile(chosen_profile())
