"""Python-side lock on the cross-language golden vectors.

``cpp/equity-var-engine/tests/test_cross_language.cpp`` and
``rust/equity-var-engine/tests/test_cross_language.rs`` assert values that
``tests/golden/generate_golden.py`` computes from this package and commits to
``tests/golden/golden_vectors.json``. The C++ header and the Rust module
``tests/golden/mod.rs`` are generated from that JSON by the tools in each
engine. This file checks the JSON against a live run of the generator and
checks the convention keys against independent hand computations. It holds
no hand-copied constants.

Regenerate after a deliberate change::

    PYTHONPATH=src python tests/golden/generate_golden.py
    python ../../../cpp/equity-var-engine/tools/gen_golden_header.py
    python ../../../rust/equity-var-engine/tools/gen_golden_rs.py

Tolerance of the live-vs-committed check: 64 ulps
---------------------------------------------------
The JSON is written with ``repr`` and round-trips exactly, but numpy and
scipy are not bit-stable across releases (numpy 2.5.4 moved ``c_ewma_cov_22``
by 1 ulp). The longest floating-point chain behind any key is an EWMA
recursion: with decay 0.94 each step contracts earlier rounding error, so the
accumulated error is bounded by about ``1 / (1 - 0.94) ~ 17`` roundings, plus
up to 4 ulps from a different libm ``sin``/``cos`` in the inputs, amplified at
most a few times by the Kupiec / chi2 tail (condition number < 4 on these
inputs). 64 ulps covers that with margin and is still five orders of
magnitude below the 1e-9 relative tolerance the compiled engines use.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
GOLDEN_DIR = HERE / "golden"
GOLDEN_JSON = GOLDEN_DIR / "golden_vectors.json"

REL, ABS = 1e-9, 1e-12
MAX_ULPS = 64


def _load_generator():
    spec = importlib.util.spec_from_file_location("eq_var_generate_golden", GOLDEN_DIR / "generate_golden.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def committed() -> dict:
    return json.loads(GOLDEN_JSON.read_text())


@pytest.fixture(scope="module")
def generator():
    return _load_generator()


@pytest.fixture(scope="module")
def regenerated(generator) -> dict:
    return generator.build()


def ulp_distance(a: float, b: float) -> int:
    """Number of representable doubles between ``a`` and ``b`` (same sign assumed)."""
    if a == b:
        return 0
    if math.copysign(1.0, a) != math.copysign(1.0, b):
        return 2**62
    ia = int(np.array(abs(a)).view(np.int64))
    ib = int(np.array(abs(b)).view(np.int64))
    return abs(ia - ib)


REL_TOL_OTHER_VERSIONS = 1e-10  # SciPy t-quantile moved 4.3e-12 rel between 1.15 and 1.18


def _minor(v: str) -> tuple:
    return tuple(v.split(".")[:2])


def _same_minor_versions(recorded: dict) -> bool:
    import scipy

    return (_minor(recorded["numpy"]) == _minor(np.__version__)
            and _minor(recorded["scipy"]) == _minor(scipy.__version__))


def _rel_close(a: float, b: float) -> bool:
    return abs(a - b) <= REL_TOL_OTHER_VERSIONS * max(abs(a), abs(b), 1e-300)


class TestUlpDistance:
    def test_adjacent_doubles(self):
        assert ulp_distance(1.0, math.nextafter(1.0, 2.0)) == 1

    def test_sign_change_is_far(self):
        assert ulp_distance(1e-300, -1e-300) > MAX_ULPS


class TestCommittedJsonIsReproducible:
    def test_same_keys(self, committed, regenerated):
        assert list(committed["values"]) == list(regenerated["values"])

    def test_every_value_within_ulp_bound(self, committed, regenerated):
        # The ulp bound holds against the numpy/scipy minor versions the JSON
        # was generated with. Other versions change special functions
        # (Student-t isf, erfc) by up to ~1e2 ulps, so there the check falls
        # back to REL_TOL_OTHER_VERSIONS, which is still far below any
        # modelling tolerance.
        same = _same_minor_versions(committed["versions"])
        for k, v in regenerated["values"].items():
            c = committed["values"][k]
            pairs = list(zip(c, v)) if isinstance(v, list) else [(c, v)]
            if isinstance(v, list):
                assert len(c) == len(v), k
            for ci, vi in pairs:
                if same:
                    assert ulp_distance(ci, vi) <= MAX_ULPS, f"{k}: committed {ci!r} vs regenerated {vi!r}"
                else:
                    assert _rel_close(ci, vi), f"{k}: committed {ci!r} vs regenerated {vi!r}"

    def test_versions_recorded(self, committed):
        assert set(committed["versions"]) == {"python", "numpy", "scipy"}

    def test_tolerance_is_the_one_the_engines_use(self, committed):
        assert committed["tolerance"] == {"rel": REL, "abs": ABS}

    def test_inputs_documented(self, committed):
        assert "sin(3t + 1)" in committed["inputs"]["case_a"]
        assert committed["inputs"]["case_b"]["exposures"] == [1.0e6, -5.0e5, 2.0e5]
        assert "floor(x * 100 + 0.5)" in committed["inputs"]["conv"]["pnl"]


SPEC_S13_KEYS = (
    [f"conv_{m}_{a}" for a in ("a010", "a025", "a050") for m in ("var", "es")]
    + [f"conv_basel_{f}_n{n}_x{x}_a{a}" for (n, x, a) in
       [(250, 4, "010"), (250, 5, "010"), (250, 9, "010"), (250, 10, "010"), (500, 8, "010"),
        (500, 9, "010"), (500, 12, "010"), (1000, 15, "010"), (250, 12, "025")]
       for f in ("zone", "k", "cdf")]
    + ["conv_kupiec_lr_n250_x5", "conv_kupiec_lr_n500_x9", "conv_kupiec_lr_n250_x0", "conv_as_z2",
       "conv_cf_valid_s0122_km0427", "conv_cf_valid_s00_k00", "conv_cf_valid_s10_k10",
       "conv_cf_valid_sm0105_k80001", "conv_cf_valid_a001_s0122_km0427"]
    + [f"conv_{m}_a{ij}" for m in ("higham", "clip") for ij in ("01", "02", "12")]
    + ["conv_ewma_sigma_last"]
)


class TestConventionKeysAgainstHandComputation:
    """Spec S13 keys checked against plain-Python computations that share no
    code with eq_var, so a change in the package cannot move both sides."""

    def test_every_spec_key_present(self, committed):
        conv = {k for k in committed["values"] if k.startswith("conv_")}
        assert conv == set(SPEC_S13_KEYS)

    @pytest.fixture()
    def losses(self):
        p = [math.floor(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i)) * 100.0 + 0.5) / 100.0
             for i in range(250)]
        return sorted((-x for x in p), reverse=True)

    @pytest.mark.parametrize("alpha, tag", [(0.010, "a010"), (0.025, "a025"), (0.050, "a050")])
    def test_var_es(self, committed, losses, alpha, tag):
        k = 250 * alpha
        m = math.ceil(k - 1e-9)
        var = losses[m - 1]
        es = (sum(losses[: m - 1]) + (k - (m - 1)) * var) / k
        assert committed["values"][f"conv_var_{tag}"] == var
        assert committed["values"][f"conv_es_{tag}"] == pytest.approx(es, rel=1e-14)

    def test_kupiec(self, committed):
        def lr(n, x, a=0.01):
            ll = lambda q: (n - x) * math.log(1 - q) + (x * math.log(q) if x else 0.0)  # noqa: E731
            return -2 * ll(a) + (2 * ll(x / n) if x else 0.0)

        for n, x in ((250, 5), (500, 9), (250, 0)):
            assert committed["values"][f"conv_kupiec_lr_n{n}_x{x}"] == pytest.approx(lr(n, x), rel=1e-12)

    def test_as_z2(self, committed, losses):
        # VaR 1000 is exceeded on several days of P, so the exception branch is exercised.
        pnl = [-x for x in losses]
        hits = [x for x in pnl if x < -1000.0]
        assert len(hits) > 0
        want = sum(x / 1300.0 for x in hits) / (250 * 0.025) + 1.0
        assert committed["values"]["conv_as_z2"] == pytest.approx(want, abs=1e-12)

    def test_higham_and_clip(self, committed):
        # The nearest correlation matrix to A keeps the signs and puts 0.5 off the diagonal.
        for ij, sign in (("01", 1), ("02", 1), ("12", -1)):
            assert committed["values"][f"conv_higham_a{ij}"] == pytest.approx(sign * 0.5, abs=1e-8)
            assert committed["values"][f"conv_clip_a{ij}"] == pytest.approx(sign * 0.5, abs=1e-9)

    def test_ewma(self, committed):
        p = [math.floor(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i)) * 100.0 + 0.5) / 100.0
             for i in range(250)]
        r = [x / 1e5 for x in p]
        s2 = sum(x * x for x in r[:30]) / 30
        for x in r:
            s2 = 0.94 * s2 + 0.06 * x * x
        assert committed["values"]["conv_ewma_sigma_last"] == pytest.approx(math.sqrt(s2), rel=1e-13)
