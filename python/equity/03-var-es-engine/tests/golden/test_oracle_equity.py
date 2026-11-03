"""W5: eq_var against the independent oracle (``var_oracle.py``).

The oracle is mpmath and exact rationals and shares no code with eq_var;
``test_oracle_imports_no_engine_code`` keeps it that way. Closed forms must
agree to 1e-10 relative; the historical VaR order statistic exactly.
"""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from eq_var.backtesting import (
    basel_traffic_light,
    christoffersen_cc,
    christoffersen_independence,
    kupiec_pof,
)
from eq_var.expected_shortfall import expected_shortfall, normal_es, student_t_es
from eq_var.historical_var import historical_var
from eq_var.parametric_var import (
    cornish_fisher_valid,
    cornish_fisher_var,
    parametric_var,
    portfolio_sigma,
)

HERE = Path(__file__).resolve().parent
ORACLE = json.loads((HERE / "oracle_vectors.json").read_text())
V = ORACLE["values"]
INP = ORACLE["inputs"]
REL = ORACLE["tolerance"]["closed_form_rel"]
ALLOWED_IMPORTS = {"__future__", "json", "math", "sys", "fractions", "pathlib", "mpmath"}


def tag(s: str) -> str:
    return s.replace("-", "m").replace(".", "")


def close(actual: float, expected: float, what: str, rel: float = REL) -> None:
    assert math.isfinite(actual), what
    assert abs(actual - expected) <= rel * abs(expected), (
        f"{what}: engine {actual!r} oracle {expected!r} rel {abs(actual / expected - 1):.3e}")


def conv_sample() -> np.ndarray:
    return np.array([math.floor(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i))
                                * 100.0 + 0.5) / 100.0 for i in range(250)])


def load_oracle():
    pytest.importorskip("mpmath")
    sys.path.insert(0, str(HERE))
    try:
        import var_oracle
    finally:
        sys.path.remove(str(HERE))
    return var_oracle


# ---------------------------------------------------------------- the oracle
@pytest.mark.parametrize("name", ["var_oracle.py", "emit_oracle_consts.py"])
def test_oracle_imports_no_engine_code(name: str) -> None:
    """The oracle and its emitter import only a few stdlib modules and mpmath."""
    tree = ast.parse((HERE / name).read_text())
    seen: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            seen |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            seen.add((node.module or "").split(".")[0])
        elif isinstance(node, ast.Call) and getattr(node.func, "id", "") in {
                "__import__", "exec", "eval"}:
            pytest.fail("dynamic import or exec in the oracle")
        elif isinstance(node, ast.Attribute) and node.attr == "import_module":
            pytest.fail("importlib in the oracle")
    assert seen <= ALLOWED_IMPORTS, seen - ALLOWED_IMPORTS
    assert not seen & {"eq_var", "fx_var", "numpy", "scipy", "importlib"}


def test_oracle_json_regenerates() -> None:
    o = load_oracle()
    for path, project in ((o.EQ_OUT, "equity"), (o.FX_OUT, "fx")):
        assert json.loads(path.read_text())["values"] == o.build(project)["values"], project


def test_emitted_cpp_and_rust_constants_are_fresh() -> None:
    sys.path.insert(0, str(HERE))
    try:
        import emit_oracle_consts
    finally:
        sys.path.remove(str(HERE))
    assert emit_oracle_consts.main(["--check"]) == 0


def test_oracle_reproduces_bcbs_1996_table() -> None:
    """BCBS (1996) backtesting framework, table 1: 250 days at 99 %."""
    o = load_oracle()
    for x, pct in o.BCBS_1996_CUM_PCT.items():
        assert round(100.0 * V[f"basel_cdf_n250_x{x}_a001"], 2) == pct, x


def test_oracle_kupiec_regions_against_published_table() -> None:
    """Kupiec (1995) non-rejection regions at 5 %, as tabulated by Jorion.

    Seven of the nine cells are reproduced exactly by the LR test. Two are
    not, and the difference is in the table, not in the statistic:
    T = 255, p = 1 %: zero exceptions gives LR = 5.126 > 3.841, so the LR
    test rejects N = 0; the table prints no lower bound. T = 255, p = 5 %:
    N = 20 gives LR = 3.727 < 3.841, so the LR test accepts it; the table
    stops at 19. Both are pinned here so a change in either is seen.
    """
    o = load_oracle()
    mp = o.mp
    known = {("0.01", 255): (1, 6), ("0.05", 255): (7, 20)}
    for (a, t), region in o.KUPIEC_1995_REGIONS.items():
        acc = [x for x in range(t + 1) if o.kupiec(t, x, mp.mpf(a))[0] < o.CHI2_1_95]
        got = (acc[0], acc[-1])
        assert acc == list(range(got[0], got[1] + 1))  # the region is an interval
        assert got == known.get((a, t), region), (a, t, got, region)
    assert abs(float(o.kupiec(255, 0, mp.mpf("0.01"))[0]) - 5.1257) < 1e-4
    assert abs(float(o.kupiec(255, 20, mp.mpf("0.05"))[0]) - 3.7272) < 1e-4


def test_oracle_mar99_third_worst_of_250() -> None:
    """MAR33/MAR99: the 99 % VaR on 250 days is the 3rd worst loss."""
    assert V["hist_kth_rank_a001"] == 3.0
    losses = sorted(-conv_sample(), reverse=True)
    assert V["hist_kth_var_a001"] == losses[2]
    # The interpolated rule sits between the 3rd and 4th worst (h = 2.49).
    assert losses[3] <= V["hist_interp_var_a001"] <= losses[2]


# ---------------------------------------------------------------- the engine
def test_parametric_closed_forms() -> None:
    one = np.array([[1.0]])
    sig, mean = (float(x) for x in INP["sigma_mean_odd"])
    for a_s in INP["alphas"]:
        a, t = float(a_s), tag(a_s)
        close(parametric_var(np.array([1.0]), one, a), V[f"normal_var_unit_a{t}"], f"N VaR {a_s}")
        close(parametric_var(np.array([sig]), one, a, mean=mean), V[f"normal_var_odd_a{t}"],
              "N VaR odd")
        close(normal_es(1.0, a), V[f"normal_es_unit_a{t}"], f"N ES {a_s}")
        close(normal_es(sig, a, mean), V[f"normal_es_odd_a{t}"], "N ES odd")
        for nu_s in INP["dfs"]:
            nu = float(nu_s)
            close(parametric_var(np.array([1.0]), one, a, dist="t", df=nu),
                  V[f"t{nu_s}_var_unit_a{t}"], f"t{nu_s} VaR {a_s}")
            close(parametric_var(np.array([sig]), one, a, dist="t", df=nu, mean=mean),
                  V[f"t{nu_s}_var_odd_a{t}"], f"t{nu_s} VaR odd {a_s}")
            close(student_t_es(1.0, a, nu), V[f"t{nu_s}_es_unit_a{t}"], f"t{nu_s} ES {a_s}")
            close(student_t_es(sig, a, nu, mean), V[f"t{nu_s}_es_odd_a{t}"],
                  f"t{nu_s} ES odd {a_s}")


def test_cornish_fisher() -> None:
    sig, mean = (float(x) for x in INP["sigma_mean_odd"])
    for s_s, k_s, a_s in INP["cornish_fisher"]:
        key = f"cf_s{tag(s_s)}_k{tag(k_s)}_a{tag(a_s)}"
        valid = cornish_fisher_valid(float(s_s), float(k_s), float(a_s))
        assert valid == bool(V[f"{key}_valid"]), key
        if valid:
            close(cornish_fisher_var(sig, float(a_s), float(s_s), float(k_s), mean),
                  V[f"{key}_var_odd"], key)


def test_case_b_parametric() -> None:
    b = INP["case_b"]
    w = np.array([float(x) for x in b["exposures"]])
    vols = np.array([float(x) for x in b["vols"]])
    cov = np.outer(vols, vols) * np.array([[float(x) for x in r] for r in b["corr"]])
    sigma = portfolio_sigma(w, cov)
    close(sigma, V["b_sigma"], "B sigma")
    for a_s in ("0.01", "0.025"):
        a, t = float(a_s), tag(a_s)
        close(parametric_var(w, cov, a), V[f"b_normal_var_a{t}"], "B N VaR")
        close(parametric_var(w, cov, a, horizon_days=10), V[f"b_normal_var_10d_a{t}"],
              "B N VaR 10d")
        close(normal_es(sigma, a), V[f"b_normal_es_a{t}"], "B N ES")
        close(parametric_var(w, cov, a, dist="t", df=5.0), V[f"b_t5_var_a{t}"], "B t5 VaR")
        close(student_t_es(sigma, a, 5.0), V[f"b_t5_es_a{t}"], "B t5 ES")


def test_historical_order_statistics() -> None:
    p = conv_sample()
    for a_s in INP["hist_alphas"]:
        a, t = float(a_s), tag(a_s)
        assert historical_var(p, a) == V[f"hist_kth_var_a{t}"], a_s
        close(expected_shortfall(p, a), V[f"hist_kth_es_a{t}"], f"ES {a_s}", rel=1e-12)


def test_backtest_statistics() -> None:
    for n, x, a_s in INP["kupiec"]:
        r = kupiec_pof(n_obs=n, n_exceptions=x, alpha=float(a_s))
        key = f"n{n}_x{x}_a{tag(a_s)}"
        if V[f"kupiec_lr_{key}"] < 1e-300:
            assert abs(r["lr"]) < 1e-12, key
        else:
            close(r["lr"], V[f"kupiec_lr_{key}"], f"kupiec LR {key}")
        close(r["pvalue"], V[f"kupiec_p_{key}"], f"kupiec p {key}")
    pats = {"eq": lambda t: t % 40 == 0 or t in (100, 101),
            "fx": lambda t: t % 37 == 5 or t in (100, 101)}
    for name, rule in pats.items():
        ex = np.array([bool(rule(t)) for t in range(250)])
        ind = christoffersen_independence(ex)
        cc = christoffersen_cc(ex, 0.01)
        close(ind["lr"], V[f"chr_{name}_ind_lr"], f"{name} ind LR")
        close(ind["pvalue"], V[f"chr_{name}_ind_p"], f"{name} ind p")
        close(cc["lr"], V[f"chr_{name}_cc_lr"], f"{name} cc LR")
        close(cc["pvalue"], V[f"chr_{name}_cc_p"], f"{name} cc p")
    for n, x, a_s in INP["basel"]:
        r = basel_traffic_light(n_obs=n, n_exceptions=x, alpha=float(a_s))
        close(r["cumulative_prob"], V[f"basel_cdf_n{n}_x{x}_a{tag(a_s)}"], f"basel {n} {x} {a_s}")
