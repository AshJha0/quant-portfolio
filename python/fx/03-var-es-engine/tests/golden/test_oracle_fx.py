"""W5: fx_var against the independent oracle.

The oracle lives in ``python/equity/03-var-es-engine/tests/golden/
var_oracle.py`` (mpmath and exact rationals, no engine imports; the equity
tests enforce that and regenerate both JSON files). This file reads only
the FX JSON. Closed forms must agree to 1e-10 relative; the historical VaR
order statistic exactly.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from fx_var.backtesting import (
    basel_traffic_light,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)
from fx_var.expected_shortfall import empirical_var_es, normal_es, normal_var, t_es, t_var
from fx_var.parametric_var import cornish_fisher_valid, cornish_fisher_var, portfolio_sigma, var_covar

HERE = Path(__file__).resolve().parent
ORACLE = json.loads((HERE / "oracle_vectors.json").read_text())
V = ORACLE["values"]
INP = ORACLE["inputs"]
REL = ORACLE["tolerance"]["closed_form_rel"]


def tag(s: str) -> str:
    return s.replace("-", "m").replace(".", "")


def close(actual: float, expected: float, what: str, rel: float = REL) -> None:
    assert math.isfinite(actual), what
    assert abs(actual - expected) <= rel * abs(expected), (
        f"{what}: engine {actual!r} oracle {expected!r} rel {abs(actual / expected - 1):.3e}")


def conv_sample() -> np.ndarray:
    return np.array([math.floor(1000.0 * math.sin(0.37 * i + 0.1) * (1.0 + 0.5 * math.cos(0.11 * i))
                                * 100.0 + 0.5) / 100.0 for i in range(250)])


def test_oracle_file_is_the_fx_one() -> None:
    assert ORACLE["project"] == "fx"
    assert "eq_var" not in ORACLE["generator"] and "fx_var" not in ORACLE["generator"]


def test_parametric_closed_forms() -> None:
    sig, mean = (float(x) for x in INP["sigma_mean_odd"])
    for a_s in INP["alphas"]:
        a, t = float(a_s), tag(a_s)
        close(normal_var(1.0, a), V[f"normal_var_unit_a{t}"], f"N VaR {a_s}")
        close(normal_var(sig, a, mean), V[f"normal_var_odd_a{t}"], "N VaR odd")
        close(normal_es(1.0, a), V[f"normal_es_unit_a{t}"], f"N ES {a_s}")
        close(normal_es(sig, a, mean), V[f"normal_es_odd_a{t}"], "N ES odd")
        for nu_s in INP["dfs"]:
            nu = float(nu_s)
            close(t_var(1.0, a, nu), V[f"t{nu_s}_var_unit_a{t}"], f"t{nu_s} VaR {a_s}")
            close(t_var(sig, a, nu, mean), V[f"t{nu_s}_var_odd_a{t}"], f"t{nu_s} VaR odd")
            close(t_es(1.0, a, nu), V[f"t{nu_s}_es_unit_a{t}"], f"t{nu_s} ES {a_s}")
            close(t_es(sig, a, nu, mean), V[f"t{nu_s}_es_odd_a{t}"], f"t{nu_s} ES odd")


def test_cornish_fisher() -> None:
    sig, mean = (float(x) for x in INP["sigma_mean_odd"])
    for s_s, k_s, a_s in INP["cornish_fisher"]:
        key = f"cf_s{tag(s_s)}_k{tag(k_s)}_a{tag(a_s)}"
        valid = cornish_fisher_valid(float(s_s), float(k_s), float(a_s))
        assert valid == bool(V[f"{key}_valid"]), key
        if valid:
            close(cornish_fisher_var(sig, float(s_s), float(k_s), float(a_s), mean),
                  V[f"{key}_var_odd"], key)


def test_case_b_parametric() -> None:
    b = INP["case_b"]
    names = ["FX:EUR", "FX:JPY", "IR:USD"]
    w = pd.Series([float(x) for x in b["exposures"]], index=names)
    cov = pd.DataFrame([[float(x) for x in r] for r in b["cov"]], index=names, columns=names)
    close(portfolio_sigma(w, cov), V["b_sigma"], "B sigma")
    for a_s in ("0.01", "0.025"):
        a, t = float(a_s), tag(a_s)
        var, es = var_covar(w, cov, a)
        close(var, V[f"b_normal_var_a{t}"], "B N VaR")
        close(es, V[f"b_normal_es_a{t}"], "B N ES")
        close(var_covar(w, cov, a, horizon_days=10.0)[0], V[f"b_normal_var_10d_a{t}"], "B 10d")
        var, es = var_covar(w, cov, a, dist="t", df=5.0)
        close(var, V[f"b_t5_var_a{t}"], "B t5 VaR")
        close(es, V[f"b_t5_es_a{t}"], "B t5 ES")


def test_historical_order_statistics() -> None:
    p = conv_sample()
    for a_s in INP["hist_alphas"]:
        t = tag(a_s)
        var, es = empirical_var_es(p, float(a_s))
        assert var == V[f"hist_kth_var_a{t}"], a_s
        close(es, V[f"hist_kth_es_a{t}"], f"ES {a_s}", rel=1e-12)


def test_backtest_statistics() -> None:
    for n, x, a_s in INP["kupiec"]:
        lr, p = kupiec_pof(n_obs=n, n_exceptions=x, alpha=float(a_s))
        key = f"n{n}_x{x}_a{tag(a_s)}"
        if V[f"kupiec_lr_{key}"] < 1e-300:
            assert abs(lr) < 1e-12, key
        else:
            close(lr, V[f"kupiec_lr_{key}"], f"kupiec LR {key}")
        close(p, V[f"kupiec_p_{key}"], f"kupiec p {key}")
    pats = {"eq": lambda t: t % 40 == 0 or t in (100, 101),
            "fx": lambda t: t % 37 == 5 or t in (100, 101)}
    for name, rule in pats.items():
        ex = [1 if rule(t) else 0 for t in range(250)]
        lr, p = christoffersen_independence(ex)
        close(lr, V[f"chr_{name}_ind_lr"], f"{name} ind LR")
        close(p, V[f"chr_{name}_ind_p"], f"{name} ind p")
        lr, p = conditional_coverage(ex, 0.01)
        close(lr, V[f"chr_{name}_cc_lr"], f"{name} cc LR")
        close(p, V[f"chr_{name}_cc_p"], f"{name} cc p")
    for n, x, a_s in INP["basel"]:
        r = basel_traffic_light(n_obs=n, n_exceptions=x, alpha=float(a_s))
        close(r.cumulative_prob, V[f"basel_cdf_n{n}_x{x}_a{tag(a_s)}"], f"basel {n} {x} {a_s}")
