"""Print the tables in docs/VALIDATION.md sections 1 and 2.

Usage::

    python examples/validation_tables.py

Synthetic data only, seeds fixed (the same seeds as tests/conftest.py and
tests/test_arch_crosscheck.py). Section 2 needs the optional ``arch``
package and is skipped without it. Nothing is written to disk.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))

from eq_vol import fit_egarch, fit_garch, fit_gjr
from eq_vol.data import synthetic as syn
from eq_vol.egarch import egarch_recursion, expected_abs_z
from eq_vol.garch import garch_loglik, gaussian_loglik
from true_params import EGARCH_TRUE, GARCH_TRUE, GJR_TRUE


def recovery_table(title: str, fit, truth: dict) -> None:
    print(f"\n{title}\n")
    print("| param | true | estimate | classic SE | robust SE | (est - true)/robust SE |")
    print("|---|---|---|---|---|---|")
    for k, v in truth.items():
        est, se, rse = fit.params[k], fit.std_errors[k], fit.robust_std_errors[k]
        print(f"| {k} | {v:.4g} | {est:.4g} | {se:.2g} | {rse:.2g} | {(est - v) / rse:+.2f} |")


def main() -> None:
    print("## Section 1: parameter recovery (synthetic data)")
    recovery_table("GARCH(1,1), n = 20,000, seed 1",
                   fit_garch(syn.simulate_garch(20_000, seed=1, **GARCH_TRUE).returns), GARCH_TRUE)
    recovery_table("GJR-GARCH(1,1), n = 15,000, seed 2",
                   fit_gjr(syn.simulate_gjr(15_000, seed=2, **GJR_TRUE).returns), GJR_TRUE)
    recovery_table("EGARCH(1,1), n = 8,000, seed 3",
                   fit_egarch(syn.simulate_egarch(8_000, seed=3, **EGARCH_TRUE).returns), EGARCH_TRUE)
    ft = fit_garch(syn.simulate_garch(20_000, dist="t", nu=8.0, seed=4, **GARCH_TRUE).returns, dist="t")
    print(f"\nGARCH-t, n = 20,000, seed 4: nu true 8.0, estimate {ft.params['nu']:.2f} "
          f"(robust SE {ft.robust_std_errors['nu']:.2f})")

    try:
        from arch import arch_model
    except ImportError:
        print("\n(arch not installed: section 2 skipped)")
        return

    print("\n## Section 2: cross-check against arch (synthetic data)\n")
    n = 5000
    r = syn.simulate_garch(n, seed=42).returns
    ours = fit_garch(r)
    theirs = arch_model(100 * r, mean="Zero", vol="GARCH", p=1, q=1).fit(disp="off")
    log100 = np.log(100.0)
    rows = [
        ("alpha", ours.params["alpha"], theirs.params["alpha[1]"]),
        ("beta", ours.params["beta"], theirs.params["beta[1]"]),
        ("omega (pct^2)", ours.params["omega"] * 1e4, theirs.params["omega"]),
        ("log-likelihood (rescaled)", ours.loglik, theirs.loglikelihood + n * log100),
    ]
    print("| quantity | eq_vol | arch | abs difference |")
    print("|---|---|---|---|")
    for name, a, b in rows:
        print(f"| {name} | {a:.10g} | {b:.10g} | {abs(a - b):.1e} |")
    ll_at = garch_loglik(r, theirs.params["omega"] / 1e4, theirs.params["alpha[1]"],
                         theirs.params["beta[1]"])
    print(f"| GARCH: our LL at arch's parameters | | | {abs(ll_at - theirs.loglikelihood - n * log100):.1e} |")

    re_ = syn.simulate_egarch(n, seed=44).returns
    oe = fit_egarch(re_)
    te = arch_model(100 * re_, mean="Zero", vol="EGARCH", p=1, o=1, q=1).fit(disp="off")
    p = te.params
    s2 = egarch_recursion(re_, p["omega"] - (1 - p["beta[1]"]) * np.log(1e4), p["alpha[1]"],
                          p["gamma[1]"], p["beta[1]"], oe.init_var, expected_abs_z())
    diff = abs(gaussian_loglik(re_, s2) - te.loglikelihood - n * log100)
    print(f"| EGARCH: our LL at arch's parameters | | | {diff:.1e} |")

    rt = syn.simulate_garch(n, dist="t", nu=5.0, seed=7).returns
    o = fit_garch(rt)
    m = arch_model(100 * rt, mean="Zero", vol="GARCH", p=1, q=1)
    c, rb = m.fit(disp="off", cov_type="classic"), m.fit(disp="off", cov_type="robust")
    print("\nGaussian QMLE on Student-t(5) data, n = 5,000, seed 7:\n")
    print("| param | classic (ours) | classic (arch) | robust (ours) | robust (arch) |")
    print("|---|---|---|---|---|")
    for k, ak in (("alpha", "alpha[1]"), ("beta", "beta[1]")):
        print(f"| {k} | {o.std_errors[k]:.5f} | {c.std_err[ak]:.5f} | "
              f"{o.robust_std_errors[k]:.5f} | {rb.std_err[ak]:.5f} |")


if __name__ == "__main__":
    main()
