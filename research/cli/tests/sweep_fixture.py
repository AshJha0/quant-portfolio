"""A deterministic toy study for the qp-run tests."""


def study(a: float, b: float, c: float) -> dict:
    return {"sharpe": a * b + c, "n": 3}
