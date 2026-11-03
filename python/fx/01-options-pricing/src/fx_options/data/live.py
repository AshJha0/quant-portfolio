"""Optional live FX reference rates from the ECB via the free Frankfurter API.

Import-guarded and NEVER used by the test suite (CONVENTIONS.md: no
network access in test code paths).  Intended only for ad-hoc exploration,
e.g. seeding examples with a current EURUSD spot.

The ECB publishes daily reference rates around 16:00 CET; they are
indicative mid rates, not tradable quotes, and are quoted EUR-based
(units of currency per 1 EUR).

Deprecated: use ``qp_data.fetch("frankfurter:EUR")`` or
``qp_data.fetch("ecb:EXR:USD")``. This function now delegates to qp_data and
is removed in v4.0.0.
"""

from __future__ import annotations

__all__ = ["LiveDataUnavailable", "fetch_ecb_rates"]


def _qp_data():  # imported lazily: qp_data is optional for this project
    try:
        from qp_data import legacy
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "live data now comes from qp_data: pip install -e python/data"
        ) from exc
    return legacy


class LiveDataUnavailable(RuntimeError):
    """Raised when the live-rate endpoint cannot be reached or parsed."""


def fetch_ecb_rates(base: str = "EUR", symbols: tuple[str, ...] = ("USD", "JPY", "GBP", "CHF"),
                    timeout: float = 10.0) -> dict[str, float]:
    """Fetch latest ECB reference rates (units of ``symbol`` per 1 ``base``).

    Parameters
    ----------
    base : str
        Base currency ISO code (the '1 unit' side).
    symbols : tuple of str
        Quote currencies to request.
    timeout : float
        Socket timeout in seconds.

    Returns
    -------
    dict
        Mapping symbol -> rate, plus ``"_date"`` (ISO date string of the
        fixing, stored under a non-currency key).

    Raises
    ------
    LiveDataUnavailable
        On any network or parsing failure: callers should degrade to
        synthetic data (`fx_options.data.synthetic`).
    """
    legacy = _qp_data()
    legacy.deprecated("fx_options.data.live.fetch_ecb_rates", 'qp_data.fetch("frankfurter:EUR")')
    try:
        frame = legacy.frankfurter(base, symbols, timeout=timeout)
        row = frame.iloc[-1]
        rates: dict[str, float] = {str(k): float(v) for k, v in row.items()}
        rates["_date"] = frame.index[-1].date().isoformat()  # type: ignore[assignment]
        return rates
    except Exception as exc:
        raise LiveDataUnavailable(f"could not fetch ECB rates for {base}: {exc}") from exc
