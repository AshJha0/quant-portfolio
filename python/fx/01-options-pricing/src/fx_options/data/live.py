"""Optional live FX reference rates from the ECB via the free Frankfurter API.

Import-guarded and NEVER used by the test suite (CONVENTIONS.md: no
network access in test code paths).  Intended only for ad-hoc exploration,
e.g. seeding examples with a current EURUSD spot.

The ECB publishes daily reference rates around 16:00 CET; they are
indicative mid rates, not tradable quotes, and are quoted EUR-based
(units of currency per 1 EUR).
"""

from __future__ import annotations

import json
import urllib.request

__all__ = ["fetch_ecb_rates", "LiveDataUnavailable"]

_API = "https://api.frankfurter.app/latest"


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
        On any network or parsing failure — callers should degrade to
        synthetic data (`fx_options.data.synthetic`).
    """
    url = f"{_API}?base={base}&symbols={','.join(symbols)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        rates = {str(k): float(v) for k, v in payload["rates"].items()}
        rates["_date"] = payload.get("date", "unknown")
        return rates
    except Exception as exc:  # noqa: BLE001 - degrade gracefully by design
        raise LiveDataUnavailable(
            f"could not fetch ECB rates from {url}: {exc}"
        ) from exc
