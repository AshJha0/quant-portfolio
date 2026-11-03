"""Credit indices (CDX, iTraxx) on the ISDA CDS Standard Model (plan W18).

``isda`` holds the single-name standard model with a portable numeric
kernel (Rust and C++ ports for parity); ``index`` adds series and rolls,
factor and defaults, quote conversion, intrinsic value and the index basis.
"""

from .index import (
    CDX_HY,
    CDX_IG,
    ITRAXX_MAIN,
    CreditIndex,
    IndexFamily,
    adjust_to_index,
    index_basis,
    index_maturity,
    intrinsic,
    roll_date,
    series_on,
)
from .isda import (
    CdsContract,
    FlatForwardCurve,
    HazardCurve,
    bootstrap_hazard,
    cds_legs,
    isda_legs_kernel,
    par_spread,
    previous_imm_date,
    quoted_spread_from_upfront,
    upfront_from_quoted_spread,
)

__all__ = [
    "CDX_HY",
    "CDX_IG",
    "ITRAXX_MAIN",
    "CdsContract",
    "CreditIndex",
    "FlatForwardCurve",
    "HazardCurve",
    "IndexFamily",
    "adjust_to_index",
    "bootstrap_hazard",
    "cds_legs",
    "index_basis",
    "index_maturity",
    "intrinsic",
    "isda_legs_kernel",
    "par_spread",
    "previous_imm_date",
    "quoted_spread_from_upfront",
    "roll_date",
    "series_on",
    "upfront_from_quoted_spread",
]
