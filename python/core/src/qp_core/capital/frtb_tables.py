"""FRTB SBM parameter tables for CSR non-securitisation, equity and commodity
(BCBS d457, January 2019, MAR21.51-21.85, Tables 3, 4, 5, 9, 10, 11, 12, 13).

Plain idea
----------
Every number the standard rule needs - how far each bucket can move and how
closely two buckets move together - is a published constant. Keeping them in
one module, transcribed once and pinned by a checksum in the tests, means a
typo cannot slip in silently: any edit changes the digest and fails a test
that names the paragraph to re-read.

Formal statement
----------------
* ``CSR_RW[b]``: Table 4 (MAR21.53), buckets 1-18, same for all tenors.
  Footnote 17: AA- or better covered bonds may use 1.5 % (``CSR_RW_COVERED_AA``).
* ``CSR_SECTOR_GAMMA``: Table 5 (MAR21.57), keyed by sector group
  ("1".."7" for buckets 1-7 / 9-15, then "8", "16", "17", "18").
* ``EQ_RW_SPOT[b]``, ``EQ_RW_REPO[b]``: Table 10 (MAR21.77), buckets 1-13.
* ``EQ_RHO[b]``: MAR21.78(2): 15 % (1-4), 25 % (5-8), 7.5 % (9), 12.5 % (10),
  80 % (12, 13); bucket 11 has no correlation (MAR21.79).
* ``COM_RW[b]``, ``COM_RHO[b]``: Tables 11 and 12 (MAR21.82-83), buckets 1-11.
* ``CSR_TENORS``: MAR21.9 (0.5, 1, 3, 5, 10 years).

Worked example
--------------
Footnote 18: a 5y Apple bond-curve sensitivity against a 10y Google CDS
sensitivity, same bucket: 35 % x 65 % x 99.9 % = 22.73 %.

>>> round(100 * CSR_RHO_NAME * CSR_RHO_TENOR * CSR_RHO_BASIS, 2)
22.73

Sources: BCBS d457 (2019) pp 40-52, read from the bis.org PDF text. Evidence
class: regulatory text.
"""

from __future__ import annotations

__all__ = [
    "COM_GAMMA",
    "COM_RHO",
    "COM_RHO_BASIS",
    "COM_RHO_TENOR",
    "COM_RW",
    "CSR_GAMMA_RATING",
    "CSR_RHO_BASIS",
    "CSR_RHO_NAME",
    "CSR_RHO_NAME_INDEX",
    "CSR_RHO_TENOR",
    "CSR_RW",
    "CSR_RW_COVERED_AA",
    "CSR_SECTOR_GAMMA",
    "CSR_TENORS",
    "EQ_GAMMA_INDEX",
    "EQ_GAMMA_MIXED",
    "EQ_GAMMA_SINGLE",
    "EQ_RHO",
    "EQ_RHO_SPOT_REPO",
    "EQ_RW_REPO",
    "EQ_RW_SPOT",
    "csr_sector",
]

CSR_TENORS: tuple[float, ...] = (0.5, 1.0, 3.0, 5.0, 10.0)

# Table 4.
CSR_RW: dict[int, float] = {
    1: 0.005,
    2: 0.010,
    3: 0.050,
    4: 0.030,
    5: 0.030,
    6: 0.020,
    7: 0.015,
    8: 0.025,
    9: 0.020,
    10: 0.040,
    11: 0.120,
    12: 0.070,
    13: 0.085,
    14: 0.055,
    15: 0.050,
    16: 0.120,
    17: 0.015,
    18: 0.050,
}
CSR_RW_COVERED_AA = 0.015  # footnote 17

# MAR21.54-55.
CSR_RHO_NAME = 0.35
CSR_RHO_NAME_INDEX = 0.80
CSR_RHO_TENOR = 0.65
CSR_RHO_BASIS = 0.999
# MAR21.57(1).
CSR_GAMMA_RATING = 0.5

# Table 5, upper triangle, by sector group.
_SECTORS = ("1", "2", "3", "4", "5", "6", "7", "8", "16", "17", "18")
_TABLE5_ROWS = (
    (0.75, 0.10, 0.20, 0.25, 0.20, 0.15, 0.10, 0.00, 0.45, 0.45),
    (0.05, 0.15, 0.20, 0.15, 0.10, 0.10, 0.00, 0.45, 0.45),
    (0.05, 0.15, 0.20, 0.05, 0.20, 0.00, 0.45, 0.45),
    (0.20, 0.25, 0.05, 0.05, 0.00, 0.45, 0.45),
    (0.25, 0.05, 0.15, 0.00, 0.45, 0.45),
    (0.05, 0.20, 0.00, 0.45, 0.45),
    (0.05, 0.00, 0.45, 0.45),
    (0.00, 0.45, 0.45),
    (0.00, 0.00),
    (0.75,),
)
CSR_SECTOR_GAMMA: dict[tuple[str, str], float] = {}
for _i, _row in enumerate(_TABLE5_ROWS):
    for _j, _v in enumerate(_row):
        _a, _b = _SECTORS[_i], _SECTORS[_i + 1 + _j]
        CSR_SECTOR_GAMMA[(_a, _b)] = _v
        CSR_SECTOR_GAMMA[(_b, _a)] = _v


def csr_sector(bucket: int) -> str:
    """Sector group of a CSR non-sec bucket for Table 5 (buckets 9-15 share
    the sector of 1-7).

    >>> csr_sector(3), csr_sector(11), csr_sector(16)
    ('3', '3', '16')
    """
    if bucket in range(1, 8):
        return str(bucket)
    if bucket in range(9, 16):
        return str(bucket - 8)
    if bucket in (8, 16, 17, 18):
        return str(bucket)
    raise ValueError(f"CSR bucket {bucket} is not 1-18")


# Table 10.
EQ_RW_SPOT: dict[int, float] = {
    1: 0.55,
    2: 0.60,
    3: 0.45,
    4: 0.55,
    5: 0.30,
    6: 0.35,
    7: 0.40,
    8: 0.50,
    9: 0.70,
    10: 0.50,
    11: 0.70,
    12: 0.15,
    13: 0.25,
}
EQ_RW_REPO: dict[int, float] = {b: round(v / 100.0, 6) for b, v in EQ_RW_SPOT.items()}
# MAR21.78(2).
EQ_RHO: dict[int, float] = {
    1: 0.15,
    2: 0.15,
    3: 0.15,
    4: 0.15,
    5: 0.25,
    6: 0.25,
    7: 0.25,
    8: 0.25,
    9: 0.075,
    10: 0.125,
    12: 0.80,
    13: 0.80,
}
EQ_RHO_SPOT_REPO = 0.999  # MAR21.78(1) and (4)
# MAR21.80.
EQ_GAMMA_SINGLE = 0.15
EQ_GAMMA_INDEX = 0.75
EQ_GAMMA_MIXED = 0.45

# Tables 11 and 12.
COM_RW: dict[int, float] = {
    1: 0.30,
    2: 0.35,
    3: 0.60,
    4: 0.80,
    5: 0.40,
    6: 0.45,
    7: 0.20,
    8: 0.35,
    9: 0.25,
    10: 0.35,
    11: 0.50,
}
COM_RHO: dict[int, float] = {
    1: 0.55,
    2: 0.95,
    3: 0.40,
    4: 0.80,
    5: 0.60,
    6: 0.65,
    7: 0.55,
    8: 0.45,
    9: 0.15,
    10: 0.40,
    11: 0.15,
}
COM_RHO_TENOR = 0.99
COM_RHO_BASIS = 0.999
COM_GAMMA = 0.20  # MAR21.85; 0 with bucket 11
