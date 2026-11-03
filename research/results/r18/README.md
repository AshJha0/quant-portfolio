# research/results/r18: Round 18 real-data runs (exploratory and verification)

Nothing here is a confirmatory result. No registration was signed, and the
2016-2026 hold-out lock stayed closed.

| File | What it is |
|---|---|
| `fetch_store.py` | Fills `data/store` (git-ignored) from keyless sources. Only the SHA-256 of each download is committed, in `data_manifest.json`, along with the series that could not be fetched. |
| `fx_panel_exploratory.py` / `.json` | **Exploratory.** The W23 carry, value and momentum panel on ECB spot rates and OECD 3-month interbank rates and CPI, 2000-01 to 2015-12. |
| `sofr_2024_check.py` / `sofr_2024_check_w13a.json` | **Verification.** Compounds the NY Fed's 2024 SOFR fixings and checks the result against the NY Fed SOFR Index, both directly and through the W13a `fi_rates.legs.compounded_rate`. |

## FX panel, 2000-2015 (exploratory)

The plain idea: currencies with higher interest rates, cheaper real value or
stronger past excess returns are expected to earn more over the next month.
The script tests that cross-sectionally, month by month (Fama and MacBeth
1973, Newey-West lag 3), and with high-minus-low portfolio sorts (Lustig,
Roussanov and Verdelhan 2011).

| signal | months | HML Sharpe (annual) | HML NW t | FM t | median live | implied monthly IC |
|---|---|---|---|---|---|---|
| carry | 189 | 0.64 | 2.34 | 2.49 | 21 | 0.090 |
| value | 129 | 0.24 | 0.77 | 0.71 | 18 | 0.022 |
| momentum | 177 | 0.29 | 1.10 | 0.39 | 21 | 0.013 |
| combined (draft score) | 177 | 0.53 | 2.00 | 2.44 | 21 | 0.067 |

How to read the table:

- Carry is the known premium, and these numbers replicate it. They are in
  the range Menkhoff et al. (2012, JF) report for the pre-2010 years.
- Value and momentum are not distinguishable from zero in this sample.
- **The median live cross-section is 21 currencies, below the W6 gate's
  minimum of 30.** The OECD has no 3-month interbank series on FRED for
  HKD, SGD, MYR, PHP, THB, BRL, INR, RON, BGN, HRK, CYP, MTL, LVL or LTL.
  So, on free data, the combination panel cannot reach the gate's panel
  size as it is designed today.
- The implied IC is the mean FM slope divided by the mean cross-sectional
  standard deviation of next-month returns. It is in-sample and comes from
  before 2016. Post-publication decay is expected (McLean and Pontiff 2016),
  so treat it as an upper bound.

Caveats, all of them also stored in the JSON:

- The rates and CPI are today's vintage, so revisions have not been removed.
- Carry uses interbank rates rather than forward points, which ignores
  deviations from covered interest parity.
- EUR CPI is Germany's, used as a proxy.
- The sample starts in 2000 because `qp_core.holidays` covers only
  2000-2099. The store computes a release date for every observation, so it
  refuses anything dated earlier.

## SOFR 2024 (verification)

In every month of 2024, and over the 1-year OIS period from 2024-01-02 to
2025-01-02, the SOFR compounded from the daily fixings matches the
index-implied rate within 0.0009 bp. That gap is the rounding of the
8-decimal index. The W13a `compounded_rate` gives the same numbers to
floating-point precision.

The realised 1-year SOFR OIS par rate for 2024 is 5.2759 %.
