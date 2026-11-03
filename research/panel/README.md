# research/panel: the FX carry, value and momentum panel (W23 part 1)

The question: across the 38-currency point-in-time universe
(`research/universe/fx.csv`), do currencies with higher carry, cheaper real
value or stronger past excess returns earn more over the next month? All
three are known premia, so a positive answer reproduces the literature; it
is not a discovery.

| File | What it does |
|---|---|
| `signals.py` | Month-end excess returns (CIP, calendar-day accrual), carry `i* - i`, value (minus the 5-year change in the real exchange rate), momentum (12-month excess return, skip 1). |
| `study.py` | `run_panel`: LRV portfolio sorts, HML with a Newey-West t-statistic, Fama-MacBeth slopes, median live cross-section and the power-gate verdict. |
| `holdout_lock.py` | The 2016-01-01 to 2026-12-31 hold-out, enforced in code. |
| `equity_factors.py` | Factor momentum on the Kenneth French library factors, behind the same lock. |
| `tests/` | Hand examples from each docstring, the lock, and the covariance oracles. |

## The hold-out lock

Every function here passes its inputs through `HoldoutLock.guard`. While the
lock is closed, any row dated on or after 2016-01-01 raises `HoldoutLocked`,
so the hold-out cannot be seen by accident. `HoldoutLock.training(frame)`
returns the rows before 2016.

The lock opens only for a new registration: a dated file
`research/prereg/YYYY-MM-DD-<name>.md`, signed with the hash of its current
text, that names the window `2016-01-01` to `2026-12-31`. Each opening is
written to `holdout_ledger.jsonl`, and a second opening with the same
registration is refused. Signed registrations are never edited; editing one
changes its hash and voids the signature (`research/tools/prereg.py`).

## Example

```python
from holdout_lock import HoldoutLock
import signals as sg, study

rx = sg.monthly_excess_returns(spot_m, rates_known_m, "USD")
sigs = {"carry": sg.carry(rates_known_m, "USD"),
        "value": sg.value(spot_m, cpi_known_m, "USD"),
        "momentum": sg.momentum(rx)}
lock = HoldoutLock()                      # closed: 1999-2015 only
res = study.run_panel(HoldoutLock.training(rx),
                      {k: HoldoutLock.training(v) for k, v in sigs.items()},
                      live_mask, lock=lock, ic_monthly=0.06, n_eff=15)
res["label"]                               # "exploratory"
```

Rates and price indices must be the values published by each month-end
(`research/lib/excess_returns.as_known`, monthly series from month m+2).

## Sources

Lustig, Roussanov and Verdelhan (2011), RFS 24(11); Asness, Moskowitz and
Pedersen (2013), JF 68(3); Menkhoff, Sarno, Schmeling and Schrimpf (2012),
JF 67(2) and JFE 106(3); Fama and MacBeth (1973); Newey and West (1987);
Ehsani and Linnainmaa (2022), JF 77(3). Covariance: Ledoit and Wolf (2004,
2020), in `research/lib/covariance.py`.
