# qp-posttrade

Post-trade, best execution and surveillance for quant-portfolio (plan item
W29). Everything here runs on simulated fills and a simulated audit log, and
every report carries the stamp `synthetic venue`.

## What is in it

| Module | Plain idea | Source |
|---|---|---|
| `settlement` | When is a trade paid for? US equities T+1, FX spot T+2, 2026 holiday tables | SEC Rule 15c6-1 (T+1 from 2024-05-28); ISDA 1998 FX Definitions; NYSE, Federal Reserve, ECB TARGET2, UK calendars |
| `booking` | Fills become booked trades; amend, cancel and novation append versions; confirmations are matched field by field | FINOS CDM lifecycle events; SWIFT MT300 fields |
| `tca` | Implementation shortfall and effective spread per fill, totals by venue | Perold (1988); SEC Rule 605 (2024) |
| `bestex` | RTS 28-layout top-venue table and Rule 605-style size buckets, reconciled to TCA totals | (EU) 2017/576 (deleted 2024), SEC Rule 605 |
| `surveillance` | Spoofing/layering, wash trades and marking the close on the attributable audit log | EU MAR Art. 12; CEA s.4c(a)(5)(C) (Dodd-Frank s.747); FINRA 5210 |
| `sim` | Seeded benign flow plus planted abusers | this package |

RTS 27 is suspended and RTS 28 was removed from EU law by the 2024 MiFIR
review and from UK rules by the FCA, so the best-ex outputs are layout
templates, not filings. The references are dated 2026-11; check them before
reuse.

## Worked example

A US equity trade on Friday 9 October 2026 settles on Tuesday 13 October:
Monday is Columbus Day, an NYSE trading day but a Federal Reserve holiday, so
DTC does not settle.

```python
from datetime import date
from qp_posttrade.settlement import us_equity_settle
us_equity_settle(date(2026, 10, 9))  # date(2026, 10, 13)
```

## Evidence

- 40-case settlement-date table (`tests/test_settlement.py`), each case worked
  by hand and labelled with the rule it exercises.
- Every planted case (spoofing, layering, wash, marking the close) is found in
  20 of 20 seeded sessions.
- False positives on 20 benign sessions (2 symbols, 3 two-sided market makers
  cancelling about 90 % of quotes, 5 directional traders each): zero for every
  detector, so below 0.15 per session at 95 % confidence (rule of three).
  With the spoofing size ratio at 3 instead of the default 5, the market
  makers produced 144.5 false alerts per session; the threshold is the
  design choice that matters.
- Best-ex totals equal TCA totals on 60 random fill sets (Hypothesis).

## Limits

- Holiday tables cover 2026 only; other years raise rather than guess. No
  CAD/JPY tables, so T+1 FX pairs are not supported.
- Benign flow is synthetic; real false-positive rates on a real desk will be
  higher. No crypto venue wash-volume estimate yet.
