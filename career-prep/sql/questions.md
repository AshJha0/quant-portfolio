# SQL and time series: knowledge check (questions only)

Ten query tasks on a small trades/quotes/closes data set, and three
discussion prompts. The schema and data are in [`fixtures.sql`](fixtures.sql)
(SQLite dialect; window functions need SQLite 3.25 or later). Reference
queries are in [`keys/answers.md`](keys/answers.md).

How it is checked: `career-prep/tests/test_sql_pack.py` loads the fixtures
into an in-memory SQLite database, runs every reference query, and compares
the rows with the same answer computed in pandas without SQL. Order the
output as each task says, so the comparison is exact.

```
trades(id, sym, t, px, qty, side)   -- side 'B' buy, 'S' sell; t in seconds
quotes(sym, t, bid, ask)            -- a quote is valid from t until the next one
closes(sym, d, close)               -- daily closes; BBB has no row on 2026-01-08
```

References: the SQLite documentation on window functions; Molinaro and de
Graaf, *SQL Cookbook* (2nd ed.); the kdb+ `aj` documentation for as-of joins
(the idea, not the syntax).

## Tasks

- **S01** Per symbol: total traded quantity and VWAP rounded to 4 places. Columns `sym, volume, vwap`; order by `sym`.
- **S02** Per symbol: signed net position (buys positive). Columns `sym, net`; order by `sym`.
- **S03** Running net position per symbol after each trade. Columns `id, sym, pos`; order by `id`.
- **S04** For each trade, the prevailing quote (last quote at or before the trade time, same symbol). Columns `id, bid, ask`, with NULLs if there is none; order by `id`.
- **S05** For each trade, signed slippage against the prevailing mid in basis points, rounded to 2 places: buys pay `(px - mid)/mid`, sells pay `(mid - px)/mid`. Columns `id, slip_bp`; only trades with a quote; order by `id`.
- **S06** The largest trade (by quantity) per symbol; break ties by earliest `t`. Columns `sym, id`; order by `sym`.
- **S07** Daily simple returns from `closes`, using the previous available close per symbol, rounded to 6 places. Columns `sym, d, ret`; skip each symbol's first day; order by `sym, d`.
- **S08** Days where a symbol's close is missing while another symbol has one. Columns `sym, d`; order by `sym, d`.
- **S09** Three-trade moving average of `px` per symbol (fewer at the start), rounded to 4 places. Columns `id, ma3`; order by `id`.
- **S10** Per symbol, the number of quote updates in each 30-second bucket (`t / 30`). Columns `sym, bucket, n`; order by `sym, bucket`.

## Discussion (not graded)

- **SD1** Why is a correlated subquery for an as-of join slow on a big table, and what do column stores do instead?
- **SD2** `ROWS` versus `RANGE` in a window frame: when do they differ?
- **SD3** What goes wrong if you compute returns with `LAG` over a table that has missing days?
