# SQL and time series: reference queries

Questions are in [`../questions.md`](../questions.md). Each query below is run
by `career-prep/tests/test_sql_pack.py` against [`../fixtures.sql`](../fixtures.sql)
and compared with a pandas computation.

### S01
```sql
SELECT sym, SUM(qty) AS volume, ROUND(SUM(px * qty) / SUM(qty), 4) AS vwap
FROM trades GROUP BY sym ORDER BY sym;
```
VWAP is notional over volume, not the mean price.

### S02
```sql
SELECT sym, SUM(CASE side WHEN 'B' THEN qty ELSE -qty END) AS net
FROM trades GROUP BY sym ORDER BY sym;
```

### S03
```sql
SELECT id, sym,
       SUM(CASE side WHEN 'B' THEN qty ELSE -qty END)
         OVER (PARTITION BY sym ORDER BY t, id ROWS UNBOUNDED PRECEDING) AS pos
FROM trades ORDER BY id;
```
Order the window by time with the id as tie-break, so equal timestamps still give one answer.

### S04
```sql
SELECT tr.id,
       (SELECT q.bid FROM quotes q WHERE q.sym = tr.sym AND q.t <= tr.t ORDER BY q.t DESC LIMIT 1) AS bid,
       (SELECT q.ask FROM quotes q WHERE q.sym = tr.sym AND q.t <= tr.t ORDER BY q.t DESC LIMIT 1) AS ask
FROM trades tr ORDER BY tr.id;
```
The CCC trade at t = 50 comes before CCC's first quote at 55, so it gets NULLs: the join must not reach forward in time.

### S05
```sql
WITH q AS (
  SELECT tr.id, tr.side, tr.px,
         (SELECT (bid + ask) / 2 FROM quotes q WHERE q.sym = tr.sym AND q.t <= tr.t
          ORDER BY q.t DESC LIMIT 1) AS mid
  FROM trades tr)
SELECT id, ROUND(CASE side WHEN 'B' THEN (px - mid) / mid ELSE (mid - px) / mid END * 10000, 2) AS slip_bp
FROM q WHERE mid IS NOT NULL ORDER BY id;
```

### S06
```sql
SELECT sym, id FROM (
  SELECT sym, id, ROW_NUMBER() OVER (PARTITION BY sym ORDER BY qty DESC, t) AS rn FROM trades)
WHERE rn = 1 ORDER BY sym;
```
`ROW_NUMBER` gives exactly one row per group; `RANK` would return ties twice.

### S07
```sql
SELECT sym, d, ROUND(close / prev - 1, 6) AS ret FROM (
  SELECT sym, d, close, LAG(close) OVER (PARTITION BY sym ORDER BY d) AS prev FROM closes)
WHERE prev IS NOT NULL ORDER BY sym, d;
```
BBB's 2026-01-09 return spans two days (from 01-07); see SD3.

### S08
```sql
SELECT s.sym, d.d
FROM (SELECT DISTINCT sym FROM closes) s CROSS JOIN (SELECT DISTINCT d FROM closes) d
LEFT JOIN closes c ON c.sym = s.sym AND c.d = d.d
WHERE c.close IS NULL ORDER BY s.sym, d.d;
```
Build the full grid, then anti-join.

### S09
```sql
SELECT id, ROUND(AVG(px) OVER (PARTITION BY sym ORDER BY t, id ROWS 2 PRECEDING), 4) AS ma3
FROM trades ORDER BY id;
```

### S10
```sql
SELECT sym, t / 30 AS bucket, COUNT(*) AS n FROM quotes GROUP BY sym, bucket ORDER BY sym, bucket;
```
Integer division in SQLite truncates; with non-negative times that is the floor.

## Discussion notes

- **SD1** It runs one indexed lookup per trade at best, and a scan per trade at worst. Column stores sort both tables by symbol and time and merge them in one pass (kdb+ `aj`, DuckDB `ASOF JOIN`, pandas `merge_asof`).
- **SD2** `ROWS` counts physical rows; `RANGE` includes every row whose ordering value ties with the current one. With duplicate timestamps a running sum under `RANGE` jumps by the whole group at once.
- **SD3** `LAG` takes the previous row, not the previous calendar day, so a gap produces a multi-day return labelled as one day (S07, BBB). Reindex to the trading calendar first, or record the gap length.
