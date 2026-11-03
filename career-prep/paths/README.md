# Study paths

Three plans by length (4, 8 and 12 weeks) and one path per role. Each plan
assumes about ten hours a week. The week-by-week plans are ordered so that
the material later weeks lean on comes first: probability before
stochastic calculus, stochastic calculus before derivatives, data
structures before system design. Every pack named here has its answers in a
separate key and its numbers checked in CI, so you can trust a mismatch
between your answer and the key to be your mistake, or a bug worth
reporting.

How to use a week: work the pack's questions cold, mark against the key,
and redo the misses two days later. End each plan with the mock rounds that
match your role, sat under time.

## Four weeks (a refresher)

| Week | Work |
| --- | --- |
| 1 | [probability](../probability/) (first 30), [games](../games/) mental-maths drill daily |
| 2 | [stats](../stats/), [algorithms](../algorithms/) (first 15) |
| 3 | The pack for your role from the table below, plus [case](../case/) |
| 4 | Two [mock rounds](../mock-rounds/) for your role, a debrief after each, then redo the misses |

## Eight weeks

| Week | Work |
| --- | --- |
| 1 | [probability](../probability/) |
| 2 | [stats](../stats/), [games](../games/) |
| 3 | [numerics](../numerics/), [python](../python/) |
| 4 | [algorithms](../algorithms/), [sql](../sql/) |
| 5 | Role pack 1 (from the table below) |
| 6 | Role pack 2, [system-design](../system-design/) |
| 7 | [case](../case/), [tools](../tools/) |
| 8 | Three [mock rounds](../mock-rounds/), debriefs, misses |

## Twelve weeks (a full preparation)

| Week | Work |
| --- | --- |
| 1-2 | [probability](../probability/), [games](../games/) |
| 3 | [stats](../stats/) |
| 4 | [numerics](../numerics/) |
| 5 | [stochastic](../stochastic/) |
| 6 | [derivatives](../derivatives/), [fixed-income](../fixed-income/) |
| 7 | [risk](../risk/), [portfolio](../portfolio/) |
| 8 | [ml](../ml/), [alpha](../alpha/) |
| 9 | [algorithms](../algorithms/), [python](../python/), [sql](../sql/) |
| 10 | [cpp](../cpp/) or [rust](../rust/) or [java](../java/), [tools](../tools/) |
| 11 | [system-design](../system-design/), [case](../case/) |
| 12 | All six [mock rounds](../mock-rounds/), debriefs, misses |

## By role

The roles and the workstreams that build their code are those of section 7
of the v3.0.0 plan. Read the code for your role after the packs; the
system-design cases name the files.

| Role | Packs, in order | Mock round |
| --- | --- | --- |
| Quant analyst / researcher | [probability](../probability/), [stats](../stats/), [ml](../ml/), [alpha](../alpha/), [python](../python/), [sql](../sql/) | [1](../mock-rounds/round-1.md) |
| Quant developer | [cpp](../cpp/), [python](../python/), [algorithms](../algorithms/), [numerics](../numerics/), [system-design](../system-design/) | [4](../mock-rounds/round-4.md) |
| Algo developer | [algorithms](../algorithms/), [system-design](../system-design/) (X01, X04), [games](../games/) | [4](../mock-rounds/round-4.md), [6](../mock-rounds/round-6.md) |
| Strats | [derivatives](../derivatives/), [fixed-income](../fixed-income/), [python](../python/), [case](../case/) | [2](../mock-rounds/round-2.md) |
| Low-latency developer | [cpp](../cpp/), [tools](../tools/), [system-design](../system-design/) (X01, X08) | [5](../mock-rounds/round-5.md) |
| Pricing quant | [stochastic](../stochastic/), [derivatives](../derivatives/), [numerics](../numerics/) | [2](../mock-rounds/round-2.md) |
| Risk quant | [risk](../risk/), [portfolio](../portfolio/), [stats](../stats/) | [3](../mock-rounds/round-3.md) |
| Hedging | [derivatives](../derivatives/), [risk](../risk/), [case](../case/) (C01) | [2](../mock-rounds/round-2.md), [3](../mock-rounds/round-3.md) |
| Execution | [case](../case/) (C04, C05), [games](../games/), [system-design](../system-design/) (X04, X07) | [6](../mock-rounds/round-6.md) |
| Market data | [cpp](../cpp/), [system-design](../system-design/) (X01, X08) | [5](../mock-rounds/round-5.md) |
| Order management (FIX/OMS) | [system-design](../system-design/) (X05, X06), [java](../java/), [sql](../sql/) | [4](../mock-rounds/round-4.md) |
| Post-trade and surveillance | [system-design](../system-design/) (X07), [sql](../sql/), [derivatives](../derivatives/), [fixed-income](../fixed-income/) | [6](../mock-rounds/round-6.md) |
| Central risk book and market making | [system-design](../system-design/) (X02, X03), [risk](../risk/), [games](../games/) | [6](../mock-rounds/round-6.md) |

There is no separate microstructure pack yet: the execution and market
making material is spread over the market-making game in
[games](../games/), cases C04 and C05, and design cases X01 to X04.
