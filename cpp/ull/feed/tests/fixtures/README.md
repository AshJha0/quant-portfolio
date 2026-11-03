# Feed handler fixtures (synthetic)

These files are **not** captures of a live exchange. `tools/make_fixtures.py`
generates them from fixed seeds. They use the message shapes of the public
Coinbase Exchange `full` channel and its REST level-3 book, and of the
Binance spot diff-depth stream and `/api/v3/depth`. They are stored in the
QPRC recording format (`include/qp/ull/recorder.hpp`).

| File | Content |
|---|---|
| `coinbase_full_11.qprc` / `.expected` | 1,500 steps after subscribing. The REST snapshot arrives 25 frames after its own sequence. |
| `coinbase_full_12.qprc` / `.expected` | 800 steps. The snapshot arrives 60 frames late. |
| `binance_depth_21.qprc` / `.expected` | 1,200 events. `lastUpdateId` falls inside one event's `[U, u]` range. Every other frame uses the combined-stream wrapper. |
| `binance_depth_22.qprc` / `.expected` | 600 events, with the same features as `binance_depth_21`. |

Each `.expected` file holds the final book as computed by the generator's
own Python reference book, which shares no code with the C++ handler. It
uses the same text form as `L2Book::dump()`. Its first lines give
`last_seq`, plus `orders` for Coinbase.

To regenerate, run this from the repo root:
`python cpp/ull/feed/tools/make_fixtures.py`.
