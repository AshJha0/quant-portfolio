# Rust: answer key

Questions are in [`../questions.md`](../questions.md). An `E` code means the
program does not compile with that error; anything else is the exact output.

- **R01** `E0382`. `let t = s` moves the `String`; `s` is no longer valid. Clone it, or borrow with `&s`.
- **R02** `5 5`. `i32` is `Copy`, so assignment copies.
- **R03** `E0499`. At most one mutable borrow may be live at a time, and `a` is still used after `b` is created.
- **R04** `E0502`. `push` may reallocate and leave `first` dangling, so the compiler refuses a mutable borrow while a shared one is live.
- **R05** `0 None 255`. The three explicit overflow policies.
- **R06** `-3 -1 1`. `/` and `%` truncate toward zero (as in C++ and Java); `rem_euclid` gives the non-negative remainder that Python's `%` gives here.
- **R07** `11 | 10`. Each `let` makes a new binding; the inner one ends with its block.
- **R08** `Some(42) None`. `.ok()` turns `Result` into `Option`, dropping the error.
- **R09** `E0106`. A returned reference needs a lifetime, and no input lifetime exists to tie it to; return the owned `String` instead.
- **R10** `b | a`. Locals drop in reverse order of declaration (Rust Reference, "Destructors").
- **R11** `E0277`. `Rc` uses a non-atomic count and is not `Send`; use `Arc`.
- **R12** `before | m1 | m2 | m3 | 12`. `map` does nothing until `sum` pulls items.
- **R13** `E0277`. `f64` is not `Ord` because NaN is unordered. Use `v.sort_by(f64::total_cmp)`.
- **R14** `E0004`. `match` must be exhaustive; `Side::Sell` is not covered.

## Discussion notes

- **RD1** `Send`: ownership can move to another thread. `Sync`: `&T` can be shared between threads. `Arc` makes shared ownership thread-safe and `Mutex` gives exclusive access, so `Arc<Mutex<T>>` is `Send + Sync` when `T: Send` (std docs for `Mutex`).
- **RD2** For bounds-check removal in a measured hot loop, SIMD intrinsics or FFI. Each `unsafe` block needs a `// SAFETY:` comment stating the invariant it relies on, and a test (ideally under Miri) that exercises it.
- **RD3** It wraps to 0 silently, because release builds turn overflow checks off by default (debug builds panic). Use `wrapping_*`, `checked_*`, `saturating_*` or `overflowing_*`, or set `overflow-checks = true` in the release profile. (A literal `255u8 + 1` is caught at compile time by the `arithmetic_overflow` lint; the run-time case is a value that arrives at run time.)
