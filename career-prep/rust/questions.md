# Rust: knowledge check (questions only)

Part A: fourteen short programs. For each, say what it prints, or, if it does
not compile, the compiler error code (for example `E0000`) and why. Part B:
three small tasks with reference solutions and unit tests in
[`solutions.rs`](solutions.rs). Answers for part A are in
[`keys/answers.md`](keys/answers.md).

How it is checked: `career-prep/tests/test_compiled_packs.py` compiles every
program with `rustc --edition 2021`, checks that the ones keyed as errors
fail with that exact code and that the others print exactly the keyed
output, and runs `rustc --test solutions.rs`. The test skips if `rustc` is
not installed. Outputs are written with " | " between lines.

This repository's Rust lives in `rust/` (pricing and risk kernels with
Python parity tests, and the low-latency crates). References: *The Rust
Programming Language* (Klabnik and Nichols), chapters 4, 8, 13, 15 and 16;
the Rust Reference on destructors; the rustc error index.

## Part A

### R01 Moves
```rust
fn main() {
    let s = String::from("a");
    let t = s;
    println!("{} {}", s, t);
}
```

### R02 Copy types
```rust
fn main() {
    let a = 5;
    let b = a;
    println!("{} {}", a, b);
}
```

### R03 Two mutable borrows
```rust
fn main() {
    let mut v = vec![1];
    let a = &mut v;
    let b = &mut v;
    a.push(2);
    b.push(3);
}
```

### R04 Borrow while mutating
```rust
fn main() {
    let mut v = vec![1, 2];
    let first = &v[0];
    v.push(3);
    println!("{}", first);
}
```

### R05 Explicit overflow handling
```rust
fn main() {
    let x: u8 = 255;
    println!("{} {:?} {}", x.wrapping_add(1), x.checked_add(1), x.saturating_add(1));
}
```

### R06 Division and remainder
```rust
fn main() {
    println!("{} {} {}", -7 / 2, -7 % 2, (-7i32).rem_euclid(2));
}
```

### R07 Shadowing
```rust
fn main() {
    let x = 5;
    let x = x * 2;
    {
        let x = x + 1;
        println!("{}", x);
    }
    println!("{}", x);
}
```

### R08 Option
```rust
fn parse(s: &str) -> Option<i32> {
    s.parse().ok()
}
fn main() {
    println!("{:?} {:?}", parse("42"), parse("x"));
}
```

### R09 Returning a reference to a local
```rust
fn make() -> &String {
    let s = String::new();
    &s
}
fn main() {
    make();
}
```

### R10 Drop order
```rust
struct D(&'static str);
impl Drop for D {
    fn drop(&mut self) {
        println!("{}", self.0);
    }
}
fn main() {
    let _a = D("a");
    let _b = D("b");
}
```

### R11 Rc across threads
```rust
use std::rc::Rc;
fn main() {
    let n = Rc::new(1);
    let h = std::thread::spawn(move || println!("{}", n));
    h.join().unwrap();
}
```

### R12 Lazy iterators
```rust
fn main() {
    let v = vec![1, 2, 3];
    let it = v.iter().map(|x| {
        println!("m{}", x);
        x * 2
    });
    println!("before");
    let s: i32 = it.sum();
    println!("{}", s);
}
```

### R13 Sorting floats
```rust
fn main() {
    let mut v = vec![2.0_f64, 1.0];
    v.sort();
    println!("{:?}", v);
}
```

### R14 Exhaustive match
```rust
enum Side { Buy, Sell }
fn sign(s: Side) -> i32 {
    match s {
        Side::Buy => 1,
    }
}
fn main() {
    println!("{}", sign(Side::Sell));
}
```

## Part B: write the function (solutions in `solutions.rs`)

- **RB1** A fixed-capacity ring buffer of `u64` with `push` (returns `false` when full) and `pop`, no allocation after construction.
- **RB2** `vwap(&[(f64, f64)]) -> Option<f64>`, `None` on zero volume.
- **RB3** `max_drawdown(&[f64]) -> f64` in one pass.

## Discussion (not graded)

- **RD1** What do `Send` and `Sync` mean, and why is `Arc<Mutex<T>>` both when `T: Send`?
- **RD2** When would you reach for `unsafe` in a pricing kernel, and what must you document when you do?
- **RD3** In release builds, what does `255u8 + 1` do at run time, and how do you make overflow behaviour explicit?
