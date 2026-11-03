# Java basics: answer key

Questions are in [`../questions.md`](../questions.md). `compile error` means
`javac` rejects the program; anything else is the exact output.

- **J01** `true false true`. JLS 5.1.7 requires boxing to return the same object for values in -128..127, so `x == y` holds. 1000 is outside that range; on a default HotSpot JVM each boxing creates a new object (the cache can be widened with `-XX:AutoBoxCacheMax`, so do not rely on `false` either). Compare boxed values with `equals`.
- **J02** `false true true`. `new String` makes a new object; `==` compares references (JLS 15.21.3); `intern` returns the pooled literal.
- **J03** `-2147483648 | -2147483648`. Integer arithmetic wraps silently (JLS 15.18.2), and the absolute value of the minimum has no `int` representation. Use `Math.addExact` and `Math.absExact` to get an exception instead.
- **J04** `-3 -1 1`. Division truncates toward zero and the remainder takes the dividend's sign (JLS 15.17.2-15.17.3); `floorMod` matches Python.
- **J05** `2`. A `return` in `finally` discards the pending return (JLS 14.20.2); compilers warn about it.
- **J06** `NPE`. Unboxing a `null` throws.
- **J07** `null 1`. The entry was stored under the hash of `[1]`; after the mutation the lookup uses the hash of `[1, 2]` and misses. Never mutate a key while it is in a hash map.
- **J08** `98 b`. `c + 1` promotes to `int`; the cast turns it back into a `char`.
- **J09** `0.30000000000000004`. Same IEEE 754 doubles as Python; `Double.toString` prints the shortest string that round-trips.
- **J10** `obj str`. Overloads are chosen at compile time from the static type (JLS 15.12.2); only overriding is dynamic.
- **J11** `compile error`. `Exception` is checked: `g` must declare `throws Exception` or catch it.
- **J12** `compile error`. A lambda may only capture effectively final locals (JLS 15.27.2). Use an `int[]`, an `AtomicInteger` or a field.

## Discussion notes

- **JD1** Autoboxing (`Map<Long, ...>`, varargs of primitives), iterators in for-each over collections, lambdas that capture state, string concatenation and logging formatters, and `Optional` returns. Check with a profiler's allocation view or JFR, and assert zero allocation in a steady-state test.
- **JD2** Visibility and ordering: a write to a `volatile` field happens-before every later read of it (JLS 17.4.5). It does not make compound actions like `count++` atomic; use `AtomicLong` or a lock.
- **JD3** To keep the producer's and consumer's counters on different cache lines, so updates by one core do not keep invalidating the line the other is reading (false sharing; Thompson et al. 2011).
