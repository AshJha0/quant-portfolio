# Java basics: knowledge check (questions only)

The v3.0.0 plan adds a Java order gateway and OMS core measured in
microseconds (decision D2). This pack covers the language basics that work
assumes. Part A: twelve short programs; say what each prints, or `compile
error` and why. Part B: a garbage-free ring buffer in
[`Solutions.java`](Solutions.java). Answers for part A are in
[`keys/answers.md`](keys/answers.md).

How it is checked: `career-prep/tests/test_compiled_packs.py` runs each
program with the JDK source launcher (`java Main.java`, JDK 11 or later),
checks compile errors and exact output, and runs `java Solutions.java`, whose
`main` asserts its own checks. The test skips if `java` is not installed.
Outputs are written with " | " between lines.

References: the Java Language Specification (JLS, Java SE 21), sections
5.1.7 (boxing), 15.17.2-15.17.3 (division and remainder), 15.21.3
(reference equality), 14.20.2 (try-finally), 15.12.2 (overload
resolution) and 15.27.2 (lambda capture); Goetz et al., *Java Concurrency in
Practice*; the LMAX Disruptor technical paper (Thompson et al. 2011).

## Part A

### J01 Boxed equality
```java
class Main { public static void main(String[] a) {
    Integer x = 127, y = 127, p = 1000, q = 1000;
    System.out.println((x == y) + " " + (p == q) + " " + p.equals(q));
}}
```

### J02 String identity
```java
class Main { public static void main(String[] a) {
    String s1 = "ab";
    String s2 = new String("ab");
    System.out.println((s1 == s2) + " " + s1.equals(s2) + " " + (s1 == s2.intern()));
}}
```

### J03 Integer overflow
```java
class Main { public static void main(String[] a) {
    System.out.println(Integer.MAX_VALUE + 1);
    System.out.println(Math.abs(Integer.MIN_VALUE));
}}
```

### J04 Division and remainder
```java
class Main { public static void main(String[] a) {
    System.out.println(-7 / 2 + " " + -7 % 2 + " " + Math.floorMod(-7, 2));
}}
```

### J05 return in finally
```java
class Main {
    static int f() { try { return 1; } finally { return 2; } }
    public static void main(String[] a) { System.out.println(f()); }
}
```

### J06 Unboxing null
```java
class Main { public static void main(String[] a) {
    Integer x = null;
    try { int y = x; System.out.println(y); }
    catch (NullPointerException e) { System.out.println("NPE"); }
}}
```

### J07 Mutable map keys
```java
import java.util.*;
class Main { public static void main(String[] a) {
    List<Integer> k = new ArrayList<>(List.of(1));
    Map<List<Integer>, String> m = new HashMap<>();
    m.put(k, "v");
    k.add(2);
    System.out.println(m.get(k) + " " + m.size());
}}
```

### J08 char arithmetic
```java
class Main { public static void main(String[] a) {
    char c = 'a';
    System.out.println((c + 1) + " " + (char) (c + 1));
}}
```

### J09 Floating point
```java
class Main { public static void main(String[] a) {
    System.out.println(0.1 + 0.2);
}}
```

### J10 Overloading is static
```java
class Main {
    static String f(Object o) { return "obj"; }
    static String f(String s) { return "str"; }
    public static void main(String[] a) {
        Object o = "x";
        System.out.println(f(o) + " " + f("x"));
    }
}
```

### J11 Checked exceptions
```java
class Main {
    static void g() { throw new Exception("x"); }
    public static void main(String[] a) { g(); }
}
```

### J12 Lambda capture
```java
class Main { public static void main(String[] a) {
    int n = 0;
    Runnable r = () -> n++;
    r.run();
}}
```

## Part B (solution in `Solutions.java`)

- **JB1** A single-producer, single-consumer ring buffer of `long` with a power-of-two capacity, preallocated, that never allocates after construction: `offer(long)` returns `false` when full, `poll()` returns a sentinel when empty.

## Discussion (not graded)

- **JD1** What allocates in a "garbage-free" Java hot path that you might not expect?
- **JD2** What does `volatile` guarantee, and what does it not?
- **JD3** Why does the Disruptor pad its sequence counters?
