// Reference solution for part B of the Java pack.
// Run with `java Solutions.java` (JDK 11 or later); main checks itself and
// exits non-zero on a failure. Called by career-prep/tests/test_compiled_packs.py.

import java.util.ArrayDeque;
import java.util.concurrent.atomic.AtomicLong;

public class Solutions {

    /** JB1: SPSC ring of longs. head is written only by the consumer, tail only by the producer. */
    static final class LongRing {
        static final long EMPTY = Long.MIN_VALUE;
        private final long[] buf;
        private final int mask;
        private final AtomicLong head = new AtomicLong(); // next index to read
        private final AtomicLong tail = new AtomicLong(); // next index to write

        LongRing(int capacity) {
            if (Integer.bitCount(capacity) != 1) {
                throw new IllegalArgumentException("capacity must be a power of two");
            }
            buf = new long[capacity];
            mask = capacity - 1;
        }

        boolean offer(long v) {
            long t = tail.get();
            if (t - head.get() == buf.length) {
                return false;
            }
            buf[(int) (t & mask)] = v;
            tail.lazySet(t + 1); // release: the slot write is visible before the new tail
            return true;
        }

        long poll() {
            long h = head.get();
            if (h == tail.get()) {
                return EMPTY;
            }
            long v = buf[(int) (h & mask)];
            head.lazySet(h + 1);
            return v;
        }
    }

    static void check(boolean ok, String what) {
        if (!ok) {
            throw new AssertionError(what);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        // Single-threaded: compare with ArrayDeque under a pseudo-random op sequence.
        LongRing r = new LongRing(4);
        ArrayDeque<Long> ref = new ArrayDeque<>();
        long seed = 7;
        for (long i = 0; i < 10_000; i++) {
            seed = seed * 6364136223846793005L + 1442695040888963407L;
            if (seed >>> 63 == 0) {
                boolean ok = r.offer(i);
                check(ok == (ref.size() < 4), "offer at " + i);
                if (ok) {
                    ref.addLast(i);
                }
            } else {
                Long want = ref.pollFirst();
                long got = r.poll();
                check(want == null ? got == LongRing.EMPTY : got == want, "poll at " + i);
            }
        }
        // Two threads: every value arrives once, in order.
        LongRing q = new LongRing(1024);
        final long n = 1_000_000;
        Thread producer = new Thread(() -> {
            for (long i = 0; i < n; i++) {
                while (!q.offer(i)) {
                    Thread.onSpinWait();
                }
            }
        });
        producer.start();
        for (long expect = 0; expect < n; ) {
            long v = q.poll();
            if (v == LongRing.EMPTY) {
                Thread.onSpinWait();
                continue;
            }
            check(v == expect, "order: got " + v + " expected " + expect);
            expect++;
        }
        producer.join();
        System.out.println("ok");
    }
}
