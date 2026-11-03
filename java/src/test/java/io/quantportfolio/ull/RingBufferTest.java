package io.quantportfolio.ull;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.quantportfolio.ull.ring.BatchEventProcessor;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.ring.Sequence;
import io.quantportfolio.ull.util.LongIntHashMap;
import io.quantportfolio.ull.util.Sorts;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

class RingBufferTest {
    static final class Slot {
        long value;
    }

    @Test
    void sizeMustBePowerOfTwo() {
        assertThrows(IllegalArgumentException.class, () -> new RingBuffer<>(Slot::new, 12));
    }

    @Test
    void tryNextRefusesToLapAConsumer() {
        RingBuffer<Slot> ring = new RingBuffer<>(Slot::new, 4);
        Sequence c = new Sequence(-1);
        ring.addGatingSequence(c);
        for (int i = 0; i < 4; i++) {
            ring.publish(ring.tryNext());
        }
        assertEquals(-1L, ring.tryNext(), "full ring must not overwrite unread slot 0");
        c.set(0);
        assertEquals(4L, ring.tryNext());
    }

    /** Two threads, 20 million events through a 1,024-slot ring: every value arrives once, in order. */
    @Test
    void crossThreadOrderAndCompleteness() throws InterruptedException {
        final long n = 20_000_000L;
        RingBuffer<Slot> ring = new RingBuffer<>(Slot::new, 1024);
        long[] state = new long[3]; // expected next, count, errors
        BatchEventProcessor<Slot> proc = new BatchEventProcessor<>(ring, (e, s, end) -> {
            if (e.value != state[0]) {
                state[2]++;
            }
            state[0] = e.value + 1;
            state[1]++;
        });
        Thread t = new Thread(proc, "consumer");
        t.start();
        for (long i = 0; i < n; i++) {
            long s = ring.next();
            ring.get(s).value = i;
            ring.publish(s);
        }
        while (proc.sequence().get() < n - 1) {
            Thread.onSpinWait();
        }
        proc.halt();
        t.join();
        assertEquals(n, state[1]);
        assertEquals(0, state[2]);
    }

    @Test
    void hashMapMatchesJdkMapUnderRandomOps() {
        Random r = new Random(7);
        LongIntHashMap m = new LongIntHashMap(512);
        Map<Long, Integer> ref = new HashMap<>();
        for (int i = 0; i < 200_000; i++) {
            long k = 1 + r.nextInt(700);
            if (r.nextBoolean() && (ref.size() < 512 || ref.containsKey(k))) {
                m.put(k, i);
                ref.put(k, i);
            } else {
                Integer v = ref.remove(k);
                assertEquals(v == null ? LongIntHashMap.MISSING : v, m.remove(k));
            }
            long q = 1 + r.nextInt(700);
            Integer v = ref.get(q);
            assertEquals(v == null ? LongIntHashMap.MISSING : v, m.get(q));
            assertEquals(ref.size(), m.size());
        }
    }

    @Test
    void heapSortSorts() {
        Random r = new Random(3);
        for (int t = 0; t < 500; t++) {
            int n = r.nextInt(200);
            long[] a = r.longs(n + 5, -1000, 1000).toArray();
            long[] b = Arrays.copyOf(a, n);
            Arrays.sort(b);
            Sorts.sort(a, n);
            assertEquals(Arrays.toString(b), Arrays.toString(Arrays.copyOf(a, n)));
        }
    }
}
