package io.quantportfolio.ull;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.quantportfolio.ull.metrics.LatencyHistogram;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class LatencyHistogramTest {
    @Test
    void exactBelow128AndWithinOneSixtyFourthAbove() {
        LatencyHistogram h = new LatencyHistogram();
        for (int v = 0; v < 100; v++) {
            h.record(v);
        }
        assertEquals(49, h.percentile(0.5));
        assertEquals(99, h.percentile(1.0));

        Random r = new Random(11);
        long[] xs = new long[100_000];
        h.reset();
        for (int i = 0; i < xs.length; i++) {
            xs[i] = (long) Math.exp(5 + 10 * r.nextDouble());
            h.record(xs[i]);
        }
        Arrays.sort(xs);
        for (double q : new double[] {0.5, 0.9, 0.99, 0.999}) {
            long exact = xs[(int) Math.ceil(q * xs.length) - 1];
            long got = h.percentile(q);
            assertTrue(got <= exact && exact - got <= exact / 64 + 1, q + ": " + got + " vs " + exact);
        }
        assertEquals(xs[xs.length - 1], h.max());
    }
}
