package io.quantportfolio.ull.metrics;

/**
 * Garbage-free log-linear histogram of non-negative nanosecond values, in
 * the spirit of HdrHistogram (Tene) with fixed 64 sub-buckets per power of
 * two.
 *
 * <p>Values below 128 are exact. Above that, a value {@code v} with highest
 * set bit {@code m} lands in a bucket of width {@code 2^(m-6)}, so a reported
 * percentile is the bucket's lower edge and under-states the true value by
 * less than 1/64 (1.6 %).
 */
public final class LatencyHistogram {
    private final long[] counts = new long[64 * 64];
    private long total;
    private long max;

    static int index(long v) {
        if (v < 128) {
            return (int) Math.max(0, v);
        }
        int m = 63 - Long.numberOfLeadingZeros(v);
        int shift = m - 6;
        return shift * 64 + (int) (v >>> shift);
    }

    static long lowerEdge(int idx) {
        if (idx < 128) {
            return idx;
        }
        int shift = idx / 64 - 1;
        long mant = idx % 64 + 64;
        return mant << shift;
    }

    public void record(long v) {
        counts[index(v)]++;
        total++;
        if (v > max) {
            max = v;
        }
    }

    public long count() {
        return total;
    }

    public long max() {
        return max;
    }

    /** Value at quantile {@code q} in [0, 1] (lower bucket edge; exact below 128). */
    public long percentile(double q) {
        if (total == 0) {
            return 0;
        }
        long rank = (long) Math.ceil(q * total);
        if (rank < 1) {
            rank = 1;
        }
        long seen = 0;
        for (int i = 0; i < counts.length; i++) {
            seen += counts[i];
            if (seen >= rank) {
                return lowerEdge(i);
            }
        }
        return max;
    }

    public void reset() {
        java.util.Arrays.fill(counts, 0);
        total = 0;
        max = 0;
    }
}
