package io.quantportfolio.ull.util;

/**
 * Open-addressing {@code long -> int} map with linear probing and
 * backward-shift deletion. Fixed capacity, no boxing, no allocation after
 * construction. Key 0 is reserved as the empty marker.
 */
public final class LongIntHashMap {
    public static final int MISSING = Integer.MIN_VALUE;
    private final long[] keys;
    private final int[] values;
    private final int mask;
    private final int maxSize;
    private int size;

    /** @param maxSize the most entries ever held at once; the table is sized at least twice that. */
    public LongIntHashMap(int maxSize) {
        int cap = Integer.highestOneBit(Math.max(4, maxSize * 2 - 1)) << 1;
        keys = new long[cap];
        values = new int[cap];
        mask = cap - 1;
        this.maxSize = maxSize;
    }

    private static int hash(long k) {
        long h = k * 0x9E3779B97F4A7C15L;
        return (int) (h ^ (h >>> 32));
    }

    public int size() {
        return size;
    }

    public int get(long key) {
        int i = hash(key) & mask;
        while (true) {
            long k = keys[i];
            if (k == key) {
                return values[i];
            }
            if (k == 0L) {
                return MISSING;
            }
            i = (i + 1) & mask;
        }
    }

    public boolean containsKey(long key) {
        return get(key) != MISSING;
    }

    /** Insert or overwrite. Throws if full; capacity is a configuration limit. */
    public void put(long key, int value) {
        if (key == 0L) {
            throw new IllegalArgumentException("key 0 is reserved");
        }
        int i = hash(key) & mask;
        while (true) {
            long k = keys[i];
            if (k == key) {
                values[i] = value;
                return;
            }
            if (k == 0L) {
                if (size == maxSize) {
                    throw new IllegalStateException("LongIntHashMap full: " + maxSize);
                }
                keys[i] = key;
                values[i] = value;
                size++;
                return;
            }
            i = (i + 1) & mask;
        }
    }

    /** Remove and return the value, or {@link #MISSING}. */
    public int remove(long key) {
        int i = hash(key) & mask;
        while (true) {
            long k = keys[i];
            if (k == 0L) {
                return MISSING;
            }
            if (k == key) {
                int v = values[i];
                shiftBack(i);
                size--;
                return v;
            }
            i = (i + 1) & mask;
        }
    }

    private void shiftBack(int hole) {
        int i = hole;
        while (true) {
            i = (i + 1) & mask;
            long k = keys[i];
            if (k == 0L) {
                break;
            }
            int home = hash(k) & mask;
            // Move k into the hole if its home is not cyclically in (hole, i].
            boolean move = (i > hole) ? (home <= hole || home > i) : (home <= hole && home > i);
            if (move) {
                keys[hole] = k;
                values[hole] = values[i];
                hole = i;
            }
        }
        keys[hole] = 0L;
        values[hole] = 0;
    }

    public void clear() {
        java.util.Arrays.fill(keys, 0L);
        size = 0;
    }

    /** Copy every key into {@code out} (unsorted); returns the count. */
    public int keys(long[] out) {
        int n = 0;
        for (long k : keys) {
            if (k != 0L) {
                out[n++] = k;
            }
        }
        return n;
    }

    /** Raw table access for allocation-free iteration: slot count. */
    public int capacity() {
        return keys.length;
    }

    /** Key at table slot {@code i}, 0 if empty. */
    public long keyAt(int i) {
        return keys[i];
    }

    /** Value at table slot {@code i}. */
    public int valueAt(int i) {
        return values[i];
    }
}
