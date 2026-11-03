package io.quantportfolio.ull.util;

/** Allocation-free in-place heap sort of a {@code long[]} prefix. */
public final class Sorts {
    private Sorts() {}

    public static void sort(long[] a, int n) {
        for (int i = n / 2 - 1; i >= 0; i--) {
            sift(a, i, n);
        }
        for (int end = n - 1; end > 0; end--) {
            long t = a[0];
            a[0] = a[end];
            a[end] = t;
            sift(a, 0, end);
        }
    }

    private static void sift(long[] a, int i, int n) {
        while (true) {
            int l = 2 * i + 1;
            if (l >= n) {
                return;
            }
            int c = (l + 1 < n && a[l + 1] > a[l]) ? l + 1 : l;
            if (a[i] >= a[c]) {
                return;
            }
            long t = a[i];
            a[i] = a[c];
            a[c] = t;
            i = c;
        }
    }
}
