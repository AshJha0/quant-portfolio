package io.quantportfolio.ull.gateway;

/**
 * Clock with a stated source and error bound (W28d clock discipline). Every
 * report the gateway emits carries both, so a reader knows how far a
 * timestamp can be trusted.
 */
public interface GatewayClock {
    int SOURCE_MONOTONIC = 1;
    int SOURCE_MANUAL = 2;

    long nanos();

    /** Upper bound on the error of {@link #nanos()} for interval measurement, in ns. */
    long errorBoundNanos();

    int source();

    /**
     * {@link System#nanoTime()}: monotonic, process-local, not wall-clock.
     * The error bound is the smallest non-zero step observed at start-up
     * (the timer's granularity), a lower bound on resolution rather than a
     * guarantee of accuracy against UTC.
     */
    static GatewayClock monotonic() {
        long min = Long.MAX_VALUE;
        for (int i = 0; i < 10_000; i++) {
            long a = System.nanoTime();
            long b;
            do {
                b = System.nanoTime();
            } while (b == a);
            min = Math.min(min, b - a);
        }
        final long bound = min;
        return new GatewayClock() {
            @Override
            public long nanos() {
                return System.nanoTime();
            }

            @Override
            public long errorBoundNanos() {
                return bound;
            }

            @Override
            public int source() {
                return SOURCE_MONOTONIC;
            }
        };
    }

    /** A test clock moved by hand. */
    final class Manual implements GatewayClock {
        public long now;

        @Override
        public long nanos() {
            return now;
        }

        @Override
        public long errorBoundNanos() {
            return 0;
        }

        @Override
        public int source() {
            return SOURCE_MANUAL;
        }
    }
}
