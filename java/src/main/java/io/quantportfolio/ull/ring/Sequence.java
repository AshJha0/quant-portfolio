package io.quantportfolio.ull.ring;

import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;

/**
 * A padded, single-writer sequence counter.
 *
 * <p>The value sits between seven-long pads on each side so that two
 * sequences do not share a 64-byte cache line (false sharing). Writes use
 * release semantics and reads use acquire semantics: the same publication
 * protocol as the LMAX Disruptor's {@code Sequence} (Thompson et al., 2011),
 * written with {@link VarHandle} instead of {@code sun.misc.Unsafe}.
 */
@SuppressWarnings("unused")
public final class Sequence {
    private static final VarHandle VALUE;

    static {
        try {
            VALUE = MethodHandles.lookup().findVarHandle(Sequence.class, "value", long.class);
        } catch (ReflectiveOperationException e) {
            throw new ExceptionInInitializerError(e);
        }
    }

    private long p1, p2, p3, p4, p5, p6, p7;
    private volatile long value;
    private long q1, q2, q3, q4, q5, q6, q7;

    public Sequence(long initial) {
        VALUE.setRelease(this, initial);
    }

    /** Acquire read. */
    public long get() {
        return (long) VALUE.getAcquire(this);
    }

    /** Release write: writes made before this call are visible to a reader that sees the new value. */
    public void set(long v) {
        VALUE.setRelease(this, v);
    }

    /** Plain read; only for the single owning writer thread. */
    public long getPlain() {
        return (long) VALUE.get(this);
    }
}
