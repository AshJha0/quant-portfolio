package io.quantportfolio.ull.ring;

import java.util.Arrays;

/**
 * A single-producer ring buffer of pre-allocated, mutable events, in the style
 * of the LMAX Disruptor.
 *
 * <p><b>Plain idea.</b> The ring is an array of event objects created once.
 * The producer claims the next slot, overwrites its fields and publishes the
 * slot number. Consumers follow behind and read slots up to the published
 * number. Nothing is allocated after construction, so the steady state makes
 * no garbage.
 *
 * <p><b>Formal statement.</b> Let {@code P} be the published cursor and
 * {@code C_i} the sequence of consumer {@code i}. The producer may claim slot
 * {@code s} only when {@code s - size <= min_i C_i}, so it never overwrites
 * an unread slot. Consumer {@code i} may read slot {@code s} only when
 * {@code s <= P}. The cursor is written with release semantics after the slot
 * fields and read with acquire semantics, so the fields are visible to the
 * consumer (JLS 17.4; {@link java.lang.invoke.VarHandle} access modes).
 * Waiting is a busy spin with {@link Thread#onSpinWait()}.
 *
 * <p><b>Worked example.</b> Size 4, one consumer at -1. The producer claims
 * 0..3 and publishes 3. Claiming 4 needs {@code 4 - 4 = 0 <= C}, so it spins
 * until the consumer has read slot 0 and set {@code C = 0}.
 *
 * <p>Single producer only: {@link #next()} and {@link #publish(long)} are
 * called from one thread. Consumers are registered with
 * {@link #addGatingSequence(Sequence)} before the first publish.
 */
public final class RingBuffer<E> {
    private final Object[] entries;
    private final int mask;
    private final int size;
    private final Sequence cursor = new Sequence(-1L);
    private Sequence[] gating = new Sequence[0];
    private long nextValue = -1L;
    private long cachedGate = -1L;

    public RingBuffer(EventFactory<E> factory, int size) {
        if (size < 1 || Integer.bitCount(size) != 1) {
            throw new IllegalArgumentException("size must be a power of two, got " + size);
        }
        this.size = size;
        this.mask = size - 1;
        this.entries = new Object[size];
        for (int i = 0; i < size; i++) {
            entries[i] = factory.newInstance();
        }
    }

    public int size() {
        return size;
    }

    public Sequence cursor() {
        return cursor;
    }

    /** Register a consumer sequence the producer must not lap. Call before publishing. */
    public void addGatingSequence(Sequence s) {
        Sequence[] g = Arrays.copyOf(gating, gating.length + 1);
        g[g.length - 1] = s;
        gating = g;
    }

    /** Claim the next slot, spinning while the ring is full. */
    public long next() {
        long next = nextValue + 1;
        long wrapPoint = next - size;
        if (wrapPoint > cachedGate) {
            long min;
            while (wrapPoint > (min = minGating(next - 1))) {
                Thread.onSpinWait();
            }
            cachedGate = min;
        }
        nextValue = next;
        return next;
    }

    /** Claim the next slot if the ring has room, else return -1 without claiming. */
    public long tryNext() {
        long next = nextValue + 1;
        long wrapPoint = next - size;
        if (wrapPoint > cachedGate) {
            long min = minGating(next - 1);
            cachedGate = min;
            if (wrapPoint > min) {
                return -1L;
            }
        }
        nextValue = next;
        return next;
    }

    @SuppressWarnings("unchecked")
    public E get(long sequence) {
        return (E) entries[(int) (sequence & mask)];
    }

    /** Make slot {@code sequence}, and every earlier claimed slot, visible to consumers. */
    public void publish(long sequence) {
        cursor.set(sequence);
    }

    private long minGating(long dflt) {
        long min = dflt;
        for (Sequence s : gating) {
            long v = s.get();
            if (v < min) {
                min = v;
            }
        }
        return min;
    }

    /**
     * Hand every event published so far to {@code handler} and advance
     * {@code consumer}. Returns the number handled (0 if none). This is the
     * non-blocking step of a busy-poll loop; {@link BatchEventProcessor} is
     * the loop.
     */
    public int poll(Sequence consumer, EventHandler<E> handler) {
        long from = consumer.getPlain() + 1;
        long available = cursor.get();
        if (available < from) {
            return 0;
        }
        for (long s = from; s <= available; s++) {
            handler.onEvent(get(s), s, s == available);
        }
        consumer.set(available);
        return (int) (available - from + 1);
    }
}
