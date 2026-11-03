package io.quantportfolio.ull.ring;

/**
 * A busy-polling consumer loop. It reads published events in batches and
 * releases the slots after each batch. It never blocks and never allocates;
 * it uses one core, which is the usual trade on a latency path.
 */
public final class BatchEventProcessor<E> implements Runnable {
    private final RingBuffer<E> ring;
    private final EventHandler<E> handler;
    private final Sequence sequence = new Sequence(-1L);
    private volatile boolean running = true;

    public BatchEventProcessor(RingBuffer<E> ring, EventHandler<E> handler) {
        this.ring = ring;
        this.handler = handler;
        ring.addGatingSequence(sequence);
    }

    public Sequence sequence() {
        return sequence;
    }

    public void halt() {
        running = false;
    }

    @Override
    public void run() {
        while (running) {
            if (ring.poll(sequence, handler) == 0) {
                Thread.onSpinWait();
            }
        }
        ring.poll(sequence, handler);
    }
}
