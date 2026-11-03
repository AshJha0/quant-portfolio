package io.quantportfolio.ull.ring;

/** Consumer callback. {@code endOfBatch} is true for the last event currently available. */
@FunctionalInterface
public interface EventHandler<E> {
    void onEvent(E event, long sequence, boolean endOfBatch);
}
