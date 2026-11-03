package io.quantportfolio.ull.ring;

/** Creates the pre-allocated slot objects of a ring. Called only at construction. */
@FunctionalInterface
public interface EventFactory<E> {
    E newInstance();
}
