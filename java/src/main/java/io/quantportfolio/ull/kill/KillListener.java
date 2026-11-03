package io.quantportfolio.ull.kill;

/**
 * Receives every kill transition (contract "wire form"). All arguments are
 * primitives so that emitting an event allocates nothing; {@code cancel}
 * is a scratch buffer valid only during the call, with {@code nCancel}
 * entries. Strings are symbol-table ids; {@code actor} may be
 * {@link KillSwitch#ACTOR_SYSTEM}; {@code reason} uses the codes on
 * {@link KillSwitch}.
 */
@FunctionalInterface
public interface KillListener {
    void onKillEvent(int seq, int event, int killId, int level, int[] scope, int actor, int state,
                     long[] cancel, int nCancel, int reason);

    KillListener NONE = (seq, event, killId, level, scope, actor, state, cancel, nCancel, reason) -> { };
}
