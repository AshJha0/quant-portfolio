package io.quantportfolio.ull.kill;

import io.quantportfolio.ull.util.LongIntHashMap;
import io.quantportfolio.ull.util.Sorts;
import io.quantportfolio.ull.util.SymbolTable;

/**
 * The Java implementation of the shared kill-switch contract
 * ({@code docs/KILL_CONTRACT.md}, rules K1-K7). The Python
 * {@code qp_oms.kill.KillSwitch} is the reference; this class must replay the
 * golden vectors in {@code python/oms/kill/vectors.json} to the same wire
 * events, byte for byte (see {@code KillContractParityTest}).
 *
 * <p>Garbage-free: kills, scopes and outstanding-cancel lists live in arrays
 * sized at construction. {@link #check(int[])} (the per-order K1 test) is a
 * scan of the active kills with no allocation. The only allocating operation
 * is {@link #cancelOnDisconnect(int[][])}, which runs once per session drop.
 *
 * <p>Not thread-safe: the gateway owns it and calls it from its single
 * event-processing thread (the single-writer principle).
 */
public final class KillSwitch {
    public static final int FIRM = 1;
    public static final int DESK = 2;
    public static final int STRATEGY = 3;
    public static final int SYMBOL = 4;

    public static final int ENGAGED = 0;
    public static final int KILLED = 1;
    public static final int RESET_PENDING = 2;
    public static final int CLEARED = 3;

    public static final int EV_ENGAGE = 0;
    public static final int EV_ENGAGE_DUP = 1;
    public static final int EV_KILLED = 2;
    public static final int EV_RESET_REQUESTED = 3;
    public static final int EV_RESET_REJECTED = 4;
    public static final int EV_CLEARED = 5;

    /** Actor id for transitions made by the switch itself ("system"). */
    public static final int ACTOR_SYSTEM = -1;
    /** Reason codes below zero; a reason >= 0 is a symbol id. */
    public static final int REASON_NONE = -1;
    public static final int REASON_SAME_ACTOR = -2;
    /** {@code REASON_STATE_BASE - state} renders as {@code "state=<STATE>"}. */
    public static final int REASON_STATE_BASE = -10;

    private static final String[] LEVEL_NAMES = {"", "FIRM", "DESK", "STRATEGY", "SYMBOL"};
    private static final String[] STATE_NAMES = {"ENGAGED", "KILLED", "RESET_PENDING", "CLEARED"};
    private static final String[] EVENT_NAMES = {
        "engage", "engage_dup", "killed", "reset_requested", "reset_rejected", "cleared"};

    private final SymbolTable symbols;
    private final KillListener listener;
    private final int codActor;
    private final int codReason;

    // Kill k has id k + 1.
    private final int maxKills;
    private final int[][] scope;
    private final int[] level;
    private final int[] state;
    private final int[] requester;
    private final long[][] outstanding;
    private final int[] nOutstanding;
    private int nKills;
    private int seq;

    // Working orders: order id -> slot in workingPath.
    private final LongIntHashMap working;
    private final int[][] workingPath;
    private final int[] freeSlots;
    private int nFree;

    private final long[] cancelScratch;

    public KillSwitch(SymbolTable symbols, int maxKills, int maxWorking, KillListener listener) {
        this.symbols = symbols;
        this.listener = listener;
        this.codActor = symbols.intern("system:cod");
        this.codReason = symbols.intern("cancel_on_disconnect");
        this.maxKills = maxKills;
        this.scope = new int[maxKills][4];
        this.level = new int[maxKills];
        this.state = new int[maxKills];
        this.requester = new int[maxKills];
        this.outstanding = new long[maxKills][];
        this.nOutstanding = new int[maxKills];
        this.working = new LongIntHashMap(maxWorking);
        this.workingPath = new int[maxWorking][4];
        this.freeSlots = new int[maxWorking];
        for (int i = 0; i < maxWorking; i++) {
            freeSlots[i] = maxWorking - 1 - i;
        }
        this.nFree = maxWorking;
        this.cancelScratch = new long[maxWorking];
    }

    // ---- scope helpers ------------------------------------------------------

    /** Level of a scope, or throws if wildcards are not a pure suffix after the firm. */
    public static int validateScope(int[] s) {
        if (s.length != 4) {
            throw new IllegalArgumentException("scope must have four fields");
        }
        int n = 0;
        while (n < 4 && s[n] != SymbolTable.WILD) {
            n++;
        }
        if (n == 0) {
            throw new IllegalArgumentException("invalid scope: firm must not be a wildcard");
        }
        for (int i = n; i < 4; i++) {
            if (s[i] != SymbolTable.WILD) {
                throw new IllegalArgumentException("invalid scope: wildcards must be a suffix after firm");
            }
        }
        return n;
    }

    static boolean covers(int[] s, int[] path) {
        for (int i = 0; i < 4; i++) {
            if (s[i] != SymbolTable.WILD && s[i] != path[i]) {
                return false;
            }
        }
        return true;
    }

    // ---- order tracking -----------------------------------------------------

    /** The order is live at the venue (or about to be sent). */
    public void onOrderWorking(long orderId, int[] path) {
        int slot = working.get(orderId);
        if (slot == LongIntHashMap.MISSING) {
            if (nFree == 0) {
                throw new IllegalStateException("too many working orders");
            }
            slot = freeSlots[--nFree];
            working.put(orderId, slot);
        }
        System.arraycopy(path, 0, workingPath[slot], 0, 4);
    }

    /** K3: a terminal report (cancel ack, fill, reject, expiry). */
    public void onOrderTerminal(long orderId) {
        int slot = working.remove(orderId);
        if (slot != LongIntHashMap.MISSING) {
            freeSlots[nFree++] = slot;
        }
        for (int k = 0; k < nKills; k++) {
            int n = nOutstanding[k];
            if (n == 0) {
                continue;
            }
            long[] arr = outstanding[k];
            for (int i = 0; i < n; i++) {
                if (arr[i] == orderId) {
                    arr[i] = arr[n - 1];
                    nOutstanding[k] = --n;
                    if (state[k] == ENGAGED && n == 0) {
                        state[k] = KILLED;
                        emit(EV_KILLED, k, ACTOR_SYSTEM, REASON_NONE, cancelScratch, 0);
                    }
                    break;
                }
            }
        }
    }

    public int workingCount() {
        return working.size();
    }

    // ---- engage (K2, K7) ----------------------------------------------------

    /**
     * Engage a kill on {@code s}. Returns the kill id. The ids to cancel
     * (sorted ascending) are copied into {@code cancelOut}; their count is
     * {@link #lastCancelCount()}.
     */
    public int engage(int[] s, int actor, int reason, long[] cancelOut) {
        int lvl = validateScope(s);
        for (int k = 0; k < nKills; k++) {
            if (state[k] != CLEARED && sameScope(scope[k], s)) {
                emit(EV_ENGAGE_DUP, k, actor, reason, cancelScratch, 0);
                lastCancelCount = 0;
                return k + 1;
            }
        }
        if (nKills == maxKills) {
            throw new IllegalStateException("kill table full: " + maxKills);
        }
        int k = nKills++;
        System.arraycopy(s, 0, scope[k], 0, 4);
        level[k] = lvl;
        requester[k] = Integer.MIN_VALUE;
        int n = 0;
        for (int i = 0, cap = working.capacity(); i < cap; i++) {
            long id = working.keyAt(i);
            if (id != 0L && covers(s, workingPath[working.valueAt(i)])) {
                cancelScratch[n++] = id;
            }
        }
        Sorts.sort(cancelScratch, n);
        if (outstanding[k] == null || outstanding[k].length < n) {
            outstanding[k] = new long[Math.max(n, 16)];
        }
        System.arraycopy(cancelScratch, 0, outstanding[k], 0, n);
        nOutstanding[k] = n;
        state[k] = n == 0 ? KILLED : ENGAGED;
        emit(EV_ENGAGE, k, actor, reason, cancelScratch, n);
        if (cancelOut != null) {
            System.arraycopy(cancelScratch, 0, cancelOut, 0, n);
        }
        lastCancelCount = n;
        return k + 1;
    }

    private int lastCancelCount;

    public int lastCancelCount() {
        return lastCancelCount;
    }

    private static boolean sameScope(int[] a, int[] b) {
        return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
    }

    /**
     * K6: engage a STRATEGY-level kill (actor {@code system:cod}) for every
     * distinct strategy among {@code sessionPaths}, in the order of their
     * names (as the Python reference sorts tuples of strings). Allocates; runs
     * once per disconnect.
     */
    public int[] cancelOnDisconnect(int[][] sessionPaths) {
        java.util.TreeMap<String, int[]> strategies = new java.util.TreeMap<>(KillSwitch::compareKeys);
        for (int[] p : sessionPaths) {
            int[] sc = {p[0], p[1], p[2], SymbolTable.WILD};
            strategies.put(symbols.name(sc[0]) + "\u0000" + symbols.name(sc[1]) + "\u0000"
                + symbols.name(sc[2]) + "\u0000*", sc);
        }
        int[] ids = new int[strategies.size()];
        int i = 0;
        for (int[] sc : strategies.values()) {
            ids[i++] = engage(sc, codActor, codReason, null);
        }
        return ids;
    }

    private static int compareKeys(String a, String b) {
        // Field-wise comparison, identical to Python tuple-of-str ordering for BMP text.
        String[] x = a.split("\u0000", -1);
        String[] y = b.split("\u0000", -1);
        for (int i = 0; i < 4; i++) {
            int c = x[i].compareTo(y[i]);
            if (c != 0) {
                return c;
            }
        }
        return 0;
    }

    // ---- block (K1) ---------------------------------------------------------

    /**
     * K1: returns 0 if an order on {@code path} may go out, else the id of
     * the widest active kill covering it (lowest level, then lowest id).
     * Allocation-free.
     */
    public int check(int[] path) {
        int best = -1;
        for (int k = 0; k < nKills; k++) {
            if (state[k] != CLEARED && covers(scope[k], path)) {
                if (best < 0 || level[k] < level[best]) {
                    best = k;
                }
            }
        }
        return best + 1;
    }

    /** Render the reject reason {@code KILLED:<level>:<scope>} for a kill id from {@link #check}. */
    public String rejectReason(int killId) {
        int k = killId - 1;
        return "KILLED:" + LEVEL_NAMES[level[k]] + ":" + scopeString(scope[k], "/");
    }

    private String scopeString(int[] s, String sep) {
        return symbols.name(s[0]) + sep + symbols.name(s[1]) + sep + symbols.name(s[2]) + sep + symbols.name(s[3]);
    }

    // ---- reset (K4) ---------------------------------------------------------

    public boolean requestReset(int killId, int actor) {
        int k = index(killId);
        if (state[k] != KILLED) {
            emit(EV_RESET_REJECTED, k, actor, REASON_STATE_BASE - state[k], cancelScratch, 0);
            return false;
        }
        state[k] = RESET_PENDING;
        requester[k] = actor;
        emit(EV_RESET_REQUESTED, k, actor, REASON_NONE, cancelScratch, 0);
        return true;
    }

    public boolean approveReset(int killId, int actor) {
        int k = index(killId);
        if (state[k] != RESET_PENDING) {
            emit(EV_RESET_REJECTED, k, actor, REASON_STATE_BASE - state[k], cancelScratch, 0);
            return false;
        }
        if (actor == requester[k]) {
            emit(EV_RESET_REJECTED, k, actor, REASON_SAME_ACTOR, cancelScratch, 0);
            return false;
        }
        state[k] = CLEARED;
        emit(EV_CLEARED, k, actor, REASON_NONE, cancelScratch, 0);
        return true;
    }

    public int state(int killId) {
        return state[index(killId)];
    }

    private int index(int killId) {
        if (killId < 1 || killId > nKills) {
            throw new IllegalArgumentException("unknown kill id " + killId);
        }
        return killId - 1;
    }

    private void emit(int ev, int k, int actor, int reason, long[] cancel, int n) {
        seq++;
        listener.onKillEvent(seq, ev, k + 1, level[k], scope[k], actor, state[k], cancel, n, reason);
    }

    // ---- rendering (off the hot path) ---------------------------------------

    /**
     * Canonical JSON of one event: keys sorted, no spaces. Byte-identical to
     * the Python {@code KillEvent.to_wire} for ASCII names. Allocates.
     */
    public String toWire(int seqNo, int ev, int killId, int lvl, int[] sc, int actor, int st,
                         long[] cancel, int nCancel, int reason) {
        StringBuilder b = new StringBuilder(160);
        b.append("{\"actor\":");
        jsonString(b, actor == ACTOR_SYSTEM ? "system" : symbols.name(actor));
        b.append(",\"cancel\":[");
        for (int i = 0; i < nCancel; i++) {
            if (i > 0) {
                b.append(',');
            }
            b.append(cancel[i]);
        }
        b.append("],\"event\":");
        jsonString(b, EVENT_NAMES[ev]);
        b.append(",\"kill_id\":").append(killId);
        b.append(",\"level\":");
        jsonString(b, LEVEL_NAMES[lvl]);
        b.append(",\"reason\":");
        jsonString(b, reasonText(reason));
        b.append(",\"scope\":[");
        for (int i = 0; i < 4; i++) {
            if (i > 0) {
                b.append(',');
            }
            jsonString(b, symbols.name(sc[i]));
        }
        b.append("],\"seq\":").append(seqNo);
        b.append(",\"state\":");
        jsonString(b, STATE_NAMES[st]);
        b.append('}');
        return b.toString();
    }

    private String reasonText(int reason) {
        if (reason >= 0) {
            return symbols.name(reason);
        }
        if (reason == REASON_NONE) {
            return "";
        }
        if (reason == REASON_SAME_ACTOR) {
            return "same_actor";
        }
        return "state=" + STATE_NAMES[REASON_STATE_BASE - reason];
    }

    /** JSON string with Python {@code json.dumps} (ensure_ascii) escaping. */
    public static void jsonString(StringBuilder b, String s) {
        b.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> b.append("\\\"");
                case '\\' -> b.append("\\\\");
                case '\n' -> b.append("\\n");
                case '\r' -> b.append("\\r");
                case '\t' -> b.append("\\t");
                case '\b' -> b.append("\\b");
                case '\f' -> b.append("\\f");
                default -> {
                    if (c < 0x20 || c > 0x7f) {
                        b.append(String.format("\\u%04x", (int) c));
                    } else {
                        b.append(c);
                    }
                }
            }
        }
        b.append('"');
    }
}
