package io.quantportfolio.ull.oms;

/** FIX OrdStatus values used by the OMS core, as int constants (no enum lookups on the hot path). */
public final class OrderStatus {
    private OrderStatus() {}

    public static final int PENDING_NEW = 0;
    public static final int NEW = 1;
    public static final int PARTIALLY_FILLED = 2;
    public static final int FILLED = 3;
    public static final int PENDING_CANCEL = 4;
    public static final int CANCELED = 5;
    public static final int REJECTED = 6;
    /** Returned by OMS operations for an unknown order or an illegal transition. */
    public static final int INVALID = -1;

    private static final String[] NAMES = {
        "PENDING_NEW", "NEW", "PARTIALLY_FILLED", "FILLED", "PENDING_CANCEL", "CANCELED", "REJECTED"};

    public static boolean isTerminal(int s) {
        return s == FILLED || s == CANCELED || s == REJECTED;
    }

    public static String name(int s) {
        return s < 0 ? "INVALID" : NAMES[s];
    }
}
