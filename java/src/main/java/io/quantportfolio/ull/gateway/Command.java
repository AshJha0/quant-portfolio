package io.quantportfolio.ull.gateway;

/** Mutable inbound event (a ring slot). Strings are symbol-table ids. */
public final class Command {
    public static final int NEW = 1;
    public static final int CANCEL = 2;
    public static final int VENUE_ACK = 3;
    public static final int VENUE_FILL = 4;
    public static final int VENUE_CANCELED = 5;
    public static final int VENUE_REJECT = 6;
    public static final int KILL_ENGAGE = 7;
    public static final int KILL_REQUEST_RESET = 8;
    public static final int KILL_APPROVE_RESET = 9;
    public static final int DISCONNECT = 10;

    public int type;
    public long clOrdId;
    public final int[] path = new int[4];
    /** +1 buy, -1 sell. */
    public int side;
    public long qty;
    /** Price in integer ticks. */
    public long px;
    public int actor;
    public int reason;
    public int killId;
    /** Caller's timestamp (for latency measurement); copied to reports. */
    public long originNanos;

    public Command set(int t, long id, int[] p, int s, long q, long price) {
        type = t;
        clOrdId = id;
        if (p != null) {
            System.arraycopy(p, 0, path, 0, 4);
        }
        side = s;
        qty = q;
        px = price;
        return this;
    }
}
