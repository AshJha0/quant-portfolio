package io.quantportfolio.ull.gateway;

/** Mutable outbound event (a ring slot): messages to the venue and reports to the client. */
public final class Report {
    /** To venue: send a NewOrderSingle (35=D). */
    public static final int SEND_NEW = 1;
    /** To venue: send an OrderCancelRequest (35=F). */
    public static final int SEND_CANCEL = 2;
    /** To client: order rejected before leaving; see {@link #rejectCode}. */
    public static final int REJECT = 3;
    /** To client: execution report (ack, fill, cancel, venue reject); see {@link #status}. */
    public static final int EXEC = 4;
    /** To client and audit: a kill transition. */
    public static final int KILL_EVENT = 5;

    public int type;
    public long clOrdId;
    public int rejectCode;
    public int status;
    public long qty;
    public long px;
    public long cum;
    public long leaves;
    public int killId;
    public int killEvent;
    public int killState;
    public long originNanos;
    public long gatewayNanos;
    public long clockErrorNanos;
    public int clockSource;
}
