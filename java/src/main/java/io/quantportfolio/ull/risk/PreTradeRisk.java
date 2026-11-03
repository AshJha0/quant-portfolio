package io.quantportfolio.ull.risk;

import io.quantportfolio.ull.oms.OmsCore;
import io.quantportfolio.ull.util.LongIntHashMap;

/**
 * Pre-trade risk checks (W28d) with no allocation per call.
 *
 * <p><b>Plain idea.</b> Before an order leaves, ask: is the size or value
 * absurd, is the price far from the market, would it take us past our
 * position or credit limit counting orders already working, have we seen
 * this ClOrdID, is the symbol restricted, would we trade with ourselves, and
 * are we sending too fast? Any yes rejects the order.
 *
 * <p><b>Formal statement.</b> Checks run in this order and the first failure
 * is returned:
 * <ol>
 *   <li>THROTTLE: at most {@code maxMsgs} accepted messages in any window of
 *       {@code windowNanos} per session. Rule: <b>reject, never queue</b>;
 *       rejected messages do not use up the window.</li>
 *   <li>DUPLICATE: ClOrdID is live, or among the last {@code dedupeWindow}
 *       terminal ids.</li>
 *   <li>RESTRICTED: symbol on the restricted list.</li>
 *   <li>FAT_FINGER: {@code qty > maxOrderQty}.</li>
 *   <li>NOTIONAL: {@code qty * px > maxOrderNotional}.</li>
 *   <li>PRICE_BAND: {@code |px - ref| * 10_000 > bandBps * ref} (ref price per symbol).</li>
 *   <li>POSITION: buys {@code pos + workingBuy + qty > maxPos}; sells
 *       {@code pos - workingSell - qty < -maxPos} (worst case if every working
 *       order fills).</li>
 *   <li>CREDIT: {@code workingNotional + qty * px > creditLimit}.</li>
 *   <li>SELF_TRADE: an open order of the same firm on the other side would be
 *       crossed. Policy: cancel-newest (the incoming order is rejected).</li>
 * </ol>
 * Sources: FIA, Best Practices for Automated Trading Risk Controls (2024);
 * SEC Rule 15c3-5(c)(1); MiFID II RTS 6 Art. 15.
 *
 * <p><b>Worked example.</b> maxPos 1,000, position +600, working buy 300.
 * A buy of 200 gives 600 + 300 + 200 = 1,100 &gt; 1,000: POSITION. A buy of
 * 100 gives exactly 1,000 and passes.
 */
public final class PreTradeRisk {
    public static final int OK = 0;
    public static final int THROTTLE = 1;
    public static final int DUPLICATE = 2;
    public static final int RESTRICTED = 3;
    public static final int FAT_FINGER = 4;
    public static final int NOTIONAL = 5;
    public static final int PRICE_BAND = 6;
    public static final int POSITION = 7;
    public static final int CREDIT = 8;
    public static final int SELF_TRADE = 9;
    public static final int KILLED = 10;
    public static final int UNKNOWN_ORDER = 11;

    private static final String[] NAMES = {
        "OK", "THROTTLE", "DUPLICATE", "RESTRICTED", "FAT_FINGER", "NOTIONAL", "PRICE_BAND",
        "POSITION", "CREDIT", "SELF_TRADE", "KILLED", "UNKNOWN_ORDER"};

    public static String name(int code) {
        return NAMES[code];
    }

    private final OmsCore oms;
    public long maxOrderQty = Long.MAX_VALUE / 4;
    public long maxOrderNotional = Long.MAX_VALUE / 4;
    public long creditLimit = Long.MAX_VALUE / 4;
    private final long[] refPx;
    private final long[] bandBps;
    private final long[] maxPos;
    private final boolean[] restricted;

    private final long[] msgTimes;
    private final long windowNanos;
    private int msgHead;
    private int msgCount;

    private final LongIntHashMap recentTerminal;
    private final long[] recentRing;
    private int recentHead;
    private int recentCount;

    public PreTradeRisk(OmsCore oms, int maxSymbols, int maxMsgs, long windowNanos, int dedupeWindow) {
        this.oms = oms;
        refPx = new long[maxSymbols];
        bandBps = new long[maxSymbols];
        maxPos = new long[maxSymbols];
        java.util.Arrays.fill(bandBps, Long.MAX_VALUE / 4);
        java.util.Arrays.fill(maxPos, Long.MAX_VALUE / 4);
        restricted = new boolean[maxSymbols];
        msgTimes = new long[maxMsgs];
        this.windowNanos = windowNanos;
        recentTerminal = new LongIntHashMap(dedupeWindow);
        recentRing = new long[dedupeWindow];
    }

    public void setSymbol(int sym, long ref, long bps, long maxPosition) {
        refPx[sym] = ref;
        bandBps[sym] = bps;
        maxPos[sym] = maxPosition;
    }

    public void setRestricted(int sym, boolean r) {
        restricted[sym] = r;
    }

    /** Throttle gate for any inbound session message; records the message if accepted. */
    public boolean admitMessage(long now) {
        int cap = msgTimes.length;
        if (msgCount == cap) {
            if (now - msgTimes[msgHead] < windowNanos) {
                return false;
            }
            msgHead = (msgHead + 1) % cap;
            msgCount--;
        }
        msgTimes[(msgHead + msgCount) % cap] = now;
        msgCount++;
        return true;
    }

    /** Checks 2-9 for a new order (the throttle is applied by {@link #admitMessage}). */
    public int checkNew(long id, int[] path, int side, long qty, long px) {
        if (oms.isOpen(id) || recentTerminal.containsKey(id)) {
            return DUPLICATE;
        }
        int sym = path[3];
        if (restricted[sym]) {
            return RESTRICTED;
        }
        if (qty <= 0 || qty > maxOrderQty) {
            return FAT_FINGER;
        }
        long notional = qty * px;
        if (px <= 0 || notional > maxOrderNotional) {
            return NOTIONAL;
        }
        long ref = refPx[sym];
        if (ref > 0 && Math.abs(px - ref) * 10_000L > bandBps[sym] * ref) {
            return PRICE_BAND;
        }
        long pos = oms.position(sym);
        if (side > 0 ? pos + oms.workingBuy(sym) + qty > maxPos[sym]
                     : pos - oms.workingSell(sym) - qty < -maxPos[sym]) {
            return POSITION;
        }
        if (oms.workingNotional() + notional > creditLimit) {
            return CREDIT;
        }
        if (oms.wouldSelfTrade(path, side, px)) {
            return SELF_TRADE;
        }
        return OK;
    }

    /** Remember a terminal ClOrdID for duplicate detection (bounded FIFO window). */
    public void onTerminal(long id) {
        if (recentTerminal.containsKey(id)) {
            return;
        }
        int cap = recentRing.length;
        if (recentCount == cap) {
            recentTerminal.remove(recentRing[recentHead]);
            recentHead = (recentHead + 1) % cap;
            recentCount--;
        }
        recentRing[(recentHead + recentCount) % cap] = id;
        recentCount++;
        recentTerminal.put(id, 1);
    }
}
