package io.quantportfolio.ull.oms;

import io.quantportfolio.ull.util.LongIntHashMap;

/**
 * Garbage-free order store and state machine (the core of W28a, in Java).
 *
 * <p><b>Plain idea.</b> Every live order is a row in a set of parallel arrays
 * (struct of arrays), found by ClOrdID through an open-addressing map. A row
 * is reused once the order is terminal, so after start-up nothing is
 * allocated.
 *
 * <p><b>Formal statement.</b> For every open order: {@code cum + leaves = qty},
 * {@code 0 <= cum <= qty}. Transitions (FIX 4.4 Vol. 4 Appendix D, reduced to
 * the single-venue set used here):
 * <pre>
 * PENDING_NEW --ack--> NEW --fill--> PARTIALLY_FILLED --fill--> FILLED
 * PENDING_NEW --reject--> REJECTED
 * {PENDING_NEW, NEW, PARTIALLY_FILLED} --cancel request--> PENDING_CANCEL
 * PENDING_CANCEL --fill--> PENDING_CANCEL (or FILLED when leaves reach 0)
 * {any open} --cancel ack--> CANCELED
 * </pre>
 * A fill larger than {@code leaves} is refused ({@link OrderStatus#INVALID}).
 * Per symbol the store also keeps net position and working buy and sell
 * quantity, which pre-trade risk reads; and the total working notional.
 *
 * <p><b>Worked example.</b> Buy 100 at 10: PENDING_NEW, cum 0, leaves 100,
 * working buy 100. Ack: NEW. Fill 40: PARTIALLY_FILLED, cum 40, leaves 60,
 * position +40, working buy 60. Cancel ack: CANCELED, leaves 0, working buy 0,
 * row freed.
 */
public final class OmsCore {
    private final int capacity;
    private final LongIntHashMap index;
    private final long[] clOrdId;
    private final int[][] path;
    private final int[] side;
    private final long[] qty;
    private final long[] px;
    private final long[] cum;
    private final long[] leaves;
    private final int[] status;
    private final int[] prevOpenStatus;
    private final int[] free;
    private int nFree;

    // Per-symbol intrusive list of open rows (for self-trade checks) and exposure.
    private final int[] symHead;
    private final int[] next;
    private final int[] prev;
    private final long[] position;
    private final long[] workingBuy;
    private final long[] workingSell;
    private long workingNotional;
    // Self-trade prefilter: best open buy / sell price per symbol and how many open orders sit there.
    // When the last order at the best price leaves, the best is recomputed lazily by a scan.
    private final long[] bestBuy;
    private final long[] bestSell;
    private final int[] nAtBestBuy;
    private final int[] nAtBestSell;
    private final boolean[] bestDirty;

    /** Status of the order after the most recent operation (also for rows already freed). */
    private int lastStatus;
    private long lastCum;
    private long lastLeaves;

    public OmsCore(int capacity, int maxSymbols) {
        this.capacity = capacity;
        index = new LongIntHashMap(capacity);
        clOrdId = new long[capacity];
        path = new int[capacity][4];
        side = new int[capacity];
        qty = new long[capacity];
        px = new long[capacity];
        cum = new long[capacity];
        leaves = new long[capacity];
        status = new int[capacity];
        prevOpenStatus = new int[capacity];
        free = new int[capacity];
        for (int i = 0; i < capacity; i++) {
            free[i] = capacity - 1 - i;
        }
        nFree = capacity;
        next = new int[capacity];
        prev = new int[capacity];
        symHead = new int[maxSymbols];
        java.util.Arrays.fill(symHead, -1);
        position = new long[maxSymbols];
        workingBuy = new long[maxSymbols];
        workingSell = new long[maxSymbols];
        bestBuy = new long[maxSymbols];
        bestSell = new long[maxSymbols];
        nAtBestBuy = new int[maxSymbols];
        nAtBestSell = new int[maxSymbols];
        bestDirty = new boolean[maxSymbols];
        java.util.Arrays.fill(bestBuy, Long.MIN_VALUE);
        java.util.Arrays.fill(bestSell, Long.MAX_VALUE);
    }

    private void bestOnAdd(int sym, int s, long p) {
        if (s > 0) {
            if (p > bestBuy[sym]) {
                bestBuy[sym] = p;
                nAtBestBuy[sym] = 1;
            } else if (p == bestBuy[sym]) {
                nAtBestBuy[sym]++;
            }
        } else {
            if (p < bestSell[sym]) {
                bestSell[sym] = p;
                nAtBestSell[sym] = 1;
            } else if (p == bestSell[sym]) {
                nAtBestSell[sym]++;
            }
        }
    }

    private void bestOnRemove(int sym, int s, long p) {
        if (s > 0 && p == bestBuy[sym] && --nAtBestBuy[sym] == 0) {
            bestDirty[sym] = true;
        } else if (s < 0 && p == bestSell[sym] && --nAtBestSell[sym] == 0) {
            bestDirty[sym] = true;
        }
    }

    private void recomputeBest(int sym) {
        bestBuy[sym] = Long.MIN_VALUE;
        bestSell[sym] = Long.MAX_VALUE;
        nAtBestBuy[sym] = 0;
        nAtBestSell[sym] = 0;
        for (int r = symHead[sym]; r >= 0; r = next[r]) {
            bestOnAdd(sym, side[r], px[r]);
        }
        bestDirty[sym] = false;
    }

    public int openCount() {
        return capacity - nFree;
    }

    public boolean isOpen(long id) {
        return index.containsKey(id);
    }

    /** Add a new order (PENDING_NEW). Returns false if the id is live or the store is full. */
    public boolean add(long id, int[] orderPath, int orderSide, long orderQty, long orderPx) {
        if (nFree == 0 || index.containsKey(id)) {
            return false;
        }
        int r = free[--nFree];
        index.put(id, r);
        clOrdId[r] = id;
        System.arraycopy(orderPath, 0, path[r], 0, 4);
        side[r] = orderSide;
        qty[r] = orderQty;
        px[r] = orderPx;
        cum[r] = 0;
        leaves[r] = orderQty;
        status[r] = OrderStatus.PENDING_NEW;
        int sym = orderPath[3];
        next[r] = symHead[sym];
        prev[r] = -1;
        if (symHead[sym] >= 0) {
            prev[symHead[sym]] = r;
        }
        symHead[sym] = r;
        bestOnAdd(sym, orderSide, orderPx);
        addWorking(r, orderQty);
        record(r);
        return true;
    }

    private void addWorking(int r, long q) {
        if (side[r] > 0) {
            workingBuy[path[r][3]] += q;
        } else {
            workingSell[path[r][3]] += q;
        }
        workingNotional += q * px[r];
    }

    private void record(int r) {
        lastStatus = status[r];
        lastCum = cum[r];
        lastLeaves = leaves[r];
    }

    public int ack(long id) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING) {
            return OrderStatus.INVALID;
        }
        if (status[r] == OrderStatus.PENDING_NEW) {
            status[r] = OrderStatus.NEW;
        } else if (status[r] == OrderStatus.PENDING_CANCEL && prevOpenStatus[r] == OrderStatus.PENDING_NEW) {
            prevOpenStatus[r] = OrderStatus.NEW;
        } else {
            return OrderStatus.INVALID;
        }
        record(r);
        return status[r];
    }

    public int fill(long id, long fillQty, long fillPx) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING || fillQty <= 0 || fillQty > leaves[r]) {
            return OrderStatus.INVALID;
        }
        cum[r] += fillQty;
        leaves[r] -= fillQty;
        int sym = path[r][3];
        position[sym] += side[r] * fillQty;
        addWorking(r, -fillQty);
        if (leaves[r] == 0) {
            status[r] = OrderStatus.FILLED;
        } else if (status[r] != OrderStatus.PENDING_CANCEL) {
            status[r] = OrderStatus.PARTIALLY_FILLED;
        }
        record(r);
        if (status[r] == OrderStatus.FILLED) {
            release(r);
        }
        return lastStatus;
    }

    public int requestCancel(long id) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING || status[r] == OrderStatus.PENDING_CANCEL) {
            return OrderStatus.INVALID;
        }
        prevOpenStatus[r] = status[r];
        status[r] = OrderStatus.PENDING_CANCEL;
        record(r);
        return status[r];
    }

    /** Cancel ack, or an unsolicited cancel (expiry) from the venue. */
    public int canceled(long id) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING) {
            return OrderStatus.INVALID;
        }
        addWorking(r, -leaves[r]);
        leaves[r] = 0;
        status[r] = OrderStatus.CANCELED;
        record(r);
        release(r);
        return OrderStatus.CANCELED;
    }

    public int reject(long id) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING || cum[r] != 0) {
            return OrderStatus.INVALID;
        }
        addWorking(r, -leaves[r]);
        leaves[r] = 0;
        status[r] = OrderStatus.REJECTED;
        record(r);
        release(r);
        return OrderStatus.REJECTED;
    }

    private void release(int r) {
        index.remove(clOrdId[r]);
        int sym = path[r][3];
        if (prev[r] >= 0) {
            next[prev[r]] = next[r];
        } else {
            symHead[sym] = next[r];
        }
        if (next[r] >= 0) {
            prev[next[r]] = prev[r];
        }
        bestOnRemove(sym, side[r], px[r]);
        free[nFree++] = r;
    }

    public int lastStatus() {
        return lastStatus;
    }

    public long lastCum() {
        return lastCum;
    }

    public long lastLeaves() {
        return lastLeaves;
    }

    public long position(int sym) {
        return position[sym];
    }

    public long workingBuy(int sym) {
        return workingBuy[sym];
    }

    public long workingSell(int sym) {
        return workingSell[sym];
    }

    public long workingNotional() {
        return workingNotional;
    }

    /**
     * Self-trade test: is there an open order on {@code sym} for the same
     * firm, on the other side, at a price the new order would cross?
     */
    public boolean wouldSelfTrade(int[] orderPath, int orderSide, long orderPx) {
        int sym = orderPath[3];
        if (bestDirty[sym]) {
            recomputeBest(sym);
        }
        // O(1) prefilter over all firms: no open opposite order at a crossable price means no self-trade.
        if (orderSide > 0 ? orderPx < bestSell[sym] : orderPx > bestBuy[sym]) {
            return false;
        }
        for (int r = symHead[orderPath[3]]; r >= 0; r = next[r]) {
            if (side[r] != orderSide && path[r][0] == orderPath[0] && status[r] != OrderStatus.PENDING_CANCEL) {
                boolean cross = orderSide > 0 ? orderPx >= px[r] : orderPx <= px[r];
                if (cross) {
                    return true;
                }
            }
        }
        return false;
    }

    /** Path of an open order (copied into {@code out}); false if not open. */
    public boolean pathOf(long id, int[] out) {
        int r = index.get(id);
        if (r == LongIntHashMap.MISSING) {
            return false;
        }
        System.arraycopy(path[r], 0, out, 0, 4);
        return true;
    }

    /** Every open ClOrdID into {@code out}; returns the count. */
    public int openIds(long[] out) {
        return index.keys(out);
    }

    public int statusOf(long id) {
        int r = index.get(id);
        return r == LongIntHashMap.MISSING ? OrderStatus.INVALID : status[r];
    }

    /** Recompute every invariant from the rows; returns null if all hold, else a description. For tests. */
    public String checkInvariants() {
        long[] wb = new long[workingBuy.length];
        long[] ws = new long[workingSell.length];
        long notional = 0;
        int open = 0;
        for (int i = 0, cap = index.capacity(); i < cap; i++) {
            if (index.keyAt(i) == 0L) {
                continue;
            }
            int r = index.valueAt(i);
            open++;
            if (cum[r] + leaves[r] != qty[r] || cum[r] < 0 || leaves[r] < 0) {
                return "cum+leaves!=qty for " + clOrdId[r];
            }
            if (OrderStatus.isTerminal(status[r])) {
                return "terminal order still open: " + clOrdId[r];
            }
            if (side[r] > 0) {
                wb[path[r][3]] += leaves[r];
            } else {
                ws[path[r][3]] += leaves[r];
            }
            notional += leaves[r] * px[r];
        }
        if (open != openCount()) {
            return "open count mismatch";
        }
        if (!java.util.Arrays.equals(wb, workingBuy) || !java.util.Arrays.equals(ws, workingSell)) {
            return "working quantity mismatch";
        }
        if (notional != workingNotional) {
            return "working notional mismatch";
        }
        return null;
    }
}
