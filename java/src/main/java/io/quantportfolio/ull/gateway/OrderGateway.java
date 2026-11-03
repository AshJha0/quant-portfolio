package io.quantportfolio.ull.gateway;

import io.quantportfolio.ull.kill.KillListener;
import io.quantportfolio.ull.kill.KillSwitch;
import io.quantportfolio.ull.oms.OmsCore;
import io.quantportfolio.ull.oms.OrderStatus;
import io.quantportfolio.ull.ring.EventHandler;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.risk.PreTradeRisk;
import io.quantportfolio.ull.util.SymbolTable;

/**
 * Garbage-free order gateway: the single thread that owns the kill switch,
 * pre-trade risk and OMS core, and turns inbound {@link Command}s into
 * outbound {@link Report}s.
 *
 * <p><b>Plain idea.</b> Every order passes three gates in a fixed order:
 * the kill switch (is this part of the business stopped?), pre-trade risk
 * (is the order sane?), then the OMS records it and the gateway sends it.
 * Venue replies update the OMS. A kill cancels every working order it covers
 * and blocks new ones until a two-person reset.
 *
 * <p><b>Formal statement.</b> Single writer: only the gateway thread calls
 * {@link #onEvent}. Order of effects for a NEW: throttle, K1 kill check,
 * risk checks, OMS insert, kill tracking, then {@code SEND_NEW} is published.
 * A rejected order never reaches the venue. On {@code KILL_ENGAGE}, every
 * covered working order gets {@code SEND_CANCEL} (K2) in ascending id order;
 * the kill reaches KILLED when the last of them is terminal (K3). Every
 * report carries the gateway timestamp, the clock source and its error bound.
 * After construction no method on this path allocates (verified by
 * {@code AllocationTest}); {@code DISCONNECT} is the one exception (K6 runs
 * once per session drop).
 */
public final class OrderGateway implements EventHandler<Command> {
    private final OmsCore oms;
    private final PreTradeRisk risk;
    private final KillSwitch kill;
    private final RingBuffer<Report> out;
    private final GatewayClock clock;
    private final long[] cancelScratch;
    private final int[] pathScratch = new int[4];
    private long now;
    private long originNanos;

    public OrderGateway(SymbolTable symbols, int maxOrders, int maxSymbols, int maxKills,
                        RingBuffer<Report> out, GatewayClock clock, int maxMsgs, long windowNanos,
                        int dedupeWindow) {
        this.oms = new OmsCore(maxOrders, maxSymbols);
        this.risk = new PreTradeRisk(oms, maxSymbols, maxMsgs, windowNanos, dedupeWindow);
        this.out = out;
        this.clock = clock;
        this.cancelScratch = new long[maxOrders];
        KillListener l = this::onKillEvent;
        this.kill = new KillSwitch(symbols, maxKills, maxOrders, l);
    }

    public OmsCore oms() {
        return oms;
    }

    public PreTradeRisk risk() {
        return risk;
    }

    public KillSwitch killSwitch() {
        return kill;
    }

    @Override
    public void onEvent(Command c, long sequence, boolean endOfBatch) {
        now = clock.nanos();
        originNanos = c.originNanos;
        switch (c.type) {
            case Command.NEW -> onNew(c);
            case Command.CANCEL -> onCancel(c);
            case Command.VENUE_ACK -> exec(c.clOrdId, oms.ack(c.clOrdId), 0, 0);
            case Command.VENUE_FILL -> {
                int st = oms.fill(c.clOrdId, c.qty, c.px);
                exec(c.clOrdId, st, c.qty, c.px);
                if (st == OrderStatus.FILLED) {
                    terminal(c.clOrdId);
                }
            }
            case Command.VENUE_CANCELED -> {
                int st = oms.canceled(c.clOrdId);
                exec(c.clOrdId, st, 0, 0);
                if (st == OrderStatus.CANCELED) {
                    terminal(c.clOrdId);
                }
            }
            case Command.VENUE_REJECT -> {
                int st = oms.reject(c.clOrdId);
                exec(c.clOrdId, st, 0, 0);
                if (st == OrderStatus.REJECTED) {
                    terminal(c.clOrdId);
                }
            }
            case Command.KILL_ENGAGE -> {
                kill.engage(c.path, c.actor, c.reason, cancelScratch);
                int n = kill.lastCancelCount();
                for (int i = 0; i < n; i++) {
                    sendCancel(cancelScratch[i]);
                }
            }
            case Command.KILL_REQUEST_RESET -> kill.requestReset(c.killId, c.actor);
            case Command.KILL_APPROVE_RESET -> kill.approveReset(c.killId, c.actor);
            case Command.DISCONNECT -> onDisconnect();
            default -> throw new IllegalArgumentException("unknown command type " + c.type);
        }
    }

    private void onNew(Command c) {
        if (!risk.admitMessage(now)) {
            reject(c.clOrdId, PreTradeRisk.THROTTLE);
            return;
        }
        if (kill.check(c.path) != 0) {
            reject(c.clOrdId, PreTradeRisk.KILLED);
            return;
        }
        int code = risk.checkNew(c.clOrdId, c.path, c.side, c.qty, c.px);
        if (code != PreTradeRisk.OK) {
            reject(c.clOrdId, code);
            return;
        }
        oms.add(c.clOrdId, c.path, c.side, c.qty, c.px);
        kill.onOrderWorking(c.clOrdId, c.path);
        Report r = claim();
        r.type = Report.SEND_NEW;
        r.clOrdId = c.clOrdId;
        r.qty = c.qty;
        r.px = c.px;
        r.status = OrderStatus.PENDING_NEW;
        publish(r);
    }

    private void onCancel(Command c) {
        if (!risk.admitMessage(now)) {
            reject(c.clOrdId, PreTradeRisk.THROTTLE);
            return;
        }
        if (!oms.isOpen(c.clOrdId)) {
            reject(c.clOrdId, PreTradeRisk.UNKNOWN_ORDER);
            return;
        }
        sendCancel(c.clOrdId);
    }

    /** Cancels from a kill bypass the throttle: stopping risk is never rate-limited. */
    private void sendCancel(long id) {
        if (oms.requestCancel(id) == OrderStatus.INVALID) {
            return; // already pending cancel
        }
        Report r = claim();
        r.type = Report.SEND_CANCEL;
        r.clOrdId = id;
        r.status = OrderStatus.PENDING_CANCEL;
        publish(r);
    }

    private void onDisconnect() {
        int n = oms.openIds(cancelScratch);
        int[][] paths = new int[n][4];
        for (int i = 0; i < n; i++) {
            oms.pathOf(cancelScratch[i], paths[i]);
        }
        kill.cancelOnDisconnect(paths);
        n = oms.openIds(cancelScratch);
        io.quantportfolio.ull.util.Sorts.sort(cancelScratch, n);
        for (int i = 0; i < n; i++) {
            if (oms.statusOf(cancelScratch[i]) != OrderStatus.PENDING_CANCEL
                    && oms.pathOf(cancelScratch[i], pathScratch) && kill.check(pathScratch) != 0) {
                sendCancel(cancelScratch[i]);
            }
        }
    }

    private void terminal(long id) {
        kill.onOrderTerminal(id);
        risk.onTerminal(id);
    }

    private void exec(long id, int st, long q, long p) {
        Report r = claim();
        r.type = Report.EXEC;
        r.clOrdId = id;
        r.status = st;
        r.qty = q;
        r.px = p;
        r.cum = oms.lastCum();
        r.leaves = oms.lastLeaves();
        publish(r);
    }

    private void reject(long id, int code) {
        Report r = claim();
        r.type = Report.REJECT;
        r.clOrdId = id;
        r.rejectCode = code;
        r.status = OrderStatus.REJECTED;
        publish(r);
    }

    private void onKillEvent(int seq, int event, int killId, int level, int[] scope, int actor, int state,
                             long[] cancel, int nCancel, int reason) {
        Report r = claim();
        r.type = Report.KILL_EVENT;
        r.killId = killId;
        r.killEvent = event;
        r.killState = state;
        publish(r);
    }

    private long claimed;

    private Report claim() {
        claimed = out.next();
        Report r = out.get(claimed);
        r.clOrdId = 0;
        r.rejectCode = 0;
        r.status = 0;
        r.qty = 0;
        r.px = 0;
        r.cum = 0;
        r.leaves = 0;
        r.killId = 0;
        return r;
    }

    private void publish(Report r) {
        r.originNanos = originNanos;
        r.gatewayNanos = now;
        r.clockErrorNanos = clock.errorBoundNanos();
        r.clockSource = clock.source();
        out.publish(claimed);
    }
}
