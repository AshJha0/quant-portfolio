package io.quantportfolio.ull;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.quantportfolio.ull.gateway.Command;
import io.quantportfolio.ull.gateway.GatewayClock;
import io.quantportfolio.ull.gateway.OrderGateway;
import io.quantportfolio.ull.gateway.Report;
import io.quantportfolio.ull.kill.KillSwitch;
import io.quantportfolio.ull.oms.OrderStatus;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.ring.Sequence;
import io.quantportfolio.ull.risk.PreTradeRisk;
import io.quantportfolio.ull.util.SymbolTable;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class GatewayTest {
    SymbolTable sym;
    RingBuffer<Report> out;
    Sequence reader;
    OrderGateway gw;
    GatewayClock.Manual clock;
    Command cmd = new Command();
    int qp, rates, mm1, mm2, ust10, ust2, alice, bob;
    int[] p1, p2;

    /** A flattened copy of one report. */
    record R(int type, long id, int code, int status, long cum, long leaves, int killId, int killState) {}

    @BeforeEach
    void setUp() {
        sym = new SymbolTable();
        qp = sym.intern("QP");
        rates = sym.intern("RATES");
        mm1 = sym.intern("MM1");
        mm2 = sym.intern("MM2");
        ust10 = sym.intern("UST10Y");
        ust2 = sym.intern("UST2Y");
        alice = sym.intern("alice");
        bob = sym.intern("bob");
        p1 = new int[] {qp, rates, mm1, ust10};
        p2 = new int[] {qp, rates, mm2, ust2};
        out = new RingBuffer<>(Report::new, 1024);
        reader = new Sequence(-1);
        out.addGatingSequence(reader);
        clock = new GatewayClock.Manual();
        gw = new OrderGateway(sym, 256, 64, 16, out, clock, 1000, 1_000_000_000L, 1024);
        gw.risk().setSymbol(ust10, 10_000, 500, 1_000);
        gw.risk().setSymbol(ust2, 10_000, 500, 1_000);
    }

    List<R> run(Command c) {
        clock.now += 1_000;
        gw.onEvent(c, 0, true);
        List<R> l = new ArrayList<>();
        out.poll(reader, (r, s, e) -> l.add(new R(r.type, r.clOrdId, r.rejectCode, r.status, r.cum, r.leaves,
            r.killId, r.killState)));
        return l;
    }

    List<R> newOrder(long id, int[] p, int side, long q, long px) {
        return run(cmd.set(Command.NEW, id, p, side, q, px));
    }

    @Test
    void orderLifecycleKeepsCumPlusLeavesEqualQty() {
        assertEquals(List.of(new R(Report.SEND_NEW, 1, 0, OrderStatus.PENDING_NEW, 0, 0, 0, 0)),
            newOrder(1, p1, 1, 100, 10_000));
        assertEquals(OrderStatus.NEW, run(cmd.set(Command.VENUE_ACK, 1, null, 0, 0, 0)).get(0).status());
        R f = run(cmd.set(Command.VENUE_FILL, 1, null, 0, 40, 10_000)).get(0);
        assertEquals(OrderStatus.PARTIALLY_FILLED, f.status());
        assertEquals(40, f.cum());
        assertEquals(60, f.leaves());
        f = run(cmd.set(Command.VENUE_FILL, 1, null, 0, 60, 10_000)).get(0);
        assertEquals(OrderStatus.FILLED, f.status());
        assertEquals(100, gw.oms().position(ust10));
        assertEquals(0, gw.oms().openCount());
        // Overfill is refused.
        newOrder(2, p1, 1, 10, 10_000);
        assertEquals(OrderStatus.INVALID, run(cmd.set(Command.VENUE_FILL, 2, null, 0, 11, 10_000)).get(0).status());
    }

    @Test
    void killStopsSendsCancelsAllAndNeedsTwoPeople() {
        newOrder(7, p1, 1, 10, 10_000);
        newOrder(8, p2, -1, 10, 10_010);
        newOrder(9, new int[] {qp, sym.intern("FX"), mm1, ust10}, 1, 10, 10_000);
        Command k = cmd.set(Command.KILL_ENGAGE, 0, new int[] {qp, rates, 0, 0}, 0, 0, 0);
        k.actor = alice;
        k.reason = sym.intern("desk loss limit");
        List<R> r = run(k);
        assertEquals(Report.KILL_EVENT, r.get(0).type());
        assertEquals(KillSwitch.ENGAGED, r.get(0).killState());
        assertEquals(List.of(7L, 8L), r.stream().filter(x -> x.type() == Report.SEND_CANCEL).map(R::id).toList(),
            "mass cancel covers the desk only, ascending ids");
        // New orders on the desk are rejected; the FX desk still trades.
        assertEquals(PreTradeRisk.KILLED, newOrder(10, p2, 1, 1, 10_000).get(0).code());
        assertEquals(Report.SEND_NEW, newOrder(11, new int[] {qp, sym.intern("FX"), mm1, ust2}, 1, 1, 10_000)
            .get(0).type());
        // A fill racing the cancel is booked and counts as terminal.
        run(cmd.set(Command.VENUE_FILL, 7, null, 0, 10, 10_000));
        List<R> last = run(cmd.set(Command.VENUE_CANCELED, 8, null, 0, 0, 0));
        assertEquals(KillSwitch.KILLED, last.get(last.size() - 1).killState(), "all cancels acknowledged");
        assertEquals(10, gw.oms().position(ust10));
        Command rr = cmd.set(Command.KILL_REQUEST_RESET, 0, null, 0, 0, 0);
        rr.killId = 1;
        rr.actor = alice;
        run(rr);
        Command ap = cmd.set(Command.KILL_APPROVE_RESET, 0, null, 0, 0, 0);
        ap.killId = 1;
        ap.actor = alice;
        assertEquals(KillSwitch.RESET_PENDING, run(ap).get(0).killState(), "same person cannot approve");
        ap.actor = bob;
        assertEquals(KillSwitch.CLEARED, run(ap).get(0).killState());
        assertEquals(Report.SEND_NEW, newOrder(12, p2, 1, 1, 10_000).get(0).type());
    }

    @Test
    void disconnectCancelsEverythingOnTheSession() {
        newOrder(1, p1, 1, 10, 10_000);
        newOrder(2, p2, 1, 10, 10_000);
        List<R> r = run(cmd.set(Command.DISCONNECT, 0, null, 0, 0, 0));
        assertEquals(List.of(1L, 2L), r.stream().filter(x -> x.type() == Report.SEND_CANCEL).map(R::id).toList());
        assertEquals(2, r.stream().filter(x -> x.type() == Report.KILL_EVENT).count(), "one STRATEGY kill each");
        assertEquals(PreTradeRisk.KILLED, newOrder(3, p1, 1, 1, 10_000).get(0).code());
    }

    @Test
    void riskRejectsEachCheck() {
        gw.risk().maxOrderQty = 500;
        gw.risk().maxOrderNotional = 4_000_000;
        assertEquals(PreTradeRisk.FAT_FINGER, newOrder(1, p1, 1, 501, 10_000).get(0).code());
        assertEquals(PreTradeRisk.NOTIONAL, newOrder(2, p1, 1, 450, 10_000).get(0).code());
        assertEquals(PreTradeRisk.PRICE_BAND, newOrder(3, p1, 1, 10, 10_501).get(0).code());
        assertEquals(Report.SEND_NEW, newOrder(4, p1, 1, 10, 10_500).get(0).type(), "band edge passes");
        assertEquals(PreTradeRisk.DUPLICATE, newOrder(4, p1, 1, 10, 10_000).get(0).code());
        assertEquals(PreTradeRisk.SELF_TRADE, newOrder(5, p1, -1, 10, 10_400).get(0).code());
        assertEquals(Report.SEND_NEW, newOrder(6, p2, -1, 10, 10_000).get(0).type(),
            "a sell on another symbol does not self-trade");
        gw.risk().setRestricted(ust2, true);
        assertEquals(PreTradeRisk.RESTRICTED, newOrder(7, p2, 1, 1, 10_000).get(0).code());
        gw.risk().maxOrderQty = 10_000;
        gw.risk().maxOrderNotional = Long.MAX_VALUE / 4;
        // Position limit counts working orders: 10 working buy on UST10Y, limit 1,000.
        assertEquals(PreTradeRisk.POSITION, newOrder(8, p1, 1, 991, 10_000).get(0).code());
        gw.risk().creditLimit = gw.oms().workingNotional() + 10 * 10_000 - 1;
        assertEquals(PreTradeRisk.CREDIT, newOrder(9, p1, 1, 10, 10_000).get(0).code());
        // Duplicate after terminal: cancel order 4 then resend id 4.
        run(cmd.set(Command.VENUE_CANCELED, 4, null, 0, 0, 0));
        assertEquals(PreTradeRisk.DUPLICATE, newOrder(4, p1, 1, 1, 10_000).get(0).code());
    }

    @Test
    void throttleRejectsAndDoesNotQueue() {
        OrderGateway g = new OrderGateway(sym, 64, 64, 4, out, clock, 3, 1_000_000L, 64);
        gw = g;
        long t0 = clock.now;
        for (int i = 1; i <= 3; i++) {
            assertEquals(Report.SEND_NEW, newOrder(i, p1, 1, 1, 10_000).get(0).type());
        }
        assertEquals(PreTradeRisk.THROTTLE, newOrder(4, p1, 1, 1, 10_000).get(0).code());
        clock.now = t0 + 1_000_000 + 1_000; // first message leaves the window
        assertEquals(Report.SEND_NEW, newOrder(5, p1, 1, 1, 10_000).get(0).type());
    }

    @Test
    void reportsCarryClockSourceAndBound() {
        clock.now = 5;
        gw.onEvent(cmd.set(Command.NEW, 1, p1, 1, 1, 10_000), 0, true);
        Report r = out.get(out.cursor().get());
        assertEquals(GatewayClock.SOURCE_MANUAL, r.clockSource);
        assertEquals(0, r.clockErrorNanos);
        assertEquals(5, r.gatewayNanos);
        assertTrue(GatewayClock.monotonic().errorBoundNanos() > 0);
    }

    /**
     * Property test: random new/ack/fill/cancel/kill sequences never break the
     * OMS invariants, and working exposure never passes the position limit.
     */
    @Test
    void randomSequencesHoldInvariants() {
        for (long seed = 1; seed <= 200; seed++) {
            setUp();
            Random rnd = new Random(seed);
            long nextId = 1;
            List<Long> live = new ArrayList<>();
            for (int step = 0; step < 400; step++) {
                int op = rnd.nextInt(10);
                if (op < 4) {
                    int[] p = rnd.nextBoolean() ? p1 : p2;
                    int side = rnd.nextBoolean() ? 1 : -1;
                    long px = 9_600 + rnd.nextInt(800);
                    List<R> r = newOrder(nextId, p, side, 1 + rnd.nextInt(300), px);
                    if (r.get(0).type() == Report.SEND_NEW) {
                        live.add(nextId);
                    }
                    nextId++;
                } else if (!live.isEmpty()) {
                    long id = live.get(rnd.nextInt(live.size()));
                    int kind = switch (op) {
                        case 4 -> Command.VENUE_ACK;
                        case 5, 6 -> Command.VENUE_FILL;
                        case 7 -> Command.CANCEL;
                        case 8 -> Command.VENUE_CANCELED;
                        default -> Command.VENUE_REJECT;
                    };
                    run(cmd.set(kind, id, null, 0, 1 + rnd.nextInt(200), 10_000));
                    if (!gw.oms().isOpen(id)) {
                        live.remove(id);
                    }
                }
                assertNull(gw.oms().checkInvariants(), "seed " + seed + " step " + step);
                for (int s : new int[] {ust10, ust2}) {
                    long pos = gw.oms().position(s);
                    assertTrue(pos + gw.oms().workingBuy(s) <= 1_000 && pos - gw.oms().workingSell(s) >= -1_000,
                        "worst-case exposure within limit, seed " + seed);
                }
            }
        }
    }
}
