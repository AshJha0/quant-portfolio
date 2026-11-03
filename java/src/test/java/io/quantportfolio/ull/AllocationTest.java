package io.quantportfolio.ull;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.quantportfolio.ull.gateway.Command;
import io.quantportfolio.ull.gateway.GatewayClock;
import io.quantportfolio.ull.gateway.OrderGateway;
import io.quantportfolio.ull.gateway.Report;
import io.quantportfolio.ull.metrics.GcPauseTracker;
import io.quantportfolio.ull.metrics.LatencyHistogram;
import io.quantportfolio.ull.ring.EventHandler;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.ring.Sequence;
import io.quantportfolio.ull.util.SymbolTable;
import java.lang.management.ManagementFactory;
import org.junit.jupiter.api.Test;

/**
 * Zero-garbage proof: after warm-up, the steady-state order path allocates
 * exactly zero bytes on the gateway thread, measured with
 * {@code com.sun.management.ThreadMXBean#getCurrentThreadAllocatedBytes}
 * (HotSpot's per-thread TLAB accounting, exact to the byte for this thread).
 */
class AllocationTest {
    static final com.sun.management.ThreadMXBean THREADS =
        (com.sun.management.ThreadMXBean) ManagementFactory.getThreadMXBean();

    /** Drives NEW, ACK, partial fill, fill / cancel cycles plus kill checks and risk rejects. */
    static final class Driver {
        final SymbolTable sym = new SymbolTable();
        final RingBuffer<Report> out = new RingBuffer<>(Report::new, 4096);
        final Sequence reader = new Sequence(-1);
        final GatewayClock clock = GatewayClock.monotonic();
        final OrderGateway gw;
        final Command cmd = new Command();
        final int[] path;
        final int[] killedPath;
        long id = 1;
        long reports;
        final EventHandler<Report> sink = (r, s, e) -> reports += r.type;

        Driver() {
            out.addGatingSequence(reader);
            int qp = sym.intern("QP");
            int rates = sym.intern("RATES");
            int ust = sym.intern("UST10Y");
            path = new int[] {qp, rates, sym.intern("MM1"), ust};
            killedPath = new int[] {qp, rates, sym.intern("MM2"), ust};
            gw = new OrderGateway(sym, 1024, 64, 16, out, clock, 1 << 20, 1L, 1 << 16);
            gw.risk().setSymbol(ust, 10_000, 500, 1L << 40);
            Command k = new Command().set(Command.KILL_ENGAGE, 0, new int[] {qp, rates, killedPath[2], 0}, 0, 0, 0);
            k.actor = sym.intern("alice");
            k.reason = sym.intern("test");
            gw.onEvent(k, 0, true);
            drain();
        }

        void drain() {
            out.poll(reader, sink);
        }

        void cycle() {
            long a = id++;
            long b = id++;
            send(cmd.set(Command.NEW, a, path, 1, 100, 10_000));
            send(cmd.set(Command.NEW, b, path, -1, 50, 10_200));
            send(cmd.set(Command.VENUE_ACK, a, null, 0, 0, 0));
            send(cmd.set(Command.VENUE_ACK, b, null, 0, 0, 0));
            send(cmd.set(Command.VENUE_FILL, a, null, 0, 40, 10_000));
            send(cmd.set(Command.VENUE_FILL, a, null, 0, 60, 10_000));
            send(cmd.set(Command.CANCEL, b, null, 0, 0, 0));
            send(cmd.set(Command.VENUE_CANCELED, b, null, 0, 0, 0));
            send(cmd.set(Command.NEW, id++, killedPath, 1, 1, 10_000)); // rejected: KILLED
            send(cmd.set(Command.NEW, a, path, 1, 1, 10_000)); // rejected: DUPLICATE
            send(cmd.set(Command.NEW, id++, path, 1, 1, 99_000)); // rejected: PRICE_BAND
            drain();
        }

        void send(Command c) {
            c.originNanos = 0;
            gw.onEvent(c, 0, true);
        }
    }

    @Test
    void steadyStateOrderPathAllocatesNothing() {
        Driver d = new Driver();
        for (int i = 0; i < 200_000; i++) {
            d.cycle(); // warm-up: JIT compiles the path
        }
        assertEquals(0, d.gw.oms().openCount());
        try (GcPauseTracker gc = new GcPauseTracker()) {
            long before = THREADS.getCurrentThreadAllocatedBytes();
            for (int i = 0; i < 500_000; i++) {
                d.cycle();
            }
            long allocated = THREADS.getCurrentThreadAllocatedBytes() - before;
            assertEquals(0L, allocated, "bytes allocated over 5.5 million gateway events");
            assertEquals(0L, gc.collectionsSinceStart(), "no GC during the steady-state window");
        }
        assertTrue(d.reports > 0);
        assertEquals(null, d.gw.oms().checkInvariants());
    }

    /** Negative control: the meter does see allocation (so a zero above is not a dead meter). */
    @Test
    void meterDetectsAllocation() {
        long before = THREADS.getCurrentThreadAllocatedBytes();
        Object[] keep = new Object[1000];
        for (int i = 0; i < keep.length; i++) {
            keep[i] = new long[16];
        }
        long allocated = THREADS.getCurrentThreadAllocatedBytes() - before;
        assertTrue(allocated >= 1000 * 16 * 8, "meter saw " + allocated + " bytes");
        assertEquals(1000, keep.length);
    }

    @Test
    void ringAndHistogramAllocateNothing() {
        RingBuffer<Report> ring = new RingBuffer<>(Report::new, 1024);
        Sequence c = new Sequence(-1);
        ring.addGatingSequence(c);
        LatencyHistogram h = new LatencyHistogram();
        long[] sum = new long[1];
        EventHandler<Report> handler = (r, s, e) -> {
            sum[0] += r.clOrdId;
            h.record(r.clOrdId & 0xFFFF);
        };
        for (int round = 0; round < 2; round++) {
            long before = THREADS.getCurrentThreadAllocatedBytes();
            for (long i = 0; i < 2_000_000; i++) {
                long s = ring.next();
                ring.get(s).clOrdId = i;
                ring.publish(s);
                if ((i & 255) == 255) {
                    ring.poll(c, handler);
                }
            }
            ring.poll(c, handler);
            long allocated = THREADS.getCurrentThreadAllocatedBytes() - before;
            if (round == 1) {
                assertEquals(0L, allocated, "bytes allocated by 2 million publish/poll after warm-up");
            }
        }
        assertTrue(h.count() > 0 && sum[0] > 0);
    }
}
