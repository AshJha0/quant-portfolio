package io.quantportfolio.ull.bench;

import io.quantportfolio.ull.gateway.Command;
import io.quantportfolio.ull.gateway.GatewayClock;
import io.quantportfolio.ull.gateway.OrderGateway;
import io.quantportfolio.ull.gateway.Report;
import io.quantportfolio.ull.metrics.GcPauseTracker;
import io.quantportfolio.ull.metrics.LatencyHistogram;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.ring.Sequence;
import io.quantportfolio.ull.util.SymbolTable;
import java.lang.management.ManagementFactory;
import java.util.Locale;

/**
 * Three-thread, in-process order-path latency harness.
 *
 * <p><b>What it measures.</b> Software in-process latency, not wire-to-wire.
 * A client thread publishes a NewOrderSingle into the client ring at a fixed
 * rate; the gateway thread runs throttle, kill check, pre-trade risk and the
 * OMS insert and publishes {@code SEND_NEW} to the outbound ring; a venue
 * thread reads it, records latency, and answers with a full fill through a
 * second (venue) ring, so the OMS stays in steady state. Latency is measured
 * from the order's <i>intended</i> send time on the fixed schedule, not its
 * actual send time, so a stall that delays later sends is charged to them
 * (coordinated-omission correction, Tene). Two ring hops and two thread
 * hand-offs are included. Busy-spin waits; no thread pinning (the JDK has no
 * portable affinity API), so the OS may migrate threads.
 *
 * <p>Usage: {@code java -cp target/classes io.quantportfolio.ull.bench.LatencyMain [ratePerSec] [seconds]}.
 * Prints one JSON line with percentiles in microseconds, GC counts during the
 * measured window, and the JVM, OS and CPU count.
 */
public final class LatencyMain {
    private LatencyMain() {}

    public static void main(String[] args) throws Exception {
        long rate = args.length > 0 ? Long.parseLong(args[0]) : 100_000;
        int seconds = args.length > 1 ? Integer.parseInt(args[1]) : 10;
        long warmupOrders = rate * 3;
        long measuredOrders = rate * seconds;
        long intervalNanos = 1_000_000_000L / rate;

        SymbolTable sym = new SymbolTable();
        int ust = sym.intern("UST10Y");
        int[] path = {sym.intern("QP"), sym.intern("RATES"), sym.intern("MM1"), ust};

        RingBuffer<Command> clientRing = new RingBuffer<>(Command::new, 1 << 14);
        // Each pending fill belongs to an open order, and the OMS holds at most 2^16 open orders,
        // so a 2^17-slot venue ring can never be full: the venue thread never blocks, which rules
        // out the gateway/venue deadlock of two full rings.
        RingBuffer<Command> venueRing = new RingBuffer<>(Command::new, 1 << 17);
        RingBuffer<Report> outRing = new RingBuffer<>(Report::new, 1 << 14);
        GatewayClock clock = GatewayClock.monotonic();
        OrderGateway gw = new OrderGateway(sym, 1 << 16, 64, 16, outRing, clock, 1 << 20, 1L, 1 << 16);
        gw.risk().setSymbol(ust, 10_000, 500, Long.MAX_VALUE / 8);

        Sequence gwClientSeq = new Sequence(-1);
        Sequence gwVenueSeq = new Sequence(-1);
        Sequence venueOutSeq = new Sequence(-1);
        clientRing.addGatingSequence(gwClientSeq);
        venueRing.addGatingSequence(gwVenueSeq);
        outRing.addGatingSequence(venueOutSeq);

        LatencyHistogram hist = new LatencyHistogram();
        LatencyHistogram toGateway = new LatencyHistogram();
        long total = warmupOrders + measuredOrders;
        Sequence seen = new Sequence(0);
        Flag running = new Flag();
        Sequence rejects = new Sequence(0);

        Thread gateway = new Thread(() -> {
            while (running.on) {
                int n = clientRing.poll(gwClientSeq, gw) + venueRing.poll(gwVenueSeq, gw);
                if (n == 0) {
                    Thread.onSpinWait();
                }
            }
        }, "gateway");

        io.quantportfolio.ull.ring.EventHandler<Report> venueHandler = (r, s, e) -> {
                    if (r.type == Report.SEND_NEW) {
                        long lat = System.nanoTime() - r.originNanos;
                        if (r.clOrdId > warmupOrders) {
                            hist.record(lat);
                            toGateway.record(r.gatewayNanos - r.originNanos);
                        }
                        long vs = venueRing.next();
                        Command f = venueRing.get(vs);
                        f.type = Command.VENUE_FILL;
                        f.clOrdId = r.clOrdId;
                        f.qty = r.qty;
                        f.px = r.px;
                        f.originNanos = 0;
                        venueRing.publish(vs);
                        seen.set(r.clOrdId);
                    } else if (r.type == Report.REJECT) {
                        rejects.set(rejects.getPlain() + 1);
                        seen.set(r.clOrdId);
                    }
                };
        Thread venue = new Thread(() -> {
            while (running.on) {
                int n = outRing.poll(venueOutSeq, venueHandler);
                if (n == 0) {
                    Thread.onSpinWait();
                }
            }
        }, "venue");

        gateway.setDaemon(true);
        venue.setDaemon(true);
        gateway.start();
        venue.start();

        // Created before the clock starts: registering JMX listeners takes hundreds of ms and
        // would otherwise show up as a coordinated-omission-corrected stall.
        GcPauseTracker gc = new GcPauseTracker();
        long gcAtMeasureStart = 0;
        long start = System.nanoTime();
        for (long id = 1; id <= total; id++) {
            if (id == warmupOrders + 1) {
                gcAtMeasureStart = GcPauseTracker.collectionCount();
            }
            long intended = start + (id - 1) * intervalNanos;
            while (System.nanoTime() < intended) {
                Thread.onSpinWait();
            }
            long s = clientRing.next();
            Command c = clientRing.get(s);
            boolean buy = (id & 1) == 0;
            c.set(Command.NEW, id, path, buy ? 1 : -1, 100, buy ? 9_990 : 10_010);
            c.originNanos = intended;
            clientRing.publish(s);
        }
        while (venueOutSeq.get() < 0 || seen.get() < total) {
            Thread.onSpinWait();
        }
        long gcCount = GcPauseTracker.collectionCount() - gcAtMeasureStart;
        running.on = false;
        gateway.join(1000);
        venue.join(1000);

        String json = String.format(Locale.ROOT,
            "{\"label\":\"software in-process latency, not wire-to-wire; client->gateway->venue thread, "
                + "2 ring hops; coordinated-omission corrected\",\"unit\":\"microseconds\","
                + "\"rate_per_s\":%d,\"orders\":%d,\"p50\":%.2f,\"p90\":%.2f,\"p99\":%.2f,\"p99_9\":%.2f,"
                + "\"p99_99\":%.2f,\"to_gateway_p50\":%.2f,\"to_gateway_p99\":%.2f,\"max\":%.2f,\"gc_collections_in_window\":%d,\"gc_max_event_ms\":%d,"
                + "\"rejects\":%d,\"open_orders_at_end\":%d,\"jvm\":\"%s\",\"os\":\"%s\",\"cpus\":%d}",
            rate, hist.count(), us(hist.percentile(0.5)), us(hist.percentile(0.9)), us(hist.percentile(0.99)),
            us(hist.percentile(0.999)), us(hist.percentile(0.9999)), us(toGateway.percentile(0.5)), us(toGateway.percentile(0.99)), us(hist.max()), gcCount,
            gc.maxPauseMillis(), rejects.get(), gw.oms().openCount(),
            System.getProperty("java.vm.name") + " " + System.getProperty("java.version"),
            System.getProperty("os.name") + " " + System.getProperty("os.arch"),
            ManagementFactory.getOperatingSystemMXBean().getAvailableProcessors());
        System.out.println(json);
        gc.close();
    }

    private static double us(long nanos) {
        return nanos / 1_000.0;
    }

    private static final class Flag {
        volatile boolean on = true;
    }
}
