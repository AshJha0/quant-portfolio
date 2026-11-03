package io.quantportfolio.ull.bench;

import io.quantportfolio.ull.gateway.Command;
import io.quantportfolio.ull.gateway.GatewayClock;
import io.quantportfolio.ull.gateway.OrderGateway;
import io.quantportfolio.ull.gateway.Report;
import io.quantportfolio.ull.ring.EventHandler;
import io.quantportfolio.ull.ring.RingBuffer;
import io.quantportfolio.ull.ring.Sequence;
import io.quantportfolio.ull.util.SymbolTable;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.Benchmark;
import org.openjdk.jmh.annotations.BenchmarkMode;
import org.openjdk.jmh.annotations.Fork;
import org.openjdk.jmh.annotations.Level;
import org.openjdk.jmh.annotations.Measurement;
import org.openjdk.jmh.annotations.Mode;
import org.openjdk.jmh.annotations.OutputTimeUnit;
import org.openjdk.jmh.annotations.Scope;
import org.openjdk.jmh.annotations.Setup;
import org.openjdk.jmh.annotations.State;
import org.openjdk.jmh.annotations.Warmup;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmarks of the single-threaded hot path, in microseconds per
 * operation. These measure software in-process cost on one thread: no ring
 * hand-off between threads (see {@link LatencyMain} for that), no network.
 *
 * <ul>
 *   <li>{@code orderRoundTrip}: NEW (throttle, kill check, risk, OMS insert,
 *       SEND_NEW), venue ACK, full FILL (terminal, kill and dedupe
 *       bookkeeping); 3 gateway events and 3 reports.</li>
 *   <li>{@code killCheck}: the K1 test against 8 active kills.</li>
 *   <li>{@code ringPublishPoll}: one publish and one poll on the same thread.</li>
 * </ul>
 * Run: {@code java -jar target/benchmarks.jar -prof gc} ({@code gc.alloc.rate.norm}
 * should read about 0 B/op for all three).
 */
@State(Scope.Thread)
@BenchmarkMode({Mode.AverageTime, Mode.SampleTime})
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Warmup(iterations = 3, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(value = 1, jvmArgsAppend = {"-Xms1g", "-Xmx1g", "-XX:+UseG1GC"})
public class GatewayBenchmarks {
    private OrderGateway gw;
    private RingBuffer<Report> out;
    private Sequence reader;
    private final Command cmd = new Command();
    private int[] path;
    private int[] probe;
    private long id;
    private long sink;
    private final EventHandler<Report> drain = (r, s, e) -> sink += r.type;

    private RingBuffer<Command> ring;
    private Sequence ringReader;
    private final EventHandler<Command> ringHandler = (c, s, e) -> sink += c.clOrdId;

    @Setup(Level.Trial)
    public void setUp() {
        SymbolTable sym = new SymbolTable();
        int ust = sym.intern("UST10Y");
        int qp = sym.intern("QP");
        path = new int[] {qp, sym.intern("RATES"), sym.intern("MM1"), ust};
        probe = new int[] {qp, sym.intern("RATES"), sym.intern("MM9"), ust};
        out = new RingBuffer<>(Report::new, 4096);
        reader = new Sequence(-1);
        out.addGatingSequence(reader);
        gw = new OrderGateway(sym, 1 << 12, 64, 16, out, GatewayClock.monotonic(), 1 << 20, 1L, 1 << 16);
        gw.risk().setSymbol(ust, 10_000, 500, Long.MAX_VALUE / 8);
        for (int k = 0; k < 8; k++) {
            Command e = new Command().set(Command.KILL_ENGAGE, 0,
                new int[] {qp, sym.intern("DESK" + k), 0, 0}, 0, 0, 0);
            e.actor = sym.intern("alice");
            e.reason = sym.intern("bench");
            gw.onEvent(e, 0, true);
        }
        out.poll(reader, drain);
        ring = new RingBuffer<>(Command::new, 1024);
        ringReader = new Sequence(-1);
        ring.addGatingSequence(ringReader);
    }

    @Benchmark
    public long orderRoundTrip() {
        long a = ++id;
        gw.onEvent(cmd.set(Command.NEW, a, path, (a & 1) == 0 ? 1 : -1, 100, (a & 1) == 0 ? 9_990 : 10_010), 0, true);
        gw.onEvent(cmd.set(Command.VENUE_ACK, a, null, 0, 0, 0), 0, true);
        gw.onEvent(cmd.set(Command.VENUE_FILL, a, null, 0, 100, 10_000), 0, true);
        out.poll(reader, drain);
        return sink;
    }

    @Benchmark
    public int killCheck() {
        return gw.killSwitch().check(probe);
    }

    @Benchmark
    public void ringPublishPoll(Blackhole bh) {
        long s = ring.next();
        ring.get(s).clOrdId = s;
        ring.publish(s);
        bh.consume(ring.poll(ringReader, ringHandler));
    }
}
