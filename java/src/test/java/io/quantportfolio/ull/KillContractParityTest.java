package io.quantportfolio.ull;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.quantportfolio.ull.kill.KillSwitch;
import io.quantportfolio.ull.util.SymbolTable;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.DynamicTest;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestFactory;

/**
 * Replays the shared golden vectors ({@code python/oms/kill/vectors.json},
 * written by the Python reference) and requires byte-identical wire events.
 */
class KillContractParityTest {

    static final Path VECTORS = Path.of("..", "python", "oms", "kill", "vectors.json");

    /** Runs one command list the way {@code qp_oms.kill.contract.replay} does. */
    static List<String> replay(List<Object> commands) {
        SymbolTable sym = new SymbolTable();
        List<String> out = new ArrayList<>();
        KillSwitch[] holder = new KillSwitch[1];
        KillSwitch ks = new KillSwitch(sym, 64, 1024,
            (seq, ev, id, lvl, sc, actor, st, cancel, n, reason) ->
                out.add(holder[0].toWire(seq, ev, id, lvl, sc, actor, st, cancel, n, reason)));
        holder[0] = ks;
        for (Object o : commands) {
            @SuppressWarnings("unchecked")
            Map<String, Object> c = (Map<String, Object>) o;
            String op = (String) c.get("op");
            switch (op) {
                case "working" -> ks.onOrderWorking((Long) c.get("id"), path(sym, c.get("path")));
                case "terminal" -> ks.onOrderTerminal((Long) c.get("id"));
                case "engage" -> ks.engage(path(sym, c.get("scope")), sym.intern((String) c.get("actor")),
                    sym.intern((String) c.getOrDefault("reason", "")), null);
                case "disconnect" -> {
                    @SuppressWarnings("unchecked")
                    List<Object> ps = (List<Object>) c.get("paths");
                    int[][] paths = new int[ps.size()][];
                    for (int i = 0; i < paths.length; i++) {
                        paths[i] = path(sym, ps.get(i));
                    }
                    ks.cancelOnDisconnect(paths);
                }
                case "request_reset" -> ks.requestReset(((Long) c.get("kill_id")).intValue(),
                    sym.intern((String) c.get("actor")));
                case "approve_reset" -> ks.approveReset(((Long) c.get("kill_id")).intValue(),
                    sym.intern((String) c.get("actor")));
                case "check" -> {
                    int[] p = path(sym, c.get("path"));
                    int k = ks.check(p);
                    StringBuilder b = new StringBuilder("{\"check\":[");
                    @SuppressWarnings("unchecked")
                    List<Object> raw = (List<Object>) c.get("path");
                    for (int i = 0; i < 4; i++) {
                        if (i > 0) {
                            b.append(',');
                        }
                        KillSwitch.jsonString(b, (String) raw.get(i));
                    }
                    b.append("],\"result\":");
                    if (k == 0) {
                        b.append("null");
                    } else {
                        KillSwitch.jsonString(b, ks.rejectReason(k));
                    }
                    out.add(b.append('}').toString());
                }
                default -> throw new IllegalArgumentException("unknown op " + op);
            }
        }
        return out;
    }

    static int[] path(SymbolTable sym, Object o) {
        @SuppressWarnings("unchecked")
        List<Object> l = (List<Object>) o;
        int[] p = new int[4];
        for (int i = 0; i < 4; i++) {
            p[i] = sym.intern((String) l.get(i));
        }
        return p;
    }

    @TestFactory
    List<DynamicTest> goldenVectors() throws IOException {
        assertTrue(Files.exists(VECTORS), "golden vectors missing: " + VECTORS.toAbsolutePath());
        @SuppressWarnings("unchecked")
        Map<String, Object> doc = (Map<String, Object>) MiniJson.parse(
            Files.readString(VECTORS, StandardCharsets.UTF_8));
        assertEquals(1L, doc.get("version"), "contract version changed: review KillSwitch");
        @SuppressWarnings("unchecked")
        List<Object> cases = (List<Object>) doc.get("cases");
        assertFalse(cases.isEmpty());
        List<DynamicTest> tests = new ArrayList<>();
        for (Object o : cases) {
            @SuppressWarnings("unchecked")
            Map<String, Object> cs = (Map<String, Object>) o;
            tests.add(DynamicTest.dynamicTest((String) cs.get("name"), () -> {
                @SuppressWarnings("unchecked")
                List<Object> cmds = (List<Object>) cs.get("commands");
                assertEquals(cs.get("expected"), replay(cmds));
            }));
        }
        return tests;
    }

    @Test
    void invalidScopesAreRejected() {
        SymbolTable sym = new SymbolTable();
        KillSwitch ks = new KillSwitch(sym, 8, 8, io.quantportfolio.ull.kill.KillListener.NONE);
        int qp = sym.intern("QP");
        int mm = sym.intern("MM1");
        assertThrows(IllegalArgumentException.class,
            () -> ks.engage(new int[] {qp, SymbolTable.WILD, mm, SymbolTable.WILD}, 1, -1, null));
        assertThrows(IllegalArgumentException.class,
            () -> ks.engage(new int[] {SymbolTable.WILD, SymbolTable.WILD, SymbolTable.WILD, SymbolTable.WILD},
                1, -1, null));
    }

    @Test
    void fillRacingCancelCountsAsTerminalAndNestingSurvivesWiderClear() {
        SymbolTable sym = new SymbolTable();
        KillSwitch ks = new KillSwitch(sym, 8, 8, io.quantportfolio.ull.kill.KillListener.NONE);
        int qp = sym.intern("QP");
        int rates = sym.intern("RATES");
        int mm1 = sym.intern("MM1");
        int ust = sym.intern("UST10Y");
        int alice = sym.intern("alice");
        int bob = sym.intern("bob");
        int[] path = {qp, rates, mm1, ust};
        ks.onOrderWorking(7, path);
        long[] out = new long[8];
        int strat = ks.engage(new int[] {qp, rates, mm1, 0}, alice, -1, out);
        assertEquals(1, ks.lastCancelCount());
        int desk = ks.engage(new int[] {qp, rates, 0, 0}, alice, -1, out);
        assertEquals(1, ks.lastCancelCount(), "order 7 is covered by both kills");
        ks.onOrderTerminal(7); // a fill raced the cancel: still terminal (K3)
        assertEquals(KillSwitch.KILLED, ks.state(strat));
        assertEquals(KillSwitch.KILLED, ks.state(desk));
        assertEquals(desk, ks.check(path), "widest kill named");
        assertTrue(ks.requestReset(desk, alice));
        assertTrue(ks.approveReset(desk, bob));
        assertEquals(strat, ks.check(path), "K5: the narrower kill stays active");
    }
}
