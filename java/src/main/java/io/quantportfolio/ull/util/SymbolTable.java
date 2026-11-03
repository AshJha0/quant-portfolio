package io.quantportfolio.ull.util;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Interns strings (firm, desk, strategy, symbol, actor names) to small ints at
 * session set-up, so the hot path compares ints and never touches a String.
 * Id 0 is always the wildcard {@code "*"}. Interning allocates; it is a
 * configuration-time operation, not a hot-path one.
 */
public final class SymbolTable {
    public static final int WILD = 0;
    private final Map<String, Integer> ids = new HashMap<>();
    private final List<String> names = new ArrayList<>();

    public SymbolTable() {
        intern("*");
    }

    public synchronized int intern(String s) {
        Integer id = ids.get(s);
        if (id != null) {
            return id;
        }
        int n = names.size();
        names.add(s);
        ids.put(s, n);
        return n;
    }

    public synchronized String name(int id) {
        return names.get(id);
    }

    public synchronized int size() {
        return names.size();
    }
}
