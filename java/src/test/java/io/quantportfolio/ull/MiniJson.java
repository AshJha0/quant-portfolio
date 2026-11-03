package io.quantportfolio.ull;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Minimal JSON reader for test vectors (objects, arrays, strings, integers, true/false/null). */
final class MiniJson {
    private final String s;
    private int i;

    private MiniJson(String s) {
        this.s = s;
    }

    static Object parse(String s) {
        MiniJson p = new MiniJson(s);
        Object v = p.value();
        p.ws();
        if (p.i != s.length()) {
            throw new IllegalArgumentException("trailing data at " + p.i);
        }
        return v;
    }

    private void ws() {
        while (i < s.length() && Character.isWhitespace(s.charAt(i))) {
            i++;
        }
    }

    private Object value() {
        ws();
        char c = s.charAt(i);
        switch (c) {
            case '{': {
                i++;
                Map<String, Object> m = new LinkedHashMap<>();
                ws();
                if (s.charAt(i) == '}') {
                    i++;
                    return m;
                }
                while (true) {
                    ws();
                    String k = string();
                    ws();
                    expect(':');
                    m.put(k, value());
                    ws();
                    if (s.charAt(i) == ',') {
                        i++;
                    } else {
                        expect('}');
                        return m;
                    }
                }
            }
            case '[': {
                i++;
                List<Object> l = new ArrayList<>();
                ws();
                if (s.charAt(i) == ']') {
                    i++;
                    return l;
                }
                while (true) {
                    l.add(value());
                    ws();
                    if (s.charAt(i) == ',') {
                        i++;
                    } else {
                        expect(']');
                        return l;
                    }
                }
            }
            case '"':
                return string();
            case 'n':
                i += 4;
                return null;
            case 't':
                i += 4;
                return Boolean.TRUE;
            case 'f':
                i += 5;
                return Boolean.FALSE;
            default: {
                int st = i;
                while (i < s.length() && "-0123456789".indexOf(s.charAt(i)) >= 0) {
                    i++;
                }
                return Long.parseLong(s.substring(st, i));
            }
        }
    }

    private void expect(char c) {
        if (s.charAt(i) != c) {
            throw new IllegalArgumentException("expected " + c + " at " + i);
        }
        i++;
    }

    private String string() {
        expect('"');
        StringBuilder b = new StringBuilder();
        while (true) {
            char c = s.charAt(i++);
            if (c == '"') {
                return b.toString();
            }
            if (c == '\\') {
                char e = s.charAt(i++);
                switch (e) {
                    case 'n' -> b.append('\n');
                    case 't' -> b.append('\t');
                    case 'r' -> b.append('\r');
                    case 'b' -> b.append('\b');
                    case 'f' -> b.append('\f');
                    case 'u' -> {
                        b.append((char) Integer.parseInt(s.substring(i, i + 4), 16));
                        i += 4;
                    }
                    default -> b.append(e);
                }
            } else {
                b.append(c);
            }
        }
    }
}
