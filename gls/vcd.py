"""Tiny VCD reader/writer (scalar + vector wires, 0/1/x/z)."""

def read(path, scope_path=None):
    """-> (timescale_str, {fullname_bit: [(t, v)]}). Vectors are split per bit 'name[i]'.
    scope_path: e.g. 'gcd_tb.gcd1' to restrict to that scope (non-recursive)."""
    ids, scope, ts = {}, [], None
    changes = {}
    with open(path) as f:
        toks = iter(f.read().split())
    t = 0
    for tok in toks:
        if tok == '$timescale':
            ts = next(toks); x = next(toks)
            if x != '$end': ts += x; next(toks)
        elif tok == '$scope':
            next(toks); scope.append(next(toks)); next(toks)
        elif tok == '$upscope':
            scope.pop(); next(toks)
        elif tok == '$var':
            typ, width, code, name = next(toks), int(next(toks)), next(toks), next(toks)
            nxt = next(toks)
            if nxt != '$end': next(toks)  # bit range
            sp = '.'.join(scope)
            if scope_path is None or sp == scope_path:
                ids.setdefault(code, []).append((name, width))
        elif tok in ('$dumpvars', '$end', '$dumpall', '$dumpon', '$dumpoff'):
            continue
        elif tok.startswith('$'):
            while next(toks) != '$end': pass
        elif tok.startswith('#'):
            t = int(tok[1:])
        elif tok[0] in 'bB':
            val = tok[1:]; code = next(toks)
            for name, w in ids.get(code, []):
                val2 = val.rjust(w, '0' if val[0] in '01' else val[0])
                for i in range(w):
                    changes.setdefault(f"{name}[{i}]", []).append((t, val2[w - 1 - i]))
        elif tok[0] in 'rR':
            next(toks)
        else:
            v, code = tok[0], tok[1:]
            for name, w in ids.get(code, []):
                changes.setdefault(name if w == 1 else f"{name}[0]", []).append((t, v))
    return ts, changes

def write(path, timescale, names, events, end_time):
    """names: list of signal names; events: list of (t, idx, v) sorted by t."""
    def code(i):
        s = ''
        i += 1
        while i:
            i, r = divmod(i - 1, 94); s += chr(33 + r)
        return s
    with open(path, 'w') as f:
        f.write(f"$timescale {timescale} $end\n$scope module top $end\n")
        for i, n in enumerate(names):
            f.write(f"$var wire 1 {code(i)} {n.replace(' ', '_')} $end\n")
        f.write("$upscope $end\n$enddefinitions $end\n")
        cur = None
        for t, i, v in events:
            if t != cur: f.write(f"#{t}\n"); cur = t
            f.write(f"{v}{code(i)}\n")
        f.write(f"#{end_time}\n")
