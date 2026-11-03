"""Minimal Liberty reader: only what the front-end needs (pin direction, function, ff, timing arcs).
Parses only the requested cells (brace-balanced slice), not the whole library grammar."""
import re
TOK = re.compile(r'\\\n|"(?:[^"\\]|\\.)*"|[A-Za-z_0-9.\-+!\[\]\']+|[{}();:,]|\S')

class Group:
    def __init__(self, kind, args):
        self.kind, self.args, self.attrs, self.groups = kind, args, {}, []
    def sub(self, kind):
        return [g for g in self.groups if g.kind == kind]
    def __repr__(self):
        return f"Group({self.kind},{self.args})"

def _strip(s):
    return s[1:-1] if s.startswith('"') else s

def _parse(tokens, i, grp):
    while i < len(tokens):
        t = tokens[i]
        if t == '}':
            return i + 1
        if t == '\\\n' or t == ';':
            i += 1; continue
        name = t; i += 1
        if tokens[i] == ':':
            val = tokens[i + 1]; grp.attrs[name] = _strip(val); i += 2
            continue
        if tokens[i] == '(':
            j = i + 1; args = []
            while tokens[j] != ')':
                if tokens[j] not in (',', '\\\n'): args.append(_strip(tokens[j]))
                j += 1
            j += 1
            if j < len(tokens) and tokens[j] == '{':
                g = Group(name, args); grp.groups.append(g)
                i = _parse(tokens, j + 1, g)
            else:  # complex attribute e.g. values(...)
                grp.attrs[name] = args; i = j
            continue
        i += 1
    return i

def read_cells(path, names):
    text = open(path).read()
    cells = {}
    for n in names:
        m = re.search(r'\bcell\s*\(\s*"?%s"?\s*\)\s*\{' % re.escape(n), text)
        if not m:
            raise KeyError(f"cell {n} not in {path}")
        depth, k = 1, m.end()
        while depth:
            c = text[k]
            depth += (c == '{') - (c == '}'); k += 1
        toks = TOK.findall(text[m.end():k])
        g = Group('cell', [n]); _parse(toks, 0, g)
        cells[n] = _cell_info(g)
    return cells

def _table_mid(tg):
    rows = [[float(x) for x in r.split(',')] for r in tg.attrs['values']]
    r = rows[len(rows) // 2]
    return r[len(r) // 2]

def _cell_info(g):
    info = {'name': g.args[0], 'pins': {}, 'ff': None, 'arcs': []}
    for ff in g.sub('ff'):
        info['ff'] = {'state': ff.args[0], 'clocked_on': ff.attrs.get('clocked_on'), 'next_state': ff.attrs.get('next_state')}
    for p in g.sub('pin'):
        pin = p.args[0]
        info['pins'][pin] = {'dir': p.attrs.get('direction'), 'function': p.attrs.get('function')}
        for t in p.sub('timing'):
            rel = t.attrs.get('related_pin')
            if rel is None or t.attrs.get('timing_type', 'combinational') not in ('combinational', 'rising_edge', 'falling_edge', 'combinational_rise', 'combinational_fall'):
                continue
            cr = t.sub('cell_rise'); cf = t.sub('cell_fall')
            info['arcs'].append({'from': rel, 'to': pin, 'when': t.attrs.get('when'), 'sense': t.attrs.get('timing_sense'),
                                 'type': t.attrs.get('timing_type', 'combinational'),
                                 'rise_ns': _table_mid(cr[0]) if cr else None, 'fall_ns': _table_mid(cf[0]) if cf else None})
    return info
