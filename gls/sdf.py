"""SDF subset: CELL / INSTANCE / ABSOLUTE IOPATH (rise fall), optional COND.
writer: derives delays from Liberty arcs at the middle point of each NLDM table (NOT signoff STA; no SPEF/slew propagation).
uniform=True collapses every arc into one (max) rise/fall per output pin so it is exactly expressible as a
Verilog `assign #(rise,fall)` for Icarus cross-validation."""
import re

def write(design, path, uniform=False, design_name='top'):
    L = [f'(DELAYFILE\n (SDFVERSION "3.0")\n (DESIGN "{design_name}")\n (TIMESCALE 1ps)']
    for c in design.cells:
        arcs = c.info['arcs']
        if not arcs: continue
        entries = []
        if uniform:
            by_out = {}
            for a in arcs:
                r, f = by_out.get(a['to'], (0, 0))
                by_out[a['to']] = (max(r, a['rise_ns'] or 0), max(f, a['fall_ns'] or 0))
            for a in arcs:
                r, f = by_out[a['to']]
                frm = f"(posedge {a['from']})" if a['type'] == 'rising_edge' else a['from']
                entries.append((None, frm, a['to'], r, f))
            entries = list(dict.fromkeys(entries))
        else:
            # Liberty may hold several arcs for one (when, from, to) (e.g. xor: positive_unate + negative_unate).
            # SDF IOPATH cannot distinguish them without COND, so we keep the max (conservative) per key.
            merged = {}
            for a in arcs:
                frm = f"(posedge {a['from']})" if a['type'] == 'rising_edge' else a['from']
                k = (a['when'], frm, a['to'])
                r, f = merged.get(k, (0, 0))
                merged[k] = (max(r, a['rise_ns'] or 0), max(f, a['fall_ns'] or 0))
            entries = [(*k, r, f) for k, (r, f) in merged.items()]
        L.append(f' (CELL (CELLTYPE "{c.type}") (INSTANCE {c.name})\n  (DELAY (ABSOLUTE')
        for when, frm, to, r, f in entries:
            rp, fp = round(r * 1000), round(f * 1000)
            io = f'(IOPATH {frm} {to} ({rp}) ({fp}))'
            L.append(f'   (COND {when} {io})' if when else f'   {io}')
        L.append('  ))\n )')
    L.append(')')
    open(path, 'w').write('\n'.join(L) + '\n')

_CELL = re.compile(r'\(CELL\s*\(CELLTYPE\s*"([^"]+)"\)\s*\(INSTANCE\s*([^)\s]*)\)(.*?)\n \)', re.S)
_IOP = re.compile(r'(?:\(COND\s+(.+?)\s+)?\(IOPATH\s+(\(posedge\s+\w+\)|\w+)\s+(\w+)\s+\(([-\d.:]+)\)\s+\(([-\d.:]+)\)\)')

def _v(s):
    return float(s.split(':')[1] if ':' in s else s)

def parse(path):
    """-> {instance: [ {cond, from, edge, to, rise_ps, fall_ps} ]}"""
    txt = open(path).read()
    ts = re.search(r'\(TIMESCALE\s+(\d+)\s*(ps|ns)\)', txt)
    scale = (int(ts.group(1)) * (1000 if ts.group(2) == 'ns' else 1)) if ts else 1
    out = {}
    for ctype, inst, body in _CELL.findall(txt):
        arcs = []
        for cond, frm, to, r, f in _IOP.findall(body):
            edge = 'posedge' if frm.startswith('(posedge') else None
            frm = frm.replace('(posedge', '').strip(' )')
            arcs.append({'cond': cond or None, 'from': frm, 'edge': edge, 'to': to,
                         'rise_ps': _v(r) * scale, 'fall_ps': _v(f) * scale})
        out[inst] = arcs
    return out
