"""runSim-ready GL0AM inputs from our netlist/SDF/stimulus.

Conventions (read from GL0AM.runSim.py / GL0AM.cupy / GL0AM.generateGraph.py, see docs/GL0AM_ADAPTER.md):
* one node per cell OUTPUT PIN, named "<inst>/<pin>"; then ports; the last two nodes are 1'b0 and 1'b1.
* comb cell_type = row index in runSim's cells_list (gl0am_port/gl0am_cells_list.json). runSim builds each truth
  table with numberToBase (most-significant digit = bits[0]) but the kernels index it with value*3^(x-1), so
  cells_list pin j (0-based) of an n-pin cell must get edge x = n - j (i.e. reversed). LUT masks use bit (n-1-j).
* posedge flop: code 200 (bit0 = inverted clock, bit1 = inverted output); edge x: D=1 SI=2 SE=3 EN=4 SDN=5 CDN=6 CP=7.
  D edges type 2, CP edges type 3; ties-high via edge_ties_*, unconnected pins default (D/SI/SE=0, EN/SDN/CDN/CP=1).
* clock-tree buffers/inverters (BUFF=21 / INV=24) take update edges (type 3) so generateGraph re-wires them.
* fullSDF_LUT row per edge: 4*2^n entries [inRise->outRise | inFall->outRise | inRise->outFall | inFall->outFall],
  each indexed by the bitmask of input pins at 1 (cells_list positions). Values in ns (runSim multiplies by 1000).
  Row 0 = 1024 zeros (runSim zeroes full_delays[0:1024]).  condXSDF_LUT: R x 4.
* seq_pushoutSDF: N x 4 [rise, fall, sr_rise, sr_fall] ns; we add clock-tree insertion delay to rise/fall.
sky130 cells with no exact cells_list equivalent are rewritten as <generic gate> plus zero-delay INVs on some
inputs and/or the output (found by exhaustive search over pin permutations and inversions)."""
import gzip, json, os, pickle, itertools
import numpy as np
import scipy.sparse as sp
from .frontend import out_pins, in_pins

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUFF, INV, DFF = 21, 24, 200

def load_cells_list(path=None):
    cl = json.load(open(path or os.path.join(HERE, 'gl0am_port', 'gl0am_cells_list.json')))
    out = []
    for name, pins, expr in cl:
        n = len(pins); tt = 0
        if not name.startswith('dummy'):
            for m in range(1 << n):
                bits = [(m >> i) & 1 for i in range(n)]
                if eval(expr, {'bits': bits}): tt |= 1 << m
        out.append((name, pins, tt if not name.startswith('dummy') else None))
    return out

def _tt(fn, pins):
    tt = 0
    for m in range(1 << len(pins)):
        if fn({p: (m >> i) & 1 for i, p in enumerate(pins)}): tt |= 1 << m
    return tt

def map_cell(fn, pins, cl):
    """-> dict(code, name, src=[(sky_pin, inverted) per generic pin], post_inv) with the fewest inserted INVs."""
    n = len(pins); target = _tt(fn, pins); best = None
    for code, (name, gp, gtt) in enumerate(cl):
        if gtt is None or len(gp) != n: continue
        for perm in itertools.permutations(range(n)):          # generic pin j <- sky pin perm[j]
            for inv in range(1 << n):
                ninv = bin(inv).count('1')
                for post in (0, 1):
                    cost = ninv + post
                    if best is not None and cost >= best[0]: continue
                    ok = True
                    for m in range(1 << n):                    # m: sky pin values
                        g = 0
                        for j in range(n):
                            g |= (((m >> perm[j]) & 1) ^ ((inv >> j) & 1)) << j
                        if (((gtt >> g) & 1) ^ post) != ((target >> m) & 1): ok = False; break
                    if ok: best = (cost, code, name, perm, inv, post)
        if best is not None and best[0] == 0: break
    if best is None: return None
    cost, code, name, perm, inv, post = best
    return {'code': code, 'name': name, 'src': [(pins[perm[j]], bool((inv >> j) & 1)) for j in range(n)], 'post_inv': bool(post)}

SPLIT_H = [('AND', 0, 2), ('AND', 1, 3), ('AND', 2, 4), ('OR', 29, 2), ('OR', 99, 3), ('OR', 100, 4)]

def split_cell(fn, pins, cl):
    """Two-level rewrite for cells with no single-gate match (e.g. a311oi = NOR3(AND3(A1,A2,A3),B1,C1)):
    f(x) = F(H(x_S), x_rest) with H an AND/OR of a pin subset S and F mapped by map_cell over ['~h'] + rest."""
    n = len(pins); best = None
    for kind, hcode, k in SPLIT_H:
        if k >= n: continue
        for S in itertools.combinations(pins, k):
            rest = [p for p in pins if p not in S]
            H = (lambda v, S=S, kind=kind: int(all(v[p] for p in S)) if kind == 'AND' else int(any(v[p] for p in S)))
            table, ok = {}, True
            for m in range(1 << n):
                v = {p: (m >> i) & 1 for i, p in enumerate(pins)}
                key = (H(v),) + tuple(v[p] for p in rest); y = int(bool(fn(v)))
                if table.setdefault(key, y) != y: ok = False; break
            if not ok or len(table) != 1 << (len(rest) + 1): continue
            F = lambda v, rest=rest, table=table: table[(v['~h'],) + tuple(v[p] for p in rest)]
            g = map_cell(F, ['~h'] + rest, cl)
            if g is None: continue
            inc = all(table[(1,) + r[1:]] >= table[(0,) + r[1:]] for r in table)
            cost = sum(i for _, i in g['src']) + g['post_inv']
            if best is None or cost < best[0]:
                best = (cost, dict(g, H={'code': hcode, 'name': cl[hcode][0], 'pins': list(S), 'h_rises_out': inc}))
    return best[1] if best else None

class Exporter:
    def __init__(self, design, arcs, clock='clk'):
        self.d, self.arcs, self.clock = design, arcs, clock
        self.cl = load_cells_list()
        self.maps, self.decomp = {}, {}
        for c in design.cells:
            if c.is_seq:
                ff = c.info['ff']; assert ff['clocked_on'] in in_pins(c) and ff['next_state'] in in_pins(c), c.type
                continue
            for op in out_pins(c):
                k = (c.type, op)
                if k in self.maps: continue
                ip = in_pins(c); m = map_cell(c.fn[op][2], ip, self.cl)
                if m is None: m = split_cell(c.fn[op][2], ip, self.cl)
                if m is None: raise RuntimeError(f'no GL0AM mapping for {k}')
                self.maps[k] = m
                if m['post_inv'] or any(i for _, i in m['src']) or m.get('H'): self.decomp[k] = m

    def _arc(self, cell, frm, to, rising, inputs):   # mirrors gls.sim.Simulator._delay
        best = None
        for a in self.arcs.get(cell.name, []):
            if a['to'] != to or a['from'] != frm: continue
            if a.get('_cond') is not None or a['cond']:
                from . import boolexpr
                f = a.get('_cond') or boolexpr.compile_fn(a['cond'])
                if f(inputs): best = a; break
            elif best is None: best = a
        if best is None:
            cands = [a for a in self.arcs.get(cell.name, []) if a['to'] == to]
            if not cands: return 0.0
            return max(a['rise_ps'] if rising else a['fall_ps'] for a in cands)
        return best['rise_ps'] if rising else best['fall_ps']

    def build(self):
        d = self.d
        names, codes = [], []
        def node(nm, code): names.append(nm); codes.append(code); return len(names) - 1
        driver = {}; seq_nodes = []; final = {}
        # pass 1: nodes
        plan = []
        for c in d.cells:
            for op in out_pins(c):
                onet = c.conn.get(op)
                if c.is_seq:
                    nid = node(f'{c.name}/{op}', DFF); seq_nodes.append((nid, c)); final[(c.name, op)] = nid
                else:
                    m = self.maps[(c.type, op)]
                    hn = node(f'{c.name}/{op}~h', m['H']['code']) if m.get('H') else None
                    pre = {j: node(f'{c.name}/{op}~inv{j}', INV) for j, (_, inv) in enumerate(m['src']) if inv}
                    if m['post_inv']:
                        core = node(f'{c.name}/{op}~core', m['code']); fin = node(f'{c.name}/{op}', INV)
                    else:
                        core = fin = node(f'{c.name}/{op}', m['code'])
                    plan.append((c, op, m, pre, core, fin, hn)); final[(c.name, op)] = fin
                if onet is not None and not isinstance(onet, tuple): driver[onet] = final[(c.name, op)]
        n_cells = len(names)
        port_nodes = {}
        for b in list(d.inputs) + list(d.outputs):
            if b not in port_nodes: port_nodes[b] = node(d.netname[b], BUFF)
        for b in d.inputs: driver[b] = port_nodes[b]
        gnd = node("1'b0", BUFF); vdd = node("1'b1", BUFF)
        N = len(names)
        # clock tree: buf/inv cells reachable from the clock PI whose outputs only feed CLK pins / other tree cells
        clk_net = [b for b in d.inputs if d.netname[b] == self.clock][0]
        fan = {}
        for c in d.cells:
            for p in in_pins(c):
                n = c.conn.get(p)
                if n is not None and not isinstance(n, tuple): fan.setdefault(n, []).append((c, p))
        tree, ins = set(), {clk_net: 0.0}; frontier = [clk_net]
        while frontier:
            n = frontier.pop()
            for c, p in fan.get(n, []):
                if c.is_seq or len(in_pins(c)) != 1: continue
                m = self.maps[(c.type, out_pins(c)[0])]
                if m['code'] not in (BUFF, INV): continue
                o = c.conn[out_pins(c)[0]]
                if all(cc.is_seq and pp == cc.info['ff']['clocked_on'] or (not cc.is_seq and len(in_pins(cc)) == 1) for cc, pp in fan.get(o, [])):
                    tree.add(c.name)
                    rising = (m['code'] == BUFF)   # posedge at the leaves for non-inverting trees
                    ins[o] = ins[n] + self._arc(c, p, out_pins(c)[0], True, {p: 1})
                    frontier.append(o)
        et = sp.lil_matrix((N, N), dtype=np.int8); pid = sp.lil_matrix((N, N), dtype=np.int16)
        rown = sp.lil_matrix((N, N), dtype=np.int32)
        lut = [[0.0] * 1024]; xc = [[0.0] * 4]
        ties = []; info = {'collisions': 0}
        def edge(s, t, x, kind, row=0):
            if et[s, t] != 0: info['collisions'] += 1; raise RuntimeError(f'edge collision {names[s]}->{names[t]}')
            et[s, t] = kind; pid[s, t] = x; rown[s, t] = row
        def src_of(n, ):
            if isinstance(n, tuple): return vdd if n[1] else gnd
            return driver.get(n)
        def add_row(vals_rise, vals_fall):   # per mask lists (len 2^n), ps -> ns
            row = [v / 1000.0 for v in vals_rise] * 2 + [v / 1000.0 for v in vals_fall] * 2
            lut.append(row); xc.append([max(vals_rise) / 1000.0, max(vals_rise) / 1000.0, max(vals_fall) / 1000.0, max(vals_fall) / 1000.0])
            return len(lut) - 1
        for c, op, m, pre, core, fin, hn in plan:
            n = len(m['src']); kind = 3 if c.name in tree else 1
            if hn is not None:
                H = m['H']; k = len(H['pins'])
                for j, hp in enumerate(H['pins']):
                    s = src_of(c.conn.get(hp)); s = gnd if s is None else s
                    r = self._arc(c, hp, op, True, {}); f = self._arc(c, hp, op, False, {})   # unconditional (max) arcs
                    if not H['h_rises_out']: r, f = f, r
                    edge(s, hn, k - j, kind, add_row([r] * (1 << k), [f] * (1 << k)))
            for j, (sp_, inv) in enumerate(m['src']):
                if sp_ == '~h':
                    s = hn
                else:
                    s = src_of(c.conn.get(sp_))
                if s is None: s = gnd
                rise, fall = [], []
                for mask in range(1 << n):
                    iv = {}
                    for k_, (pk, ik) in enumerate(m['src']): iv[pk] = ((mask >> (n - 1 - k_)) & 1) ^ int(ik)
                    if '~h' in iv:
                        for hp in m['H']['pins']: iv[hp] = iv['~h']
                    if sp_ == '~h': r = f = 0.0
                    else: r = self._arc(c, sp_, op, True, iv); f = self._arc(c, sp_, op, False, iv)
                    if m['post_inv']: r, f = f, r
                    rise.append(r); fall.append(f)
                row = add_row(rise, fall) if sp_ != '~h' else 0
                if inv:
                    edge(s, pre[j], 1, kind, 0); edge(pre[j], core, n - j, kind, row)
                else:
                    edge(s, core, n - j, kind, row)
            if m['post_inv']: edge(core, fin, 1, kind, 0)
        seq_push = np.zeros((N, 4), np.float32)
        for nid, c in seq_nodes:
            ff = c.info['ff']; clkp, dp = ff['clocked_on'], ff['next_state']
            for p, x in ((dp, 1), (clkp, 7)):
                n = c.conn.get(p)
                if isinstance(n, tuple):
                    if n[1] == 1 and x not in (4, 5, 6, 7): ties.append((nid, x))
                    if n[1] == 0 and x in (4, 5, 6, 7): edge(gnd, nid, x, 2)
                    continue
                s = driver.get(n)
                if s is not None: edge(s, nid, x, 3 if x == 7 else 2)
            q = out_pins(c)[0]
            ins_d = ins.get(c.conn.get(clkp), 0.0)
            seq_push[nid, 0] = (ins_d + self._arc(c, clkp, q, True, {})) / 1000.0
            seq_push[nid, 1] = (ins_d + self._arc(c, clkp, q, False, {})) / 1000.0
        for b in d.outputs:
            if b in driver and driver[b] != port_nodes[b]: edge(driver[b], port_nodes[b], 1, 1, 0)
        data = {'adjacency_pin_ids': pid.tocsr(), 'adjacency_edge_types': et.tocsr(),
                'SDF_net_rise': sp.csr_matrix((N, N), dtype=np.float32), 'SDF_net_fall': sp.csr_matrix((N, N), dtype=np.float32),
                'seq_pushoutSDF': seq_push, 'cell_index': {i: names[i] for i in range(n_cells)}, 'num_of_std_cells': n_cells,
                'port_index': {i: names[i] for i in range(n_cells, N)}, 'cell_ids': np.array(codes, np.int64),
                'edge_ties_node_nums': np.array([t[0] for t in ties], np.int64), 'edge_ties_pin_positions': np.array([t[1] for t in ties], np.int8),
                'pin4SDF_rownum': rown.tocsr(), 'condXSDF_LUT': np.array(xc, np.float32), 'fullSDF_LUT': lut,
                'ram_rows': [], 'ram_cols': []}
        self.names, self.N, self.n_cells, self.port_nodes, self.tree, self.ins = names, N, n_cells, port_nodes, tree, ins
        return data

    def pin_net_lines(self):
        d = self.d; L = []
        for c in d.cells:
            for op in out_pins(c):
                n = c.conn.get(op)
                if n is not None and not isinstance(n, tuple): L.append(f'{c.name}/{op} {d.netname[n]}')
        for b in d.inputs: L.append(f'{d.netname[b]} {d.netname[b]}')
        return L

def waveforms(design, init, stim):
    W, V = {}, {}
    for b in design.inputs:
        nm = design.netname[b]; t = [0]; v = [init.get(b, 0)]
        for tt, n, vv in stim:
            if n == b:
                if tt == 0: v[0] = vv
                else: t.append(tt); v.append(vv)
        W[nm] = np.array(t, np.int64); V[nm] = np.array(v, np.int8)
    return {'waveforms': W, 'waveforms_value': V}

def write_all(design, arcs, init, stim, outdir, block, clock='clk'):
    os.makedirs(outdir, exist_ok=True)
    ex = Exporter(design, arcs, clock); data = ex.build()
    with gzip.GzipFile(f'{outdir}/{block}_GraphPrecompile.pkl.gz', 'w') as f: pickle.dump(data, f)
    open(f'{outdir}/{block}.pin_net', 'w').write('\n'.join(ex.pin_net_lines()) + '\n')
    with gzip.GzipFile(f'{outdir}/{block}.waveforms.pkl.gz', 'w') as f: pickle.dump(waveforms(design, init, stim), f)
    open(f'{outdir}/{block}.clkTree', 'w').write('\n'.join(sorted(ex.tree)) + ('\n' if ex.tree else ''))
    rep = {'nodes': ex.N, 'cell_output_nodes_incl_inserted': ex.n_cells, 'ports_incl_consts': ex.N - ex.n_cells,
           'comb_edges': int((data['adjacency_edge_types'] == 1).sum()), 'update_edges': int(((data['adjacency_edge_types'] == 2) + (data['adjacency_edge_types'] == 3)).sum()),
           'lut_rows': len(data['fullSDF_LUT']), 'clock_tree_cells': sorted(ex.tree),
           'clock_insertion_ps': {design.netname[k]: v for k, v in ex.ins.items()},
           'mapping': {f'{t}:{o}': {'gl0am': m['name'], 'code': m['code'], 'pins': [('!' if i else '') + p for p, i in m['src']], 'post_inv': m['post_inv'], **({'H': m['H']} if m.get('H') else {})} for (t, o), m in sorted(ex.maps.items())},
           'decomposed': sorted(f'{t}:{o}' for t, o in ex.decomp)}
    json.dump(rep, open(f'{outdir}/{block}_adapter.json', 'w'), indent=1)
    return ex, data, rep
