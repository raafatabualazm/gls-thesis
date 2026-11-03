"""Static check of an exported GL0AM precompile: evaluate the GL0AM-coded comb graph (cells_list functions, pin x-1)
on random PI/flop values and compare every cell-output net with our sky130 Liberty functions."""
import sys, os, gzip, pickle, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gls import frontend
from gls.frontend import out_pins, in_pins
v, top = sys.argv[1], sys.argv[2]; outdir = 'out/gl0am_in'
d = frontend.load(v, 'designs/gcd/sky130hd_tt.lib', top, '/tmp/fe', yosys='/workspace/oss-cad-suite/bin/yosys')
data = pickle.load(gzip.open(f'{outdir}/{top}_GraphPrecompile.pkl.gz'))
cl = json.load(open('gl0am_port/gl0am_cells_list.json'))
N = data['adjacency_edge_types'].shape[0]; et = data['adjacency_edge_types'].tocoo(); pid = data['adjacency_pin_ids']
names = {**data['cell_index'], **data['port_index']}; idx = {n: i for i, n in names.items()}
ins = {i: [] for i in range(N)}
for s, t, k in zip(et.row, et.col, et.data): ins[t].append((s, int(pid[s, t]), int(k)))
codes = data['cell_ids']; rnd = random.Random(1); bad = 0; checked = 0
for trial in range(200):
    val = {}
    net = {}
    for b in d.inputs: net[b] = rnd.randint(0, 1); val[idx[d.netname[b]]] = net[b]
    val[idx["1'b0"]] = 0; val[idx["1'b1"]] = 1
    for c in d.cells:
        if c.is_seq:
            q = out_pins(c)[0]; x = rnd.randint(0, 1); val[idx[f'{c.name}/{q}']] = x; net[c.conn[q]] = x
    def ev(i):
        if i in val: return val[i]
        e = [(s, x) for s, x, k in ins[i] if k in (1, 3)]
        bits = [0] * len(cl[codes[i]][1])
        for s, x in e: bits[len(bits) - x] = ev(s)   # runSim: pin x <-> bits[n-x] (numberToBase is MSB-first)
        val[i] = int(eval(cl[codes[i]][2], {'bits': bits})); return val[i]
    # reference: settle sky130 netlist
    for _ in range(len(d.cells)):
        for c in d.cells:
            if c.is_seq: continue
            iv = {p: net.get(c.conn[p], 0) for p in in_pins(c)}
            for op in out_pins(c): net[c.conn[op]] = int(bool(c.fn[op][2](iv)))
    for c in d.cells:
        if c.is_seq: continue
        for op in out_pins(c):
            checked += 1
            if ev(idx[f'{c.name}/{op}']) != net[c.conn[op]]: bad += 1
print(json.dumps({'design': top, 'cell_outputs_checked': checked, 'mismatches': bad}))
