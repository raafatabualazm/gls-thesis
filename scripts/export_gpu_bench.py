"""Export a design + the partitioner configs to a flat int32 file for gpu/partbench.cu.
Uses the same steps/args as run_pipeline.py (default SDF mode 'full', inertial CPU sim for event counts)."""
import sys, os, argparse, struct
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter
from gls import frontend, sdf, sim, stimulus, graph, partition
from gls.frontend import in_pins, out_pins
ap = argparse.ArgumentParser()
ap.add_argument('--verilog', default='designs/gcd/gcd_sky130hd.v'); ap.add_argument('--lib', default='designs/gcd/sky130hd_tt.lib')
ap.add_argument('--top', default='gcd'); ap.add_argument('--vcd', default='designs/gcd/gcd_sky130hd.vcd'); ap.add_argument('--scope', default='gcd_tb.gcd1')
ap.add_argument('--clock', default='clk'); ap.add_argument('--k', type=int, default=8); ap.add_argument('--gammas', default='1024,16')
ap.add_argument('--random-cycles', type=int, default=0); ap.add_argument('--reset', default=None)
ap.add_argument('--out', default='gpu/data')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); tmp = f'/tmp/exp_{a.top}'
d = frontend.load(a.verilog, a.lib, a.top, tmp, yosys='/workspace/oss-cad-suite/bin/yosys')
sp = f'{tmp}/{a.top}.sdf'; sdf.write(d, sp, uniform=False, design_name=a.top); arcs = sdf.parse(sp)
G = graph.GateGraph(d, arcs); feats = G.features()
if a.random_cycles:
    init, stim, t_end = stimulus.random_clocked(d, a.clock, a.random_cycles, reset=a.reset); t_end -= 10000
else:
    ts, init, stim, t_end, missing = stimulus.from_vcd(d, a.vcd, a.scope)
clk_net = [b for b in d.inputs if d.netname[b] == a.clock][0]
rises = sorted(t for t, n, v in stim if n == clk_net and v == 1); n_cycle = len(rises)
S = sim.Simulator(d, arcs, 'inertial'); S.init(init); tr = S.run(stim, t_end + 10000)
out_events = {i: S.out_events.get(i, 0) for i in range(G.n_cells)}
net_tr = Counter(n for _, n, _ in tr)
gate_net_events = {i: sum(net_tr[n] for n in set(c.conn.values()) if not isinstance(n, tuple)) for i, c in enumerate(d.cells)}
k = a.k; R = []
R.append(partition.gl0am_style(G, k, np.ones(G.N), 'gl0am_unit', hgr_path=f'{tmp}/u.hgr'))
R.append(partition.gl0am_style(G, k, np.array([out_events.get(i, 0) + 1 for i in range(G.N)], float), 'gl0am_event_weighted', hgr_path=f'{tmp}/e.hgr'))
for g in [int(x) for x in a.gammas.split(',')]: R.append(partition.simpart(G, k, gamma=g))
wr, per = partition.wang_density(G, gate_net_events, n_cycle, k); R.append(wr)
R.append(partition.cost_model(G, feats, out_events, k, None))
# ---- node arrays
N, nc = G.N, G.n_cells
kind = np.zeros(N, np.int32)  # 0 comb cell, 1 flop, 2 PI, 3 PO
for i in range(N):
    kind[i] = (1 if G.is_seq[i] else 0) if i < nc else (2 if G.types[i] == 'PI' else 3)
fan = -np.ones((N, 6), np.int32); nfan = np.zeros(N, np.int32)
for s, t, pos, kd, p in G.edges: fan[t, pos] = s
for node, pos, v in G.ties: fan[node, pos] = -2 - int(v)   # -2 const0, -3 const1
tt = np.zeros(N, np.int64); drise = np.zeros(N, np.int32); dfall = np.zeros(N, np.int32); dpos = np.zeros(N, np.int32)
for i, c in enumerate(d.cells):
    ip = in_pins(c); nfan[i] = len(ip)
    ar = arcs.get(c.name, [])
    drise[i] = int(round(max([x['rise_ps'] for x in ar] or [0]))); dfall[i] = int(round(max([x['fall_ps'] for x in ar] or [0])))
    if c.is_seq: dpos[i] = ip.index(c.info['ff']['next_state']); continue
    fn = c.fn[out_pins(c)[0]][2]; bits = 0
    for m in range(1 << len(ip)):
        if fn({p: (m >> j) & 1 for j, p in enumerate(ip)}): bits |= 1 << m
    tt[i] = bits
for i in range(nc, N):
    if kind[i] == 3: nfan[i] = 1; tt[i] = 2   # PO = buffer
lvl = np.array(feats['level'], np.int32)
# ---- stimulus: PI values at each clk rise (after all events strictly before the next rise)
pis = [i for i in range(nc, N) if kind[i] == 2]; pnet = {G.port_nodes[b]: b for b in G.port_nodes}
cur = dict(init); st = sorted(stim); j = 0; vec = np.zeros((n_cycle, len(pis)), np.int32)
for c_ in range(n_cycle):
    lim = rises[c_ + 1] if c_ + 1 < n_cycle else 1 << 62
    while j < len(st) and st[j][0] < lim: cur[st[j][1]] = st[j][2]; j += 1
    vec[c_] = [cur.get(pnet[p], 0) for p in pis]
# ---- CPU reference of the same cycle-level 2-value timed eval (global, unpartitioned)
order = sorted([i for i in range(N) if kind[i] in (0, 3)], key=lambda i: lvl[i])
val = np.zeros(N, np.int64); arr = np.zeros(N, np.int64); ref_ev = 0; ref_sig = []
def fv(s): return 0 if s == -2 else 1 if s == -3 else val[s]
for c_ in range(n_cycle):
    old = val.copy(); arr[:] = -1
    for i in range(nc):
        if kind[i] == 1:
            nv = old[fan[i, dpos[i]]] if fan[i, dpos[i]] >= 0 else fv(fan[i, dpos[i]])
            if nv != val[i]: val[i] = nv; arr[i] = drise[i] if nv else dfall[i]; ref_ev += 1
    for q, p in enumerate(pis):
        if vec[c_, q] != val[p]: val[p] = vec[c_, q]; arr[p] = 0; ref_ev += 1
    for i in order:
        m = 0; t = -1
        for jj in range(nfan[i]):
            s = fan[i, jj]; b = fv(s) if s < -1 else (val[s] if s >= 0 else 0); m |= int(b) << jj
            if s >= 0 and arr[s] >= 0: t = max(t, arr[s])
        nv = (tt[i] >> m) & 1
        if nv != val[i]: val[i] = nv; arr[i] = t + (drise[i] if nv else dfall[i]); ref_ev += 1
    ref_sig.append(int(sum(int(val[i]) * ((i * 2654435761) % 1000003) for i in range(N)) % 1000000007))
# ---- write
def wa(f, x): x = np.asarray(x).astype(np.int64).ravel(); f.write(struct.pack('<q', len(x))); f.write(x.astype('<i8').tobytes())
with open(f'{a.out}/{a.top}.bin', 'wb') as f:
    wa(f, [N, nc, n_cycle, len(pis), len(R)])
    for x in (kind, nfan, fan, tt, drise, dfall, dpos, lvl, pis, vec): wa(f, x)
    for r in R:
        parts = [sorted(p, key=lambda i: (lvl[i], i)) for p in r.parts]
        sizes = [len(p) for p in parts]
        wa(f, sizes); wa(f, [g for p in parts for g in p])
names = [r.name for r in R]
open(f'{a.out}/{a.top}_meta.txt', 'w').write('\n'.join(names) + f'\nref_events {ref_ev}\nref_final_sig {ref_sig[-1]}\nn_cycle {n_cycle}\n')
print(a.top, 'N', N, 'cells', nc, 'cycles', n_cycle, 'configs', names, [len(r.parts) for r in R], 'ref_events', ref_ev, 'sig', ref_sig[-1])
