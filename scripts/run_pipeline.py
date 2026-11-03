"""End-to-end CPU pipeline on one design: front-end -> SDF -> features -> CPU timed sim (event counts)
-> CSR + GL0AM-style precompile -> 5 partitioner configs. Writes out/<top>_*.json/.md."""
import sys, os, json, argparse, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import Counter
from gls import frontend, sdf, sim, stimulus, graph, partition, vcd

ap = argparse.ArgumentParser()
ap.add_argument('--verilog', default='designs/gcd/gcd_sky130hd.v'); ap.add_argument('--lib', default='designs/gcd/sky130hd_tt.lib')
ap.add_argument('--top', default='gcd'); ap.add_argument('--vcd', default='designs/gcd/gcd_sky130hd.vcd'); ap.add_argument('--scope', default='gcd_tb.gcd1')
ap.add_argument('--clock', default='clk'); ap.add_argument('--k', type=int, default=8)
ap.add_argument('--gammas', default='1024,16'); ap.add_argument('--weights', default=None)
ap.add_argument('--sdf-mode', default='full', choices=['full', 'uniform']); ap.add_argument('--delay-mode', default='inertial')
ap.add_argument('--random-cycles', type=int, default=0); ap.add_argument('--reset', default=None)
ap.add_argument('--bin', default='/workspace/oss-cad-suite/bin'); ap.add_argument('--out', default='out')
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); T = {}
t0 = time.time()
d = frontend.load(a.verilog, a.lib, a.top, a.out, yosys=f'{a.bin}/yosys'); T['frontend_s'] = time.time() - t0
sp = f'{a.out}/{a.top}.sdf'; sdf.write(d, sp, uniform=(a.sdf_mode == 'uniform'), design_name=a.top); arcs = sdf.parse(sp)
G = graph.GateGraph(d, arcs)
feats = G.features()
json.dump(feats, open(f'{a.out}/{a.top}_features.json', 'w'))
csr = G.csr(); np.savez(f'{a.out}/{a.top}_csr.npz', **csr)
data, autocodes = G.gl0am_precompile(f'{a.out}/{a.top}_GraphPrecompile.pkl.gz')
chk = graph.load_like_gl0am(f'{a.out}/{a.top}_GraphPrecompile.pkl.gz')
pairs = Counter((s, t) for s, t, *_ in G.edges); collisions = sum(c - 1 for c in pairs.values() if c > 1)
# simulation
if a.random_cycles:
    init, stim, t_end = stimulus.random_clocked(d, a.clock, a.random_cycles, reset=a.reset); t_end -= 10000
else:
    ts, init, stim, t_end, missing = stimulus.from_vcd(d, a.vcd, a.scope)
clk_net = [b for b in d.inputs if d.netname[b] == a.clock][0]
n_cycle = sum(1 for t, n, v in stim if n == clk_net and v == 1)
t0 = time.time(); S = sim.Simulator(d, arcs, a.delay_mode); S.init(init); tr = S.run(stim, t_end + 10000); T['sim_s'] = time.time() - t0
out_events = {i: S.out_events.get(i, 0) for i in range(G.n_cells)}
net_tr = Counter(n for _, n, _ in tr)
gate_net_events = {i: sum(net_tr[n] for n in set(c.conn.values()) if not isinstance(n, tuple)) for i, c in enumerate(d.cells)}
nets = sorted({n for c in d.cells for n in c.conn.values() if not isinstance(n, tuple)} | set(d.inputs)); idx = {n: i for i, n in enumerate(nets)}
vcd.write(f'{a.out}/{a.top}_{a.sdf_mode}_{a.delay_mode}.vcd', '1ps', [d.netname.get(n, f'n{n}') for n in nets], [(t, idx[n], v) for t, n, v in tr], t_end + 10000)
json.dump({'out_events': out_events, 'evals': dict(S.evals), 'connected_net_events': gate_net_events}, open(f'{a.out}/{a.top}_events.json', 'w'))
# partitions
k = a.k; R = []
unit = np.ones(G.N)
ev_w = np.array([out_events.get(i, 0) + 1 for i in range(G.N)], float)
R.append(partition.gl0am_style(G, k, unit, 'gl0am_unit', hgr_path=f'{a.out}/{a.top}_unit.hgr'))
R.append(partition.gl0am_style(G, k, ev_w, 'gl0am_event_weighted', hgr_path=f'{a.out}/{a.top}_events.hgr'))
for g in [int(x) for x in a.gammas.split(',')]:
    R.append(partition.simpart(G, k, gamma=g))
wr, per = partition.wang_density(G, gate_net_events, n_cycle, k); R.append(wr)
weights = json.load(open(a.weights)) if a.weights else None
R.append(partition.cost_model(G, feats, out_events, k, weights))
# predicted cost per partition under the (unfitted) cost model, for every config
cvec, _ = partition.cost_vector(G, feats, out_events, weights)
for r in R:
    r.extra['predicted_cost_per_part(unfitted_model)'] = [round(float(sum(cvec[g] for g in p)), 1) for p in r.parts]
nc = G.n_cells
dt = np.array(feats['dt_arr_ps'][:nc]); comb = ~G.is_seq[:nc]
report = {
  'design': a.top, 'cells': nc, 'seq_cells': int(G.is_seq[:nc].sum()), 'ports': G.N - nc, 'edges': len(G.edges),
  'cell_types': len(set(G.types[:nc])), 'sdf_mode': a.sdf_mode, 'sdf_iopaths': sum(len(v) for v in arcs.values()),
  'sdf_cond_iopaths': sum(1 for v in arcs.values() for x in v if x['cond']),
  'matrix_edge_collisions(same src,dst; GL0AM matrix keeps one)': collisions,
  'gl0am_loader_mirror': {'n_nodes': chk['n_nodes'], 'comb_edges': chk['comb_edges'], 'update_edges': chk['update_edges'], 'lut_rows': len(data['fullSDF_LUT'])},
  'features': {'max_level': int(max(feats['level'])), 'fanin_mean': float(np.mean(feats['fanin'][:nc])), 'fanout_max': int(max(feats['fanout'][:nc])),
               'iopath_size_hist': dict(Counter(feats['iopath_table_size'][:nc])), 'dt_arr_ps_comb_mean': float(dt[comb].mean()), 'dt_arr_ps_comb_max': float(dt[comb].max()),
               'dt_arr_ps_comb_p50': float(np.median(dt[comb]))},
  'sim': {'mode': a.delay_mode, 'n_cycle(clk rises)': n_cycle, 't_end_ps': t_end + 10000, 'net_transitions': len(tr),
          'gate_output_events_total': int(sum(out_events.values())), 'gates_with_zero_output_events': sum(1 for v in out_events.values() if v == 0),
          'max_gate_output_events': int(max(out_events.values())), 'gate_evaluations_total': int(sum(S.evals.values()))},
  'timing_s': {k_: round(v, 3) for k_, v in T.items()},
  'partitions': [r.summary() for r in R]}
json.dump(report, open(f'{a.out}/{a.top}_report.json', 'w'), indent=1, default=float)
print(json.dumps({k_: v for k_, v in report.items() if k_ != 'partitions'}, indent=1, default=float))
for r in R:
    s = r.summary()
    print(f"{s['config']:<22} k={s['k']:<3} bal={s['balance_max_over_mean']:<7} rep={s['replication_factor']:<7} cut={s.get('cut_hyperedges','-')} km1={s.get('km1','-')} sizes={s['part_sizes']}")
