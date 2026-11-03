"""Compare GL0AM runSim output (waveform dump + SAIF) with Icarus and our CPU simulator on the same netlist,
uniform SDF and stimulus that scripts/gl0am_export.py used."""
import sys, os, gzip, pickle, json, re, argparse, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from gls import frontend, sdf, sim, stimulus, iverilog_ref, vcd
from gls.frontend import out_pins
ap = argparse.ArgumentParser()
ap.add_argument('--verilog'); ap.add_argument('--top'); ap.add_argument('--lib', default='designs/gcd/sky130hd_tt.lib')
ap.add_argument('--vcd'); ap.add_argument('--scope'); ap.add_argument('--random-cycles', type=int, default=0); ap.add_argument('--reset')
ap.add_argument('--clock', default='clk'); ap.add_argument('--duration', type=int, required=True); ap.add_argument('--warmup', type=int, default=5000)
ap.add_argument('--gl0am', default='/workspace/laptop/gl0am_out2'); ap.add_argument('--inp', default='out/gl0am_in')
ap.add_argument('--bin', default='/workspace/oss-cad-suite/bin'); ap.add_argument('--period', type=int, default=5000)
a = ap.parse_args(); top = a.top; od = f'{a.inp}/cmp'; os.makedirs(od, exist_ok=True)
d = frontend.load(a.verilog, a.lib, top, od, yosys=f'{a.bin}/yosys')
arcs = sdf.parse(f'{a.inp}/{top}_uniform.sdf')
if a.random_cycles: init, stim, t_end = stimulus.random_clocked(d, a.clock, a.random_cycles, reset=a.reset)
else: _, init, stim, t_end, _ = stimulus.from_vcd(d, a.vcd, a.scope)
T = a.duration
S = sim.Simulator(d, arcs, 'inertial'); S.init(init); tr = S.run(stim, T)
vf, rv = f'{od}/{top}_ref.v', f'{od}/{top}_icarus.vcd'
iverilog_ref.emit(d, arcs, stim, init, T, vf, rv)
subprocess.run([f'{a.bin}/iverilog', '-o', f'{od}/{top}.vvp', vf], check=True)
subprocess.run([f'{a.bin}/vvp', '-n', f'{od}/{top}.vvp'], check=True, stdout=subprocess.DEVNULL)
_, ich = vcd.read(rv, 'ref')
# GL0AM waveforms
data = pickle.load(gzip.open(f'{a.inp}/{top}_GraphPrecompile.pkl.gz'))
names = {**data['cell_index'], **data['port_index']}; idx = {n: i for i, n in names.items()}
z = np.load(f'{a.gl0am}/{top}_waves_0.npz'); W, Sx, Ex = z['w'], z['s'], z['e']
END_T = 1073738823
import math
FOLD = None
def gl_wave(node):
    global FOLD
    if FOLD is None: FOLD = math.ceil(math.ceil(T / Sx.shape[1]) / a.period) * a.period   # runSim fold_split
    ev = []
    for k in range(Sx.shape[1]):
        for x in W[Sx[node, k]:Ex[node, k]]:
            t = int(x) & 0x3FFFFFFF
            if t == END_T: continue
            ev.append((k * FOLD + t, 'x' if (int(x) >> 30) == 2 else str((int(x) >> 30) & 1)))
    out = []   # keep split order; within a split entries are time-ordered; the first entry of a split is its start value
    for t, v in ev:
        if out and out[-1][1] == v: continue
        if out and out[-1][0] == t: out[-1] = (t, v); continue
        out.append((t, v))
    return out
net_node = {}
for c in d.cells:
    for op in out_pins(c):
        n = c.conn.get(op)
        if n is not None and not isinstance(n, tuple): net_node[n] = idx[f'{c.name}/{op}']
for b in d.inputs: net_node[b] = idx[d.netname[b]]
mine = {n: [] for n in net_node}
for t, n, v in tr:
    if n in mine: mine[n].append((t, str(v)))
def after(ev, w): return [(t, v) for t, v in ev if t > w]
res = {'design': top, 'duration_ps': T, 'warmup_ps': a.warmup, 'nets': len(net_node)}
mm_ic, mm_cpu, ic_cpu, n_ic, n_gl, ex = 0, 0, 0, 0, 0, []
mm_x = mm_timing = 0
for n, node in net_node.items():
    g = after(gl_wave(node), a.warmup); i = after([(t, v) for t, v in ich.get(f'n{n}', [])], a.warmup); c = after(mine[n], a.warmup)
    # Icarus prints the initial value; drop repeats
    i2 = []
    for t, v in i:
        if i2 and i2[-1][1] == v: continue
        i2.append((t, v))
    n_ic += len(i2); n_gl += len(g)
    if g != i2:
        mm_ic += 1
        if any(v == 'x' for _, v in g): mm_x += 1
        elif [v for _, v in g] == [v for _, v in i2]: mm_timing += 1
        if len(ex) < 6: ex.append({'net': d.netname.get(n, n), 'gl0am': g[:5], 'icarus': i2[:5]})
    if g != c: mm_cpu += 1
    if i2 != c: ic_cpu += 1
res.update({'transitions_after_warmup_icarus': n_ic, 'transitions_after_warmup_gl0am': n_gl,
            'nets_mismatch_gl0am_vs_icarus': mm_ic, 'nets_mismatch_gl0am_vs_cpusim': mm_cpu, 'nets_mismatch_icarus_vs_cpusim': ic_cpu,
            'mismatch_with_gl0am_X': mm_x, 'mismatch_same_values_timing_only': mm_timing, 'examples': ex})
# SAIF
txt = gzip.open(f'{a.gl0am}/outGL0AM.{top}.saif.gz').read().decode()
saif = {}
for nm, t0, t1, tx, tc in re.findall(r'\(([^\s()]+)\s*\(T0 (\d+)\) \(T1 (\d+)\) \(TX (\d+)\) \(TZ 0\)\s*\(TC (\d+)\)', txt):
    saif[nm.replace('\\', '')] = (int(t0), int(t1), int(tx), int(tc))
def stats(ev):
    t0 = t1 = tx = tc = 0; prev_t, prev_v = 0, 'x'
    for t, v in ev + [(T, None)]:
        dt = min(t, T) - prev_t
        if prev_v == '0': t0 += dt
        elif prev_v == '1': t1 += dt
        else: tx += dt
        if v is None: break
        if prev_v in '01' and v in '01' and v != prev_v: tc += 1
        prev_t, prev_v = t, v
    return t0, t1, tx, tc
tc_exact = t1_exact = cmp_n = self_ok = t0_exact = t1x_exact = 0; t1x = []; tc_ic = tc_gl = 0; worst = []
for n in net_node:
    nm = d.netname.get(n)
    if nm not in saif: continue
    i = [(t, v) for t, v in ich.get(f'n{n}', [])]
    st = stats(i); g = saif[nm]; cmp_n += 1; tc_ic += st[3]; tc_gl += g[3]
    tc_exact += (st[3] == g[3]); t0_exact += (st[0] == g[0]); t1x_exact += (st[1] + st[2] == g[1] + g[2]); t1_exact += (st[1] == g[1])
    worst.append((abs(st[3] - g[3]), nm, st, g))
    gs = stats(gl_wave(net_node[n])); self_ok += (tuple(gs) == tuple(g))
    if st[1] != g[1] and len(t1x) < 3: t1x.append({'net': nm, 'icarus': st, 'gl0am_saif': g, 'gl0am_wave': gs})
worst.sort(reverse=True)
res['saif'] = {'nets_compared': cmp_n, 'TC_exact_match': tc_exact, 'T1_exact_match': t1_exact, 'T0_exact_match': t0_exact, 'T1_plus_TX_exact_match': t1x_exact, 'TC_total_icarus': tc_ic, 'TC_total_gl0am': tc_gl, 'gl0am_saif_equals_own_waves': self_ok, 'T1_mismatch_examples': t1x,
               'worst_TC': [{'net': w[1], 'icarus(T0,T1,TX,TC)': w[2], 'gl0am(T0,T1,TX,TC)': w[3]} for w in worst[:4]]}
json.dump(res, open(f'{od}/{top}_gl0am_compare.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
if os.environ.get('DBG'):
    for c in d.cells:
        if c.is_seq:
            n = c.conn['Q']; print('FF', c.name, d.netname[n], 'GL', gl_wave(net_node[n])[:8], 'IC', ich.get(f'n{n}', [])[:8])
    for b in d.inputs[:6]: print('PI', d.netname[b], 'GL', gl_wave(net_node[b])[:6], 'IC', ich.get(f'n{b}', [])[:6])
if os.environ.get('DBGX'):
    drv = {}
    for c in d.cells:
        for op in out_pins(c):
            n = c.conn.get(op)
            if n is not None and not isinstance(n, tuple): drv[n] = (c.name, c.type, {p: d.netname.get(v, v) for p, v in c.conn.items()})
    xs = []
    for n, node in net_node.items():
        g = gl_wave(node); t = [tt for tt, v in g if v == 'x' and tt > 0]
        if t: xs.append((min(t), d.netname.get(n, n), drv.get(n)))
    for r in sorted(xs, key=lambda r: r[0])[:12]: print('X', r, file=sys.stderr)
if os.environ.get('DBGD'):
    drv = {}
    for c in d.cells:
        for op in out_pins(c):
            n = c.conn.get(op)
            if n is not None and not isinstance(n, tuple): drv[n] = (c.name, c.type, {p: d.netname.get(v, v) for p, v in c.conn.items()})
    rows = []
    for n, node in net_node.items():
        g = after(gl_wave(node), a.warmup); i = []
        for t, v in after(ich.get(f'n{n}', []), a.warmup):
            if i and i[-1][1] == v: continue
            i.append((t, v))
        if g == i: continue
        k = next((j for j in range(min(len(g), len(i))) if g[j] != i[j]), min(len(g), len(i)))
        rows.append((min(g[k][0] if k < len(g) else 1e18, i[k][0] if k < len(i) else 1e18), d.netname.get(n, n), g[max(0,k-2):k+3], i[max(0,k-2):k+3], drv.get(n)))
    for r in sorted(rows, key=lambda r: r[0]): print('D', r, file=sys.stderr)
if os.environ.get('DBGN'):
    t0, t1 = map(int, os.environ['DBGW'].split(','))
    rn = {v: k for k, v in d.netname.items()}
    for nm in os.environ['DBGN'].split(','):
        n = rn.get(nm, nm)
        print('N', nm, 'IC', [(t, v) for t, v in ich.get(f'n{n}', []) if t0 <= t <= t1], 'GL', [(t, v) for t, v in gl_wave(net_node[n]) if t0 <= t <= t1] if n in net_node else None, file=sys.stderr)
