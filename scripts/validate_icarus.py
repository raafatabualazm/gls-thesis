"""Cross-validate the CPU simulator against Icarus Verilog on a design, with a uniform derived SDF."""
import sys, os, subprocess, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gls import frontend, sdf, sim, stimulus, iverilog_ref, vcd

ap = argparse.ArgumentParser()
ap.add_argument('--verilog', default='designs/gcd/gcd_sky130hd.v'); ap.add_argument('--lib', default='designs/gcd/sky130hd_tt.lib')
ap.add_argument('--top', default='gcd'); ap.add_argument('--vcd', default='designs/gcd/gcd_sky130hd.vcd'); ap.add_argument('--scope', default='gcd_tb.gcd1')
ap.add_argument('--bin', default='/workspace/oss-cad-suite/bin'); ap.add_argument('--out', default='out')
ap.add_argument('--warmup', type=int, default=5000)
ap.add_argument('--random-cycles', type=int, default=0); ap.add_argument('--clock', default='clk'); ap.add_argument('--reset', default=None); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--sdf-mode', default='uniform', choices=['uniform','full'])
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
d = frontend.load(a.verilog, a.lib, a.top, a.out, yosys=f'{a.bin}/yosys')
usdf = f'{a.out}/{a.top}_{a.sdf_mode}.sdf'; sdf.write(d, usdf, uniform=(a.sdf_mode == 'uniform'), design_name=a.top)
arcs = sdf.parse(usdf)
if a.random_cycles:
    init, stim, t_end = stimulus.random_clocked(d, a.clock, a.random_cycles, reset=a.reset, seed=a.seed); missing = []
else:
    ts, init, stim, t_end, missing = stimulus.from_vcd(d, a.vcd, a.scope)
t_end += 10000
s = sim.Simulator(d, arcs, 'inertial'); s.init(init); trace = s.run(stim, t_end)
vfile = f'{a.out}/{a.top}_{a.sdf_mode}_ref.v'; rvcd = f'{a.out}/{a.top}_{a.sdf_mode}_icarus.vcd'
if a.sdf_mode == 'uniform':
    iverilog_ref.emit(d, arcs, stim, init, t_end, vfile, rvcd); flags = []
else:
    iverilog_ref.emit_specify(d, arcs, stim, init, t_end, vfile, rvcd); flags = ['-gspecify']
subprocess.run([f'{a.bin}/iverilog', *flags, '-o', f'{a.out}/{a.top}_ref.vvp', vfile], check=True)
subprocess.run([f'{a.bin}/vvp', '-n', f'{a.out}/{a.top}_ref.vvp'], check=True, stdout=subprocess.DEVNULL)
_, ich = vcd.read(rvcd, 'ref')
nets = sorted({n for c in d.cells for n in c.conn.values() if not isinstance(n, tuple)} | set(d.inputs))
mine = {n: [] for n in nets}
for t, n, v in trace: mine[n].append((t, str(v)))
mism, compared, nev = [], 0, 0
for n in nets:
    ref = [(t, v) for t, v in ich.get(f'n{n}', []) if t > a.warmup]
    my = [(t, v) for t, v in mine[n] if t > a.warmup]
    # Icarus starts at x: drop leading duplicates of the same value
    compared += 1; nev += len(ref)
    if ref != my: mism.append((d.netname.get(n, n), ref[:4], my[:4]))
names = [d.netname.get(n, f'n{n}') for n in nets]; idx = {n: i for i, n in enumerate(nets)}
vcd.write(f'{a.out}/{a.top}_cpusim.vcd', '1ps', names, [(t, idx[n], v) for t, n, v in trace], t_end)
res = {'design': a.top, 'sdf_mode': a.sdf_mode, 'cells': len(d.cells), 'nets_compared': compared, 'icarus_transitions_after_warmup': nev,
       'cpusim_transitions_total': len(trace), 'mismatching_nets': len(mism), 'missing_inputs_in_vcd': missing,
       'warmup_ps': a.warmup, 't_end_ps': t_end, 'first_mismatches': mism[:5]}
json.dump(res, open(f'{a.out}/{a.top}_{a.sdf_mode}_validation.json', 'w'), indent=1, default=str)
print(json.dumps(res, indent=1, default=str))
