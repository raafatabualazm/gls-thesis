"""Write runSim-ready GL0AM inputs (precompile, waveform pickle, pin_net, clkTree) for one design."""
import sys, os, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gls import frontend, sdf, stimulus, gl0am_adapter
ap = argparse.ArgumentParser()
ap.add_argument('--verilog'); ap.add_argument('--lib', default='designs/gcd/sky130hd_tt.lib'); ap.add_argument('--top')
ap.add_argument('--vcd', default=None); ap.add_argument('--scope', default=None); ap.add_argument('--random-cycles', type=int, default=0)
ap.add_argument('--reset', default=None); ap.add_argument('--seed', type=int, default=0); ap.add_argument('--clock', default='clk')
ap.add_argument('--sdf-mode', default='uniform', choices=['uniform', 'full']); ap.add_argument('--bin', default='/workspace/oss-cad-suite/bin')
ap.add_argument('--out', default='out/gl0am_in')
a = ap.parse_args()
d = frontend.load(a.verilog, a.lib, a.top, a.out, yosys=f'{a.bin}/yosys')
sp = f'{a.out}/{a.top}_{a.sdf_mode}.sdf'; sdf.write(d, sp, uniform=(a.sdf_mode == 'uniform'), design_name=a.top); arcs = sdf.parse(sp)
if a.random_cycles: init, stim, t_end = stimulus.random_clocked(d, a.clock, a.random_cycles, reset=a.reset, seed=a.seed)
else: _, init, stim, t_end, _ = stimulus.from_vcd(d, a.vcd, a.scope)
ex, data, rep = gl0am_adapter.write_all(d, arcs, init, stim, a.out, a.top, a.clock)
rep['t_end_ps'] = t_end
json.dump(rep, open(f'{a.out}/{a.top}_adapter.json', 'w'), indent=1)
print(json.dumps({k: v for k, v in rep.items() if k != 'mapping'}, indent=1))
