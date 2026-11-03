import os, json, subprocess, sys, pytest
from gls import sim, sdf
from conftest import ROOT

def mk(d, delays):
    """delays: inst -> list of (from, to, rise, fall[, cond])"""
    return {i: [{'cond': (x[4] if len(x) > 4 else None), 'from': x[0], 'edge': None, 'to': x[1], 'rise_ps': x[2], 'fall_ps': x[3]} for x in v] for i, v in delays.items()}

def names(d):
    return {d.netname[b]: b for b in d.netname}

def test_inertial_vs_transport_pulse(tiny):
    n = names(tiny); cells = {c.type.split('__')[1] + c.name: c for c in tiny.cells}
    u = {c.name: c for c in tiny.cells}
    inv1 = [c for c in tiny.cells if c.conn.get('A') == n['a']][0].name
    others = {c.name: [] for c in tiny.cells}
    D = {k: [('A', 'Y', 10, 10), ('B', 'Y', 10, 10), ('CLK', 'Q', 10, 10)] for k in others}
    D[inv1] = [('A', 'Y', 100, 100)]
    arcs = mk(tiny, D)
    stim = [(1000, n['a'], 1), (1050, n['a'], 0)]   # 50ps pulse on a, inverter delay 100ps
    out = {}
    for mode in ('inertial', 'transport'):
        s = sim.Simulator(tiny, mk(tiny, D), mode); s.init({n['a']: 0, n['b']: 1, n['clk']: 0})
        tr = s.run(stim, 5000)
        out[mode] = [(t, v) for t, net, v in tr if net == n['y']]
    assert out['inertial'] == []                         # pulse swallowed
    assert out['transport'] == [(1120, 0), (1170, 1)]    # y = !a & b; 1000+100+10+10 ; 1050+100+10+10

def test_cond_arc_selection(tiny):
    n = names(tiny)
    nand = [c for c in tiny.cells if 'nand2' in c.type][0].name
    D = {c.name: [('A', 'Y', 10, 10), ('CLK', 'Q', 10, 10)] for c in tiny.cells}
    D[nand] = [('A', 'Y', 500, 500, 'B'), ('A', 'Y', 50, 50), ('B', 'Y', 10, 10)]
    s = sim.Simulator(tiny, mk(tiny, D)); s.init({n['a']: 0, n['b']: 1, n['clk']: 0})
    tr = s.run([(1000, n['a'], 1)], 5000)
    # a rises -> n1 falls @1010 -> nand Y rises; COND B true -> 500ps -> @1510 ; y falls @1520
    assert [(t, v) for t, net, v in tr if net == n['y']] == [(1520, 0)]

def test_flop_samples_at_posedge(tiny):
    n = names(tiny)
    D = {c.name: [('A', 'Y', 10, 10), ('B', 'Y', 10, 10), ('CLK', 'Q', 30, 40)] for c in tiny.cells}
    s = sim.Simulator(tiny, mk(tiny, D)); s.init({n['a']: 0, n['b']: 1, n['clk']: 0})
    # y = !(nand(!a, b)) = !a & b -> 1 initially
    tr = s.run([(100, n['clk'], 1), (200, n['clk'], 0), (250, n['a'], 1), (300, n['clk'], 1)], 1000)
    q = [(t, v) for t, net, v in tr if net == n['q']]
    assert q == [(130, 1), (340, 0)]

@pytest.mark.slow
@pytest.mark.parametrize('args', [
    ['--sdf-mode', 'uniform'], ['--sdf-mode', 'full'],
    ['--verilog', 'designs/acc8/acc8_sky130hd.v', '--top', 'acc8', '--random-cycles', '200', '--reset', 'rst', '--sdf-mode', 'uniform']])
def test_icarus_crossvalidation(tmp_path, args):
    r = subprocess.run([sys.executable, 'scripts/validate_icarus.py', '--out', str(tmp_path), *args], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    res = json.loads(r.stdout)
    assert res['mismatching_nets'] == 0 and res['icarus_transitions_after_warmup'] > 0
