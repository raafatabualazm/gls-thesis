import math, numpy as np
from collections import Counter
from gls import graph, partition

def test_features_and_dt(gcd):
    d, arcs, G, out = gcd
    f = G.features()
    nc = G.n_cells
    assert all(f['dt_arr_ps'][i] >= 0 for i in range(G.N))
    assert all(f['tmax_ps'][i] >= f['tmin_ps'][i] for i in range(G.N))
    seq = [i for i in range(nc) if G.is_seq[i]]
    # flops launch at clk->Q: window = max - min of their CLK->Q arc
    i = seq[0]; a = arcs[d.cells[i].name][0]
    assert math.isclose(f['dt_arr_ps'][i], max(a['rise_ps'], a['fall_ps']) - min(a['rise_ps'], a['fall_ps']))
    assert sum(f['fanin'][:nc]) + sum(f['fanin'][nc:]) == len(G.edges) == sum(f['fanout'])

def test_dt_arr_hand_computed(tiny, tmp_path):
    from gls import sdf
    p = tmp_path / 't.sdf'; sdf.write(tiny, str(p)); arcs = sdf.parse(str(p))
    G = graph.GateGraph(tiny, arcs); tmin, tmax, dt = G.arrival_window()
    by = {c.type.split('__')[1] + ':' + c.name: i for i, c in enumerate(tiny.cells)}
    nand = [i for i, c in enumerate(tiny.cells) if 'nand2' in c.type][0]
    inv1 = [i for i, c in enumerate(tiny.cells) if c.conn.get('A') in tiny.inputs and 'inv' in c.type][0]
    A = lambda i, pin: [x for x in arcs[tiny.cells[i].name] if x['from'] == pin][0]
    a1 = A(inv1, 'A')
    assert tmin[inv1] == min(a1['rise_ps'], a1['fall_ps']) and tmax[inv1] == max(a1['rise_ps'], a1['fall_ps'])
    aA, aB = A(nand, 'A'), A(nand, 'B')
    assert tmin[nand] == min(tmin[inv1] + min(aA['rise_ps'], aA['fall_ps']), min(aB['rise_ps'], aB['fall_ps']))
    assert tmax[nand] == max(tmax[inv1] + max(aA['rise_ps'], aA['fall_ps']), max(aB['rise_ps'], aB['fall_ps']))

def test_gl0am_precompile_loads(gcd):
    d, arcs, G, out = gcd
    p = str(out / 'gp.pkl.gz'); data, _ = G.gl0am_precompile(p)
    chk = graph.load_like_gl0am(p)
    kinds = Counter(e[3] for e in G.edges)
    assert chk['n_nodes'] == G.N and chk['comb_edges'] == kinds['comb'] and chk['update_edges'] == kinds['seq_d'] + kinds['seq_clk']
    assert (chk['end_ptr'] >= chk['start_ptr']).all() and chk['end_ptr'].max() <= len(chk['flat_delays'])
    assert len(chk['ports']) == G.N - G.n_cells

def test_gl0am_style_cover_and_replication(gcd):
    d, arcs, G, out = gcd
    r = partition.gl0am_style(G, 4, np.ones(G.N), 'u')
    covered = set().union(*r.parts)
    assert covered == set(range(G.n_cells)) and r.replication >= 1.0 and len(r.parts) == 4
    assert r.extra['km1'] >= r.extra['cut_hyperedges']

def test_simpart_toy():
    class T:  # 2-level toy: sources 0,1 ; mid 2,3 ; sinks 4,5
        n_cells = 6; N = 6; is_seq = np.zeros(6, bool)
        succ = {0: [(2, 'A'), (3, 'A')], 1: [(3, 'B')], 2: [(4, 'A'), (5, 'A')], 3: [(5, 'B')]}
        def comb_order(self):
            from collections import defaultdict
            s = defaultdict(list, self.succ); return [0, 1, 2, 3, 4, 5], s
        def levels(self): return np.array([0, 0, 1, 1, 2, 2])
    r = partition.simpart(T(), 2, gamma=100)       # no level > gamma -> pure RAP
    assert r.extra['l_gamma'] == -1 and r.extra['n_dsp_parts'] == 0
    # groups: src0 succ{2,3}->max(2%2,3%2)=1 ; src1 succ{3}->1 ; all nodes group 1 -> part1 has all, part0 empty
    assert r.parts[1] == {0, 1, 2, 3, 4, 5} and r.parts[0] == set()
    r2 = partition.simpart(T(), 2, gamma=1)        # every level has 2 nodes > 1 -> l_gamma=2 -> all DSP
    assert r2.extra['l_gamma'] == 2 and r2.replication == 1.0

def test_wang_rule_matches_formula(gcd):
    d, arcs, G, out = gcd
    ev = {i: (i * 7) % 300 for i in range(G.n_cells)}
    for ncyc in (25, 5000):
        r, per = partition.wang_density(G, ev, ncyc, 8)
        tgt = max(math.ceil(ncyc / 512), 1); fl = 8 / tgt
        for i, (e, n, th) in per.items():
            assert n == max(1, min(ncyc, round(8 / (e / ncyc + fl)))) and th == math.ceil(ncyc / n)

def test_cost_model_weights_are_inputs(gcd):
    d, arcs, G, out = gcd
    f = G.features(); ev = {i: 1 for i in range(G.n_cells)}
    c0, _ = partition.cost_vector(G, f, ev, {'w0': 0, 'w_events': 0, 'w_iopath': 0, 'w_dt_arr_per_ns': 0, 'w_fanin': 1})
    assert list(c0[:G.n_cells]) == f['fanin'][:G.n_cells]
    c1, _ = partition.cost_vector(G, f, ev, {'w0': 2, 'w_events': 0, 'w_iopath': 0, 'w_dt_arr_per_ns': 0, 'w_fanin': 0})
    assert (c1[:G.n_cells] == 2).all()
