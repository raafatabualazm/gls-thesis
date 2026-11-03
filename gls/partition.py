"""Five partitioner baselines on the gate graph (see proposal §3).
All produce a Result with per-partition gate sets (with replication), loads and predicted cost."""
import math, json, os
import numpy as np
from collections import defaultdict, Counter

HERE = os.path.dirname(os.path.abspath(__file__))

class Result:
    def __init__(self, name, parts, weights, n_gates, extra=None):
        self.name, self.parts, self.extra = name, parts, extra or {}
        loads = [sum(weights[g] for g in p) for p in parts]
        self.loads = loads
        mean = (sum(loads) / len(loads)) if loads else 0
        self.balance = (max(loads) / mean) if mean else float('nan')
        self.replication = (sum(len(p) for p in parts) / n_gates) if n_gates else float('nan')
    def summary(self):
        d = {'config': self.name, 'k': len(self.parts), 'balance_max_over_mean': round(self.balance, 4),
             'replication_factor': round(self.replication, 4), 'part_sizes': [len(p) for p in self.parts],
             'part_loads': [round(x, 3) for x in self.loads]}
        d.update(self.extra); return d

# ----------------------------------------------------------------------------- helpers
def comb_gates(G):
    return [i for i in range(G.n_cells) if not G.is_seq[i]]

def endpoints(G):
    """Endpoints = sequential cells (their D/clk input cones) and primary outputs."""
    return [i for i in range(G.N) if (i < G.n_cells and G.is_seq[i]) or (i >= G.n_cells and G.types[i] == 'PO')]

def endpoint_sets(G):
    """For every comb gate: frozenset of endpoints reachable through combinational fan-out."""
    order, succ = G.comb_order()
    eps = set(endpoints(G))
    seqsucc = defaultdict(list)
    for s, t, pos, kind, p in G.edges:
        if kind != 'comb': seqsucc[s].append(t)
    reach = {}
    for u in reversed(order):
        r = set()
        for v, _ in succ[u]:
            if v in eps: r.add(v)
            else: r |= reach.get(v, set())
        for v in seqsucc[u]: r.add(v)   # edge into flop D/CLK
        reach[u] = frozenset(r)
    return reach

def kahypar_partition(n_nodes, nets, net_w, node_w, k, eps=0.03, seed=1, ini=None):
    import kahypar
    ini = ini or os.path.join(HERE, 'kahypar_cut.ini')
    idx = [0]; flat = []
    for n in nets: flat.extend(n); idx.append(len(flat))
    h = kahypar.Hypergraph(n_nodes, len(nets), idx, flat, k, [int(w) for w in net_w], [int(max(1, w)) for w in node_w])
    ctx = kahypar.Context(); ctx.loadINIconfiguration(ini)
    ctx.setK(k); ctx.setEpsilon(eps); ctx.setSeed(seed); ctx.suppressOutput(True)
    kahypar.partition(h, ctx)
    blocks = [h.blockID(i) for i in range(n_nodes)]
    cut = sum(w for n, w in zip(nets, net_w) if len({blocks[v] for v in n}) > 1)
    km1 = sum(w * (len({blocks[v] for v in n}) - 1) for n, w in zip(nets, net_w))
    return blocks, cut, km1

def write_hgr(path, n_nodes, nets, net_w, node_w):
    """hMetis fmt 11 file, as GL0AM writes (partitioningSetup.py:466-487)."""
    with open(path, 'w') as f:
        f.write(f"{len(nets)} {n_nodes} 11\n")
        for n, w in zip(nets, net_w): f.write(f"{int(w)} " + ' '.join(str(v + 1) for v in n) + '\n')
        for w in node_w: f.write(f"{int(max(1, w))}\n")

# ----------------------------------------------------------------------------- 1/2/5: GL0AM-style endpoint hypergraph
def gl0am_style(G, k, gate_weight, name, eps=0.03, seed=1, hgr_path=None, max_endpoints=1000, rep_penalty=0.0):
    """Clean re-implementation of GL0AM partitioningSetup.py on CPU: cluster comb gates by identical endpoint set;
    hypergraph nodes = endpoints, node weight = Σ_cluster W(C)/|E(C)| (int-truncated, as cp.int32 cast);
    hyperedges = multi-endpoint clusters, weight ceil(W(C)/|E(C)|), member list truncated to max_endpoints.
    W(C) uses `gate_weight` (unit -> config 1, events -> config 2, cost model -> config 5).
    Each partition then simulates the full fan-in cone of its endpoints (gates in several cones are replicated)."""
    reach = endpoint_sets(G)
    eps_list = endpoints(G); eidx = {e: i for i, e in enumerate(eps_list)}
    clusters = defaultdict(list)
    for g in comb_gates(G):
        if reach[g]: clusters[reach[g]].append(g)
    node_w = np.zeros(len(eps_list))
    for e in eps_list:
        node_w[eidx[e]] += gate_weight[e] if e < G.n_cells else 0
    nets, net_w = [], []
    for es, gs in clusters.items():
        W = sum(gate_weight[g] for g in gs)
        for e in es: node_w[eidx[e]] += W / len(es)
        if len(es) > 1:
            nets.append(sorted(eidx[e] for e in es)[:max_endpoints]); net_w.append(math.ceil(W / len(es)))
    node_w_int = np.floor(node_w).astype(int)
    if hgr_path: write_hgr(hgr_path, len(eps_list), nets, net_w, node_w_int)
    blocks, cut, km1 = kahypar_partition(len(eps_list), nets, net_w, node_w_int, k, eps, seed)
    parts = [set() for _ in range(k)]
    for e, b in zip(eps_list, blocks):
        if e < G.n_cells: parts[b].add(e)
    for es, gs in clusters.items():
        for b in {blocks[eidx[e]] for e in es}: parts[b].update(gs)
    unreached = [g for g in comb_gates(G) if not reach[g]]
    for g in unreached: parts[0].add(g)
    r = Result(name, parts, gate_weight, G.n_cells, {
        'hypergraph_nodes': len(eps_list), 'hyperedges': len(nets), 'cut_hyperedges': cut, 'km1': km1,
        'gates_with_no_endpoint(dangling)': len(unreached)})
    if rep_penalty:
        mult = Counter(g for p in parts for g in p)
        r.loads = [sum(gate_weight[g] + rep_penalty * (mult[g] > 1) for g in p) for p in parts]
        m = sum(r.loads) / k; r.balance = max(r.loads) / m
    return r

# ----------------------------------------------------------------------------- 3: SimPart (clean-room from paper Alg. 1/2)
def simpart(G, k, gamma=1024, gate_weight=None):
    """Clean-room from the SimPart paper text (Alg. 1 majority grouping; Alg. 2 DSP-RAP hybrid).
    Our reading: Group[PI] = argmax over successors of (idx mod k) -- read literally as the successor index mod k;
    other nodes take the majority Group of predecessors in topological order. Levels from inputs.
    l_gamma = max level with > gamma nodes (-1 if none). Nodes with level > l_gamma: RAP -- each output-side node's
    fan-in cone (restricted to levels > l_gamma) goes into part Group[v]; nodes replicated across groups.
    Levels <= l_gamma: DSP -- one disjoint partition per (level, group)."""
    order, succ = G.comb_order()
    lvl = G.levels()
    pred = defaultdict(list)
    for u in order:
        for v, _ in succ[u]: pred[v].append(u)
    group = {}
    sources = [u for u in order if not pred[u]]
    for u in sources:
        ss = [v for v, _ in succ[u]]
        group[u] = max((v % k for v in ss), default=u % k)
    for u in order:
        if u in group: continue
        c = Counter(group[p] for p in pred[u])
        group[u] = max(sorted(c), key=lambda g: c[g])
    gw = gate_weight if gate_weight is not None else np.ones(G.N)
    counts = Counter(lvl[i] for i in range(G.N))
    l_gamma = max([l for l, c in counts.items() if c > gamma], default=-1)
    sinks = [u for u in range(G.N) if not succ[u] and lvl[u] > l_gamma]
    rap = defaultdict(set)
    for v in sinks:
        stack, seen = [v], set()
        while stack:
            x = stack.pop()
            if x in seen or lvl[x] <= l_gamma: continue
            seen.add(x); stack.extend(pred[x])
        rap[group[v]] |= seen
    dsp = defaultdict(set)
    for u in range(G.N):
        if lvl[u] <= l_gamma: dsp[(lvl[u], group[u])].add(u)
    keep = lambda s: {g for g in s if g < G.n_cells}
    rap_parts = [keep(rap[g]) for g in range(k)]
    dsp_parts = [keep(s) for _, s in sorted(dsp.items())]
    parts = rap_parts + [p for p in dsp_parts if p]
    mult = Counter(g for p in rap_parts for g in p)
    r = Result(f'simpart_gamma{gamma}', parts, gw, G.n_cells, {
        'gamma': gamma, 'l_gamma': int(l_gamma), 'max_level': int(lvl.max()), 'n_rap_parts': k,
        'n_dsp_parts': len([p for p in dsp_parts if p]),
        'rap_replicated_gates': sum(1 for g, m in mult.items() if m > 1),
        'rap_balance_max_over_mean': round(max(sum(gw[g] for g in p) for p in rap_parts) / max(1e-9, np.mean([sum(gw[g] for g in p) for p in rap_parts])), 4)})
    return r

# ----------------------------------------------------------------------------- 4: Wang density rule
def wang_density(G, net_events, n_cycle, k, e_target=8, floor_cycles=512):
    """GTPower Gate::setNCyclePerThreadByEventCount (power/cuda/Gate.cuh:354-376, GlobalConfig.hh:36-43):
    N_evt = Σ transitions on all nets connected to the gate; rho = N_evt/N_cyc; n_target = ceil(N_cyc/floor);
    rho_floor = E_target/n_target; n = clamp(round(E_target/(rho+rho_floor)), 1, N_cyc); threads = ceil(N_cyc/n).
    The rule is a per-gate cycle segmentation, NOT a graph partitioner. To compare with the others we bin
    per-gate thread work (N_evt + threads) into k parts with greedy LPT; that binning is ours, not GTPower's."""
    n_target = max(math.ceil(n_cycle / floor_cycles), 1)
    rho_floor = e_target / n_target
    per = {}
    for i in range(G.n_cells):
        ev = net_events.get(i, 0); rho = ev / n_cycle
        n = max(1, min(n_cycle, round(e_target / (rho + rho_floor))))
        per[i] = (ev, n, math.ceil(n_cycle / n))
    w = np.zeros(G.N)
    for i, (ev, n, th) in per.items(): w[i] = ev + th
    parts = [set() for _ in range(k)]; load = [0.0] * k
    for i in sorted(per, key=lambda i: -w[i]):
        b = int(np.argmin(load)); parts[b].add(i); load[b] += w[i]
    ns = [p[1] for p in per.values()]; ths = [p[2] for p in per.values()]
    return Result('wang_density', parts, w, G.n_cells, {
        'n_cycle': n_cycle, 'n_cycle_target': n_target, 'rho_floor': rho_floor,
        'n_per_thread_hist': dict(Counter(ns)), 'total_threads': int(sum(ths)), 'gates': len(per)}), per

# ----------------------------------------------------------------------------- 5: cost model
DEFAULT_WEIGHTS = {'w0': 1.0, 'w_events': 1.0, 'w_iopath': 1.0, 'w_dt_arr_per_ns': 1.0, 'w_fanin': 1.0,
                   'w_mem_bytes': 0.0, 'w_replication': 1.0, 'bytes_per_pin': 8, '_status': 'UNFITTED placeholder weights; fit requires GPU kernel timings'}

def cost_vector(G, feats, events, weights):
    w = dict(DEFAULT_WEIGHTS); w.update(weights or {})
    c = np.zeros(G.N)
    for i in range(G.n_cells):
        mem = (feats['fanin'][i] + feats['fanout'][i]) * w['bytes_per_pin']
        c[i] = (w['w0'] + w['w_events'] * events.get(i, 0) + w['w_iopath'] * feats['iopath_table_size'][i]
                + w['w_dt_arr_per_ns'] * feats['dt_arr_ps'][i] / 1000.0 + w['w_fanin'] * feats['fanin'][i] + w['w_mem_bytes'] * mem)
    return c, w

def cost_model(G, feats, events, k, weights=None, **kw):
    c, w = cost_vector(G, feats, events, weights)
    r = gl0am_style(G, k, c, 'cost_model', rep_penalty=w['w_replication'], **kw)
    r.extra['weights'] = {k_: v for k_, v in w.items()}
    return r
