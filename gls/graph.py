"""Gate graph, static features, Δt_arr, CSR export and a GL0AM-style GraphPrecompile.pkl.gz.

Node numbering follows GL0AM.generateGraph.py: std cells 0..num_of_std_cells-1, then ports.
Fields and semantics we could verify from GL0AM.generateGraph.py (lines 31-123) are noted; everything
else is our best guess and listed in docs/GL0AM_FORMAT_GAPS.md."""
import gzip, pickle
import numpy as np
from collections import defaultdict, deque
from .frontend import out_pins, in_pins

class GateGraph:
    def __init__(self, design, sdf_arcs):
        self.d = design
        cells = design.cells
        self.n_cells = len(cells)
        ports = list(design.inputs) + list(design.outputs)
        self.port_nodes = {}
        for b in ports:
            if b not in self.port_nodes: self.port_nodes[b] = self.n_cells + len(self.port_nodes)
        self.N = self.n_cells + len(self.port_nodes)
        self.names = [c.name for c in cells] + [design.netname[b] for b in self.port_nodes]
        self.types = [c.type for c in cells] + ['PI' if b in design.inputs else 'PO' for b in self.port_nodes]
        self.is_seq = np.array([c.is_seq for c in cells] + [False] * len(self.port_nodes))
        driver = {}
        for b in design.inputs: driver[b] = (self.port_nodes[b], None)
        for i, c in enumerate(cells):
            for p in out_pins(c):
                n = c.conn.get(p)
                if n is not None and not isinstance(n, tuple): driver[n] = (i, p)
        self.edges = []   # (src, dst, dst_pin, kind)  kind: 'comb' | 'seq_d' | 'seq_clk'
        self.ties = []    # (node, pin_pos, value)
        for i, c in enumerate(cells):
            ip = in_pins(c)
            clk = c.info['ff']['clocked_on'] if c.is_seq else None
            for pos, p in enumerate(ip):
                n = c.conn.get(p)
                if n is None: continue
                if isinstance(n, tuple): self.ties.append((i, pos, n[1])); continue
                if n not in driver: continue
                kind = 'comb' if not c.is_seq else ('seq_clk' if p == clk else 'seq_d')
                self.edges.append((driver[n][0], i, pos, kind, p))
        for b in design.outputs:
            if b in driver: self.edges.append((driver[b][0], self.port_nodes[b], 0, 'comb', 'PO'))
        self.sdf = sdf_arcs
        self.fanin = np.zeros(self.N, int); self.fanout = np.zeros(self.N, int)
        for s, t, *_ in self.edges:
            self.fanout[s] += 1; self.fanin[t] += 1

    # ---- static features -------------------------------------------------
    def iopath_table_size(self, i):
        if i >= self.n_cells: return 0
        return len(self.sdf.get(self.d.cells[i].name, []))

    def cond_iopath_count(self, i):
        if i >= self.n_cells: return 0
        return sum(1 for a in self.sdf.get(self.d.cells[i].name, []) if a['cond'])

    def arc_delays(self, i, pin):
        arcs = [a for a in self.sdf.get(self.d.cells[i].name, []) if a['from'] == pin]
        if not arcs: return 0.0, 0.0
        return min(min(a['rise_ps'], a['fall_ps']) for a in arcs), max(max(a['rise_ps'], a['fall_ps']) for a in arcs)

    def comb_order(self):
        """Topological order over combinational edges (sequential cells and PIs are sources)."""
        indeg = np.zeros(self.N, int); succ = defaultdict(list)
        for s, t, pos, kind, p in self.edges:
            if kind == 'comb': indeg[t] += 1; succ[s].append((t, p))
        q = deque(i for i in range(self.N) if indeg[i] == 0); order = []
        while q:
            u = q.popleft(); order.append(u)
            for v, _ in succ[u]:
                indeg[v] -= 1
                if indeg[v] == 0: q.append(v)
        if len(order) != self.N: raise RuntimeError('combinational cycle')
        return order, succ

    def arrival_window(self):
        """Δt_arr = tmax - tmin of the output arrival time, launching from PIs at 0 and flop Q at clk->Q.
        Combinational STA-like min/max propagation with SDF IOPATH delays (no slew, no wire delay)."""
        order, succ = self.comb_order()
        tmin = np.full(self.N, np.inf); tmax = np.full(self.N, -np.inf)
        for i in range(self.N):
            if i >= self.n_cells and self.types[i] == 'PI': tmin[i] = tmax[i] = 0.0
            elif i < self.n_cells and self.is_seq[i]:
                c = self.d.cells[i]; lo, hi = self.arc_delays(i, c.info['ff']['clocked_on'])
                tmin[i], tmax[i] = lo, hi
        in_min = np.full(self.N, np.inf); in_max = np.full(self.N, -np.inf)
        for u in order:
            if not (self.is_seq[u] or self.types[u] == 'PI'):
                if u < self.n_cells and np.isfinite(in_min[u]):
                    pass  # computed below via per-pin
            for v, p in succ[u]:
                if v < self.n_cells and not self.is_seq[v]:
                    lo, hi = self.arc_delays(v, p)
                    if np.isfinite(tmin[u]):
                        in_min[v] = min(in_min[v], tmin[u] + lo); in_max[v] = max(in_max[v], tmax[u] + hi)
                elif v >= self.n_cells and np.isfinite(tmin[u]):
                    in_min[v] = min(in_min[v], tmin[u]); in_max[v] = max(in_max[v], tmax[u])
            # finalize successors lazily: tmin/tmax of comb nodes = in_min/in_max once all preds seen
            for v, _ in succ[u]:
                if not self.is_seq[v]: tmin[v], tmax[v] = in_min[v], in_max[v]
        tmin[~np.isfinite(tmin)] = 0; tmax[~np.isfinite(tmax)] = 0
        return tmin, tmax, tmax - tmin

    def levels(self):
        order, succ = self.comb_order()
        lvl = np.zeros(self.N, int)
        for u in order:
            for v, _ in succ[u]: lvl[v] = max(lvl[v], lvl[u] + 1)
        return lvl

    def features(self):
        tmin, tmax, dt = self.arrival_window()
        lvl = self.levels()
        return {'name': self.names, 'type': self.types, 'fanin': self.fanin.tolist(), 'fanout': self.fanout.tolist(),
                'iopath_table_size': [self.iopath_table_size(i) for i in range(self.N)],
                'cond_iopath_count': [self.cond_iopath_count(i) for i in range(self.N)],
                'tmin_ps': tmin.tolist(), 'tmax_ps': tmax.tolist(), 'dt_arr_ps': dt.tolist(),
                'level': lvl.tolist(), 'is_seq': self.is_seq.tolist()}

    # ---- exports ---------------------------------------------------------
    def csr(self):
        src = np.array([e[0] for e in self.edges]); dst = np.array([e[1] for e in self.edges])
        o = np.lexsort((dst, src)); src, dst = src[o], dst[o]
        indptr = np.zeros(self.N + 1, np.int64); np.add.at(indptr, src + 1, 1); indptr = np.cumsum(indptr)
        kinds = np.array([{'comb': 1, 'seq_d': 2, 'seq_clk': 3}[self.edges[k][3]] for k in o], np.int8)
        pins = np.array([self.edges[k][2] for k in o], np.int16)
        return {'indptr': indptr, 'indices': dst.astype(np.int64), 'edge_kind': kinds, 'dst_pin_pos': pins}

    def gl0am_precompile(self, path, cell_type_codes=None):
        import scipy.sparse as sp
        N = self.N
        et = sp.lil_matrix((N, N), dtype=np.int8); pid = sp.lil_matrix((N, N), dtype=np.int16)
        nr = sp.lil_matrix((N, N), dtype=np.float32); nf = sp.lil_matrix((N, N), dtype=np.float32)
        rown = sp.lil_matrix((N, N), dtype=np.int32)
        lut, xcond = [[0.0, 0.0]], [0.0]   # row 0 = dummy (edges without a cell arc, e.g. to POs)
        code = cell_type_codes or {}
        auto = {}
        cell_ids = []
        for i, t in enumerate(self.types):
            if t in code: cell_ids.append(code[t])
            else: cell_ids.append(code.setdefault(t, auto.setdefault(t, 1000 + len(auto))))
        for s, t, pos, kind, p in self.edges:
            et[s, t] = {'comb': 1, 'seq_d': 2, 'seq_clk': 3}[kind]
            pid[s, t] = pos
            # no SPEF/SDF INTERCONNECT in our flow -> net delays are 0
            if t < self.n_cells:
                arcs = [a for a in self.sdf.get(self.d.cells[t].name, []) if a['from'] == p]
                if arcs:
                    rown[s, t] = len(lut)
                    lut.append([v for a in arcs for v in (a['rise_ps'], a['fall_ps'])])
                    xcond.append(max(max(a['rise_ps'], a['fall_ps']) for a in arcs))
        seq_push = np.zeros(N, np.float32)
        for i in range(self.n_cells):
            if self.is_seq[i]:
                seq_push[i] = self.arc_delays(i, self.d.cells[i].info['ff']['clocked_on'])[1]
        data = {'adjacency_pin_ids': pid.tocsr(), 'adjacency_edge_types': et.tocsr(),
                'SDF_net_rise': nr.tocsr(), 'SDF_net_fall': nf.tocsr(), 'seq_pushoutSDF': seq_push,
                'cell_index': {i: self.names[i] for i in range(self.n_cells)}, 'num_of_std_cells': self.n_cells,
                'port_index': {i: self.names[i] for i in range(self.n_cells, N)}, 'cell_ids': np.array(cell_ids, np.int64),
                'edge_ties_node_nums': np.array([t[0] for t in self.ties], np.int64),
                'edge_ties_pin_positions': np.array([t[1] for t in self.ties], np.int8),
                'pin4SDF_rownum': rown.tocsr(), 'condXSDF_LUT': np.array(xcond, np.float32), 'fullSDF_LUT': lut}
        with gzip.GzipFile(path, 'w') as f:
            pickle.dump(data, f)
        return data, auto

def load_like_gl0am(path):
    """Mirror of the numpy/scipy part of GL0AM.generateGraph.build_graph (lines 33-58, 101-112) without DGL/torch,
    to check our file is consumable: same keys, same indexing patterns."""
    with gzip.GzipFile(path, 'r') as f:
        data = np.load(f, allow_pickle=1)
    et = data['adjacency_edge_types']
    comb = (et == 1); upd = (et == 2) + (et == 3)
    src, dst = comb.nonzero()
    x = np.asarray(data['adjacency_pin_ids'][src, dst]).squeeze(0) if len(src) else np.array([])
    rr = np.asarray(data['SDF_net_rise'][src, dst]).squeeze(0) if len(src) else np.array([])
    rows = np.asarray(data['pin4SDF_rownum'][src, dst]).squeeze(0).astype(int) if len(src) else np.array([], int)
    lens = np.array([len(r) for r in data['fullSDF_LUT']]); endp = np.cumsum(lens); beg = np.concatenate(([0], endp))
    flat = np.array([v for r in data['fullSDF_LUT'] for v in r], np.float32)
    n = data['num_of_std_cells']
    ports = {k: data['port_index'][k] for k in range(n, et.shape[0])}
    return {'n_nodes': et.shape[0], 'comb_edges': len(src), 'update_edges': int(upd.nnz), 'pin_x': x, 'net_rise': rr,
            'start_ptr': beg[rows], 'end_ptr': endp[rows], 'flat_delays': flat, 'cell_types': np.asarray(data['cell_ids']),
            'ports': ports}
