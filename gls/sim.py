"""CPU reference: timed, event-driven, 2-value gate-level simulator with SDF IOPATH delays.
Delay modes: 'inertial' (Verilog continuous-assign semantics: an evaluation whose value differs from the
projected value cancels the pending output event; schedule only if the new value differs from the current output) or 'transport'
(every change of the projected value is queued; no cancellation).
Sequential cells: Liberty ff(clocked_on: CLK, next_state: D) for single-edge positive flops
(dfxtp). When several inputs of a cell change in the same timestep, the smallest IOPATH delay
among the changed pins is used (IEEE 1364 module-path rule, matches Icarus -gspecify). D is sampled with the values present when the clock edge is processed; CLK->Q delay from SDF.
COND arcs: the first arc whose COND evaluates true on the current input values is used,
falling back to unconditional arcs (approximation of SDF COND matching)."""
import heapq
from collections import defaultdict
from . import boolexpr

class Simulator:
    def __init__(self, design, sdf_arcs, mode='inertial'):
        assert mode in ('inertial', 'transport')
        self.d, self.mode = design, mode
        self.val = defaultdict(int)
        self.fanout = defaultdict(list)    # net -> [(cell_idx, pin)]
        self.drv = {}
        self.arcs = []
        for ci, c in enumerate(design.cells):
            for p, n in c.conn.items():
                if isinstance(n, tuple): continue
                if c.info['pins'][p]['dir'] == 'input': self.fanout[n].append((ci, p))
                else: self.drv[n] = (ci, p)
            arcs = sdf_arcs.get(c.name, [])
            for a in arcs:
                a['_cond'] = boolexpr.compile_fn(a['cond']) if a['cond'] else None
            self.arcs.append(arcs)
        self.pending = {}         # (ci, pin) -> (t, value, seq) of last scheduled event
        self.heap, self.seq = [], 0
        self.out_events = defaultdict(int)   # ci -> committed output transitions (events)
        self.evals = defaultdict(int)        # ci -> evaluations
        self.trace = []                      # (t, net, v)

    def _inputs(self, c):
        v = {}
        for p, n in c.conn.items():
            if c.info['pins'][p]['dir'] == 'input':
                v[p] = n[1] if isinstance(n, tuple) else self.val[n]
        return v

    def _delay(self, ci, from_pin, to_pin, rising, inputs):
        best = None
        for a in self.arcs[ci]:
            if a['to'] != to_pin or a['from'] != from_pin: continue
            if a['_cond'] is not None:
                if a['_cond'](inputs): best = a; break
            elif best is None: best = a
        if best is None:  # no arc for this pin: take worst arc to output
            cands = [a for a in self.arcs[ci] if a['to'] == to_pin]
            if not cands: return 0
            return max(a['rise_ps'] if rising else a['fall_ps'] for a in cands)
        return best['rise_ps'] if rising else best['fall_ps']

    def _schedule(self, ci, pin, net, t, nv, from_pins, inputs):
        key = (ci, pin)
        cur = self.val[net]
        if self.mode == 'inertial':
            # Verilog continuous assign: nothing happens unless the evaluated RHS value changes.
            p = self.pending.get(key)
            if nv == (p[1] if p else cur): return
            self.pending.pop(key, None)        # cancel
            if nv == cur: return
            d = min(self._delay(ci, fp, pin, nv == 1, inputs) for fp in from_pins)
        else:
            last = self.pending.get(key)
            proj = last[1] if last and last[0] >= t else cur
            if nv == proj: return
            d = min(self._delay(ci, fp, pin, nv == 1, inputs) for fp in from_pins)
        self.seq += 1
        ev = (t + int(round(d)), nv, self.seq)
        self.pending[key] = ev
        heapq.heappush(self.heap, (ev[0], ev[2], ci, pin, net, nv))

    def init(self, input_values):
        """Settle at t=0 with zero delay: flops=0, inputs given, combinational logic evaluated topologically."""
        for n, v in input_values.items(): self.val[n] = v
        changed = True; it = 0
        while changed:
            changed = False; it += 1
            for c in self.d.cells:
                if c.is_seq: continue
                iv = self._inputs(c)
                for p, fn in c.fn.items():
                    n = c.conn.get(p)
                    if n is None or isinstance(n, tuple): continue
                    nv = fn[2](iv)
                    if self.val[n] != nv: self.val[n] = nv; changed = True
            if it > 10000: raise RuntimeError('combinational loop at init')

    def run(self, stimulus, t_end):
        """stimulus: list of (t, net, v)."""
        for t, n, v in stimulus:
            self.seq += 1
            heapq.heappush(self.heap, (t, self.seq, -1, None, n, v))
        while self.heap and self.heap[0][0] <= t_end:
            t = self.heap[0][0]
            changed = []
            while self.heap and self.heap[0][0] == t:
                _, sq, ci, pin, net, v = heapq.heappop(self.heap)
                if ci >= 0:
                    p = self.pending.get((ci, pin))
                    if p is None or p[2] != sq:
                        if self.mode == 'inertial': continue   # cancelled
                    if p is not None and p[2] == sq: del self.pending[(ci, pin)]
                    if self.val[net] == v: continue
                    self.out_events[ci] += 1
                old = self.val[net]
                if old == v: continue
                self.val[net] = v
                self.trace.append((t, net, v))
                changed.append((net, old, v))
            # flops first sample with post-update values of this timestep except their own outputs (none change at t from CLK)
            touched = {}
            for net, old, v in changed:
                for ci, p in self.fanout[net]:
                    touched.setdefault(ci, []).append((p, old, v))
            for ci, pins in touched.items():
                c = self.d.cells[ci]
                self.evals[ci] += 1
                iv = self._inputs(c)
                for op, fn in c.fn.items():
                    onet = c.conn.get(op)
                    if onet is None or isinstance(onet, tuple): continue
                    if fn[0] == 'ff':
                        clk, dpin = fn[1], fn[2]
                        if any(p == clk and old == 0 and v == 1 for p, old, v in pins):
                            self._schedule(ci, op, onet, t, iv[dpin], [clk], iv)
                    else:
                        nv = fn[2](iv)
                        self._schedule(ci, op, onet, t, nv, [p for p, _, _ in pins], iv)
        return self.trace
