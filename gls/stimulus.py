"""Turn a reference VCD (e.g. the testbench trace shipped with GTPower's gcd) into primary-input stimulus."""
from . import vcd

def from_vcd(design, path, scope):
    ts, ch = vcd.read(path, scope)
    name2net = {design.netname[b]: b for b in design.inputs}
    init, stim = {}, []
    missing = []
    for nm, net in name2net.items():
        evs = ch.get(nm)
        if evs is None: missing.append(nm); init[net] = 0; continue
        cur = None
        for t, v in evs:
            v = 1 if v == '1' else 0   # x/z -> 0 (2-value)
            if t == 0: init[net] = v; cur = v; continue
            if v != cur: stim.append((t, net, v)); cur = v
        init.setdefault(net, 0)
    t_end = max((t for t, _, _ in stim), default=0)
    return ts, init, sorted(stim), t_end, missing

def random_clocked(design, clock, n_cycles, period=5000, skew=1000, seed=0, reset=None, reset_cycles=2):
    """Clock with rising edge at period/2 + i*period; other inputs change `skew` ps after each rising edge."""
    import random
    rnd = random.Random(seed)
    clk = [b for b in design.inputs if design.netname[b] == clock][0]
    rst = [b for b in design.inputs if design.netname[b] == reset][0] if reset else None
    others = [b for b in design.inputs if b not in (clk, rst)]
    init = {b: 0 for b in design.inputs}
    if rst is not None: init[rst] = 1
    stim = []
    for i in range(n_cycles):
        t = period // 2 + i * period
        stim += [(t, clk, 1), (t + period // 2, clk, 0)]
        for b in others: stim.append((t + skew, b, rnd.randint(0, 1)))
        if rst is not None and i == reset_cycles: stim.append((t + skew, rst, 0))
    t_end = n_cycles * period + period
    # drop no-change entries
    cur = dict(init); out = []
    for t, b, v in sorted(stim):
        if cur[b] != v: out.append((t, b, v)); cur[b] = v
    return init, out, t_end
