"""Emit a self-contained flat Verilog model (+ stimulus) for Icarus cross-validation.
Combinational outputs: `assign #(rise,fall) y = f(...)` (Verilog continuous-assign inertial semantics).
Flops: `always @(posedge CLK) q_int <= D; assign #(r,f) Q = q_int;`. Requires a uniform SDF
(one rise/fall per output pin) because Verilog continuous assigns can't select delay per input pin."""
from . import boolexpr

def emit(design, sdf_arcs, stimulus, init_inputs, t_end, path, vcd_out):
    W = lambda n: f"n{n}"
    L = ["`timescale 1ps/1ps", "module ref;"]
    nets = set()
    for c in design.cells:
        for n in c.conn.values():
            if not isinstance(n, tuple): nets.add(n)
    nets |= set(design.inputs)
    for n in sorted(nets):
        L.append(f"  {'reg' if n in design.inputs else 'wire'} {W(n)};")
    for c in design.cells:
        arcs = sdf_arcs.get(c.name, [])
        for op, fn in c.fn.items():
            onet = c.conn.get(op)
            if onet is None: continue
            a = [x for x in arcs if x['to'] == op]
            r = max(x['rise_ps'] for x in a); f = max(x['fall_ps'] for x in a)
            assert all(x['rise_ps'] == r and x['fall_ps'] == f for x in a), f"non-uniform SDF for {c.name}"
            def rn(p, c=c):
                n = c.conn[p]
                return f"1'b{n[1]}" if isinstance(n, tuple) else W(n)
            if fn[0] == 'ff':
                L.append(f"  reg q_{c.name} = 1'b0;")
                L.append(f"  always @(posedge {rn(fn[1])}) q_{c.name} <= {rn(fn[2])};")
                L.append(f"  assign #({int(r)},{int(f)}) {W(onet)} = q_{c.name};")
            else:
                L.append(f"  assign #({int(r)},{int(f)}) {W(onet)} = {boolexpr.to_verilog(fn[1], rn)};")
    L.append("  initial begin")
    L.append(f'    $dumpfile("{vcd_out}"); $dumpvars(0, ref);')
    for n, v in init_inputs.items(): L.append(f"    {W(n)} = 1'b{v};")
    t = 0
    for ts, n, v in sorted(stimulus, key=lambda x: x[0]):
        if ts != t: L.append(f"    #{ts - t};"); t = ts
        L.append(f"    {W(n)} = 1'b{v};")
    L.append(f"    #{t_end - t}; $finish;")
    L.append("  end\nendmodule")
    open(path, 'w').write('\n'.join(L) + '\n')

def emit_specify(design, sdf_arcs, stimulus, init_inputs, t_end, path, vcd_out):
    """Per-pin IOPATH variant: one module per instance with a `specify` block holding that instance's
    IOPATH rise/fall (unconditional arcs only). Compile with `iverilog -gspecify`."""
    W = lambda n: f"n{n}"
    L = ["`timescale 1ps/1ps"]
    inst = []
    for ci, c in enumerate(design.cells):
        arcs = [a for a in sdf_arcs.get(c.name, []) if not a['cond']]
        ins = [p for p, x in c.info['pins'].items() if x['dir'] == 'input']
        outs = [p for p, x in c.info['pins'].items() if x['dir'] == 'output']
        m = f"c{ci}"
        L.append(f"module {m}(" + ', '.join([f"input {p}" for p in ins] + [f"output {p}" for p in outs]) + ");")
        for op, fn in c.fn.items():
            if fn[0] == 'ff':
                L.append(f"  reg q = 1'b0; always @(posedge {fn[1]}) q <= {fn[2]}; assign {op} = q;")
            else:
                L.append(f"  assign {op} = {boolexpr.to_verilog(fn[1], lambda p: p)};")
        L.append("  specify")
        for a in arcs:
            L.append(f"    ({a['from']} => {a['to']}) = ({int(a['rise_ps'])}, {int(a['fall_ps'])});")
        L.append("  endspecify\nendmodule")
        conns = []
        for p in ins + outs:
            n = c.conn.get(p)
            if n is None: continue
            conns.append(f".{p}({chr(39).join(['1', 'b' + str(n[1])]) if isinstance(n, tuple) else W(n)})")
        inst.append(f"  {m} u{ci}(" + ', '.join(conns) + ");")
    L.append("module ref;")
    nets = set()
    for c in design.cells:
        for n in c.conn.values():
            if not isinstance(n, tuple): nets.add(n)
    nets |= set(design.inputs)
    for n in sorted(nets):
        L.append(f"  {'reg' if n in design.inputs else 'wire'} {W(n)};")
    L += inst
    L.append("  initial begin")
    L.append(f'    $dumpfile("{vcd_out}"); $dumpvars(0, ref);')
    for n, v in init_inputs.items(): L.append(f"    {W(n)} = 1'b{v};")
    t = 0
    for ts, n, v in sorted(stimulus, key=lambda x: x[0]):
        if ts != t: L.append(f"    #{ts - t};"); t = ts
        L.append(f"    {W(n)} = 1'b{v};")
    L.append(f"    #{t_end - t}; $finish;")
    L.append("  end\nendmodule")
    open(path, 'w').write('\n'.join(L) + '\n')
