"""Gate-level Verilog + Liberty -> in-memory netlist via Yosys JSON (read_liberty -lib; write_json)."""
import json, os, subprocess, re
from . import liberty, boolexpr

SKIP = re.compile(r'__(tap|fill|decap|tapvpwrvgnd|diode)')

class Cell:
    __slots__ = ('name', 'type', 'conn', 'info', 'fn', 'is_seq')
    def __init__(self, name, type_, conn):
        self.name, self.type, self.conn = name, type_, conn

class Design:
    def __init__(self):
        self.cells, self.inputs, self.outputs, self.netname, self.const = [], [], [], {}, {}
        self.libcells = {}

def run_yosys(verilog, lib, top, json_out, yosys='yosys'):
    cmd = [yosys, '-q', '-p', f"read_liberty -lib {lib}; read_verilog {verilog}; hierarchy -top {top}; write_json {json_out}"]
    subprocess.run(cmd, check=True)

def load(verilog, lib, top, workdir='out', yosys='yosys'):
    os.makedirs(workdir, exist_ok=True)
    js = os.path.join(workdir, f'{top}.json')
    run_yosys(verilog, lib, top, js, yosys)
    m = json.load(open(js))['modules'][top]
    d = Design()
    for name, nn in m['netnames'].items():
        bits = nn['bits']
        for i, b in enumerate(bits):
            if isinstance(b, int):
                nm = name if len(bits) == 1 else f"{name}[{i + nn.get('offset', 0)}]"
                # prefer port / non-hidden names
                if b not in d.netname or (not nn.get('hide_name') and d.netname[b].startswith('$')):
                    d.netname[b] = nm
    for pname, p in m['ports'].items():
        for i, b in enumerate(p['bits']):
            (d.inputs if p['direction'] == 'input' else d.outputs).append(b)
            d.netname[b] = pname if len(p['bits']) == 1 else f"{pname}[{i}]"
    types = sorted({c['type'] for c in m['cells'].values() if not SKIP.search(c['type'])})
    d.libcells = liberty.read_cells(lib, types)
    for cname, c in m['cells'].items():
        if SKIP.search(c['type']): continue
        conn = {}
        for pin, bits in c['connections'].items():
            b = bits[0]
            conn[pin] = b if isinstance(b, int) else ('const', int(b))
        cell = Cell(cname, c['type'], conn)
        cell.info = d.libcells[c['type']]
        cell.is_seq = cell.info['ff'] is not None
        outs = [p for p, x in cell.info['pins'].items() if x['dir'] == 'output']
        cell.fn = {}
        for o in outs:
            f = cell.info['pins'][o]['function']
            if cell.is_seq:
                ff = cell.info['ff']
                assert f == ff['state'], f"unsupported seq output fn {f} in {c['type']}"
                cell.fn[o] = ('ff', ff['clocked_on'], ff['next_state'])
            else:
                ast = boolexpr.parse(f)
                cell.fn[o] = ('comb', ast, boolexpr.compile_fn(f))
        d.cells.append(cell)
    return d

def out_pins(cell):
    return [p for p, x in cell.info['pins'].items() if x['dir'] == 'output']

def in_pins(cell):
    return [p for p, x in cell.info['pins'].items() if x['dir'] == 'input']
