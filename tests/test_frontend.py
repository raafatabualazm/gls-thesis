import itertools, os
from gls import boolexpr, liberty, sdf
from conftest import LIB

def test_boolexpr_ops():
    f = boolexpr.compile_fn("!((A1&A2) | (B1&B2))")
    for a1, a2, b1, b2 in itertools.product([0, 1], repeat=4):
        assert f({'A1': a1, 'A2': a2, 'B1': b1, 'B2': b2}) == 1 - ((a1 & a2) | (b1 & b2))
    g = boolexpr.compile_fn("A B' + C^D")   # juxtaposition AND, postfix not, + as OR, ^ binds tighter than +
    for a, b, c, d in itertools.product([0, 1], repeat=4):
        assert g({'A': a, 'B': b, 'C': c, 'D': d}) == ((a & (1 - b)) | (c ^ d))

def test_liberty_cells():
    c = liberty.read_cells(LIB, ['sky130_fd_sc_hd__dfxtp_1', 'sky130_fd_sc_hd__nand2_1'])
    ff = c['sky130_fd_sc_hd__dfxtp_1']
    assert ff['ff'] == {'state': 'IQ', 'clocked_on': 'CLK', 'next_state': 'D'}
    assert [(a['from'], a['to'], a['type']) for a in ff['arcs']] == [('CLK', 'Q', 'rising_edge')]
    n = c['sky130_fd_sc_hd__nand2_1']
    assert n['pins']['Y']['function'] and {a['from'] for a in n['arcs']} == {'A', 'B'}
    assert all(a['rise_ns'] > 0 and a['fall_ns'] > 0 for a in n['arcs'])

def test_frontend_tiny(tiny):
    assert sorted(c.type.split('__')[1] for c in tiny.cells) == ['dfxtp_1', 'inv_1', 'inv_1', 'nand2_1']
    assert sum(c.is_seq for c in tiny.cells) == 1
    assert {tiny.netname[b] for b in tiny.inputs} == {'a', 'b', 'clk'}

def test_sdf_roundtrip_and_cond(tmp_path, tiny):
    p = tmp_path / 't.sdf'; sdf.write(tiny, str(p), design_name='tiny')
    arcs = sdf.parse(str(p))
    nand = [c for c in tiny.cells if 'nand2' in c.type][0]
    lib = {(a['from']): (round(a['rise_ns'] * 1000), round(a['fall_ns'] * 1000)) for a in nand.info['arcs']}
    assert {a['from']: (a['rise_ps'], a['fall_ps']) for a in arcs[nand.name]} == lib
    ff = [c for c in tiny.cells if c.is_seq][0]
    assert arcs[ff.name][0]['edge'] == 'posedge' and arcs[ff.name][0]['from'] == 'CLK'
    txt = '(DELAYFILE (TIMESCALE 1ns)\n (CELL (CELLTYPE "x") (INSTANCE u9)\n  (DELAY (ABSOLUTE\n   (COND A&!B (IOPATH A Y (0.1:0.2:0.3) (0.4)))\n   (IOPATH B Y (1) (2))\n  ))\n )\n)\n'
    q = tmp_path / 'c.sdf'; q.write_text(txt)
    a = sdf.parse(str(q))['u9']
    assert a[0]['cond'] == 'A&!B' and a[0]['rise_ps'] == 200 and a[0]['fall_ps'] == 400
    assert a[1]['cond'] is None and a[1]['rise_ps'] == 1000
