import os, sys, pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
BIN = os.environ.get('OSS_CAD_BIN', '/workspace/oss-cad-suite/bin')
LIB = os.path.join(ROOT, 'designs/gcd/sky130hd_tt.lib')

@pytest.fixture(scope='session')
def tiny(tmp_path_factory):
    from gls import frontend
    return frontend.load(os.path.join(ROOT, 'designs/tiny/tiny.v'), LIB, 'tiny', str(tmp_path_factory.mktemp('t')), yosys=f'{BIN}/yosys')

@pytest.fixture(scope='session')
def gcd(tmp_path_factory):
    from gls import frontend, sdf, graph
    out = tmp_path_factory.mktemp('g')
    d = frontend.load(os.path.join(ROOT, 'designs/gcd/gcd_sky130hd.v'), LIB, 'gcd', str(out), yosys=f'{BIN}/yosys')
    p = str(out / 'gcd.sdf'); sdf.write(d, p, design_name='gcd')
    return d, sdf.parse(p), graph.GateGraph(d, sdf.parse(p)), out
