import pickle, sys, glob, os, numpy as np, torch
W = os.path.expanduser('~/gls-work')
def load(p):
    return pickle.load(open(p, 'rb'))
def canon(o):
    if isinstance(o, torch.Tensor): return ('T', str(o.dtype), tuple(o.shape), o.cpu().numpy().tobytes())
    if hasattr(o, 'get') and type(o).__module__.startswith('cupy'): o = o.get()
    if isinstance(o, np.ndarray): return ('A', str(o.dtype), o.shape, o.tobytes())
    if isinstance(o, dict): return ('D', tuple((k, canon(v)) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))))
    if isinstance(o, (list, tuple)): return ('L', tuple(canon(x) for x in o))
    if isinstance(o, (set, frozenset)): return ('S', tuple(sorted(map(repr, o))))
    if hasattr(o, 'ndata'):  # DGL graph
        s, d = o.edges(); return ('G', o.num_nodes(), canon(s), canon(d), canon(dict(o.ndata)), canon(dict(o.edata)))
    return ('O', repr(o))
for b in ['gcd', 'acc8']:
    ps = sorted(glob.glob(W + '/bench_cu118/run*/%s.8.partitions' % b)) + sorted(glob.glob(W + '/bench_cu12/run*/%s.8.partitions' % b))
    o = load(ps[0]); print(b, 'type', type(o).__name__, len(o) if hasattr(o, '__len__') else '')
    c0 = canon(o); print(b, 'all 10 runs (5 cu118 + 5 cu12) semantically identical:', all(canon(load(p)) == c0 for p in ps[1:]))
