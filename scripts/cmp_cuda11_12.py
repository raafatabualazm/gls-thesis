"""Compare upstream GL0AM on CUDA 11.8 vs CUDA 12 port: outputs bit-exact, partitions identical, stage timings (median of N)."""
import gzip, hashlib, json, os, re, statistics, sys, glob, numpy as np
W = os.path.expanduser('~/gls-work'); A, B = W + '/bench_cu118', W + '/bench_cu12'
def h(p):
    d = open(p, 'rb').read()
    if p.endswith('.gz'): d = gzip.decompress(d)
    if p.endswith('.saif.gz'): d = b'\n'.join(l for l in d.split(b'\n') if not re.match(rb'\s*\((DATE|VENDOR|PROGRAM_NAME|VERSION)', l))
    return hashlib.sha256(d).hexdigest()
def npz_eq(p, q):
    a, b = np.load(p), np.load(q)
    return sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
def times(root):
    out = {}
    for r in sorted(glob.glob(root + '/run*')):
        # per-run log section: parse bench.log split by '##### '
        pass
    log = open(root + '/bench.log').read()
    blocks = re.split(r'##### (\w+)\n', log)[1:]
    for b, txt in zip(blocks[0::2], blocks[1::2]):
        d = out.setdefault(b, {})
        for k, v in re.findall(r'(\w+) wall_s=([\d.]+)', txt): d.setdefault(k + '_wall_s', []).append(float(v))
        for k, pat in [('zero_delay_sim_s', r'0-delay sim done in ([\d.]+)'), ('signal_T_s', r'signal T time: ([\d.]+)'),
                       ('waveform_creation_s', r'Waveform creation time: ([\d.]+)'), ('resim_s', r're-sim time: ([\d.]+)'),
                       ('graph_compile_s', r'compile time is: ([\d.]+)')]:
            for v in re.findall(pat, txt): d.setdefault(k, []).append(float(v))
    return out
res = {'designs': {}, 'notes': []}
ta, tb = times(A), times(B)
for b in ['gcd', 'acc8']:
    files = ['outGL0AM.%s.saif.gz' % b, '%s.hgr' % b, '%s.hgr.part.8' % b, '%s.8.partitions' % b]
    r = {'bit_exact': {}, 'run_to_run_deterministic': {}}
    for f in files + ['%s_waves_0.npz' % b, '%s_regs.npz' % b]:
        pa, pb = A + '/run1/' + f, B + '/run1/' + f
        if not (os.path.exists(pa) and os.path.exists(pb)): r['bit_exact'][f] = 'missing'; continue
        r['bit_exact'][f] = npz_eq(pa, pb) if f.endswith('.npz') else h(pa) == h(pb)
        for tag, root in [('cu118', A), ('cu12', B)]:
            ps = sorted(glob.glob(root + '/run*/' + f))
            r['run_to_run_deterministic'].setdefault(f, {})[tag] = all((npz_eq(ps[0], p) if f.endswith('.npz') else h(ps[0]) == h(p)) for p in ps)
    r['timings_median_s'] = {}
    for k in sorted(set(ta.get(b, {})) | set(tb.get(b, {}))):
        xa, xb = ta[b].get(k, []), tb[b].get(k, [])
        r['timings_median_s'][k] = {'cu118': round(statistics.median(xa), 3) if xa else None, 'cu12': round(statistics.median(xb), 3) if xb else None, 'n': [len(xa), len(xb)]}
    res['designs'][b] = r
print(json.dumps(res, indent=1))
