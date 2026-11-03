"""Liberty boolean function -> AST -> Python evaluator and Verilog expression.
Precedence (high->low): ! and postfix ', & (and juxtaposition), ^, | / +."""
import re
_T = re.compile(r"\s*([A-Za-z_][A-Za-z_0-9\[\]]*|[!&|^()'+*]|1|0)")

def tokenize(s):
    out, pos = [], 0
    s = s.strip()
    while pos < len(s):
        m = _T.match(s, pos)
        if not m: raise ValueError(f"bad expr {s!r} at {pos}")
        out.append(m.group(1)); pos = m.end()
    return out

def parse(s):
    toks = tokenize(s); i = [0]
    def peek(): return toks[i[0]] if i[0] < len(toks) else None
    def take(): t = toks[i[0]]; i[0] += 1; return t
    def atom():
        t = take()
        if t == '!': n = ('not', atom())
        elif t == '(':
            n = orx()
            assert take() == ')'
        elif t in ('0', '1'): n = ('const', int(t))
        else: n = ('var', t)
        while peek() == "'":
            take(); n = ('not', n)
        return n
    def andx():
        n = atom()
        while peek() in ('&', '*') or (peek() is not None and peek() not in ('|', '+', '^', ')')):
            if peek() in ('&', '*'): take()
            n = ('and', n, atom())
        return n
    def xorx():
        n = andx()
        while peek() == '^': take(); n = ('xor', n, andx())
        return n
    def orx():
        n = xorx()
        while peek() in ('|', '+'): take(); n = ('or', n, xorx())
        return n
    n = orx()
    if i[0] != len(toks): raise ValueError(f"trailing tokens in {s!r}")
    return n

def to_py(n):
    k = n[0]
    if k == 'var': return f"v[{n[1]!r}]"
    if k == 'const': return str(n[1])
    if k == 'not': return f"(1-{to_py(n[1])})"
    op = {'and': '&', 'or': '|', 'xor': '^'}[k]
    return f"({to_py(n[1])}{op}{to_py(n[2])})"

def to_verilog(n, rename):
    k = n[0]
    if k == 'var': return rename(n[1])
    if k == 'const': return f"1'b{n[1]}"
    if k == 'not': return f"(~{to_verilog(n[1], rename)})"
    op = {'and': '&', 'or': '|', 'xor': '^'}[k]
    return f"({to_verilog(n[1], rename)}{op}{to_verilog(n[2], rename)})"

def variables(n, acc=None):
    acc = set() if acc is None else acc
    if n[0] == 'var': acc.add(n[1])
    for c in n[1:]:
        if isinstance(c, tuple): variables(c, acc)
    return acc

def compile_fn(s):
    return eval("lambda v: " + to_py(parse(s)))
