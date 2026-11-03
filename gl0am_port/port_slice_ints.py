# CUDA 12 / CuPy 13 port: CuPy 13 rejects 0-d torch CUDA tensors as slice bounds; wrap with int().
import re
for p in ['GL0AM/GL0AM.partitioningSetup.py','GL0AM/GL0AM.runSim.py']:
    s=open(p,newline='').read(); out=[]; i=0; n=0
    for m in re.finditer(r"(listOfEndpointsStorage|waveforms_total)\[(?=bg\.ndata\[)", s):
        st=m.end()
        if st<i: continue
        d=0; j=st; parts=[]; last=st
        while True:
            c=s[j]
            if c in '([{': d+=1
            elif c in ')]}':
                if d==0: break
                d-=1
            elif c==':' and d==0: parts.append(s[last:j]); last=j+1
            j+=1
        parts.append(s[last:j])
        if len(parts)!=2: continue
        out.append(s[i:st]); out.append(':'.join('int(%s)'%x.strip() if 'int(' not in x[:4] else x for x in parts)); i=j; n+=1
    out.append(s[i:]); s=''.join(out); open(p,'w',newline='').write(s); print(p,'wrapped',n)
