#!/bin/bash
W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba
eval "$($W/bin/micromamba shell hook -s bash)"; micromamba activate gl0am
pip -q install nvidia-cuda-nvrtc-cu12==12.1.105 2>&1 | tail -1
NV=$(python -c "import nvidia.cuda_nvrtc,os;print(os.path.dirname(nvidia.cuda_nvrtc.__file__)+'/lib')")
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$NV
[ -x $CONDA_PREFIX/bin/nvcc ] || micromamba install -y -n gl0am -c conda-forge cuda-nvcc=12.4 cuda-cudart-dev=12.4 cuda-version=12.4 gxx_linux-64=12 >/dev/null 2>&1; G=$CONDA_PREFIX; nvcc --version | tail -1; export CUPY_NVCC_PATH=$G/bin/nvcc NVCC_PREPEND_FLAGS="-ccbin $G/bin/x86_64-conda-linux-gnu-g++"
python -c "import cupy;a=cupy.arange(10);print('cupy ok',int(a.sum()))"
cd $W/gl0am_run; sed "1,8d" $W/GL0AM/GL0AM/GL0AM.cupy > GL0AM.cupy; head -2 GL0AM.cupy; S=$W/GL0AM/GL0AM; K=8
for b in gcd acc8; do echo "##### $b"
 /usr/bin/time -f "partitioningSetup wall_s=%e maxrss_kB=%M" python $S/GL0AM.partitioningSetup.py --block $b --graph4 ${b}_combo_DGLGraph 2>&1 | tail -8
 head -1 $b.hgr
 python - $b $K <<'P'
import sys,kahypar
b,k=sys.argv[1],int(sys.argv[2]); L=open(b+'.hgr').read().split('\n'); h=L[0].split(); ne,nv=int(h[0]),int(h[1]); fmt=h[2] if len(h)>2 else '0'
nets=[];nw=[]
for l in L[1:1+ne]:
    x=[int(t) for t in l.split()]
    if fmt in('1','11'): nw.append(x[0]); x=x[1:]
    else: nw.append(1)
    nets.append([v-1 for v in x])
vw=[1]*nv
if fmt in('10','11'): vw=[int(l) for l in L[1+ne:1+ne+nv]]
idx=[0];fl=[]
for n in nets: fl+=n; idx.append(len(fl))
H=kahypar.Hypergraph(nv,ne,idx,fl,k,nw,vw); c=kahypar.Context(); c.loadINIconfiguration('kahypar_cut.ini'); c.setK(k); c.setEpsilon(0.03); c.setSeed(1); c.suppressOutput(True)
kahypar.partition(H,c); bl=[H.blockID(i) for i in range(nv)]
open(f'{b}.hgr.part.{k}','w').write('\n'.join(map(str,bl))+'\n')
from collections import Counter; print('hgr fmt',fmt,'nets',ne,'verts',nv,'part sizes',sorted(Counter(bl).values()))
P
 /usr/bin/time -f "partitioningPostprocessing wall_s=%e maxrss_kB=%M" python $S/GL0AM.partitioningPostprocessing.py --block $b --graph2 ${b}_update_DGLGraph --graph3 ${b}_sram_DGLGraph --graph4 ${b}_combo_DGLGraph --hMetisResults $b.hgr.part.$K --partitions $K --graphPrecompile ${b}_GraphPrecompile.pkl.gz --graph ${b}_fullSDF_DGLGraph 2>&1 | tail -15
 ls -la $b.$K.partitions 2>&1
done
nvidia-smi --query-gpu=temperature.gpu,memory.used --format=csv,noheader
