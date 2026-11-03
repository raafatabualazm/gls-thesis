#!/bin/bash
W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba; MM=$W/bin/micromamba
eval "$($MM shell hook -s bash)"
[ -d $MAMBA_ROOT_PREFIX/envs/gl0am ] || $MM create -y -n gl0am -c conda-forge python=3.10 pip >/dev/null 2>&1
micromamba activate gl0am; python -V
pip -q install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -2
pip -q install dgl -f https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html 2>&1 | tail -2
pip -q install "numpy<2" scipy networkx cupy-cuda12x kahypar pydantic pyyaml 2>&1 | tail -2
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
python - <<'P'
import torch, dgl, cupy, numpy
print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))
print('dgl', dgl.__version__, 'cupy', cupy.__version__, 'numpy', numpy.__version__)
f,t=torch.cuda.mem_get_info(); print('free/total GB', f/1e9, t/1e9)
g=dgl.graph(([0,1],[1,2])).to('cuda'); print('dgl cuda graph ok', g.device)
a=cupy.arange(10); print('cupy ok', int(a.sum()))
P
mkdir -p $W/gl0am_run && cd $W/gl0am_run && cp "/mnt/c/Users/Raafat Abualazm/gls-work/gl0am_in/"* .
[ -d $W/GL0AM ] || git clone -q https://github.com/NVlabs/GL0AM $W/GL0AM
S=$W/GL0AM/GL0AM
for b in gcd acc8; do echo "##### $b"
 /usr/bin/time -f "generateGraph wall_s=%e maxrss_kB=%M" python $S/GL0AM.generateGraph.py --block $b --graphPrecompile ${b}_GraphPrecompile.pkl.gz 2>&1 | tail -15
 ls
 /usr/bin/time -f "splitLargeFOs wall_s=%e" python $S/GL0AM.splitLargeFOs.py --combo_graph ${b}_fullSDF_DGLGraph --target_combo_graph ${b}_combo_DGLGraph 2>&1 | tail -10
 /usr/bin/time -f "partitioningSetup wall_s=%e" python $S/GL0AM.partitioningSetup.py --block $b --graph4 ${b}_combo_DGLGraph 2>&1 | tail -10
 ls -la
done
