# Isolated CUDA 11.8 env for upstream GL0AM (all under ~/gls-work; nothing system-wide)
W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba; eval "$($W/bin/micromamba shell hook -s bash)"
set -x
if [ ! -d $W/mamba/envs/gl0am118 ]; then
micromamba create -y -n gl0am118 -c nvidia/label/cuda-11.8.0 -c conda-forge python=3.10 \
  cuda-nvcc cuda-cudart-dev cuda-libraries-dev cuda-nvrtc-dev cuda-profiler-api cuda-cccl \
  gcc_linux-64=11 gxx_linux-64=11 "cython<3" "numpy<1.25" fastrlock pip || exit 1
fi
micromamba activate gl0am118
export CUDA_PATH=$CONDA_PREFIX CUDA_HOME=$CONDA_PREFIX
nvcc --version | tail -2; $CC --version | head -1
python -c "import torch" 2>/dev/null || pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu118 || exit 1
python -c "import dgl" 2>/dev/null || pip install dgl==2.0.0 -f https://data.dgl.ai/wheels/cu118/repo.html || exit 1
pip install "setuptools<70" "torchdata<0.8" >/dev/null || exit 1
pip install "scipy==1.10.1" "networkx==2.6.3" "pandas<2" kahypar "pydantic" "pyyaml" >/dev/null || exit 1
if ! python -c "import cupy, cupyx; cupyx.segmented_sort" 2>/dev/null; then
  cd $W; [ -d cupy-fork ] || git clone -q https://github.com/leofang/cupy.git --branch segmented_sort --recursive cupy-fork || exit 1
  cd cupy-fork; export NVCC="$CONDA_PREFIX/bin/nvcc -ccbin $CXX" CUPY_NUM_BUILD_JOBS=8 CUPY_NVCC_GENERATE_CODE="arch=compute_86,code=sm_86"
  export LDFLAGS="-L$CONDA_PREFIX/lib -L$CONDA_PREFIX/lib/stubs" CFLAGS="-I$CONDA_PREFIX/include" 
  pip install --no-build-isolation -v . > $W/cupy118_build.log 2>&1 || { tail -40 $W/cupy118_build.log; exit 1; }
fi
cd $W; [ -d GL0AM-upstream ] || git -C GL0AM worktree add --detach ../GL0AM-upstream main || exit 1
git -C GL0AM-upstream log --oneline -1; git -C GL0AM-upstream status --short | head
LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib python -c "
import cupy, cupyx, torch, dgl, numpy; print('cupy', cupy.__version__, 'cuda rt', cupy.cuda.runtime.runtimeGetVersion(), 'torch', torch.__version__, torch.cuda.is_available(), 'dgl', dgl.__version__, 'numpy', numpy.__version__)
print(cupy.arange(10).sum(), hasattr(cupyx,'segmented_sort'))"
echo ENV118_DONE
