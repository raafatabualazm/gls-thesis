#!/bin/bash
# usage: run_bench.sh [reps]   (median of reps after 1 warm-up)
R=${1:-21}
W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba
eval "$($W/bin/micromamba shell hook -s bash)"; micromamba activate gls
mkdir -p $W/gpu && cd $W/gpu && cp -r "/mnt/c/Users/Raafat Abualazm/gls-work/gpu/"* . 
nvcc -O3 -arch=sm_86 -ccbin $CXX -o partbench partbench.cu 2>&1 | grep -v -i warning | head -20
[ -x partbench ] && [ partbench -nt partbench.cu ] || { echo BUILD_FAILED; exit 1; }
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$CONDA_PREFIX/lib
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,pstate --format=csv,noheader
for d in gcd acc8; do echo "### $d"; cat data/${d}_meta.txt | tr '\n' ' '; echo; ./partbench data/$d.bin 1 $R 128; done
echo "### gcd loops=40 (1000 cycles)"; ./partbench data/gcd.bin 40 $R 128
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,power.draw,pstate --format=csv,noheader
