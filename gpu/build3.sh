#!/bin/bash
W=~/gls-work; export MAMBA_ROOT_PREFIX=$W/mamba
eval "$($W/bin/micromamba shell hook -s bash)"; micromamba activate gls
swig -version | grep Version; micromamba install -y -n gls -c conda-forge "swig=4.1.*" >/dev/null 2>&1; hash -r; swig -version | grep Version
cd $W/GTPower; git log --oneline -1; rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDD_DIR=$W/cudd -DCMAKE_CUDA_ARCHITECTURES=86 \
   -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DUSE_TCL_READLINE=OFF > $W/gtp_cmake.log 2>&1 || { echo CMAKE_FAIL; tail -40 $W/gtp_cmake.log; exit 1; }
/usr/bin/time -f "build_wall_s=%e" cmake --build build -j12 > $W/gtp_build.log 2>&1 || { echo BUILD_FAIL; grep -n -i -m10 -B3 -A4 ' error\|Error [0-9]\|undefined ref' $W/gtp_build.log; exit 1; }
tail -1 $W/gtp_build.log; ls -la app/gtpower
cd test/gcd; mkdir -p res res-cpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
for i in 1 2 3; do /usr/bin/time -f "cuda_run$i wall_s=%e maxrss_kB=%M" ../../app/gtpower -exit -report_cuda_power_thread_alloc_stat ./power_vcd.tcl > res/run$i.log 2>&1; echo rc=$?; done
/usr/bin/time -f "cpu_run wall_s=%e" ../../app/gtpower -exit -result_dir ./res-cpu -disable_cuda_power_analysis ./power_vcd.tcl > res-cpu/run.log 2>&1; echo rc=$?
echo ==== CUDA LOG; cat res/run1.log | grep -v '^$' | head -150
echo ==== CPU LOG; grep -i -E 'power|total|time|runtime' res-cpu/run.log | head -40
ls -la res res-cpu; head -5 res/cuda_cell_power.txt
nvidia-smi --query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw --format=csv
