#!/bin/bash
# user-level toolchain in ~/gls-work (WSL) -- no sudo, nothing system-wide
set -x
W=~/gls-work; mkdir -p $W; cd $W
WIN="/mnt/c/Users/Raafat Abualazm/gls-work"
export MAMBA_ROOT_PREFIX=$W/mamba
if [ ! -x $W/bin/micromamba ]; then
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C $W bin/micromamba || exit 1
fi
MM=$W/bin/micromamba
if [ ! -d $MAMBA_ROOT_PREFIX/envs/gls ]; then
  $MM create -y -n gls -c conda-forge -c nvidia python=3.11 cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-version=12.8 \
     cmake swig flex bison tk eigen zlib gxx_linux-64=13 gcc_linux-64=13 make autoconf automake libtool readline >$W/mamba_create.log 2>&1 || { tail -30 $W/mamba_create.log; exit 1; }
fi
eval "$($MM shell hook -s bash)"; micromamba activate gls
which nvcc cmake swig flex bison tclsh x86_64-conda-linux-gnu-g++; nvcc --version | tail -2
# CUDD
if [ ! -f $W/cudd/lib/libcudd.a ]; then
  git clone -q https://github.com/ivmai/cudd.git $W/cudd-src && cd $W/cudd-src && \
  (autoreconf -fi >/dev/null 2>&1; ./configure --prefix=$W/cudd CC=$CC CXX=$CXX >$W/cudd_conf.log 2>&1 && make -j16 >$W/cudd_make.log 2>&1 && make install >>$W/cudd_make.log 2>&1) || { echo CUDD_FAIL; tail -20 $W/cudd_conf.log $W/cudd_make.log; }
  cd $W
fi
ls $W/cudd/lib $W/cudd/include 2>&1 | head
# GTPower source from the Windows clone (commit 7da030d)
[ -d $W/GTPower ] || cp -r "$WIN/GTPower" $W/GTPower
cd $W/GTPower; git log --oneline -1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDD_DIR=$W/cudd -DCMAKE_CUDA_ARCHITECTURES=86 \
   -DCMAKE_CUDA_HOST_COMPILER=$CXX -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DUSE_TCL_READLINE=OFF > $W/gtp_cmake.log 2>&1 || { echo CMAKE_FAIL; tail -40 $W/gtp_cmake.log; exit 1; }
cmake --build build -j12 > $W/gtp_build.log 2>&1 || { echo BUILD_FAIL; grep -m20 -B2 -A5 "error" $W/gtp_build.log; exit 1; }
ls -la build/sta* build/*/sta* 2>/dev/null; echo BUILD_OK
