# GL0AM on CUDA 12 / CuPy 13 (local port, branch `cuda12-port`, not pushed)

Base: NVlabs/GL0AM main @ 14b5dd0. Patch: `cuda12.patch` (git diff, CRLF line endings preserved, applies with `git apply`).
Tested 2026-11-03 on RTX 3060 Laptop (6 GB, sm_86), WSL Ubuntu 26.04, driver 581.95.

Stack: python 3.10, torch 2.2.0+cu121, dgl 2.4.0+cu121, cupy-cuda12x 13.6.0, conda cuda-nvcc 12.4 + gxx 12,
nvidia-cuda-nvrtc-cu12 12.1.105, KaHyPar (hMetis replacement, writes hMetis `.part.K` format).
Env: `CUPY_NVCC_PATH=$CONDA_PREFIX/bin/nvcc`, `NVCC_PREPEND_FLAGS="-ccbin $CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"`,
`LD_LIBRARY_PATH=/usr/lib/wsl/lib:<nvidia/cuda_nvrtc/lib>`.

## Changes (3 files, +99/-75, most lines are the mechanical RawKernel rename)
1. **GL0AM.cupy header**: the `/* ... */` licence block is not valid Python but the file is `exec()`'d; turned into `#` comments.
2. **cuda_bf16.h / fp16 compile error** (`no instance of overloaded function "__half::__half"`, then libcudacxx
   `call to non-constexpr __half::__half(const __half_raw&)`): CuPy 13 puts its bundled, older `cuda_fp16.h`
   (`cupy/_core/include/cupy/_cuda/cuda-12`) ahead of the toolkit on the nvcc `-I` path. Fix: every kernel gets
   `-include <toolkit>/cuda_fp16.h`, so the toolkit header is seen first and the bundled one is skipped by the shared
   `__CUDA_FP16_H__` guard. Done via a `_RawKernel12` wrapper (all 65 `cp.RawKernel(` calls) and the same options on
   runSim's `cp.RawModule`. `-D__CUDA_NO_HALF_CONVERSIONS__` alone was not enough (it fixed bf16, not libcudacxx).
3. **`cupyx.segmented_sort`** (only in the CuPy 11.0.0b2 leofang fork): replaced by `_segmented_sort` =
   `lexsort((vals, segment_id))` in partitioningSetup.py. Unit check `[5,3,9|·|1,2,8,7]` -> `[3,5,9,1,2,7,8]`.
4. **Torch CUDA tensors used as CuPy indices / slice bounds**: CuPy 13 routes them through `__array__`, which torch
   refuses for CUDA tensors. (a) 6 slice expressions wrapped in `int()` (partitioningSetup x4, runSim x2;
   `port_slice_ints.py` does it). (b) Fancy-index cases (e.g. partitioningSetup line ~185) handled by a shim in
   GL0AM.cupy that makes `torch.Tensor.__array__` copy CUDA tensors to host. Correct but adds a D2H copy per such
   index; a proper fix would be `cp.asarray(tensor)` at each site.
No `__shfl`-without-`_sync` or other deprecated intrinsics were found; no kernel source changes were needed.

## Results
- All 64 module-level RawKernels compile (nvcc 12.4, sm_86) — 0 failures.
- partitioningSetup -> KaHyPar(K=8, eps 0.03) -> partitioningPostprocessing run end-to-end on gcd and acc8;
  GL0AM's own built-in checks ("traverse cones ... True", "Cluster size of N seems correct") all pass.

| block | setup wall s | setup maxRSS | hgr nets/verts | part sizes | dup factor | postproc wall s | output |
|---|---|---|---|---|---|---|---|
| gcd  | 16.79 | 1.22 GB | 67 / 57 | 6,6,6,7,7,7,8,10 | 2.86 | 3.60 | gcd.8.partitions (124 KB) |
| acc8 | 4.48  | 1.14 GB | 15 / 19 | 1,1,1,2,2,3,4,5  | 2.54 | 3.90 | acc8.8.partitions (78 KB) |

(Wall times include ~3 s python/CUDA start-up and first-run nvcc compiles; designs are tiny, GPU memory use negligible vs 6 GB.)

## Not done: runSim.py
Blocked on input format, not on CUDA. runSim needs (a) `cell_type` codes = index into its hard-coded
GENERIC cell list (`cells_list`, AND2=0, AND3=1, ...) with GL0AM pin order — our precompile uses auto codes >=1000 for
sky130 cells, so the logic tables would not match; (b) a gzipped pickle of `{'waveforms','waveforms_value'}` per net;
(c) `--pin_net_file` (pin<->net pairs) and `--clkTreeFile`. Next step is an adapter in our precompile
(sky130 -> GENERIC function/pin mapping + waveform/pin-net export from our VCD), then compare against Icarus.
