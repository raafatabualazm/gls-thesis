# GL0AM runSim input adapter (sky130 -> GL0AM)

Code: `gls/gl0am_adapter.py`, `scripts/gl0am_export.py`, `scripts/gl0am_check_graph.py`, `scripts/gl0am_compare.py`.
Laptop runner: `gl0am_port/rs_run.sh`; runSim patch: `gl0am_port/rs_patch.py` (env-sized buffers, optional
INIT_SEQ_ZERO, npz wave dump). GPU: RTX 3060 Laptop 6 GB, CUDA 12, WSL Ubuntu-26.04.

## Mapping
Every sky130 comb cell is matched to a GL0AM cells_list entry by exhaustive pin permutation x input/output
inversion (fewest inserted INVs). Decomposed (no direct match):
- a311oi_4 = AOI211(AND2(A1,A2),A3,B1,C1)
- a311o_1  = AO211(AND2(A1,A2),A3,B1,C1)
- o311a_2  = OA31(A1,A2,A3,AND2(B1,C1))
- o311ai_4 = OAI31(A1,A2,A3,AND2(B1,C1))

**Pin-order rule (the bug that caused wrong flop data):** runSim builds truth tables with numberToBase
(MSB-first), but kernels index by value*3^(x-1). So cells_list pin j of an n-pin cell gets edge x = n-j;
delay-LUT mask bit = n-1-j. Flop pins are not reversed (D=1 ... CP=7).

## Approximations
- Edges into inserted H gates carry unconditional arcs; H->G edge is zero delay; inserted INVs are zero delay.
- Clock-tree insertion delay is folded into the flop pushout.
- Flops initialised to 0 (GL0AM_INIT_SEQ_ZERO) instead of X.

## Results (laptop, 2026-11-03 ~14:00 Africa/Cairo, UTC+2)
See results/gl0am_cmp_*.json. acc8: 7/140 nets differ (glitch timing), SAIF TC exact on 133/140.
gcd: flops and PIs match Icarus exactly; 77/288 comb nets go X in GL0AM re-sim from t=55000 ps (open issue).
