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

## Decomposition timing (exact for single-output inertial cells)
The inserted H gate (AND2) and inserted INVs have **zero delay**; the original cell's arc delay is put on the
H->core edge (and on the core's other edges). Every original input therefore reaches the core at its real time,
which reproduces one inertial assign per output. (The first version put the delay on the H inputs; when A3 fell
before the delayed H rose, the a311o output rose 88 ps late in acc8 - fixed.)

## GL0AM bug found: zero-delay fold boundaries must be clock-aligned
runSim bins zero-delay register events into 80 folds of ceil(duration/80) ps and
`setFullWaveformRegisterPseudoInputsBlockProcess` assumes the event buffer is ordered by fold. Events from
one clock edge are appended in cycle order but binned by edge+clk->Q pushout, so when a fold boundary lies
between edge+pushout of two flops, events land in the wrong fold, a slot keeps its cp.zeros value
(time 0), and `calculateNumberOfPerSplitSwitches` then restarts the waveform -> register switches dropped,
end-token X at the next split (gcd: X from 55 ns). Workaround in our inputs (no kernel change):
choose `--duration` = 80*period*m so folds are period-aligned and edges+pushouts never straddle
(gcd 400000, acc8 10400000). Not fixed in the kernels.

## Approximations
- Clock-tree insertion delay is folded into the flop pushout.
- Flops initialised to 0 (GL0AM_INIT_SEQ_ZERO) instead of X.

## Results (laptop RTX 3060 6 GB, 3 Nov 2026, Africa/Cairo UTC+2)
results/gl0am_cmp_{gcd,acc8}.json, warmup 5 ns, Icarus with the same uniform SDF:
- gcd: 0/288 nets differ, 1892/1892 edges, SAIF TC exact 288/288.
- acc8: 0/140 nets differ, 178815/178815 edges, SAIF TC exact 140/140.
- GL0AM's SAIF equals its own dumped waveforms on every net.
- SAIF T0/T1/TX differ on some nets only by the start-up interval: Icarus holds X until the first evaluation
  and reset; GL0AM flops start at 0 (GL0AM_INIT_SEQ_ZERO). With warmup 0/1/2 ns there are 86/14/0 (acc8) and
  235/45/30 (gcd) differing nets; from 5 ns on, none.
