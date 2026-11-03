# gls-thesis: open front-end, CPU reference simulator and partitioner baselines

Local-only repo. CPU-only: no GPU kernels are included.

## Layout
- `gls/liberty.py`, `gls/boolexpr.py`: Liberty cell reader (pins, functions, ff, NLDM arcs) and boolean-function compiler.
- `gls/frontend.py`: gate-level Verilog → Yosys `write_json` (`read_liberty -lib`) → netlist. `slang --ignore-unknown-modules` is used as a parse check only.
- `gls/sdf.py`: SDF writer (delays *derived* from Liberty at the middle NLDM table point; this is **not signoff STA**) and an IOPATH/COND parser.
  - `uniform` mode gives one rise/fall per output, for exact Icarus `assign #(r,f)` matching.
- `gls/graph.py`: gate graph, static features (fan-in, fan-out, cell type, IOPATH table size, COND count, level, tmin/tmax, Δt_arr), CSR `.npz`, and a GL0AM-style `GraphPrecompile.pkl.gz`. See `docs/GL0AM_FORMAT_GAPS.md`.
- `gls/sim.py`: timed, event-driven, 2-value simulator.
  - Inertial (Verilog continuous-assign semantics) and transport delay modes.
  - COND arc selection.
  - Smallest-delay rule when several inputs change at once.
  - Per-gate event and evaluation counts.
- `gls/partition.py`: the five configs:
  1. `gl0am_unit`
  2. `gl0am_event_weighted`
  3. `simpart_gamma*`
  4. `wang_density`
  5. `cost_model` (weights are inputs; **unfitted**)
- `scripts/validate_icarus.py`: cross-validates against Icarus.
  - `--sdf-mode uniform` uses continuous assigns.
  - `--sdf-mode full` uses per-instance `specify` IOPATHs with `-gspecify`.
- `scripts/run_pipeline.py`: end to end on one design. Writes `out/<top>_report.json`.

## Run
```
export PATH=/workspace/oss-cad-suite/bin:$PATH
/workspace/gls-venv/bin/python -m pytest -q
/workspace/gls-venv/bin/python scripts/validate_icarus.py --sdf-mode full
/workspace/gls-venv/bin/python scripts/run_pipeline.py            # gcd
/workspace/gls-venv/bin/python scripts/run_pipeline.py --verilog designs/acc8/acc8_sky130hd.v --top acc8 --random-cycles 2000 --reset rst --gammas 1024,8
```

## Designs
- `designs/gcd`: copied from GTPower `test/gcd/data` (sky130hd netlist, tt lib, and the Icarus testbench VCD used as stimulus).
- `designs/acc8`: an 8-bit accumulator written here and mapped to sky130hd with Yosys `dfflibmap`/`abc`.
- `designs/tiny`: 4 cells, for unit tests.

## Known limitations
- 2-value only (x/z in stimulus become 0).
- Positive-edge D flops only (dfxtp). No latches, set/reset flops, clock gates or SRAMs.
- No wire delays and no slew propagation.
- The sky130 tt Liberty has no `when` on timing arcs, so the COND path is exercised only by unit tests.
- Icarus starts every net at x, so comparisons skip a 5 ns warm-up.
- With warm-up 0, all nets differ because of the x→value initialisation.
