# GL0AM GraphPrecompile.pkl.gz: what we match, and what we don't

The reference is `GL0AM/GL0AM.generateGraph.py` (commit 14b5dd0), lines 31-123. We have no real GL0AM
precompile file to diff against: the dataset is on Google Drive and was not fetched. That script's own
pre-processor, which writes the precompile, is not in the repo.

## Verified from the loader's code
These are the keys and indexing patterns the loader uses:
- gzip + `np.load(allow_pickle=1)` of a dict.
- Node ids: std cells come first (`0..num_of_std_cells-1`), then ports.
- `cell_index` and `port_index` are dicts keyed by node id.
- `adjacency_edge_types` is a sparse NxN matrix:
  - `==1` builds g_full (combinational graph).
  - `==2|==3` builds g_update.
  - `==5` builds the SRAM graph.
- `adjacency_pin_ids[src,dst]` becomes edata `x`, the destination pin position. `x==7` is treated as the clock pin of clock gates.
- `SDF_net_rise` / `SDF_net_fall` are sparse NxN matrices indexed `[src,dst]`.
- `pin4SDF_rownum[src,dst]` gives the row into `fullSDF_LUT`, a list of per-row delay lists that the loader flattens. Pointers come from the cumsum of the row lengths. `condXSDF_LUT` is indexed by the same row.
- `seq_pushoutSDF` is a per-node array.
- `cell_ids` is a per-node int.
- `edge_ties_node_nums` / `edge_ties_pin_positions` are per-tie arrays.
- `scripts`/`gls.graph.load_like_gl0am` mirrors these numpy/scipy steps without DGL/torch and loads our file (this is tested).

## Guessed / unverified (our choices)
| field | ours | why unverified |
|---|---|---|
| edge type 2 vs 3 | 2 = data pin into a flop, 3 = clock pin into a flop | the loader ORs them together; their meaning is set in the missing pre-processor |
| `adjacency_pin_ids` | index of the pin in Liberty pin declaration order (inputs only) | the GL0AM pin order table is not public; clock-gate CLK==7 implies a fixed per-cell pin map |
| `cell_ids` | auto codes from 1000 up, by cell type | GL0AM uses codes like 21 (buf-like), 24 (inv-like) and 300-399 (clock gates); its full table is not public, and its kernels dispatch logic on these codes, so **GL0AM kernels will not run on our file without its code table** |
| `fullSDF_LUT` row layout | `[rise,fall]` per IOPATH from that input pin (several if COND) | the kernel-side layout (`GL0AM.cupy`) has not been reverse-engineered |
| `condXSDF_LUT` | max delay of the row | the semantics ("X-condition" delay) are a guess |
| `SDF_net_*` | all 0 | our flow has no SPEF/INTERCONNECT delays |
| `seq_pushoutSDF` | CLK->Q max delay on flop nodes | guessed from the name |
| clock tree buf/inv polarity pass | not needed for gcd/acc8 (clock is a primary input straight to the flops) | untested on designs with clock trees |
| SRAM (type 5) | never emitted | no SRAM macros |
| (src,dst) collisions | counted; 0 on gcd/acc8 | a scipy matrix keeps one entry per (src,dst), so a driver feeding two pins of one cell loses one edge. GL0AM shares this limitation |
