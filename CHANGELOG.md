# Changelog

Network counts are for the analysis network (stage 5): metabolite nodes /
reaction nodes / edges. The reasons behind each change are in
[docs/decisions.md](docs/decisions.md).

## 0.2.0.dev0 (2026-09-29, in progress)

Network unchanged from v0.1.1 (3,958 / 6,377 / 16,356; edge-list SHA-256
`942cd8ac…`).

- **Absorbing random walk** (experimental, branch `directed-rw`; `--mode walk`,
  `metaboenrich/walk.py`): walkers start at the sources weighted by |log2FC|,
  follow reaction directions, and stop at the targets (optional partial
  absorption, `--walk-kappa`). Only walks that reach a target are scored.
  `reactions.tsv` gains `net_forward`. Benchmark: `--method walk`. Not the
  default; evidence in D17.

- **Benchmark** (`metaboenrich/benchmark.py`, `benchmarks/`): inborn errors of
  metabolism from Miller et al. 2015 and Thistlethwaite et al. 2020 (plasma
  z-scores, fetched from the CTD package on CRAN). Group- and patient-level
  runs; rank of the causal gene for Metaboenrich (raw and leave-one-out) and
  for a neighbour baseline.
- **Scoring default is now raw** conductivity + bottleneck (`--scoring raw`);
  leave-one-out is `--scoring loo` and always written to `reactions.tsv`.
- **One-sided flow**: when nothing significant goes down (or up), current is
  exchanged evenly with the measured, unchanged metabolites (`--one-sided auto`).
- **Pathways**: `pathways.tsv` tests Human-GEM subsystems (from the pinned
  v2.0.1 YAML) for over-representation among the top reactions.
- **Transporters**: `transporters.tsv` scores transporter genes by the
  metabolites they carry (transport reactions are not in the graph).
- **Name mapping**: `--id-translation` synonym table, then an unambiguous
  loose-name match; each match is labelled in `daa_mapping.tsv`. The loose
  match applies to every run, so some inputs now map more metabolites.
- **Provenance**: every `summary.json` records the version, scoring, one-sided
  status and the network's edge-list fingerprint (also in
  `network/network_info.json`).
- **Pruning-stage diagrams** (`metaboenrich/stage_diagrams.py`) and the
  structural check (`metaboenrich/structure.py`, Pruning Atlas).
- Scoring moved to `metaboenrich/scoring.py` (outputs verified byte-identical).
- **Experimental carbon-skeleton channels** (`--atom-pairs`, `metaboenrich/atoms.py`):
  reactions split into carbon-sharing substrate–product channels from KEGG
  RCLASS and RXNMapper atom maps. Off by default; no clear benchmark gain yet
  (decision D15). Needs `pip install -r requirements-atoms.txt`.
- **Carbon Channel Atlas** (`docs/carbon_atlas.html`, `tools/atlas/build_carbon_atlas.py`)
  and its Cytoscape diagrams; `--stage-set carbon` for the structure check and
  diagrams; `--stage SET:KEY` benchmarks any named network (decision D16).

## 0.1.1 (2026-09-28)

Network: 3,958 / 6,377 / 16,356.

- Cofactor partner forms are paired one-to-one, closest formulas first, so
  adenosine kinase's AMP synthesis is kept (ATP pairs with ADP).
- A loaded carrier must be named after the carrier (UDP-glucose, udp-ribose,
  acetyl-CoA) as well as contain its formula; other cofactors never qualify
  (ATP is not a loaded CMP).
- 33 cofactor decisions changed vs 0.1.0; 10 nucleotide-salvage reactions
  restored (ADK, CTPS1/2, UCK1/2, GMPR, NUDT2).

## 0.1.0 (2026-09-28; network built 2026-09-25)

Network: 3,957 / 6,367 / 16,332.

- Kirchhoff–Ohm current flow on Human-GEM v2.0.1 (compartments merged);
  sources = significant increases, targets = significant decreases.
- Currency handling: inorganics and generic carriers removed everywhere;
  cofactors kept only in their own synthesis/catabolism; lumped-pool reactions
  dropped.
- Reaction edit: SCLY (MAR07133) B6 vitamers removed.
- Leave-one-out scoring (default in this version).
- Directed mode (path filter), Cytoscape GraphML output, locked network
  snapshot with test.
