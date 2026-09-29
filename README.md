# Metaboenrich (prototype)

Kirchhoff–Ohm current-flow analysis of a metabolomics differential abundance
analysis (DAA) on the [Human-GEM](https://github.com/SysBioChalmers/Human-GEM)
metabolic network. Significantly increased metabolites act as current
sources and significantly decreased metabolites as sinks. Reactions,
metabolites and genes are then ranked by how much current they carry
(*conductivity*) and how much they constrain it (*bottleneck*).

This implements the Metaboenrich protocol (Frasketi, Reyna, Montefusco,
Aug 2026): §1 (undirected) fully, and §2 (directed) as a path filter. Where the
protocol leaves a choice open, the choice made here is listed under
[Decisions still open](#decisions-still-open).

- [Setup](#setup)
- [Running an analysis](#running-an-analysis)
- [Outputs](#outputs)
- [Viewing results in Cytoscape](#viewing-results-in-cytoscape)
- [The locked network](#the-locked-network)
- [How it works](#how-it-works)
- [Decisions still open](#decisions-still-open)
- [Known limitations](#known-limitations)
- [Development](#development)

## Setup

Python 3.9 or later.

```sh
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt        # Windows
# .venv/bin/python -m pip install -r requirements.txt          # macOS / Linux
```

On Windows, use `py` instead of `python` if `python` opens the Microsoft Store.

The first run downloads Human-GEM v2.0.1 (46 MB SBML) to `data/Human-GEM.xml`
and checks it against a pinned SHA-256, so every machine uses the same model.

## Running an analysis

```sh
.venv\Scripts\python -m metaboenrich --daa examples/example_daa.csv --out results/example
```

A run on the full network takes a few seconds. **`examples/example_daa.csv`
is synthetic**: a made-up, hypoxia-like pattern for exercising the
pipeline. Don't interpret its results.

### Input: the DAA table

A CSV or TSV file with one row per metabolite. Column names are set on the
command line:

| option | default | contents |
|---|---|---|
| `--id-col` | `metabolite` | One or more identifier columns, tried in order for each row, e.g. `--id-col HMDB KEGG Name`. Accepts Human-GEM ids (`MAM01306`, `MAM01306m`), HMDB (`HMDB0000208` or `HMDB00208`), KEGG (`C00026`), ChEBI (`CHEBI:30915`) or Human-GEM names (case-insensitive) |
| `--fc-col` | `log2FC` | log2 fold change (add `--fc-linear` if the column is a plain fold change) |
| `--p-col` | `padj` | p-value, compared with `--alpha` (default 0.05) |
| `--min-log2fc` | `0` | sources/targets must also have \|log2FC\| ≥ this |

Metabolites with p < alpha and log2FC > 0 are **sources**, and those with
log2FC < 0 are **targets**.

**MetaboAnalyst-style `statistic_analysis.csv`** files hold several
comparisons. Pick one comparison with `--fc-col` and `--p-col`:

```sh
.venv\Scripts\python -m metaboenrich --daa statistic_analysis.csv --out results/C_vs_E ^
    --id-col HMDB KEGG Name ^
    --fc-col "Log2_Fold_Change(AlbCre_C vs AlbCre_E)" ^
    --p-col "Tukey_HSD_P_Value(AlbCre_C vs AlbCre_E)"
```

In those files `Fold_Change(A vs B)` is B / A, so sources are metabolites
higher in B. For multi-group designs, use the Tukey HSD p-values; the t-test
p-values are not corrected for multiple comparisons.

### Common options

| option | effect |
|---|---|
| `--mode directed` | keep only source–target pairs joined by a directed path (protocol §2.2) |
| `--pair-weight fc` | weight each pair by (\|log2FC_s\| + \|log2FC_t\|) / 2 instead of 1 |
| `--no-leave-one-out` | rank on raw scores instead of leave-one-out scores (see below) |
| `--currency MAM01261 ...` | remove extra metabolites from every reaction |
| `--currency-mode remove` | remove every currency metabolite everywhere (the original approach) |
| `--reaction-edits FILE` / `none` | use a different reaction-edits table, or none |
| `--cytoscape-min-current X` | edge cutoff for the Cytoscape subnetwork (default 0.1) |

`python -m metaboenrich --help` lists everything.

## Outputs

Everything is written to the `--out` directory:

| file | contents |
|---|---|
| `genes.tsv` | genes ranked by the best final score of any reaction they catalyse |
| `reactions.tsv` | every reaction carrying current, with scores, genes, and `only_reaction_of` (sources/targets whose only reaction this is) |
| `intermediate_metabolites.tsv` | unmeasured or non-significant metabolites that relay current |
| `source_target_metabolites.tsv` | scores for the sources and targets |
| `cytoscape/current_subnetwork.graphml` | the part of the network that carries current (see below) |
| `cytoscape/full_network.graphml` | the whole network, with every score attached |
| `daa_mapping.tsv`, `daa_unmapped.tsv` | how each DAA row mapped onto the model, and the rows that didn't |
| `excluded_pairs.tsv` | source–target pairs dropped (disconnected, or no directed path) |
| `currency_metabolites.tsv` | every metabolite's reaction count and currency handling |
| `currency_edges.tsv` | every cofactor–reaction decision (kept or removed, and why) |
| `summary.json` | run parameters and counts |

**Score columns.** Throughput is the fraction of one source–target pair's unit
current passing through a node.

| column | meaning |
|---|---|
| `conductivity` | mean throughput over all pairs |
| `bottleneck` | largest throughput for any single pair (1 = the only route between some source and target) |
| `conductivity_loo`, `bottleneck_loo` | the same, but the lowest value after leaving out any one source or target; **these are used for ranking** |
| `final_score` | mean percentile rank of the two ranking columns, within reactions (or within metabolites) |
| `accumulated_current` | summed throughput over pairs (the protocol's reaction "voltage" update) |
| `bottleneck_frac` | fraction of pairs where throughput ≥ `--bottleneck-tau` (informational) |

For sources and targets, `conductivity` is instead the mean effective
conductance (1 / resistance) to their partners.

## Viewing results in Cytoscape

Each run writes two GraphML files to `<out>/cytoscape/`:

- **`current_subnetwork.graphml`** holds every edge that carries at least 10%
  of the current for at least one source–target pair, plus all sources and
  targets. This is usually 100–300 nodes: the routes the analysis is using.
  Change the cutoff with `--cytoscape-min-current`.
- **`full_network.graphml`** is the whole network (about 10,000 nodes) with
  the same attributes, for finding a node's neighbourhood.

To open one: **File → Import → Network from File…**, then choose the
`.graphml` file. Apply a layout, e.g. **Layout → Prefuse Force Directed**.

Edges point substrate → reaction → product; check the `reversible`
attribute before reading direction into them. Useful mappings in the
**Style** panel:

| visual property | column | mapping |
|---|---|---|
| Node fill colour | `role` | discrete: source = red, target = blue, reaction = grey, intermediate = white |
| Node shape | `type` | discrete: metabolite = ellipse, reaction = rectangle |
| Node size | `final_score` | continuous, e.g. 20 → 80 |
| Node label | `label` (metabolites) or `genes` (reactions) | passthrough |
| Edge width | `max_current` | continuous, e.g. 1 → 8 |

Node attributes: `label`, `type`, `role`, `log2fc`, `p`, `genes`, `rank`,
`final_score`, `conductivity(_loo)`, `bottleneck(_loo)`,
`accumulated_current`, `only_reaction_of`. Edge attributes: `mean_current`,
`max_current`, `side`, `reversible`.

### Pruning-stage diagrams

`python -m metaboenrich.stage_diagrams` (after `python -m metaboenrich.structure`)
writes one Cytoscape file per pruning stage to `results/stage_diagrams/`,
all on **the same layout**, so a node sits in the same place at every
stage. It draws the nodes the structural check is about: nodes that alone cut
off 5+ metabolites and their pockets, the busiest hubs, and islands of 4+
metabolites (about 1,300 nodes). Regions: nodes removed by pruning on the left
(grouped by the stage that removed them), the main network in the centre,
islands on the right.

To open: **File → Import → Network from File** and pick a `.cyjs` file (the
layout comes with it; don't re-run a layout). Then **File → Import → Styles
from File**, pick `metaboenrich_pruning_style.xml`, and choose the
"Metaboenrich pruning" style in the Style panel. Encodings match the Pruning
Atlas: fill = pocket size (grey 0, blues 1 → 20+), size = structural current,
circle = metabolite, square = reaction, thick dark border = cuts off a pocket,
dashed border = inside a pocket, dotted border = island, faint = removed by
this stage. Every value is also a column in the node table (`role`, `pocket`,
`pocket_of`, `structural_current`, `removed_at`, …).

## The locked network

`network/` holds the exact network every analysis runs on, as built from
Human-GEM v2.0.1 with the currency rules and reaction edits in this
repository:

| file | contents |
|---|---|
| `network_info.json` | provenance: model version, URL and SHA-256, and every currency, pool and reaction-edit setting |
| `nodes.tsv` | 3,958 metabolites (with formula, HMDB/KEGG/ChEBI ids, cofactor family) and 6,377 reactions (with genes, reversibility) |
| `edges.tsv` | 16,356 metabolite–reaction edges, marked substrate/product |
| `network.graphml` | the same network for Cytoscape, without scores |
| `currency_edges.tsv` | every cofactor–reaction decision |

`tests/test_network_snapshot.py` rebuilds the network from the model and
fails if it no longer matches `network/`. After an *intentional* change to
the currency rules or reaction edits, regenerate the snapshot and commit it:

```sh
.venv\Scripts\python -m metaboenrich.export_network --out network
```

The Human-GEM SBML itself is not committed. It is fetched from the pinned
release URL and verified by hash.

## How it works

1. **Graph (protocol §1.1).** One node per metabolite and one per reaction,
   with an edge between a metabolite and each reaction it takes part in; all
   edges have conductance 1. Compartments are collapsed, because a DAA
   measures total abundance. Dropped: the biomass objective, blocked
   reactions, reactions with more than 20 metabolites
   (`--max-reaction-size`), and reactions left with fewer than 2 distinct
   metabolites (transport, exchange).

2. **Currency metabolites.** H⁺ is in 4,540 Human-GEM reactions and H₂O in
   3,496. Left in, these and the cofactors would connect nearly every
   metabolite in one or two steps, so current would bypass real pathways.
   The defaults are in `metaboenrich/network.py`:
   - **Inorganic** (H⁺, H₂O, Pi, PPi, O₂, CO₂, NH₃, ions) and **generic
     carriers** (the generic `[protein]` node, ferredoxins, cytochrome b5,
     thioredoxins) are removed from every reaction.
   - **Cofactors stay in the network through their own synthesis and
     breakdown, and are removed only where they act as exchangers in a
     cycle**, i.e. where a partner form of the same cofactor is on the other
     side of the reaction. The families (`COFACTOR_FAMILIES`) are
     ATP/ADP/AMP, GTP/GDP/GMP, UTP/UDP/UMP, CTP/CDP/CMP, NAD⁺/NADH,
     NADP⁺/NADPH, FAD/FADH₂, CoA, SAM/SAH, PAPS/PAP, ubiquinone/ubiquinol and
     PLP/PMP. **Loaded carriers** also count as partners. A metabolite
     qualifies if its name contains the carrier as a word (UDP-glucose,
     udp-ribose, CDP-choline, CMP-sialic acid, acetyl-CoA; not dUDP) and its
     formula contains the carrier's formula plus extra carbon. Members of
     other cofactor groups never qualify; ATP's formula contains every atom
     of CMP, but ATP is not a loaded CMP.

     Partners are **paired one-to-one** within each group, closest formulas
     first, and a cofactor left unpaired is kept. In adenosine kinase
     (ATP + adenosine → ADP + AMP), ATP pairs with ADP, so the AMP made from
     adenosine stays in the network as AMP synthesis. Loaded carriers may
     partner more than one cofactor, because reactions are stored as sets
     and lose stoichiometry (thiolase makes 2 acetyl-CoA).

     Examples: hexokinase's ATP → ADP is removed, while
     adenylosuccinate → AMP, adenosine → AMP, UTP → CTP, dephospho-CoA → CoA
     and NAD⁺ → nicotinamide + ADP-ribose are kept. In Human-GEM this keeps
     168 cofactor–reaction links and removes 6,808.
   - Reactions feeding **lumped pools** (biomass pools, "cofactors and
     vitamins", "xenobiotics", "steroids", "vitamin A/D/E derivatives") are
     dropped entirely.

3. **Reaction edits.** `metaboenrich/reaction_edits.tsv` removes specific
   metabolites from specific reactions to correct annotation problems. Each
   row gives its reason. There is currently one: SCLY (MAR07133) is written
   with PLP → pyridoxine-phosphate as reactant and product. That made it the
   vitamin B6 pool's only link to the rest of the network, so it ranked
   first whenever B6 vitamers changed. Without it, the B6 vitamers form their
   own island. Pairs within B6 still run; pairs between B6 and other
   metabolites are listed in `excluded_pairs.tsv`.

4. **Sources and targets (§1.2–1.3).** Significant DAA metabolites with
   log2FC > 0 are sources and those with log2FC < 0 are targets. φ =
   |log2FC| is recorded.

5. **Current flow (§1.4).** For every pair (s, t), solve L x = b with
   b_s = +1 and b_t = −1. Instead of forming the pseudoinverse, each
   connected component's Laplacian is grounded and LU-factorised once. By
   linearity x_st = y_s − y_t, so |S| + |T| sparse solves cover all |S| × |T|
   pairs. Edge currents follow from Ohm's law, I = B x, and a node's
   throughput is ½ Σ|I| over its edges.

6. **Scoring (§1.5).** Conductivity and bottleneck are defined as in the
   score table above. **Leave-one-out** scoring handles a specific artifact.
   A measured metabolite with only one route into the network forces all of
   its current through that route. That would give the route the top
   bottleneck score from that one metabolite alone, however small its fold
   change. Taking the lowest score after leaving out each source or target
   in turn means a reaction ranks highly only if its score doesn't depend on
   any single metabolite.

7. **Directed mode (§2.2).** Reaction direction comes from the flux bounds:
   irreversible reactions go substrate → reaction → product only, and
   reversible reactions go both ways. Pairs with no directed s → t path are
   excluded; the remaining pairs use the undirected calculation.

## Decisions still open

These are placeholders for the three of us to settle, not settled methodology:

- **Conductivity and bottleneck** are not defined in the protocol; the
  definitions above are the prototype's.
- **Final score**: the mean of percentile ranks is a placeholder.
- **Fold-change magnitude** doesn't enter L x = b as the protocol is written,
  so by default it has no effect. `--pair-weight fc` is one way to include it.
- **Directed current (§2.3)** is not implemented. The protocol doesn't
  specify the directed Laplacian, and a linear solve can't by itself enforce
  one-way current.
- **Genes** take the best score among their reactions; GPR and/or logic is
  ignored.
- **More cofactors?** Glutathione (GSH/GSSG, 174 reactions), THF,
  acetyl-CoA, glutamate/AKG and carnitine are treated as ordinary
  metabolites. Each is one line to add to `COFACTOR_FAMILIES`.

## Known limitations

- **Dead-end reactions are never scored.** A reaction left with one
  metabolite after currency handling is dropped, because current that enters
  it has no second edge to leave by: it would always carry zero current. In
  Human-GEM, 26 reactions fall in this group, including carbamoyl-phosphate
  synthetase (CPS1: ATP + CO₂ + NH₃ → carbamoyl-phosphate), superoxide
  dismutase, the peroxidases, sulfite oxidase and formate dehydrogenase.
  Their genes can never appear in `genes.tsv`, whatever the data. The other
  dropped reactions are transport (4,129), exchange/sink/demand (1,656) and
  reactions made only of currency (577, mostly ATPases, ATP-driven transport
  and catalase).

- **Forced routes shared by several metabolites.** Leave-one-out handles a
  route forced by one metabolite, but not a single-exit pocket holding two or
  more significant metabolites. The one case found so far (vitamin B6 via
  SCLY) is corrected by a reaction edit; others would need their own edits.
- **Nucleotide shortcuts.** Ribonucleotide reductase (ADP → dADP, …) and
  RNA/DNA synthesis and breakdown are kept, so the nucleotide families
  connect through dNTPs and an RNA node. The exodeoxyribonuclease reactions
  (RAD1/TREX1) rank highly in every comparison so far and may reflect this.
- **Mapping coverage.** Only about half of Human-GEM metabolites carry an
  HMDB id, and drugs, food compounds and predicted HMDB entries have no
  reactions. Check `daa_unmapped.tsv`.
- **Dead ends.** Some metabolites are only produced or only consumed in
  Human-GEM, e.g. (R)-2-hydroxyglutarate, so directed mode excludes all of
  their pairs.
- **No statistics on scores yet**, e.g. permutation of DAA labels.

## Development

```sh
.venv\Scripts\python -m unittest discover -s tests -t . -v
```

| path | contents |
|---|---|
| `metaboenrich/gem.py` | SBML parser and pinned Human-GEM download |
| `metaboenrich/network.py` | graph construction, currency rules, reaction edits |
| `metaboenrich/daa.py` | DAA loading and identifier mapping |
| `metaboenrich/flow.py` | current flow and leave-one-out scores |
| `metaboenrich/cytoscape.py` | GraphML writer |
| `metaboenrich/__main__.py` | command-line entry point and scoring |
| `metaboenrich/export_network.py` | writes `network/` |
| `metaboenrich/structure.py` | data-free structural check across pruning stages: per-node degree, structural current (random sources/targets) and pocket size; `python -m metaboenrich.structure --out results/structure` |
| `metaboenrich/reaction_edits.tsv` | model corrections |
| `tests/` | toy-network checks of the solver, the leave-one-out scores (against brute force), the currency rule, and the network snapshot |
