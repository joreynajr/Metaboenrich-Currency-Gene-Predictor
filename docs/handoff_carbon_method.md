# Handoff: the carbon method in Metaboenrich

For Claude Science, continuing the atom-mapping discussion. Written 2026-09-30.
Repository: Metaboenrich-Currency-Gene-Predictor (public). No unpublished lab data in this note.

## Background

Metaboenrich ranks enzymes (genes) from a metabolomics result. It runs electrical current flow
(Kirchhoff/Ohm) on the Human-GEM v2.0.1 network, with compartments merged. Significantly
decreased metabolites are current sources and increased ones are sinks. Reactions and genes are
ranked by how much current passes through them. Target data: bulk untargeted metabolomics of
polar (methanol-soluble) metabolites, with no compartment information.

Orientation (updated 2026-10-16): decreases as sources is the default since decision D20. It
departs from protocol §1.3, which makes increased metabolites the sources. This note first
followed the protocol; `--sources increased` restores it. Current flow does not depend on the
choice: swapping sources and sinks negates every current, and scores use the absolute current.

The main difficulty is currency metabolites (H2O, H+, ATP, NAD(P)H, CoA ...). Left in, they
create false shortcuts: current flows glucose → ATP → anything.

- **First method (current default).** Hand-made lists. Inorganics are removed everywhere.
  Cofactors are removed only where they exchange with a partner form (ATP → ADP) and kept where
  they are made or broken down. Pools and pseudo-reactions are dropped. One manual edit: the
  B6 vitamers are removed from SCLY, MAR07133.
- **Carbon method (the idea from our earlier discussion, now implemented).** No hand-made lists.

## The carbon method as implemented

1. **Start and pairs.** Start from raw Human-GEM and drop the biomass and pool pseudo-reactions.
   Take substrate→product carbon pairs from two sources:
   - KEGG RCLASS pairs (REST API) where KEGG has the reaction;
   - otherwise, RXNMapper atom maps computed from Human-GEM's own SMILES.

   The raw maps invent cross-pairs in transaminases, so they are cleaned. A pair is kept if the
   substrate is the product's main carbon source, or the product is the substrate's main carbon
   destination. Against KEGG on about 2,000 reactions this gives precision 90%, recall 96%.
2. **Rule 1.** Split each reaction into one link per carbon-sharing pair, and drop pairs where
   either molecule has no carbon (checked from the formula). KEGG pairs include non-carbon pairs
   such as H2O2/O2, which this removes. The enzyme's genes attach to all of its links.
3. **Rule 2.** Count how many reactions each carbon pair appears in. A pair found in 10 or more
   reactions is a "cofactor couple" and is removed everywhere, which gives 48 couples. A reaction
   whose only pairs were couples is dropped (619 reactions, mostly kinases).
4. **Reactions with no atom map** (about 1,100) are kept whole. They have no structure in
   Human-GEM or are too large for the mapper. The two rules are applied from formulas alone:
   carbon-free molecules are removed, and a couple is removed when one partner is a substrate
   and the other a product.

Rule 2 is counted after Rule 1, but the order makes no difference: every couple contains carbon
on both sides. The counts come only from mapped reactions.

**Detected couples (reactions per pair).**
- Most frequent: NADP+/NADPH 825, ADP/ATP 776, NAD+/NADH 427, AMP/ATP 243, FAD/FADH2 156,
  acetyl-CoA/CoA 137, SAH/SAM 77, UDP/UDP-glucuronate 64, CoA/malonyl-CoA 54,
  CO2/malonyl-CoA 50, 2-oxoglutarate/glutamate 41 …
- Borderline, and arguably real metabolism rather than bookkeeping: CO2/malonyl-CoA,
  2-oxoglutarate/CO2, 2-oxoglutarate/succinate, glutamate/glutamine.
- Couple count by cutoff: 181 at a cutoff of 5, 48 at 10, 18 at 20.

## Terms used here and in the atlas pages

All of these are structural: they come from the network's wiring alone, with no metabolomics data.
The network is bipartite (metabolite nodes and reaction nodes). "Main network" means its largest
connected component.

- **Links (degree).** The number of edges a node has. For a metabolite, that is the number of
  reactions (or carbon links) it takes part in. For a reaction, it is the number of metabolites it
  connects.
- **Structural current.** Pick one random metabolite as source and another as sink in the main
  network, and send one unit of current between them. Measure how much passes through each node
  (half the sum of absolute current on its edges). Average this over 150 × 150 = 22,500 pairs.
  A value of 0.01 means about 1% of a random pair's current passes through that node; 0 means the
  node is off the main network. High values mark nodes the method will tend to rank highly for
  any dataset.
- **Current through the busiest 1% of nodes.** The share of total structural current carried by
  the top 1% of nodes. Lower means current is spread out rather than funnelled through hubs.
- **Metabolites cut off from the main network.** Metabolites in any connected component other
  than the largest one ("islands").
- **Pocket size.** Our own term, not a standard one; it could be called the "disconnection size"
  of an articulation point. For each node: remove it, and count the metabolites in its component
  that are no longer connected to the largest remaining piece of that component. When removal
  splits off several pieces, the pocket size is the total across all of them, not the largest one.
  For example, in the raw network H2O has pocket size 26, made of 8 small pockets of peptide
  fragments and other hydrolysis products. A node with pocket size > 0 is an articulation point
  (cut vertex), found with Tarjan's algorithm. The nearest established idea is the chokepoint
  reaction (Yeh et al. 2004, *Genome Research*), which is narrower: a chokepoint is judged one
  metabolite at a time. Pocket size matters because current into or out of a pocket must pass
  through its gatekeeper, so the gatekeeper scores highly whenever anything in the pocket changes.

## The six networks compared

| Network | Metabolites | Reaction links | Largest hub (links) | Current through busiest 1% of nodes | Metabolites cut off from main network |
|---|---|---|---|---|---|
| 1. Raw Human-GEM | 4,035 | 7,080 | H+ (4,061) | 34% | 4 |
| 2. Rule 1 only | 3,890 | 12,181 | CoA (1,235) | 24% | 208 |
| 3. Rules 1 + 2 (the carbon method) | 3,884 | 8,720 | CoA (674) | 19% | 293 |
| 4. Carbon method, unmapped reactions dropped | 3,101 | 7,610 | CoA (597) | 18% | 279 |
| Reference: first method | 3,958 | 6,377 | acetyl-CoA (319) | 21% | 199 |
| Reference: first method + carbon split | 3,935 | 8,208 | L-carnitine (247) | 18% | 255 |

"Current through busiest 1%" is structural: current averaged over 22,500 random
source–sink pairs, with no data.

## Benchmark

The test uses 20 inborn errors of metabolism with a known faulty enzyme. The data are group-level
plasma metabolomics z-scores: Miller 2015 and Thistlethwaite 2022, via the CTD R package.
Significance is BH q < 0.05 and |mean z| ≥ 1. The score is the rank of the faulty gene among all
network genes (about 2,300); lower is better, and random is about 1,150.

| Network | Median rank | Top 100 (of 20) |
|---|---|---|
| Raw | 161.5 | 8 |
| Rule 1 | 130 | 9 |
| Carbon method, cutoff 10 | 129.5 | 9 |
| Carbon method, cutoff 5 | 85.5 | 11 |
| Carbon method, cutoff 20 | 113.5 | 10 |
| Carbon method, mapped only | 92 (of 1,896 genes) | 11 |
| First method | 88 | 11 |
| First method + carbon split | 101.5 | 10 |

Reading: following carbon helps compared with raw. At cutoff 5 the carbon method matches the
first method with no hand-made lists. Results depend on the cutoff, and 20 disorders is a small
test, so none of these differences is decisive. Per-disorder ranks swing a lot in both
directions.

## Open problems (where input would help)

1. **CoA stays a hub (674 links).** Each acyl-CoA/CoA pair is individually rare (10–19
   reactions), so Rule 2 misses the pattern "one carrier paired with many different partners". A
   rule on a molecule's number of distinct pair partners, rather than on the frequency of each
   pair, might catch it. Is there precedent (for example Arita 2004; Faust, Croes and van Helden
   on weighted paths)?
2. **The cutoff is arbitrary.** 5, 10 and 20 give quite different results. Is there a principled
   way to set it, such as a frequency-distribution break or comparison with KEGG's "cofactor"
   RCLASS roles?
3. **Real chemistry removed as couples.** 2-oxoglutarate/glutamate (transamination) and
   glutamate/glutamine are real carbon flow in some reactions and bookkeeping in others. Should
   Rule 2 be role-aware, as in the first method?
4. **More pieces break off.** The carbon method cuts off 293 metabolites versus 199. Some are
   real dead ends. Others may be real connections lost because a needed pair was removed as a
   couple.
5. **Coverage.** About 1,100 reactions have no atom map. 718 metabolites lack structures in
   Human-GEM; they are mostly drugs, glycans and lipid intermediates, so largely outside a polar
   panel.

## Where things are (repository)

- `metaboenrich/atoms.py`: pair extraction and cleaning, `carbon_method_pairs` (Rules 1 + 2),
  `carbon_method_rules` (handling of unmapped reactions), `COUPLE_MIN_REACTIONS = 10`.
- `metaboenrich/structure.py`: `_carbon_stages` defines the six networks.
- `metaboenrich/benchmark.py`: `--stage carbon:carbon_method --couple-min N`.
- `docs/carbon_atlas.html`: interactive report (charts, linked network diagrams, all tables above).
- `docs/pruning_atlas.html`: the same for the first method.
- `docs/decisions.md`: decision log (D15, D16 on carbon and pruning; D20 on orientation).
- `docs/benchmark.md`: benchmark method.
- KEGG-derived pair files are not in the repository (licence). They regenerate with
  `python -m metaboenrich.atoms`.
