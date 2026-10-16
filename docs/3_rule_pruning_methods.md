# 3-rule pruning of Human-GEM: methods

Draft methods for Metaboenrich's carbon-based network (October 2026). Every number here can be
regenerated with `python tools/reproduce_three_rule.py`. The per-reaction detail is in
`network_before_after.xlsx` (written by `tools/rules_audit.py`).

## Purpose

Metaboenrich ranks enzymes from untargeted metabolomics by running electrical current through a
metabolic network:
- significantly decreased metabolites are current sources;
- significantly increased metabolites are sinks;
- reactions and their genes are scored by the current they carry.

This orientation is the default since decision D20 (2026-10-16). It departs from protocol §1.3,
which makes increased metabolites the sources; `--sources increased` restores that. Current flow
does not depend on the choice: swapping sources and sinks negates every current, and scores use
the absolute current. No result in this note changes.

Genome-scale models link every substrate of a reaction to every product. That creates false
shortcuts through molecules that appear in thousands of reactions (H+, water, ATP, NAD(P)H, CoA), so
current flows glucose → ATP → anything. 3-rule pruning removes those shortcuts from the chemistry
itself, with no hand-made list of currency metabolites:
- a substrate is linked to a product only where carbon atoms pass between them;
- links that act as cofactor bookkeeping are removed;
- links that load or unload carrier molecules are removed.

The scope is bulk untargeted metabolomics of polar (methanol-soluble) metabolites, with no
compartment information.

## Inputs

| Input | Version / source | SHA-256 |
|---|---|---|
| Human-GEM, SBML | v2.0.1 (SysBioChalmers/Human-GEM) | `6ce49b620391f0ad76be24fdbdfc884fa376ff534dd3813c95abbe9d8c66fa5e` |
| Human-GEM, YAML (subsystems) | v2.0.1 | `3b944902a44f5f0e9dfcf23f8b9b4a54891bfb85ae711f8540e72ffbfcfa1f9a` |
| Human-GEM metabolites.tsv (SMILES, cross-references) | v2.0.1 | `c31439f997a6d4a68ba6d00971243a600d1c9523b736d876b285a8f8a4c9ab34` |
| Atom maps | RXNMapper 0.4.3 (Schwaller et al., *Sci. Adv.* 2021) on Human-GEM SMILES, computed 2026-09-30 | `data/atom_pairs_raw.json` |
| Curated substrate–product pairs | KEGG RCLASS via the KEGG REST API, retrieved 2026-09-30 (1,794 KEGG reactions) | `data/kegg_rclass_pairs.json` |

Software: Python 3.9, NumPy 2.0, SciPy 1.13, pandas 2.3, RDKit 2024.09.6, PyTorch 2.8 (CPU). The
manifest written with the network (`results/three_rule/network/manifest.json`) records the SHA-256
of every input and the exact versions used.

KEGG data are not redistributed in the public repository (KEGG licence). The REST cache is kept
locally and a re-download may differ.

## Network construction

Human-GEM has 12,877 reactions and 4,165 compartment-free metabolites.

**Compartments are merged** (a metabolite in any compartment is one node). Reactions then left with
fewer than two distinct metabolites are dropped: transport and exchange reactions, where the same
metabolite is on both sides. The result is the raw network: **7,080 reactions and 4,035
metabolites**. It is a bipartite graph of metabolite and reaction nodes. Current flows on its
graph Laplacian, with unit conductance per edge.

### Carbon pairs

For each reaction, the substrate→product pairs that share carbon atoms come from two sources.
- **KEGG RCLASS** pairs, mapped through the KEGG reaction IDs that Human-GEM records.
- **RXNMapper atom maps.** A computed pair is kept only if the substrate is the product's main
  carbon source, or the product is the substrate's main carbon destination. This removes cross-pairs
  the mapper invents in transaminases. On the 2,027 reactions with KEGG pairs, this cleaning rule
  agrees with KEGG at 90.0% precision and 96.3% recall; the uncleaned maps reach 85.4% and 98.3%.

The two sets are combined as their union, sorted per reaction
(`python -m metaboenrich.atoms --combine union`). KEGG lists only main pairs, so the union keeps
minor carbon donors such as carbamoyl-phosphate → citrulline in ornithine transcarbamylase.
- 5,800 reactions have pairs.
- 1,294 Human-GEM reactions have no atom map: 1,067 because a participant has no structure in
  Human-GEM, the rest because the mapper failed. These are mostly drugs, glycans and lipid
  intermediates.

A reaction with pairs becomes **one reaction node per pair** (a carbon link). A reaction without
pairs stays whole. The enzyme's genes attach to every link.

### Step 1: pseudo-reactions and transporters

Whole reactions are dropped if they:
- are the biomass objective;
- involve a **pool pseudo-metabolite** (102 by label: the 92 that Human-GEM names "pool", plus
  lumped entities such as "biomass", "xenobiotics", "steroids" and "vitamin A derivatives");
- **carry a carbon-containing metabolite across a membrane**: the same carbon-containing metabolite
  appears on both sides before compartments are merged (4,492 Human-GEM reactions).

An ATP-driven pump is left with ATP → ADP as its only carbon pair once compartments are merged.
Transporter genes are reported separately (the transporter table), linked to their cargo.

Dropped at this step: **766 reactions**.

### Rule 1: carbon

A link is kept only if both molecules contain carbon (from the formula). Water, H+, phosphate,
pyrophosphate, O2, H2O2, NH3, sulfate and ions (99 molecules) therefore link nothing. In reactions
with no atom map, carbon-free molecules are removed.

Removed: 141 links. Dropped: 37 reactions (catalase, superoxide dismutase, peroxidases,
pyrophosphatase).

### Rule 2: cofactor couples

For every carbon pair, count the atom-mapped reactions it appears in (once per reaction). A pair
found in **10 or more reactions** is a cofactor couple and is removed everywhere. This gives
48 couples, for example:
- NADP+/NADPH 825, ADP/ATP 776, NAD+/NADH 427, AMP/ATP 243, FAD/FADH2 156;
- acetyl-CoA/CoA 137, SAH/SAM 77;
- UDP/UDP-sugars, the nucleotide di/triphosphates, and short series of acyl-CoA/CoA pairs.

In a reaction with no atom map, both members of a couple are removed when one is a substrate and
the other a product.

Removed: 2,742 links (from reactions still present after Step 1). Dropped: 11 reactions, the
ion-pumping ATPases, ATP synthase and cytochrome c oxidase, whose only links are cofactor couples.

The couples list contains pairs that are real carbon flow in some reactions, for example
2-oxoglutarate/glutamate (41) and glutamate/glutamine (13).

### Rule 3: carriers

A **carrier** is found from structure:
- it is found whole inside at least **10** of its carbon pair partners;
- every one of those partners attaches to it at the **same atom**;
- the attachment is **through a heteroatom** (a thioester, ester or amide bond).

The substructure search uses uncharged, stereo-free RDKit molecules. A partner attached by a
carbon–carbon bond is an extension of the carbon skeleton, not a loaded carrier. So
glycine → 5-aminolevulinate (heme synthesis) and acetyl-CoA → acetoacetyl-CoA (thiolase) are not
carrier links.

Carriers found: **CoA** (157 loaded forms), **L-carnitine** (74), **glutathione** (38) and
**cholesterol** (33). Glycine just misses: 36 of its 37 conjugates attach at the nitrogen.

The links between a carrier and its loaded forms are removed everywhere. Examples are CoA /
methylmalonyl-CoA and carnitine / propionylcarnitine. Links between two loaded forms, and the
carrier's own synthesis and breakdown, stay. In reactions with no atom map, the carrier is removed
when one of its loaded forms is on the other side.

Removed: 884 links. Dropped: none (see the safeguard).

Why Rule 3 is needed: without it, CoA links to every short-chain CoA ester through rare,
individually uncounted pairs. In the methylmalonic acidemia benchmark, CoA carried about 31% of all
current and bypassed methylmalonyl-CoA mutase.

### Safeguard: keep the rarest link

If Rules 2 and 3 leave a reaction with no link, it keeps the removed pair that is found in the
fewest reactions (ties broken by metabolite ID), instead of dropping out with its genes. For
example, glutamate dehydrogenase keeps glutamate/2-oxoglutarate (41) over NAD+/NADH (427).
Reactions that carry any metabolite across a membrane are never rescued, so ion pumps do not return
ATP links.

Reactions kept by the safeguard: **212**. Among them are GLUD1/2, GLS/GLS2, GLUL, GSR and the
nucleotide kinases.

### Size limit

Reactions still listing more than 20 metabolites are dropped: 47 reactions, mostly protein
synthesis pseudo-reactions built from tRNAs.

### Result

| | Raw Human-GEM | 3-rule pruning |
|---|---|---|
| Metabolites | 4,035 | 3,808 |
| Human-GEM reactions represented | 7,080 | 6,219 |
| Reaction nodes (one per carbon link) | 7,080 | 7,676 |
| Edges | 38,766 | 16,060 |
| Genes | 2,483 | 2,191 |
| Metabolites in 100 or more reactions | 32 | 4 |
| Largest hub | H+ (4,061 reactions) | cholesterol (143) |
| Share of structural current through the busiest 1% of nodes | 34.1% | 17.1% |
| Metabolites cut off from the main network | 4 | 395 |

Structural current is the mean current per node over 150 × 150 random source–sink pairs, with no
data.

The 395 cut-off metabolites include 84 that Rule 3 separates. Most of those are xenobiotics
(naphthalene derivatives, halogenated solvents) and their glutathione conjugates.

## Benchmark

The benchmark asks whether the network ranks the known defective enzyme highly in 20 inborn errors
of metabolism.
- **Data:** group-level plasma metabolomics z-scores from Miller et al. 2015 and Thistlethwaite et
  al. 2022 (*Sci. Rep.* 12:6556; the CTD R package's `Thistlethwaite2020` dataset), via the CTD R
  package.
- **Significance:** BH q < 0.05 and |mean z| ≥ 1.
- **Name matching:** metabolite names are matched to Human-GEM through HMDB, KEGG, Human-GEM names
  and the lab's id_translation synonym table.
- **Score:** the rank of the causal gene among all network genes (best rank where a disorder has
  several causal genes). Lower is better; a random ranking gives about 1,100.

| Network | Median rank | Causal gene in top 5% (of 20) |
|---|---|---|
| Raw Human-GEM | 171 | 9 |
| Carbon method (Rules 1 + 2) | 130 | 9 |
| + Step 1 | 118 | 10 |
| + Rule 3 | 114.5 | 10 |
| **3-rule pruning (with safeguard)** | **112.5** | **10** |
| First method (hand-made currency lists) | 88 | 11 |

By disorder, Rule 3 brings the largest gain in disorders of CoA-ester metabolism, for example
HMG-CoA lyase deficiency (445 → 83).

3-rule pruning outperforms the first method in:
- propionic acidemia (760 against 1,142);
- glutaric aciduria (254 against 930).

It underperforms in:
- methylmalonic acidemia (719 against 468);
- citrullinemia (142 against 80).

With 20 disorders, differences of a few dozen ranks are within the noise.

## Known limitations

1. **Cross-links from tied atom maps.** About 65 reactions keep false carbon links, for example
   aspartate → glutamate in aspartate aminotransferase. The union adds RXNMapper pairs that the
   mapper scored as tied with the correct ones. A fix is under evaluation.
2. **Couple cutoff.** The cutoff of 10 is a convention shared by Rules 2 and 3. On the carbon method
   (Rules 1 + 2, without name lookup) the benchmark median was 85.5 at a cutoff of 5, 129.5 at 10
   and 113.5 at 20, so results are sensitive to it.
3. **Couples that are real chemistry.** 2-oxoglutarate/glutamate and glutamate/glutamine are removed
   as couples. The safeguard restores them only where they are a reaction's last link.
4. **Reactions with no atom map** (about 1,100 in the raw network) are kept whole, with Rules 1–3
   applied from formulas. They are the least resolved part of the network.
5. **Pools are identified by label**, not by a structural criterion. A generic-formula criterion
   matches 87 of the 102 pools but also 120 ordinary acyl-carrier-protein intermediates.
6. **Markers missing from Human-GEM** cannot contribute, for example 2-methylmalonylcarnitine in
   methylmalonic acidemia.

## Reproduction

```
pip install -r requirements-atoms.txt            # includes RDKit and RXNMapper
python tools/reproduce_three_rule.py             # all steps, about 15 minutes on a laptop CPU
```

Code: `metaboenrich/atoms.py` (pairs, Rules 1–3, safeguard, carriers, pools, transporters),
`metaboenrich/network.py` (graph construction) and `metaboenrich/structure.py` (`_three_rule_stages`,
the order of steps). Tests: `tests/test_three_rule.py`.
