# Handoff: 3-rule pruning revision

> **Corrections (28 Oct 2026, after steps 1–3 were implemented on `three-rule-revision`).**
> The handoff below is left as written; where it disagrees with this box, this box is right.
>
> 1. **Step 3 is wrong by an order of magnitude.** `carbon_balance_audit.csv` counts every
>    stoichiometric coefficient as 1 and only the first `C` of a formula (it reproduces that way for
>    99.94% of reactions). With stoichiometry, **21** of the 6,219 pruned reactions are unbalanced,
>    not 301 (78 of the raw 7,080). The "+6" cluster is mostly two-hexose releases such as
>    alpha-mannosidase. `MAR03536`, `MAR03598` and `MAR01711`, named below as CO₂ artefacts,
>    **balance**. Use `metaboenrich.atoms.carbon_unbalanced`, not the CSV.
> 2. **Step 4 inherits the error.** `reaction_participants.tsv`, Rule 4's only input, has no
>    stoichiometry column, so the 88 impossible links and the ATP validation table must be
>    recomputed with coefficients before any hand review.
> 3. **Step 6's CO₂ estimate** (out-degree 8 → ~2, shortcuts 904 → ~230) counted those three
>    reactions as fixed by step 3. Re-measure.
> 4. **Baseline gate.** `3_rule_pruning` has **33** tests, all of which run with `data/` present
>    (none skipped); the 49 are on `flip-orientation`. That branch is not two documentation
>    commits: it merges `3_rule_pruning` into `directed-rw` and adds ~10 commits (`walk.py`,
>    `--sources`, tests). Pruning code (`atoms.py`, `network.py`, `structure.py`) is identical on
>    both; the final benchmark re-run will likely need `flip-orientation`'s scoring code.
> 5. **The `.venv` works** on this machine (Python 3.9.10 is installed).
> 6. **Step 1** reproduced every number in its table, but the keep-rarest safeguard fires **once**
>    (216 → 217: ACAT `MAR03657`, both carrier links now removed, CoA / tetracosanoyl-CoA restored).
>    No node is lost. 14 pravastatin metabolites become cut off (395 → 409), since they reached the
>    network only through the pravastatin-CoAs. "157 → 190 for CoA is wrong" in step 1 is garbled;
>    the table (CoA 157 → 172, cholesterol 33 → 66) is right.
> 7. **Step 2** merges link-level, not one copy per group: 2 groups' copies have different atom
>    maps, 121 mix directions, 573 differ in genes. Result 6,099 reaction nodes, 12,754 edges,
>    largest hub CO₂ (103); genes, reactions and every metabolite-to-metabolite step unchanged.
> 8. The methods doc's baseline figures "212 kept by safeguard" and "84 that Rule 3 separates"
>    measure 216 and 82 on the delivered code.

Repo: `C:\Users\dmontefusco\Documents\Github\Metaboenrich-Currency-Gene-Predictor`
Target branch: **`three-rule-revision`**, to be created off `3_rule_pruning`.

Work out of an audit of the 3-rule pruned network (27 Oct 2026). Eleven findings; this
handoff covers the two that are unambiguous and fully validated, then the rest in order.

---

## 0. Before anything

```bash
git checkout 3_rule_pruning
git checkout -b three-rule-revision
```

**Do not commit to `flip-orientation`.** It tracks `origin/flip-orientation` and belongs to a
collaborator (Jo Reyna); its two commits are documentation (an HTML report index, and
OMIM/GeneReviews citations for the benchmark's causal genes). Nothing there touches the rules.

**The repo's `.venv` is dead** — built against a Python 3.9 that no longer exists on this
machine. Rebuild it, or use any env with `numpy scipy pandas rdkit openpyxl`. RXNMapper and
PyTorch are NOT needed: `data/atom_pairs_raw.json` and `data/atom_pairs_union.json` both exist
and are reused.

`metaboenrich` must be importable — run from the repo root with `PYTHONPATH` set to it, or
`pip install -e .`.

### Baseline gate — do not proceed until this passes

```bash
python -m unittest discover -s tests      # expect: 49 tests, OK, 0 skipped
```

49 (not 33) because `data/` is present, so the model-dependent tests actually run, including the
locked structure hashes in `tests/carbon_stage_hashes.json`.

Then confirm the network reproduces:

| Quantity | Expected |
|---|---|
| metabolite nodes | 3,808 |
| reaction nodes | 7,676 |
| edges | 16,060 |
| Human-GEM reactions represented | 6,219 |
| genes | 2,191 |
| carriers found | CoA 157, L-carnitine 74, GSH 38, cholesterol 33, fatty acid pool 16 |
| cholesterol degree | 144 |
| CoA degree | 121 |

Reproduction snippet (verified working):

```python
from metaboenrich.gem import load_sbml
from metaboenrich.atoms import (carbon_method_pairs, carbon_method_rules, find_carriers,
                                load_atom_pairs, pool_metabolites, transport_reactions,
                                COUPLE_MIN_REACTIONS)
from metaboenrich.network import build_graph
from metaboenrich.structure import _pruning_stages

model  = load_sbml("data/Human-GEM.xml")
pairs  = load_atom_pairs("data/atom_pairs_union.json")
pools  = set(_pruning_stages(model)[4][3].pools) | pool_metabolites(model)
cargo  = transport_reactions(model, carbon_cargo=True)
shared = transport_reactions(model, carbon_cargo=False)
_, couples = carbon_method_pairs(model, pairs, COUPLE_MIN_REACTIONS)
carriers, carrier_pairs = find_carriers(model, pairs, "data/Human-GEM_metabolites.tsv")

log = []
p, _ = carbon_method_pairs(model, pairs, COUPLE_MIN_REACTIONS, keep_rarest=True,
                           no_rescue=shared, log=log, carriers=carrier_pairs)
ru   = carbon_method_rules(model, p, couples, pools, keep_rarest=True, no_rescue=shared,
                           log=log, carriers=carriers)
g    = build_graph(model, ru, max_reaction_size=20, atom_pairs=p, drop_reactions=cargo)
# g.node_ids 11484, g.is_reaction.sum() 7676, len(g.edges) 16060
```

---

## 1. Rule 3 formula fallback — VALIDATED, implement first

### The defect

Rule 3 classifies cholesterol as a carrier and should delete every
cholesterol ↔ cholesterol-ester link. It deletes the ones it can see and silently misses the
rest, because `find_carriers` is an RDKit substructure search and `atoms.py:544` reads

```python
if ma is None or mb is None:
    continue
```

A metabolite with no SMILES in Human-GEM is skipped before any test runs. The split is absolute:

| cholesterol-esters (65 in the pruned network) | link removed | link survived |
|---|---|---|
| has SMILES (32) | 15 | 17 |
| **no SMILES (33)** | **0** | **33** |

117 of cholesterol's 144 links go to esters; 99 involve structure-less ones. Network-wide, 701
of 3,808 metabolites (18.4%) have no SMILES, carrying 1,564 links. Carrier families are hit very
unevenly — CoA 5% structure-less, carnitine 5%, GSH 0%, **cholesterol 51%** — which is why Rule 3
took CoA from 674 links to 121 but left cholesterol as the largest hub.

**"Largest hub: cholesterol (143)" in `docs/3_rule_pruning_methods.md` is therefore not a
biological fact.** It is 99 links to molecules Human-GEM never gave a structure.

### Where the fix goes — NOT where you would first think

The 99 links are in reactions with **no atom map** (a structure-less ester cannot be
atom-mapped), so they never enter `carbon_rule1_pairs` and `find_carriers` never sees them.
They are handled by `carbon_method_rules`, `atoms.py:412-414`:

```python
if carriers:
    hit |= {c for c in subs if carriers.get(c, set()) & prods}
    hit |= {c for c in prods if carriers.get(c, set()) & subs}
```

The carrier is dropped from an unmapped reaction only when a **known** loaded form sits
opposite it. The structure-less esters are not in `carriers[c]`, so nothing happens.

**The fix extends carrier membership, scanned over reactions, not over pairs.** An earlier
attempt that scanned `carbon_rule1_pairs` added exactly zero forms — do not repeat it.

### The change

Carrier *discovery* is untouched: still structural, ≥10 loaded forms, same attachment atom,
heteroatom bond. Only *membership* gains a second route, under three conditions, all required:

1. the partner has **no usable SMILES** (so the structural test could never see it);
2. its heavy-atom formula **contains** the carrier's, and
   `partner − carrier + H₂O` matches the heavy-atom formula of some metabolite in the model;
3. its name **contains the carrier's name**.

Condition 3 is the project's own precedent — CHANGELOG 0.1.1: *"A loaded carrier must be named
after the carrier (UDP-glucose, udp-ribose, acetyl-CoA) as well as contain its formula"*.

Note `network.parse_formula` **ignores hydrogen**, which removes the protonation problem
entirely: Human-GEM stores fatty acids as carboxylate anions, one H short of the neutral acid
the esterification arithmetic predicts. On heavy atoms the match is exact, no tolerance needed.

Add to `metaboenrich/atoms.py`:

```python
def formula_loaded_forms(model, carriers, smiles):
    """Loaded forms Human-GEM gives no structure, found from formula and name.

    find_carriers is a substructure search, so a loaded form with no SMILES is
    invisible to it and keeps its link to the carrier (51% of cholesterol's
    esters). A partner qualifies here only if it has no structure, its formula
    contains the carrier's with the difference plus water matching some
    metabolite in the model, and it is named after the carrier (CHANGELOG 0.1.1).
    Carrier discovery is unchanged; this only extends membership.

    Returns {carrier: {extra loaded forms}}.
    """
    from collections import Counter
    from .network import parse_formula

    form = {s.met_id: s.formula for s in model.species.values()}
    name = {s.met_id: s.name for s in model.species.values()}
    heavy = set()
    for f in form.values():
        if f:
            heavy.add(tuple(sorted(parse_formula(f).items())))

    add = {}
    for c, known in carriers.items():
        cf, cn = parse_formula(form.get(c, "")), str(name.get(c, "")).lower()
        if not cf or not cn:
            continue
        for rxn in model.reactions.values():
            subs = {model.species[s].met_id for s in rxn.substrates}
            prods = {model.species[s].met_id for s in rxn.products}
            if c in subs:
                others = prods
            elif c in prods:
                others = subs
            else:
                continue
            for x in others:
                if x == c or x in smiles or x in known:
                    continue
                xf = parse_formula(form.get(x, ""))
                if not xf or any(xf[el] < n for el, n in cf.items()):
                    continue
                acyl = xf - cf
                if not acyl or tuple(sorted((acyl + Counter({"O": 1})).items())) not in heavy:
                    continue
                if cn in str(name.get(x, "")).lower():
                    add.setdefault(c, set()).add(x)
    return add
```

**Beware `Counter.__sub__` drops non-positive counts**, so `xf - cf` cannot be used to test
containment — the explicit `any(xf[el] < n ...)` line is load-bearing. Without it a partner
*smaller* than the carrier can still produce a positive remainder and match something.

Wire it into `find_carriers`, replacing the `return` at `atoms.py:553`:

```python
    carriers = {c: set(d) for c, d in loaded.items()
                if len(d) >= min_loaded and len(Counter(d.values())) == 1}
    for c, extra in formula_loaded_forms(model, carriers, smiles).items():
        carriers[c] |= extra
    remove = {tuple(sorted((c, x))): c for c, xs in carriers.items() for x in xs}
    return carriers, remove
```

### Expected result — these are measured, not estimated

Adds **48** loaded forms: cholesterol +33 (157 → 190 for CoA is wrong; see table), CoA +15
(`3-hydroxyhexadecenoyl-CoA`, the pravastatin-CoAs). Carnitine, GSH and fatty acid pool gain
nothing.

| | Baseline | After |
|---|---|---|
| metabolite nodes | 3,808 | **3,808** |
| reaction nodes | 7,676 | **7,676** |
| edges | 16,060 | **15,938** |
| cholesterol degree | 144 | **45** |
| CoA degree | 121 | **97** |
| CO₂ degree | 121 | 121 |
| glutamate degree | 122 | 122 |
| largest hub | cholesterol (144) | **glutamate (122)** |
| carrier loaded forms | 33 / 157 | 66 / 172 |

**Node counts must not change.** No reaction dropped, no gene lost, no orphan created, and the
keep-rarest safeguard must not fire. If reaction nodes move off 7,676, the change is wrong.

Then: re-run the 49 tests, and update the carrier counts in
`docs/3_rule_pruning_methods.md` (Rule 3 section and the result table's largest-hub row).

Input list of the 48, with formulas and matched acyl groups: `rule3_fallback_validated.csv`
(attached to this handoff).

---

## 2. Deduplicate reactions identical after the compartment merge

**2,478 pruned reactions in 1,102 groups** have an identical substrate set and product set once
compartments are merged. Each copy contributes its own node and links, so hub degrees are
inflated by whatever multiplicity Human-GEM happened to carry.

Largest groups: 12-HPETE hydroxymutase (6 copies), long-chain-fatty-acid-CoA ligase (5),
1-phosphatidylinositol 4-kinase (5), AMP phosphohydrolase (4). Three identical carbonic
anhydrase reactions (`MAR03955`, `MAR03956`, `MAR05407`) are each contributing a CO₂ ↔ HCO₃⁻
link.

Collapse each group to one node carrying the **union of the group's genes**. Deletes no
chemistry and needs no judgement call, which is why it goes early — but it changes reaction
node counts, so run it as a **separate commit** after step 1 is green.

Group list: `duplicate_reactions_after_merge.csv`.

Detection (reproduces the 1,102):

```python
sig = {}
for rid, rxn in model.reactions.items():
    subs = frozenset(model.species[s].met_id for s in rxn.substrates)
    prods = frozenset(model.species[s].met_id for s in rxn.products)
    sig.setdefault((subs, prods), []).append(rid)
dups = {k: v for k, v in sig.items() if len(v) > 1}
```

---

## 3. Carbon-unbalanced reactions — decide, then act

**301 of 6,219 pruned reactions (4.8%) do not balance carbon.** Distribution: 43 off by +6, 28
by +1, 27 by −1, 14 by +8, 13 by −10. Inherited from Human-GEM, not created by the pruning.

It matters because in an unbalanced reaction "did carbon pass from A to B" has no well-defined
answer, so both the atom mapper and step 4's budget rule are working on bad input. Three of
CO₂'s eight outlet reactions are in this set — `MAR03536`, `MAR03598` (sterol demethylation,
CO₂ on the substrate side *and* a One Carbon Unit emitted) and `MAR01711` ("Peroxisomal
Thiolase 2", a thiolase that consumes CO₂) — each off by exactly the spurious CO₂ carbon.

Either exclude them from Rule 4, or flag them in the workbook. Not a code change on its own.

Full table: `carbon_balance_audit.csv` (all 12,877 reactions, carbon in/out/imbalance).

---

## 4. Rule 4 — carbon budget (prototype validated, needs review before shipping)

### Why it is needed

The union of KEGG and RXNMapper pairs admits links asserting carbon transfer the reaction's own
carbon accounting forbids. The methods document already estimates "about 65 reactions keep false
carbon links"; an independent carbon-budget test found **88 links across 61 reactions**, and it
catches the document's own named example.

Confirmed: 7 aminoacyl-tRNA synthetase ATP links. Human-GEM gives tRNA(ala) the formula `RH`
(zero carbons) and alanyl-tRNA `C3H7NOR` (three, all from alanine); ATP's ten all go to AMP, so
no carbon remains for ATP → alanyl-tRNA. KEGG RCLASS for R03038 agrees — its only pairs are
alanine ↔ alanyl-tRNA and tRNA ↔ alanyl-tRNA. Signature: 53 tRNA reactions in the model, 37
survive pruning, 7 kept an ATP link; a chemically uniform family does not split 7/37 for
chemical reasons. Also confirmed: the transaminase cross-pairs (AST, ALT, PSAT, sulfinoalanine).

### The rule

For each product, enumerate which subsets of substrates sum exactly to its carbon count. If no
admissible subset contains substrate *S*, the link *S → P* is impossible. Formulas only.

Validated on ATP's 39 surviving links, classified by hand beforehand:

| chemistry | IMPOSSIBLE | admissible | inconclusive |
|---|---|---|---|
| adenosyl genuinely transferred (24) | **0** | 22 | 2 |
| tRNA ligases (7) | **7** | 0 | 0 |
| safeguard ATP→ADP/AMP (8) | 0 | 8 | 0 |

Perfect separation, and cofactor couples are left to Rule 2 — the rules stay orthogonal.

### Limits — read before implementing

- **Inconclusive more often than not**: 4,517 of 8,408 links. Whenever a substrate splits across
  products, no subset sums to a single product. Ornithine transcarbamylase is exactly that case
  (Human-GEM writes it in reverse, citrulline splitting into carbamoyl-phosphate + ornithine), so
  it returns inconclusive and survives — which is correct, since carbamoyl-phosphate → citrulline
  is the real carbon transfer the union exists to recover. The rule proves impossibility; it
  never certifies correctness.
- **Review all 88 by hand before deleting any.** Spot-checks found some flags are genuine model
  imbalances (step 3) rather than mapping errors.
- **Do not condition on KEGG backing.** It was tried and it fails badly: of ATP's 39 links, KEGG
  backs 3 of 24 correct ones, 0 of 7 wrong ones, and **7 of 8** ATP→ADP couples. RCLASS
  classifies structural transformation patterns, not which atoms moved in a given reaction.
  Conditioning deletion on "no KEGG backing" would delete 21 correct links to remove 7 wrong ones.

**Manuscript consequence:** the reported 90.0% precision / 96.3% recall of the cleaned atom maps
measures *concordance with a differently-defined reference*, not accuracy. Reframe rather than
restate.

Files: `rule4_impossible_links.csv` (the 88, named), `rule4_carbon_budget_audit.csv` (all 8,408
verdicts), `reaction_participants.tsv` (the rule's only input — side, carbon count, formula for
all 12,877 reactions).

---

## 5. Keep-rarest safeguard — measure before changing

All eight surviving ATP→ADP/AMP links are `kept by safeguard`; none was kept on merit. They
include `ATP phosphohydrolase`, `ATP:phosphorylase-b phosphotransferase` (PHKA1/2, PHKB, PHKG2),
`ATP:rhodopsin phosphotransferase` (GRK1/7) and `ATP diphosphohydrolase` (ENTPD1/2/3/5).

For a protein kinase the only small-molecule chemistry *is* ATP→ADP. Rule 2 correctly deletes
it; the safeguard finds the reaction has no links left and restores the rarest — which is the
link just deleted. The reaction re-enters carrying exactly the shortcut the rule exists to
prevent. 212 rescues network-wide; ATP accounts for 8.

**Measure first:** how many of the 212 restore a top-10 couple, and what removing only those does
to the benchmark median of 112.5. If the answer is "a handful, unchanged", the fix is free.

Proposed fix if it is not: let the safeguard rescue a reaction's **gene** without rescuing its
**link** — keep the reaction node attached to its genes but unlinked, or attach only to a
non-couple partner. Gene coverage, the safeguard's actual purpose, is preserved.

Rescue list: `results/three_rule/rescued_reactions.csv`.

---

## 6. CO₂ — last, and probably leave it alone

| | two-step shortcuts | share | rank |
|---|---|---|---|
| directed (RWR arm) | 904 | 1.95% | 7th |
| undirected (conductance arm) | 7,260 | 5.89% | **3rd** |

CO₂ is produced by 113 reactions and consumed by 8, so direction funnels it narrowly — but the
conductance arm discards direction and CO₂ becomes the third-largest shortcut hub. Both arms
share the same pruning, so it is one decision for both.

**Seven of the eight outlet reactions are artefacts fixed by steps 2 and 3** (3 duplicate
carbonic anhydrases, 3 carbon-unbalanced, 1 CPS1-written-with-CO₂ that is redundant with
`MAR08654` using HCO₃⁻). Those steps take CO₂'s out-degree from 8 to about 2 and the directed
shortcuts from 904 to roughly 230, with no currency-metabolite judgement.

**Do not drop CO₂, HCO₃⁻ and CO as a group.** The biotin-dependent carboxylases (PC, ACACA,
PCCA/PCCB, MCCC1) all use bicarbonate, and that route carries the urea and pyrimidine entry
steps. Dropping CO₂ *alone* is safe — carbamoyl-phosphate keeps 6 of its 8 links including OTC
and the CPS1 bicarbonate route.

Note the precedent: `network/network_info.json` (the **first method**, 3,958 / 6,377 / 16,356)
lists CO₂, HCO₃⁻, H⁺, H₂O, Pi, PPi, O₂ under `removed_everywhere` as "inorganic" — and the first
method still scores best on the benchmark, median 88 against 112.5. So the older hand-curated
method removed CO₂ deliberately.

Recommendation: leave CO₂ in and report the directed/undirected asymmetry as a worked
illustration of why the graph is directed — it is the cleanest example of that argument.

Files: `co2_link_audit.csv`, `metabolite_hubs_by_carbon.csv`.

---

## 7. Smaller items

- **Two metabolites escaped the compartment merge**: `MAM00208m`
  ([protein]-N6-(dihydrolipoyl)lysine, degree 2) and `MAM20076n` (N-hexadecanoylsphing-4-enine,
  degree 1) still carry compartment letters. Everywhere else the merge worked.
- **`results/three_rule/network/edges.tsv` does not match its own manifest.** The file hashes to
  `768bf58119ee120e8f213b65f7728b589fc23a784646eb68605410dbcd065cd6`; the `manifest.json` beside
  it declares `6aa5eaba64a54fb7266fa6c35a4828004f05288201827af0376063162f44c1a6`. The manifest
  was written from different bytes than the file next to it. Find this before the manifest goes
  into a paper as a reproducibility record.
- **Docs**: largest hub stated as cholesterol (143); `metrics.tsv` and an independent rebuild
  both give 144. Moot after step 1, but fix the provenance.
- **Raw-network recipe is ambiguous.** "Reactions then left with fewer than two distinct
  metabolites are dropped" yields 8,970. The rule that reproduces 7,080 exactly is: *cancel
  metabolites appearing on both sides after merging, then require ≥2 distinct remaining*, then
  exclude 8 pseudo-reactions (`MAR00021`, `MAR04413`, `MAR09931`, `MAR12140` biomass, plus
  `MAR03911`, `MAR08759`, `MAR09932`, `MAR13081`). The doc also places biomass removal in Step 1,
  but the raw network of 7,080 already excludes it.
- **Atom-mapping paper trail.** The Links tab records the decision but not the evidence. Add per
  link: `kegg_rclass_id`, `map_source` (kegg / rxnmapper / both), `rxnmapper_score`, `was_tied`,
  `carbon_budget_verdict`, `removed_by`. `map_source` and `was_tied` matter most — every false
  link found in the audit was rxnmapper-only, and the methods document notes ties are promoted
  into the union, so the pipeline currently discards the one signal that predicts the error. Both
  exist in `atom_pairs_raw.json` at the moment the union is computed.

---

## Files referenced by this handoff

**Travelling with this document** (in the same folder — audit outputs, not in the repo):

`rule3_fallback_validated.csv`, `duplicate_reactions_after_merge.csv`,
`rule4_impossible_links.csv`, `rule4_carbon_budget_audit.csv`, `carbon_balance_audit.csv`,
`co2_link_audit.csv`, `metabolite_hubs_by_carbon.csv`, `cholesterol_link_audit.csv`,
`reaction_participants.tsv`, `network_verification.csv`, `unmerged_compartment_nodes.csv`,
and a copy of `rescued_reactions.csv`.

**Already in the repo** (cited by repo path, nothing to fetch): `data/Human-GEM.xml`,
`data/Human-GEM_metabolites.tsv`, `data/atom_pairs_raw.json`, `data/atom_pairs_union.json`,
`data/kegg_rclass_pairs.json`, `network/network_info.json`, `tests/carbon_stage_hashes.json`,
`results/three_rule/metrics.tsv`, `results/three_rule/rescued_reactions.csv`,
`results/three_rule/network/edges.tsv`, `results/three_rule/network/nodes.tsv` and their
`manifest.json`.

## Order of work

1. Branch; rebuild the environment; **baseline gate** (49 tests + the network numbers).
2. **Step 1**, Rule 3 fallback. Commit. Re-run tests. Confirm the measured table.
3. **Step 2**, deduplication. Separate commit. Re-run tests.
4. **Step 3**, decide on the 301 unbalanced reactions.
5. **Step 4**, Rule 4, after hand-reviewing the 88.
6. **Step 5**, safeguard, after measuring.
7. Re-run the benchmark **once**, at the end — not between steps, so the effects stay separable.
   Target: beat median rank 112.5 and 10 of 20 causal genes in the top 5%.
8. CO₂ last; step 1 changes its ranking.

With 20 disorders, a shift of a few dozen ranks is within the noise — the methods document says
so, and it applies to improvements as much as regressions.

## What was NOT done

The pruning has never been independently re-executed. The raw network was rebuilt from
Human-GEM v2.0.1 SBML and matched on 6 of 6 numbers; the pruned-network figures were *measured
from the delivered `edges.tsv`/`nodes.tsv`*, which confirms the files agree with their
documentation but is not independent confirmation that the rules are correct. A second
implementation from the workbook's Links sheet (11,187 candidate pairs, substrate and product
names resolving 100% to metabolite IDs) remains the way to get that, and was not started.
