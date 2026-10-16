# Decision log

Methodological choices for Metaboenrich, with the evidence behind each. Newest
decisions build on older ones; superseded entries are kept and marked.
"Decided by" names who made the call (DM = David Montefusco).

## Scope

### D1. Polar bulk metabolomics; no compartment data (2026-09-29, DM)
The tool analyses bulk untargeted metabolomics of water/methanol-soluble
metabolites, not lipidomics. Input tables carry no cell-compartment
information. Consequences: compartments stay merged (D2); lipid-species
mapping is not a priority; lipid metabolism stays in the network as
intermediates; benchmark cases use polar-signature disorders (D12).

### D2. Merge Human-GEM compartments into one node per metabolite (2026-09-24)
A DAA measures whole-sample abundance, so ATP in cytosol and mitochondria are
one measurement. Alternative: a compartment-aware network with transport
reactions as edges. Rejected for this data (D1); it would also make every
transporter a structural bottleneck.

## Network construction

### D3. Currency metabolites: role-based handling (2026-09-25, DM)
Keep cofactors in the network through their own synthesis and breakdown;
remove them only where they act as exchangers in a cycle (a partner form of
the same cofactor on the other side of the reaction). Inorganics (H⁺, H₂O, Pi,
PPi, O₂, CO₂, NH₃, ions) are removed everywhere, having no meaningful
synthesis. Alternative: remove all cofactors everywhere (the original
approach, `--currency-mode remove`).
Evidence: H⁺ is in 4,540 Human-GEM reactions and H₂O in 3,496; the structural
check shows raw-network current is hub-driven (top 1% of nodes carry 34%).
The role-based network keeps 30 more metabolites and 78 more reactions than
the remove-everywhere network with no extra pockets.

### D4. PLP is a cofactor (2026-09-25, DM)
Pyridoxal-phosphate joined the cofactor list (PLP/PMP family under D3).

### D5. Remove lumped pools and generic carriers (2026-09-25)
Reactions feeding biomass pools, "cofactors and vitamins", "xenobiotics",
"steroids" and vitamin-derivative pools are dropped; the generic `[protein]`
node, ferredoxins, cytochrome b5 and thioredoxins are removed everywhere.
Evidence: MAR00022 lumps CoA, NADH, NADPH, FADH₂, THF and others into one pool
reaction, which would link every cofactor in one step; `[protein]` linked
every amino acid through "[protein] to [protein]-L-X" reactions.

### D6. Reaction edit: SCLY B6 vitamers (2026-09-25, DM)
Human-GEM writes SCLY (MAR07133) with PLP → pyridoxine-phosphate as
reactant/product. SCLY uses PLP as a bound cofactor; the annotation made SCLY
the vitamin B6 pool's only link to the network, and it ranked first whenever
B6 vitamers changed. The two metabolites are removed from MAR07133
(`metaboenrich/reaction_edits.tsv`). Consequence: the B6 vitamers form an
island; pairs between B6 and other metabolites are excluded.

### D7. Cofactor pairing fix (2026-09-28, DM approved; v0.1.1)
Pair partner forms one-to-one by closest formula; loaded carriers must be
named after the carrier. Fixed two errors found by inspection: adenosine
kinase's AMP counted as exchange; ATP counted as "loaded CMP". 33 decisions
changed; 10 salvage reactions restored.

## Scoring

### D8. Leave-one-out scoring (2026-09-25) — superseded by D9
Introduced so a reaction forced by a single metabolite with only one route
out would not rank first on that metabolite alone. Evidence at the time: in
the lab's exploratory data it removed several top hits that each rested on a
single measured metabolite.

### D9. Raw scoring is the default (2026-09-29)
Evidence (inborn-error benchmark, 20 disorder groups, rank of the causal gene
among 2,302 network genes): raw median 88 (top 3.8%), 11 of 20 in the top 100;
leave-one-out median 236.5, 9 of 20; neighbour baseline 1,165 (random).
Leave-one-out penalises exactly the single substrate–product signal typical of
an enzyme defect (GAMT: rank 2 raw, 1,888 leave-one-out). Leave-one-out stays
available (`--scoring loo`) and in every `reactions.tsv`.

### D10. One-sided flow (2026-09-29, DM: work around limitations)
When nothing significant goes down (or up), current is exchanged evenly with
the measured, unchanged metabolites. Without it, 4 of 20 benchmark groups
(PKU, ABAT, HMGCL, HLCS) could not run. Checked against the pseudoinverse.

### D11. Transporters as a separate list (2026-09-29)
Transport reactions vanish when compartments merge, so transporter genes
(e.g. SLC2A1/GLUT1) are scored by the metabolites they carry, in
`transporters.tsv`. Kept apart from enzyme ranking because every changed
metabolite has transporters and they would crowd enzymes out of the top.

## Evaluation

### D12. Benchmark: inborn errors with polar signatures (2026-09-29, DM: hard data only)
Plasma z-scores from Miller et al. 2015 and Thistlethwaite et al. 2020
(Baylor), via the CTD package on CRAN. Causal genes are standard assignments
for each diagnosis. Excluded: Zellweger spectrum and RCDP (lipid), MCAD and
VLCAD (fatty-acid oxidation), lysinuric protein intolerance (transporter;
now testable via D11). Cobalamin defects kept as secondary (many genes).
Baseline: rank each enzyme by significant metabolites its reactions touch.

### D13. Name mapping (2026-09-29, DM: use the lab's id_translation)
Metabolites given only by name are mapped through the lab's KEGG/HMDB synonym
table, then by an unambiguous loose-name match (stereo prefixes, punctuation
and acid/salt forms ignored; cis/trans kept; keys under 5 characters
rejected). Every match is labelled in `daa_mapping.tsv`.

### D14. Blind protocol for published datasets (2026-09-29)
Tool settings frozen and committed (8d4a8da) before running; predictions
committed before building or reading the answer key (each paper's conclusions
and the "Expected tool prediction" sheet). Answers are judged against paper
conclusions (genes, pathways, processes), not only single genes.

### D15. Carbon-skeleton channels: built, kept optional (2026-09-30, DM proposed)
Idea: keep a substrate–product link only where carbon atoms pass between the
two molecules (atom mapping; after Arita 2004, PNAS), so group transfers
(phosphate, amino group, electrons) create no path. Implementation: each
reaction becomes one channel per carbon-sharing pair (`--atom-pairs`,
`metaboenrich/atoms.py`); unmapped reactions are kept whole and flagged.
Pairs: KEGG RCLASS (curated, 2,058 Human-GEM reactions) and RXNMapper on
Human-GEM's own SMILES (5,794 reactions mapped; 1,067 lack structures, 227
too long for the mapper). Raw atom maps invent cross-pairs in transaminases;
the cleaning rule (keep a pair if it is the product's main carbon donor or the
substrate's main destination) was chosen by agreement with KEGG on 2,027
reactions: precision 90.0%, recall 96.3% (raw: 85.4% / 98.3%). KEGG itself
omits minor carbon donors (e.g. carbamoyl-phosphate → citrulline in OTC).
Evidence (group benchmark, raw scoring, median rank / top 100 of 20):
current 88 / 11; channels with KEGG overriding the mapper 101 / 10; union
101.5 / 10; mapper first 78.5 / 11. Large per-disorder swings both ways
(glutaric aciduria 930 → 293 with union; OTC 168 → 971 with KEGG override).
Hub concentration falls (top 1% of nodes 21% → 19%; acetyl-CoA no longer the
largest hub) but islands grow (199 → 274 metabolites). Not adopted as default:
no clear gain on a 20-disorder benchmark. Revisit with a larger benchmark.
Carbon links alone, with no currency rules, do not solve the hub problem: CoA
stays a hub in 1,237 reactions and the busiest 1% of nodes still carry 26% of
current (raw 34%, current network 21%); benchmark median 153 vs 88. Documented
in the Carbon Channel Atlas (`docs/carbon_atlas.html`).

### D16. Which pruning steps matter for finding the faulty enzyme (2026-09-30)
Group benchmark, raw scoring, median rank (of ~2,300 genes) / top 100 of 20:
raw network 161.5 / 8; inorganics removed 137 / 9; cofactors role-filtered
87 / 11; pools dropped 88 / 11; reaction edits 88 / 11; cofactors removed
everywhere 86 / 11. Almost all of the gain comes from removing inorganics and
filtering cofactors. Dropping pools and the SCLY edit fix specific structural
artifacts but do not change this benchmark. Role-based filtering (D3) matches
removing cofactors everywhere while keeping 30 more metabolites.

### D17. Absorbing random walk: built, kept optional (2026-10-01, Reyna: plainest version, hooks 1–2) — orientation-dependent parts superseded by D20
Direction-respecting alternative to current flow (`--mode walk`,
`metaboenrich/walk.py`). Walkers start at the sources in proportion to
|log2FC| (hook 1) and follow reaction directions from the flux bounds. All
targets absorb together; with `--walk-kappa K` a target absorbs a walker with
probability 1 − exp(−K·|log2FC|) and lets the rest walk on (hook 2). Reversible
reactions are split into a forward and a backward state, and a walker may not
reverse the step it just took. Without that rule, forward-then-backward through
one reversible reaction is a co-substrate hop in two steps. Dead ends leak, and
scores use only walks that reach a target (Doob h-transform). One sparse LU per
run (21,787 states, 0.3 s on Human-GEM). Scores reuse the current-flow tables:
throughput from net traffic per edge, plus `net_forward` (> 0: the reaction is
used in its forward direction). Leave-one-out is not defined. Checked against
a Monte Carlo simulation on a toy network (max difference 0.005).
Evidence, inborn-error benchmark, raw scoring, rank of the causal gene among
2,302 genes. Group level, 20 primary disorders, median / top 100 / top 10:
current flow 88.5 / 11 / 3; walk 48 / 13 / 5. Patient level, 159 primary
patients: current flow median 134, 47% in the top 100, 5% in the top 10; walk
67, 55%, 28%; the walk is better for 71% of patients and in 14 of 20
disorders. Largest gains: MSUD 33 → 1, PKU 186 → 16, PA 1,187 → 176,
citrullinemia 82 → 20 (group). Partial absorption changes almost nothing:
group medians 45.5–47 for K = 0.5, 1 and 2 vs 48 with full absorption; patient
median 70 at K = 1 vs 67.
Losses explain where direction hurts. AADC (88 → 1,317): the main source is
3-O-methyldopa, which COMT makes irreversibly *from* L-DOPA, the blocked
substrate. Walkers cannot run back to L-DOPA and then forward through DDC, so
accumulated shunt products point away from the block. Homocystinuria
(181 → 765): the only decreased metabolite is cortisol, so there is no
informative sink. In both, under 4% of walkers reach a target.
Only 20% of released walkers reach a target on the synthetic example: most end
in directed dead ends, which is why conditioning on success matters.
Not adopted as default: it is a methods change to §2.3, the gain comes from
one benchmark, and the blind predictions (D14) were made with current flow.
Decide before the next blind round. Not tried: biased steps (hook 3), unchanged
metabolites as leaks (hook 4), the reverse "activation" orientation.
Superseded by D20: all walk numbers above, the two loss cases and the 20%
figure were made with sources = increased. The activation orientation has now
been tried and is the default.

### D18. 3-rule pruning (2026-10-02 to 2026-10-06; DM with Claude Code and Claude Science)
A carbon-based network with no hand-made currency lists, replacing the carbon
method of D15 as the candidate for publication (`docs/3_rule_pruning_methods.md`).
Steps: drop pools (by label), biomass and transporters with carbon cargo; Rule 1
carbon links; Rule 2 cofactor couples (pair in 10+ reactions); Rule 3 carriers
found from structure (whole inside 10+ partners, always at the same atom, through
a heteroatom: CoA, carnitine, glutathione, cholesterol); safeguard: a reaction
emptied by the rules keeps its rarest link (never for membrane-crossing
reactions). Evidence (group benchmark with the id_translation lookup, median
rank / top 5% of 20): carbon method 130 / 9; + Step 1 118 / 10; + Rule 3
114.5 / 10; + safeguard 112.5 / 10; first method 88 / 11. Rule 3 was found by
tracing methylmalonic acidemia: CoA carried ~31% of all current and bypassed
MMUT. Rejected along the way: the safeguard as first specified (would restore
~510 ATP links from pumps, Step 0 review), and merging long acyl-CoAs (removes
the CoA hub but does not help the benchmark). Open: false cross-links from tied
atom maps in ~65 reactions; couple cutoff sensitivity.

### D19. Benchmark uses the id_translation lookup (2026-10-02)
`--id-translation` matches patient metabolite names through the lab's synonym
table; without it propionylcarnitine (listed as O-propanoylcarnitine in
Human-GEM) was missed in methylmalonic acidemia. All 3-rule pruning comparisons
use it; earlier benchmark numbers (D15–D17) were without it.

### D20. Significant decreases are the sources (2026-10-16, Reyna: the original orientation was backwards)
New option `--sources {decreased,increased}` on `python -m metaboenrich`, the
benchmark and `benchmarks/published_predictions.py`; default `decreased`.
Significant metabolites with log2FC < 0 are the sources and log2FC > 0 the
targets. A log2FC of exactly 0 gets no role, and phi = |log2FC| is unchanged.
This departs from protocol §1.3, which makes increases the sources; the
protocol PDF was not edited. `--sources increased` restores the protocol's
orientation. `summary.json` and the benchmark's `run_info.json` record
`"orientation"`; runs without that key used `increased`.
Reyna judged the original assignment backwards. This is the
"activation" orientation D17 listed as not tried. One-sided data (D10, rule
unchanged): with only increases there are now only targets, and the measured,
unchanged metabolites act as the missing sources.
Current flow does not change. Swapping sources and targets negates every
current, all scores use |current|, and the pair weight is symmetric. Checked:
example DAA genes and reactions identical (max difference 2e-15); group
benchmark raw and leave-one-out ranks identical for all 21 disorders; patient
raw ranks identical for all 166 patients, and one leave-one-out rank differs
at the tail (HEP-GAMT-8: 1,887 vs 1,873 of 2,302, a float tie). Unit tests
lock this in. Directed mode (`--mode directed`) is not symmetric; no stored
output used it.
The absorbing walk changes and is much worse on the inborn-error benchmark
(rank of the causal gene among 2,302 genes; 20 primary disorders, 159
patients with a walk rank). Group level, median / top 10 / top 100: walk with
sources decreased 284 (over the 19 disorders it can score) / 0 / 6; walk with
sources increased (D17) 48 / 5 / 13; current flow 88.5 / 3 / 11. Patient
level, median / top 10 / top 100 / top 500: walk decreased 459 / 5.6% /
18.8% / 51.9%; walk increased 67 / 28.1% / 55.0% / 78.8%; current flow 140
(145 over the 159) / 5.0% / 46.9% / 78.8%. The walk ranks the causal gene
above current flow for 28% of patients (was 71%) and in 4 of 20 disorders by
patient median (was 14); 4 patients have it first (was 25). Median share of
walkers that reach a target: 1.8% group, 2.1% patient (was 6.6%, 3.8%).
Partial absorption still changes almost nothing: group medians 284–285 for
K = 0.5, 1 and 2; patient median 467 at K = 1 vs 459. Largest losses vs
current flow (group): TMLHE deficiency 4 → 1,172, TYMP deficiency 1 → 212, MCC
deficiency 23 → 1,649, argininemia 89 → 2,013, PKU 186 → 1,134. Largest
gains: ABAT 282 → 18, PA 1,187 → 170, ADSL deficiency 362 → 71, MMA
439 → 106. Isovaleric aciduria can no longer be scored at group level: no
walker reaches a target (was 64; current flow 17). One of its two patients is
also unscored.
A reading, not a test: in an enzyme defect the substrate side usually rises and
the product side falls. Walkers now start on the product side and must follow
reaction directions to the increased metabolites, which lie upstream of the
block, so the faulty enzyme is seldom on a successful route. The D17 loss
cases read differently. AADC: 3-O-methyldopa is now a target; walk rank 1,502
(was 1,317; current flow 88), 0.1% of walkers reach a target. Homocystinuria:
cortisol, the only significant decrease, is now the only source; rank 604
(was 765; current flow 181), 8.7% reach a target. On the synthetic example
26.5% of walkers reach a target (was 19.8%) and all 72 pairs are reached (was
63).
Reyna kept `decreased` as the default after seeing these results. The blind
predictions in `predictions/published_v1` stay sealed (D14).
`predictions/published_v2` holds the same 111 units rebuilt under the new
orientation from v1's `daa_mapping.tsv` (`--from-run`). All 66 run units keep
the diagnosed gene's rank; 55 of 66 top-15 gene lists are identical and the
other 11 reorder only within exact ties.
Open: test the walk on data where a flux increase is expected (an activated
pathway); directed mode under the new orientation is untested.

## Open

- Final score: mean of percentile ranks is a placeholder.
- Fold-change magnitude does not enter L x = b (`--pair-weight fc` exists).
- Directed current (protocol §2.3): no directed Laplacian; the absorbing walk
  (D17) is an experimental alternative. Whether to make it the default is open.
- Orientation (D20): the walk is much worse with decreases as sources on the
  inborn-error benchmark; test it on data where a flux increase is expected.
- Deoxynucleotides as cofactor families; RNA/DNA polymer nodes as carriers
  (likely source of HK1–3, PKM, RAD1/TREX1 recurring hits).
- Organic acidurias (MMA, PA, GA) rank poorly with every method.
- Candidate cofactors not yet handled: glutathione, THF, acetyl-CoA,
  glutamate/AKG, carnitine.
