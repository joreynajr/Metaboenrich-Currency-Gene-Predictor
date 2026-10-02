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

### D17. Absorbing random walk: built, kept optional (2026-10-01, Reyna: plainest version, hooks 1–2)
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

## Open

- Final score: mean of percentile ranks is a placeholder.
- Fold-change magnitude does not enter L x = b (`--pair-weight fc` exists).
- Directed current (protocol §2.3): no directed Laplacian; the absorbing walk
  (D17) is an experimental alternative. Whether to make it the default is open.
- Deoxynucleotides as cofactor families; RNA/DNA polymer nodes as carriers
  (likely source of HK1–3, PKM, RAD1/TREX1 recurring hits).
- Organic acidurias (MMA, PA, GA) rank poorly with every method.
- Candidate cofactors not yet handled: glutathione, THF, acetyl-CoA,
  glutamate/AKG, carnitine.
