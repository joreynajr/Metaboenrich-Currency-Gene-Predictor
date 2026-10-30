# Benchmark: finding the defective enzyme in inborn errors of metabolism

Question: given untargeted plasma metabolomics from patients with a known
single-enzyme defect, pooled by disorder, where does Metaboenrich rank the
causal gene?

Run: Metaboenrich 0.2.0.dev0, network v0.1.1 (3,958 metabolites, 6,377
reactions, 16,356 edges), 2026-09-29. Reproduce with
`python benchmarks/get_benchmark_data.py` then
`python -m metaboenrich.benchmark --out results/benchmark_v02`. Since
decision D21 (2026-10-19) results are reported at group level only, and the
benchmark's default is `--levels group`.

## Data

Per-sample metabolite z-scores against reference populations, with each
sample's confirmed diagnosis, distributed in the CTD R package on CRAN
(v1.3, MIT licence; SHA-256 `f59dcdf6…`):

- **Thistlethwaite et al. 2022** (Sci Rep 12:6556, doi:10.1038/s41598-022-10415-5):
  545 plasma samples, 1,364 metabolites, 16 disorders plus references. Main set.
- **Miller et al. 2015** (J Inherit Metab Dis 38:1029, doi:10.1007/s10545-015-9843-7):
  186 plasma samples, 1,203 metabolites. Its heparin-plasma cohorts are also in
  Thistlethwaite 2022; used here only for disorders not in that set.

## Cases

`benchmarks/cases.tsv`: 20 primary disorders with polar-metabolite signatures
and their causal genes (standard assignments per diagnosis), one secondary
(cobalamin defects, many possible genes), and five excluded with reasons
(Zellweger spectrum and RCDP: lipid; MCAD and VLCAD: fatty-acid oxidation;
lysinuric protein intolerance: transporter).

Sources (added 2026-10-16): the gene assignments were first made without a
recorded source. Each row now cites its OMIM phenotype entries
(`omim_phenotype`) and GeneReviews chapter (`genereviews`, NBK accession and
PMID; 20 of 26 have one), retrieved through NCBI E-utilities and Europe PMC.
Every gene in `causal_genes` matches its OMIM phenotype entry, except TMLHE
deficiency, which has no OMIM phenotype entry. `source_check` notes where OMIM
lists further genes that the table leaves out (MSUD: DLD, PPM1K; isolated MMA:
MCEE, MMADHC; cobalamin: HCFC1, THAP11; homocystinuria: other causes).

## Method

- **Group level**: all patients with a disorder. Mean z per metabolite; z-test
  on the mean (values are already standardised), BH-adjusted; significant if
  q < 0.05 and |mean z| ≥ 1.
- **Group level only** (D21): a single patient's z-scores against a
  reference population are not a differential analysis, so per-patient runs
  are not reported. `--levels patient` still exists, for reference only.
- **Ranking universe**: the 2,302 genes on network reactions; the causal gene's
  rank is its best rank among the listed causal genes. Random expectation:
  median ≈ 1,151.
- **Neighbour baseline**: rank each enzyme by the number of significant
  metabolites its own reactions touch. Current flow must beat it to add
  information beyond "this enzyme's substrate changed".
- One-sided flow is used when a disorder has significant changes in only one
  direction.
- **Orientation**: since decision D20 (2026-10-16), significant decreases are
  the sources and significant increases the targets by default
  (`--sources decreased`). This departs from protocol §1.3. The run reported
  here predates D20 and used the protocol's orientation (increases as sources).
  Current flow is symmetric, so the ranks are the same under either
  orientation. Two later reruns, one per orientation (2026-10-01 with
  increases as sources, 2026-10-16 under D20), gave identical raw and
  leave-one-out ranks for every disorder. Source and target counts swap.
  Both reruns differ from the tables below in a few ranks (for example
  argininemia raw 89, not 88). The absorbing walk (`--mode walk`) is not
  symmetric and does change; see "Absorbing walk, both orientations" below
  and `docs/rw_report.html`.

## Results

### Summary

| Level | Method | Median rank (percentile) | Top 10 | Top 50 | Top 100 |
|---|---|---|---|---|---|
| Group (20) | **Metaboenrich, raw** | **88 (3.8%)** | 3 | 7 | **11** |
| Group (20) | Metaboenrich, leave-one-out | 236.5 (10.3%) | 4 | 7 | 9 |
| Group (20) | Neighbour baseline | 1,165 (50.6%) | 1 | 5 | 7 |

### Per disorder (group level)

| Disorder | Gene(s) | Patients | Raw | Leave-one-out | Baseline |
|---|---|---|---|---|---|
| Thymidine phosphorylase deficiency | TYMP | 2 | **1** | 1 | 1,169 |
| GAMT deficiency | GAMT | 8 | **2** | 1,888 | 34 |
| TMLHE deficiency | TMLHE | 4 | **4** | 5 | 4 |
| Isovaleric aciduria | IVD | 2 | **17** | 15 | 1,184 |
| 3-MCC deficiency | MCCC1/2 | 4 | **23** | 19 | 1,160 |
| Argininosuccinic aciduria | ASL | 13 | **24** | 2 | 94 |
| Maple syrup urine disease | BCKDHA/B, DBT | 18 | **33** | 5 | 46 |
| HMG-CoA lyase deficiency* | HMGCL | 2 | 55 | 220 | 1,158 |
| Citrullinemia type I | ASS1 | 9 | 82 | 352 | 75 |
| AADC deficiency | DDC | 3 | 88 | 253 | 1,218 |
| Argininemia | ARG1 | 17 | 88 | 63 | 24 |
| OTC deficiency | OTC | 34 | 168 | 48 | 1,241 |
| Homocystinuria | CBS | 2 | 181 | 493 | 1,246 |
| Phenylketonuria* | PAH | 8 | 186 | 73 | 16 |
| GABA-transaminase deficiency* | ABAT | 7 | 282 | 282 | 1,152 |
| ADSL deficiency | ADSL | 3 | 362 | 384 | 1,172 |
| Methylmalonic aciduria | MMUT/MMAA/MMAB | 9 | 439 | 676 | 1,280 |
| Holocarboxylase synthetase deficiency* | HLCS | 1 | 717 | 647 | 1,172 |
| Glutaric aciduria type I | GCDH | 5 | 930 | 843 | 1,174 |
| Propionic aciduria | PCCA/B | 9 | 1,187 | 1,088 | 1,285 |
| *(secondary)* Cobalamin defects | several | 6 | 61 | 23 | 1,174 |

\* one-sided run (only increases significant).

### Absorbing walk, both orientations (group level)

The experimental walk (`--mode walk`, decision D17) follows reaction
directions, so it depends on which side starts. It is reported under both
orientations:

- **sources = up**: significantly increased metabolites are the sources,
  decreased are the targets (`--sources increased`; protocol §1.3; the
  orientation before D20).
- **sources = down**: significantly decreased metabolites are the sources,
  increased are the targets (`--sources decreased`; the default since D20).

Runs: sources = down on 2026-10-16 (`results/bench_rw/`); sources = up on
2026-10-19 with current code (`results/bench_rw_up/`), identical to the
archived 2026-10-01 runs. Raw scoring, rank of the causal gene among 2,302
genes, 20 primary disorders.

| Method | Median | Top 10 | Top 50 | Top 100 |
|---|---|---|---|---|
| Current flow (either orientation) | 88.5 | 3 | 7 | 11 |
| Walk, sources = up | **48** | **5** | **10** | **13** |
| Walk, sources = down | 284 (19 scored) | 0 | 2 | 6 |
| Walk κ = 0.5 / 1 / 2, sources = up | 45.5 / 47 / 46.5 | 5 | 10 | 13 |
| Walk κ = 0.5 / 1 / 2, sources = down | 284 / 284 / 285 | 0 | 2 | 6 |

κ: partial absorption (`--walk-kappa`). Isovaleric aciduria cannot be scored
with sources = down: no walker reaches a target.

| Disorder | Current flow | Walk, sources = up | Walk, sources = down | Reach a target, up | Reach a target, down |
|---|---|---|---|---|---|
| Thymidine phosphorylase deficiency | 1 | 1 | 212 | 1.7% | 2.3% |
| GAMT deficiency | 2 | 3 | 85 | 5.5% | 1.3% |
| TMLHE deficiency | 4 | 1 | 1,172 | 8.1% | 1.3% |
| Isovaleric aciduria | 17 | 64 | not scored | 0.02% | 0% |
| 3-MCC deficiency | 23 | 25 | 1,649 | 5.1% | 1.1% |
| Argininosuccinic aciduria | 24 | 9 | 40 | 6.0% | 9.5% |
| Maple syrup urine disease | 33 | 1 | 888 | 4.1% | 10.8% |
| HMG-CoA lyase deficiency* | 55 | 18 | 1,001 | 43.2% | 1.1% |
| Citrullinemia type I | 82 | 20 | 56 | 29.3% | 3.9% |
| AADC deficiency | 88 | 1,317 | 1,502 | 3.6% | 0.1% |
| Argininemia | 89 | 32 | 2,013 | 13.5% | 7.8% |
| OTC deficiency | 168 | 72 | 60 | 7.2% | 1.1% |
| Homocystinuria | 181 | 765 | 604 | 1.0% | 8.7% |
| Phenylketonuria* | 186 | 16 | 1,134 | 30.8% | 4.7% |
| GABA-transaminase deficiency* | 282 | 237 | 18 | 72.8% | 1.1% |
| ADSL deficiency | 362 | 292 | 71 | 0.1% | 0.1% |
| Methylmalonic aciduria | 439 | 99 | 106 | 20.7% | 17.2% |
| Holocarboxylase synthetase deficiency* | 717 | 320 | 284 | 36.9% | 4.8% |
| Glutaric aciduria type I | 930 | 1,054 | 1,581 | 0.2% | 0.3% |
| Propionic aciduria | 1,187 | 176 | 170 | 20.3% | 11.0% |
| *(secondary)* Cobalamin defects | 61 | 70 | 662 | 8.3% | 2.1% |

"Reach a target": share of released walkers absorbed at a target (full
absorption). \* one-sided run.

## Reading the results

- Current flow ranks the causal gene far above chance and above the
  neighbour baseline in most disorders, often where the enzyme's own
  substrates and products are not among the significant metabolites
  (TYMP, IVD, MCC, OTC, AADC).
- Raw scoring beats leave-one-out overall. Leave-one-out discounts a single
  substrate–product signal, which is typical of an enzyme defect (GAMT).
  This is why raw scoring became the default (decision D9).
- The baseline wins where the defect's signal sits directly on the enzyme
  (PKU, argininemia, citrullinemia).
- The walk with sources = up beats current flow (median 48 against 88.5);
  with sources = down it is much worse (284). A reading, not a test: in an
  enzyme defect the substrate side usually rises and the product side falls.
  With sources = up, walkers start at the accumulated substrates and the
  blocked enzyme lies on their way to the depleted products. With sources =
  down they start on the product side, so the faulty enzyme is seldom on a
  successful route.
- Organic acidurias (MMA, PA, GA) rank poorly with current flow and the
  baseline. The walk does better for MMA and PA under either orientation
  (99 and 176 with sources = up, 106 and 170 with sources = down) but not for
  GA (1,054 and 1,581). Their main markers (acylcarnitines, organic acids)
  may map poorly or sit behind CoA handling. Open question.

## Caveats

- Small: 20 disorder groups, several with 1–4 patients.
- Plasma, not cells or tissue.
- The scoring choice (raw vs leave-one-out) was made on this benchmark; the
  published-dataset blind test is the held-out check.
- Name matching changed between the first benchmark run and this one (loose
  matching maps some additional Metabolon names), which slightly changed
  source/target counts for a few disorders.
