# Benchmark: finding the defective enzyme in inborn errors of metabolism

Question: given untargeted plasma metabolomics from patients with a known
single-enzyme defect, where does Metaboenrich rank the causal gene?

Run: Metaboenrich 0.2.0.dev0, network v0.1.1 (3,958 metabolites, 6,377
reactions, 16,356 edges), 2026-09-29. Reproduce with
`python benchmarks/get_benchmark_data.py` then
`python -m metaboenrich.benchmark --out results/benchmark_v02`.

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

## Method

- **Group level**: all patients with a disorder. Mean z per metabolite; z-test
  on the mean (values are already standardised), BH-adjusted; significant if
  q < 0.05 and |mean z| ≥ 1.
- **Patient level**: each patient alone; significant if |z| ≥ 2.
- **Ranking universe**: the 2,302 genes on network reactions; the causal gene's
  rank is its best rank among the listed causal genes. Random expectation:
  median ≈ 1,151.
- **Neighbour baseline**: rank each enzyme by the number of significant
  metabolites its own reactions touch. Current flow must beat it to add
  information beyond "this enzyme's substrate changed".
- One-sided flow is used when a disorder has significant changes in only one
  direction.

## Results

### Summary

| Level | Method | Median rank (percentile) | Top 10 | Top 50 | Top 100 |
|---|---|---|---|---|---|
| Group (20) | **Metaboenrich, raw** | **88 (3.8%)** | 3 | 7 | **11** |
| Group (20) | Metaboenrich, leave-one-out | 237 (10.3%) | 4 | 7 | 9 |
| Group (20) | Neighbour baseline | 1,165 (50.6%) | 1 | 5 | 7 |
| Patient (160) | **Metaboenrich, raw** | **139 (6.0%)** | 8 | 55 | 75 |
| Patient (160) | Metaboenrich, leave-one-out | 224 (9.7%) | 12 | 43 | 58 |
| Patient (160) | Neighbour baseline | 1,168 (50.7%) | 5 | 51 | 71 |

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
- Organic acidurias (MMA, PA, GA) rank poorly with every method; their main
  markers (acylcarnitines, organic acids) may map poorly or sit behind CoA
  handling. Open question.

## Caveats

- Small: 20 disorder groups, several with 1–4 patients.
- Plasma, not cells or tissue.
- The scoring choice (raw vs leave-one-out) was made on this benchmark; the
  published-dataset blind test is the held-out check.
- Name matching changed between the first benchmark run and this one (loose
  matching maps some additional Metabolon names), which slightly changed
  source/target counts for a few disorders.
