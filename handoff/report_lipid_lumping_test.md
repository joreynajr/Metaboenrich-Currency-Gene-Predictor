# Report: lipid lumping test (reply to handoff_lipid_lumping_test.md)

Repo `Metaboenrich-Currency-Gene-Predictor`, branch `three-rule-revision`. Nothing committed; the default network and the `three_rule` stages are unchanged (tests pass). The `rescue_pick` switch is untouched.

Benchmark: the 20 primary inborn errors, group level, `--id-translation`, raw scoring, rank of the causal gene (lower is better). Baseline is `three_rule:three_rule`: median rank 140.5, 10/20 in the top 5%, 2,191 genes.

## Recommendation

**(ii), on the scoring side, using the "collapse" form.** The bias is real and large, it lives in the gene score rather than in the network, and merging lipids barely touches it.

- Collapse gives the best result of any change tried: median 140.5 → 95.5. 16 diseases improve and 3 get worse; sign-flip p = 0.026; bootstrap 95% CI on the median difference (−72, +3.5).
- That p-value is nominal. Eight variants were tried, so treat it as a strong lead, not a settled result.
- Merging lipids is close to neutral (median 136–137, CI including 0). The emptied reactions come back attached through NAD(P)/FAD couples, and the fatty-acid genes keep most of their nodes.

## Q1. How a gene's score is built (`scoring.py: gene_table`)

**Max, with no per-gene normalisation.**

- A reaction node's `final_score` is the mean percentile rank of its conductivity and its bottleneck, among the reaction nodes that carry current.
- A gene takes the **best** score among its reaction nodes. Ties are broken by the **sum** of accumulated current over its nodes.
- GPR and/or logic is ignored.
- The benchmark ranks genes in that order; genes with no current share the tail rank.

A max is not neutral to node count. A gene with k nodes gets k draws, so its best percentile is inflated: under the null, the best of k uniform draws has an expected value of k/(k+1). The sum tie-break adds a smaller bias in the same direction.

## Q2. The bias on the current network

| Gene scoring | Mean Spearman (node count vs rank) over 20 diseases | Median percentile of the 15 fatty-acid genes | Median percentile of the rest |
|---|---|---|---|
| max (current) | **−0.53** (range −0.59 to −0.44) | **0.048** | 0.50 |
| Šidák (best^k) | −0.18 | 0.50 | 0.50 |
| collapse (best^k_eff) | −0.40 | 0.24 | 0.50 |

- With the current scoring, the fatty-acid genes sit in the top 5% in every disease, whatever was measured. On the network as built, node counts are a median of 2, a 90th percentile of 16, and a maximum of 235 (HADHA).
- None of the 20 diseases has a causal gene in the fatty-acid set; there is no fatty-acid oxidation disorder in the benchmark. The benchmark therefore can only show the cost of the bias (these genes crowding out causal genes), never a benefit.

## Q3 and Q4. Benchmark

### Variants

| Run | What changed |
|---|---|
| base | `three_rule:three_rule` |
| sidak | Gene score = best^k, k = the gene's current-carrying nodes (Šidák) |
| collapse | Gene score = best^k_eff, k_eff = number of distinct gene sets among those nodes. Parallel chain-length copies with the same genes count once. |
| acyl_merged, acyl_merged_kr | The existing stages. They sit on the Rules 1–2 network (no Rule 3, no Rule 4), so each differs from base by more than one change. |
| coa_classes | Final network with fatty acyl-CoAs merged into C13–C21 and C22+ |
| coa12 | Fatty acyl-CoAs of C12 and longer merged into one node |
| coa12_acp | The same, plus acyl-[ACP] of C12 and longer |
| coa12_acp_ag3p | The same, plus 1-acylglycerol-3P of C12 and longer |

### Summary

Better and worse count diseases against base.

| Run | Median | Top 5% | Better / worse | Median paired difference | Bootstrap 95% CI, median difference | Sign-flip p |
|---|---|---|---|---|---|---|
| base | 140.5 | 10 | – | – | – | – |
| sidak | 137.0 | 9 | 11 / 7 | −5.5 | (−93, +102) | 0.40 |
| **collapse** | **95.5** | 10 | **16 / 3** | −10.0 | (−72, +3.5) | **0.026** |
| acyl_merged | 164.5 | 9 | 7 / 9 | 0 | (−79, +120) | 1.0 |
| acyl_merged_kr | 171.5 | 9 | 7 / 9 | 0 | (−71, +120) | 1.0 |
| coa_classes | 136.0 | 10 | 8 / 4 | 0 | (−29, +1.5) | 1.0 |
| coa12 | 136.0 | 10 | 9 / 3 | 0 | (−29, +2) | 1.0 |
| coa12_acp | 137.0 | 10 | 9 / 5 | 0 | (−29, +2.5) | 1.0 |
| coa12_acp_ag3p | 137.0 | 10 | 9 / 4 | 0 | (−29, +2) | 1.0 |

### Per-disease ranks

| Disease | base | sidak | collapse | acyl_merged | acyl_merged_kr | coa_classes | coa12 | coa12_acp | coa12_acp_ag3p |
|---|---|---|---|---|---|---|---|---|---|
| pku | 224 | 154 | 172 | 189 | 190 | 195 | 197 | 197 | 197 |
| msud | 39 | 72 | 43 | 26 | 29 | 34 | 31 | 29 | 29 |
| abat | 362 | 367 | 465 | 398 | 464 | 461 | 464 | 453 | 453 |
| aadc | 16 | 165 | 14 | 16 | 16 | 16 | 16 | 16 | 16 |
| gamt | 9 | 1 | 1 | 9 | 9 | 9 | 9 | 9 | 9 |
| otc | 232 | 124 | 156 | 206 | 204 | 205 | 203 | 203 | 203 |
| cit | 196 | 150 | 136 | 184 | 186 | 187 | 187 | 187 | 187 |
| asld | 31 | 24 | 28 | 33 | 36 | 34 | 35 | 34 | 34 |
| arg | 85 | 45 | 55 | 81 | 83 | 85 | 85 | 87 | 87 |
| mma | 703 | 338 | 551 | 909 | 1202 | 725 | 724 | 725 | 725 |
| pa | 741 | 575 | 578 | 924 | 874 | 746 | 734 | 737 | 737 |
| ga | 292 | 185 | 231 | 145 | 157 | 286 | 283 | 281 | 281 |
| adsl | 313 | 426 | 420 | 479 | 442 | 293 | 285 | 282 | 280 |
| mcc | 19 | 30 | 17 | 22 | 22 | 19 | 19 | 21 | 19 |
| tmlhe | 7 | 3 | 3 | 7 | 7 | 7 | 7 | 7 | 7 |
| hmgcl | 54 | 54 | 42 | 208 | 191 | 53 | 53 | 53 | 53 |
| hcu | 236 | 413 | 200 | 211 | 233 | 236 | 236 | 236 | 236 |
| iva | 17 | 5 | 12 | 36 | 36 | 17 | 17 | 17 | 17 |
| tp | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| hlcs | 421 | 694 | 319 | 602 | 806 | 400 | 398 | 403 | 400 |

### Šidák vs collapse

Šidák removes the bias completely but over-corrects. It penalises genes that really are multi-node:

| Gene | Disease | Nodes | Rank, base → Šidák |
|---|---|---|---|
| DDC | aadc | 16 | 16 → 165 |
| CBS | hcu | 5 | 236 → 413 |
| HLCS | hlcs | 5 | 421 → 694 |

Collapse only discounts copies that share a gene set, which are mostly chain-length ladders. It keeps those genes where they were or better.

- **Caveat on collapse:** "same gene set" is a stand-in for "same enzyme step at a different chain length". A gene that catalyses distinct steps with the same partner genes is also collapsed.
- **Possible refinement:** key the groups on gene set plus EC number.

## Q5. Merged variants: network, genes, and what happens to emptied reactions

| Run | Metabolites | Reaction nodes | Edges | Cut-off metabolites | Genes | Genes in main | Top-1% flow |
|---|---|---|---|---|---|---|---|
| base | 3,808 | 6,066 | 12,688 | 411 | 2,191 | 2,046 | 0.1546 |
| acyl_merged | 3,497 | 5,855 | 12,254 | 311 | 2,131 | 1,988 | 0.1797 |
| acyl_merged_kr | 3,503 | 6,262 | 13,110 | 313 | 2,191 | 2,047 | 0.1862 |
| coa_classes | 3,536 | 5,941 | 12,430 | 407 | 2,191 | 2,046 | 0.1831 |
| coa12 | 3,513 | 5,910 | 12,361 | 407 | 2,191 | 2,046 | 0.1845 |
| coa12_acp | 3,480 | 5,890 | 12,317 | 407 | 2,191 | 2,046 | 0.1862 |
| coa12_acp_ag3p | 3,422 | 5,890 | 12,317 | 407 | 2,191 | 2,046 | 0.1804 |

Classes, from structure (fatty acyl-CoAs) and formula (ACP, acylglycerol-3P):

| Class | Merged | Kept separate |
|---|---|---|
| Fatty acyl-CoA | 296 (C12+); 274 with the existing C13–C21 / C22+ boundary | 104 below the boundary |
| Acyl-[ACP] | 34 (C12+; acyl carbons = C − 11) | 37 below |
| 1-Acylglycerol-3P | 59 (C12+; acyl carbons = C − 3) | – |

- **34 non-fatty CoAs excluded.** A CoA counts as fatty only if the acyl part adds no ring and no N, P or S. The excluded set:
  - choloyl-, chenodeoxycholoyl-, deoxycholoyl-, ursodeoxycholoyl- and three muricholoyl-CoAs;
  - the DHCA, THCA and tetraHCA CoAs and the cholestanoyl/cholestenoyl intermediates;
  - 11 cysteinyl-leukotriene (LTE4/LTE5) CoAs;
  - retinoyl-CoA.
- Your section 6 reading was correct: the existing `acyl_coa_merge` would put the bile-acid CoAs into C22+.
- Acylcarnitines are never merged.

Reactions the merge leaves with no pair (every remaining pair is now inside one merged node):

| Run | Emptied | Rescued | Rescued through a couple (≥10 reactions) | Dropped (no other pair) | Genes on dropped reactions | Genes lost from network |
|---|---|---|---|---|---|---|
| coa_classes | 435 | 321 | 320 | 114 | 24 | 0 |
| coa12 | 491 | 360 | 359 | 131 | 24 | 0 |
| coa12_acp | 501 | 367 | 366 | 134 | 25 | 0 |
| coa12_acp_ag3p | 501 | 367 | 366 | 134 | 25 | 0 |

- **Your FAD/FADH₂ concern is confirmed, and it is broader.** Almost every rescue goes through a cofactor couple. For coa12:
  - NADP⁺/NADPH ×117
  - FAD/FADH₂ ×76
  - NAD⁺/NADH ×75
  - acetyl-CoA/CoA ×46
  - CO₂/malonyl-CoA ×43
- An emptied β-oxidation step therefore stays in the network, attached only to its electron carrier.
- No gene leaves the network, because each gene on a dropped reaction has other nodes.
- The merge removes few reaction nodes, because a reaction node is per reaction rather than per metabolite:

  | Gene | Nodes, base → coa12 |
  |---|---|
  | HADHA | 235 → 169 |
  | EHHADH | 167 → 92 |
  | ACAA1 | 190 → 147 |
  | CPT1A, CPT1B, CPT2, CRAT | unchanged (acylcarnitines are not merged) |

  So the merge cannot fix the gene-score bias.
- The merge also concentrates flow: top-1% share 0.155 → 0.18. The merged node becomes a hub.
- The existing `acyl_merged` stages do worse (median 164.5 and 171.5). They sit on the Rules 1–2 network, so the difference is not attributable to the merge alone.
- **Random-merge control: not run.** Merging ~300 random metabolites into one node creates an unrelated mega-hub that short-circuits current, so it would not be a fair null.

## Implementation (uncommitted)

- `atoms.py`: `fatty_acyl_coas` (structure, with the fatty test), `lipid_merge` (classes, boundary, members table), and constants `ACP_CARBONS`, `ACYLGLYCEROL3P_CARBONS`, `LIPID_MIN_CARBONS`.
- `structure.py`: new stage set `lipid_merge`, with stages:
  - `three_rule`
  - `lipid_coa_classes`
  - `lipid_coa12`
  - `lipid_coa12_acp`
  - `lipid_coa12_acp_ag3p`

  The final network's pairs are renamed through the merge and pairs inside one node are dropped. Nothing else is recomputed. An emptied reaction gets the rarest remaining carbon pair, never one Rule 4 removed and never for a membrane-crossing reaction. The logs are `LIPID_MERGE_LOG` and `LIPID_MERGE_MEMBERS`. Benchmark with `--stage lipid_merge:<key>`.
- The scoring variants are in an experiment script only (not in `scoring.py`).

## Suggested next steps

1. Put collapse into `scoring.py` as an option, off by default. Test the gene set + EC grouping.
2. Re-run the safeguard experiment (rarest / carbon / central) under collapse scoring. Part of "central"'s gain may have come from the same bias.
3. Add a benchmark case with a fatty-acid oxidation disorder (MCAD or VLCAD, if plasma acylcarnitine data exist). Without one, nothing tests whether a change hurts those genes.
