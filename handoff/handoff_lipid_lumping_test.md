# Handoff for Claude Code: should the lipid-derived nodes be lumped, and the test to decide it

Repo: Metaboenrich-Currency-Gene-Predictor, branch `three-rule-revision`. Please do not change the default network or the committed default behaviour. The uncommitted `rescue_pick` switch from the safeguard experiment is still in the working tree; leave it as it is.

## 1. The question
The audit of "underpruned" metabolites has reached the lipid-derived classes: acyl-CoAs, acylcarnitines, acyl carrier protein (ACP) intermediates and 1-acylglycerol-3P species with many chain lengths. The idea on the table is to merge the long-chain members of these classes (for example every acyl-CoA of C12 or more) into one node per class. The aim is to shrink the node set with species that polar metabolomics will not measure. The alternative is to leave them as they are. We need a test that tells us which is better. We do not yet know whether leaving them costs anything.

## 2. What is already in the code
- `atoms.py: acyl_coa_merge(model, smiles_path, classes)` finds acyl-CoAs from structure (CoA substructure plus a thioester) and merges them into two classes by acyl carbon count (carbons minus 21): long-chain C13-C21 and very-long-chain C22+. Short and medium chains, acetyl-CoA and malonyl-CoA stay separate. Pool pseudo-metabolites are never merged.
- `structure.py: _three_rule_stages` already builds `acyl_merged` and `acyl_merged_keep_rarest` as alternative stages. Pairs are renamed first, and a pair that ends up inside one merged node is dropped. Couples are counted on the merged network.
- `--stage SET:KEY` benchmarks any named stage (CHANGELOG, decision D16). I do not know whether these two stages have been benchmarked against the final network. Please check the existing logs before running anything.

## 3. Numbers from the workbook (name-parsed, so approximate)
Source: `network_before_after_revised.xlsx`. Chain lengths come from node names, with some truncated names unparsed; replace these with the structure-based lengths from `acyl_coa_merge` where possible.

| class | nodes >= C12 | raw reactions | with KEGG/HMDB ID |
|---|---|---|---|
| acyl-CoA | 285 | 2,389 | 84 |
| 1-acylglycerol-3P | 40 (+19 unparsed) | 361 | 8 |
| ACP | 32 (+6 unparsed) | 63 | 10 |
| acylcarnitine | 76 | 203 | 29 |

- All four classes together: 691 of the 3,808 pruned nodes (18%). Only about 8 of them are cut off from the main network, so this is not mainly a connectivity problem.
- Merging acyl-CoAs of C12 or more takes 285 nodes to 1. Estimated distinct reaction signatures in the pruned network fall from 4,820 to 4,399 (acyl-CoA only) or 4,253 (all four classes). This is a rough estimate from raw equations.
- 572 pruned reactions have a long acyl-CoA on both sides, involving 49 genes (ACAA1/2, ACADL/M/S/VL, ACOX1-3, DECR1/2, ECHS1, EHHADH, ELOVL1-6 and others). After a merge their chain-changing pair would sit inside one node and be dropped. This is an upper bound on the reactions that lose a pair; I have not counted how many lose all pairs.

## 4. The concern with leaving them as they are
The median gene in the pruned network has 2 reaction nodes (90th percentile 18). The 15 genes with the most reaction nodes are all fatty acid oxidation or activation genes:

HADHA 341, EHHADH 269, ACAA1 265, CPT1A 228, CPT1B 225, CPT1C 216, SLC27A2 196, HSD17B4 187, CRAT 182, CPT2 176, ACOX1 163, ACSL1 155, ACSL5/6/4 153-154.

Each chain length gets its own parallel copy of the same enzyme step. Hypothesis (untested): any gene score that sums or counts reaction nodes per gene favours these genes, and the long chain-length ladders (one node per length) change path lengths for current flow. If the score takes a max or a mean per gene, the effect may be small.

## 5. What I need you to do
1. **How is a gene score built from reaction nodes?** Read `scoring.py`. State whether a gene's score aggregates over its reaction nodes by sum, max, mean or something else, and whether any per-gene normalisation exists.
2. **Does the bias exist on the current network?** On the final network, report the Spearman correlation between a gene's number of reaction nodes and its score or rank. Compare the fatty acid genes in section 4 against the rest. If you use the benchmark runs, report which of the 20 diseases have a causal gene in this set.
3. **Benchmark the variants, one change at a time, each against the current final network (`three_rule:three_rule`):**
   - (a) existing `acyl_merged` and `acyl_merged_keep_rarest`;
   - (b) acyl-CoA merge with a C12 boundary (long = C12 and above), bile acid and other non-fatty CoAs excluded (see section 6);
   - (c) (b) plus ACP intermediates;
   - (d) (c) plus 1-acylglycerol-3P;
   - acylcarnitines are never merged in any variant (see section 6).
4. **Scoring-side alternative, on the unchanged network:** normalise each gene's score by its number of reaction nodes, or collapse parallel reaction nodes that carry the same gene set and differ only by chain length, for scoring only. Benchmark that too. If it removes the bias without touching the network, it is the lower-risk fix.
5. **For each merged variant, report:** metabolite, reaction-node and edge counts; cut-off metabolites (baseline 411); genes on the network and genes in the main component (baseline 2,191 and 2,046); top-1% flow share (baseline 0.1546); how many reactions lose every pair after the merge, which genes they carry, and what the safeguard keeps for them. In `carbon_method_rules` the un-mapped rescue path only considers pairs that are already couples (`present` is restricted to `couples`), and 60 of the 217 current rescues are un-mapped. Please check whether any of these genes end up attached through a couple such as FAD/FADH2.
6. **Metrics:** per-disease rank for all 20, the median, top-5% count, and paired per-disease differences against the baseline. With only 20 diseases, add a bootstrap or permutation interval on the median difference, and a random-merge control (merge random groups of the same sizes) if it is cheap.

## 6. Constraints and cautions
- **Non-fatty CoAs must not enter the merge.** By my reading of `acyl_coa_merge`, it counts acyl carbons only, so a bile acid CoA such as choloyl-CoA (C24) would fall into the C22+ class. About 35 non-fatty CoAs showed up in my name parse: choloyl-CoA, chenodeoxycholoyl-CoA, 25(R)DHCA-CoA, HMG-CoA, 10,11-dihydro-LTB4-CoA, dephospho-CoA and similar. Please confirm and exclude them.
- **Acylcarnitines stay unmerged.** Long-chain acylcarnitines are routinely reported in untargeted plasma metabolomics, and 29 of 76 long ones already have HMDB IDs. They are the markers for fatty acid oxidation disorders. Only 6 genes touch them (CPT1A/B/C, CPT2, CRAT, CROT).
- **Boundary.** The existing code uses C13-C21 and C22+. The proposal is C12 and above. Test both, since chain length also separates enzyme specificity (ACADM vs ACADVL, the ELOVL family).
- **Do not use my chain-length parse for the final merge.** It is name-based and approximate. Use the structure-based carbon counts.
- **Keep the default network unchanged.** Put new variants behind new named stages, and keep the existing tests passing.

## 7. Deliverable
A short report with: the answer to question 1, the bias check from question 2, a table of benchmark results for the variants (per-disease ranks and paired differences), the node, edge and gene table from question 5, and a recommendation among (i) leave as is, (ii) scoring-side normalisation, (iii) merge acyl-CoAs, (iv) merge more classes.
