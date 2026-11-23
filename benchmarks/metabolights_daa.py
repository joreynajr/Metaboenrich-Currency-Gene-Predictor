"""Build differential-abundance (DAA) tables for the active D_metabolights units (D23).

MTBLS3628 (Zheng 2023, FH-deficient RCC): FH-deficient RCC (case) vs FH wildtype
RCC (control), 10 vs 10, normal kidney unused; one unit per polar assay (HILIC,
PHILIC). Same rule as the Workbench tables (D22 item 3): Welch t-test on
log2(value), non-positive or missing -> missing, >= 2 values per group,
log2FC = mean log2(case) - mean log2(control), significant if raw p < 0.05.

Writes results/metabolights_daa/<study>_<assay>.csv (name, CHEBI, log2FC, p,
n_case, n_control) and contrasts.tsv, which tools/null_calibration.py reads.
Units added under D26 read per-sample tables built by benchmarks/metabolights_per_sample.py.

    python benchmarks/metabolights_daa.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "metabolights_daa"
DATA = Path.home() / "Desktop/Electrical Conductivity/Metabolights Benchmark Data"
REGISTRY = REPO / "benchmarks" / "data_registry.tsv"

# (study, assay) -> (per-sample file, case group, control group, reason)
CONTRASTS = {
    ("MTBLS3628", "HILIC"): ("Zheng2023/MTBLS3628_HILIC_per_sample.csv", "FH-deficient RCC", "FH wildtype RCC",
                             "tumour vs tumour isolates FH loss from the tumour-vs-normal shift (D23)"),
    ("MTBLS3628", "PHILIC"): ("Zheng2023/MTBLS3628_PHILIC_per_sample.csv", "FH-deficient RCC", "FH wildtype RCC",
                              "as HILIC; same 30 samples (D23)"),
}

# D26: per-sample tables written by benchmarks/metabolights_per_sample.py from the MAF + ISA sheets.
GLY = "Sample type:Celly lysate | Medium:{} | Treatment:{}"
PRODH = "Pluripotency:{} | Genotype:{}"
for _assay in ("RP_pos", "RP_neg"):
    for _state, _lv in (("naive", "Naïve"), ("primed", "Primed")):
        CONTRASTS[("MTBLS7840", f"{_state}_{_assay}")] = (
            f"MTBLS7840/MTBLS7840_{_assay}_per_sample.csv", PRODH.format(_lv, "shPRODH"), PRODH.format(_lv, "PLKO.1"),
            f"shPRODH vs empty pLKO.1 in {_state} H9 hESC (Replicate factor dropped; QC and blank excluded)")
    CONTRASTS[("MTBLS11858", f"AK4_{_assay}")] = (
        f"MTBLS11858/MTBLS11858_{_assay}_per_sample.csv", "adenylate kinase 4, mitochondrial", "Control",
        "shAK4 vs control shRNA, MT-4 ATL cells (shRHOC arm: RHOC not in the network; QC excluded)")
for _assay in ("HILIC", "RP"):
    for _gly, _lv in (("withGly", "with glycine"), ("noGly", "No glycine")):
        CONTRASTS[("MTBLS405", f"{_gly}_{_assay}")] = (
            f"MTBLS405/MTBLS405_Cell_{_assay}_per_sample.csv", GLY.format(_lv, "SHMT2-Down-regulated"),
            GLY.format(_lv, "control cell"),
            f"IPTG-induced shSHMT2 vs uninduced HeLa-SHMT2-shSHMT2 cells, medium {_lv} on both sides "
            "(SHMT2-up arm, benzoate and xenograft experiments not used; standards excluded)")
    CONTRASTS[("MTBLS10469", _assay)] = (
        f"MTBLS10469/MTBLS10469_{_assay}_per_sample.csv", "0.2μm dznep", "DMSO",
        "Pharmacological: DZNep (AHCY inhibitor) 0.2 uM vs DMSO, human erythroid cells")
CONTRASTS.update({
    ("MTBLS4722", "LCMS"): ("MTBLS4722/MTBLS4722_LCMS_per_sample.csv", "Gene Knockout", "Wild Type",
                            "Only design: IDH1 KO vs WT MiaPaCa-2"),
    ("MTBLS4223", "HILIC_neg"): ("MTBLS4223/MTBLS4223_HILIC_neg_per_sample.csv", "Gene Knockdown", "Wild Type",
                                 "Only design: OGDH knockdown vs WT primed hESC (Replicate factor dropped; QC and "
                                 "blank excluded)"),
})


def daa(path, case, control):
    raw = pd.read_csv(path, dtype={"CHEBI": str})
    groups = raw.iloc[0, 2:]
    body = raw.iloc[1:].reset_index(drop=True)
    vals = body.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
    vals = vals.where(vals > 0)
    # A name listed twice: keep the row with the higher median intensity.
    order = vals.median(axis=1).sort_values(ascending=False).index
    dup = body.loc[order, "metabolite"].duplicated()
    keep = order[~dup.to_numpy()]
    dropped = body.loc[order[dup.to_numpy()], "metabolite"].tolist()
    lv = np.log2(vals.loc[keep])
    a, b = lv.loc[:, groups == case], lv.loc[:, groups == control]
    rows = []
    for i in keep:
        x, y = a.loc[i].dropna(), b.loc[i].dropna()
        if len(x) < 2 or len(y) < 2:
            continue
        rows.append({"name": body.at[i, "metabolite"], "CHEBI": body.at[i, "CHEBI"],
                     "log2FC": x.mean() - y.mean(), "p": stats.ttest_ind(x, y, equal_var=False).pvalue,
                     "n_case": len(x), "n_control": len(y)})
    return pd.DataFrame(rows), int((groups == case).sum()), int((groups == control).sum()), dropped


def main():
    reg = pd.read_csv(REGISTRY, sep="\t", dtype=str).fillna("")
    act = reg[(reg.collection == "D_metabolights") & (reg.status == "active")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in act.itertuples(index=False):
        row = {"study": r.study, "analysis": r.sub_unit}
        if (r.study, r.sub_unit) not in CONTRASTS:
            rows.append({**row, "status": "not computable: no contrast defined"})
            continue
        f, case, control, reason = CONTRASTS[(r.study, r.sub_unit)]
        res, n_case, n_control, dropped = daa(DATA / f, case, control)
        res.to_csv(OUT / f"{r.study}_{r.sub_unit}.csv", index=False)
        rows.append({**row, "case": case, "control": control, "n_case": n_case, "n_control": n_control,
                     "reason": reason, "n_metabolites_tested": len(res), "n_p_lt_0.05": int((res.p < 0.05).sum()),
                     "duplicates_dropped": "; ".join(dropped), "status": "ok"})
        print(f"{r.study} {r.sub_unit}: {n_case} vs {n_control}, {len(res)} tested, "
              f"{(res.p < 0.05).sum()} with p < 0.05, {len(dropped)} duplicate names dropped")
    pd.DataFrame(rows).to_csv(OUT / "contrasts.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
