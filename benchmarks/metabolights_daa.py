"""Build differential-abundance (DAA) tables for the active D_metabolights units (D23).

MTBLS3628 (Zheng 2023, FH-deficient RCC): FH-deficient RCC (case) vs FH wildtype
RCC (control), 10 vs 10, normal kidney unused; one unit per polar assay (HILIC,
PHILIC). Same rule as the Workbench tables (D22 item 3): Welch t-test on
log2(value), non-positive or missing -> missing, >= 2 values per group,
log2FC = mean log2(case) - mean log2(control), significant if raw p < 0.05.

Writes results/metabolights_daa/<study>_<assay>.csv (name, CHEBI, log2FC, p,
n_case, n_control) and contrasts.tsv, which tools/null_calibration.py reads.

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
