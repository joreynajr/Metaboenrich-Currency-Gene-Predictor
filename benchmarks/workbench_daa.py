"""Build differential-abundance (DAA) tables for the active C_workbench analyses.

For each active Metabolomics Workbench analysis in benchmarks/data_registry.tsv
(collection == C_workbench, status == active) this script:
  1. selects ONE case-vs-control contrast (hard-coded in CONTRASTS below, with the
     reason for the choice);
  2. cleans metabolite names (drops " (CAS# ...)" and the adduct after ";"), merges
     duplicate rows of the same cleaned name by keeping the row with the highest
     median intensity;
  3. runs a Welch t-test per metabolite on log2(value) (non-positive / missing ->
     missing; >= 2 non-missing values per group required), log2FC = mean log2(case)
     - mean log2(control); significance = raw p < 0.05 (no multiple-testing correction);
  4. attaches KEGG / HMDB ids from the METABOLITES section of the raw .mwtab file when
     present (first valid id if several are listed);
  5. writes <STUDY>_<SUB_UNIT>.csv and contrasts.tsv into results/workbench_daa/,
     which tools/null_calibration.py reads (D22). The sub_unit is the analysis id, or
     <ANALYSIS>_<suffix> when one analysis gives more than one contrast (D26).

The per-sample inputs are written by benchmarks/workbench_per_sample.py.

    python benchmarks/workbench_daa.py
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "results" / "workbench_daa"
DATA = Path(r"C:\Users\dmontefusco\Desktop\Electrical Conductivity\Metabolomics_Workbench_Data")
PER_SAMPLE = DATA / "workbench_per_sample_data"
MWTAB = DATA.parent / "workbench_data"
REGISTRY = REPO / "benchmarks" / "data_registry.tsv"
P_CUT = 0.05

# ---------------------------------------------------------------------------
# Contrast definitions. Each selector is a list of factor dicts (OR across dicts,
# AND within a dict); factor values are matched exactly against the parsed
# "Factor:Value | Factor:Value" #GROUP label; the key "@sample" is instead a regex on the
# sample id. A sub_unit "AN..._<suffix>" is a further contrast from analysis AN... (D26).
# ---------------------------------------------------------------------------
C = {}
SAMPLE_KEY = "@sample"   # selector key matched as a regex against the sample id instead of a factor


def add(study, analyses, case, control, reason, not_computable=None):
    for an in analyses:
        C[(study, an)] = dict(case=case, control=control, reason=reason,
                              not_computable=not_computable)


add("ST000276", ["AN000440", "AN000441"],
    [{"IDH1 Knockdown": "yes"}], [{"IDH1 Knockdown": "No"}],
    "Only design: shIDH1 knockdown vs empty-vector control (genetic).")
add("ST000458", ["AN000000"],
    [{"Time": "144(hrs)", "Treatment": "Doxycycline"}], [{"Time": "144(hrs)", "Treatment": "Vehicle"}],
    "Dox induces shBCAT1 (Tet-on hairpin). Latest time point (144 h) Dox vs time-matched vehicle.")
add("ST000687", ["AN001060"],
    [{"TYPE": "AK4"}], [{"TYPE": "Control"}],
    "Only design: AK4 knockdown ('AK4') vs control (study title: AK4 knockdown).")
add("ST000698", ["AN001083", "AN001084"],
    [{"KD#1": "yes"}, {"KD#2": "yes"}], [{"Condition": "yes"}],
    "SIRT5 KD: two independent hairpins (KD#1, KD#2) pooled as case vs control "
    "('Condition:yes' = non-targeting control); pooling avoids an arbitrary hairpin choice.")
add("ST000824", ["AN001310"],
    [{"Cell line": "MGG8", "IDH1-R132H EXPRESSION": "YES", "IDH1-R132H INHIBITOR": "NO"}],
    [{"Cell line": "MGG8", "IDH1-R132H EXPRESSION": "NO", "IDH1-R132H INHIBITOR": "NO"}],
    "Genetic: MGG8 stably expressing IDH1-R132H (neomorphic mutant) vs MGG8 control, both without "
    "inhibitor. MGG8 chosen over SJGBM2 because SJGBM2 R132H/no-inhibitor has n=2. NOTE: gain-of-function "
    "mutant, not loss.")
add("ST001375", ["AN002295"],
    [{"Genotype": "FN3K KO"}], [{"Genotype": "WT"}],
    "Only design: FN3K KO vs WT HepG2.")
add("ST002115", ["AN003513", "AN003514"],
    [{"Treatment": "RSL3"}], [{"Treatment": "DMSO"}],
    "Pharmacological (no genetic arm): RSL3, the canonical GPX4 inhibitor, vs DMSO "
    "(ML162/ML210 not used).")
add("ST003262", ["AN005346"],
    [{"Sample source": "Brain", "Treatment": "8hrs"}], [{"Sample source": "Brain", "Treatment": "CTL"}],
    "Pharmacological CKB inhibitor time course; latest time (8 h) vs CTL.")
add("ST003267", ["AN005352", "AN005353"],
    [{"DOX": "OFF"}], [{"DOX": "On"}],
    "FDX2-iKO JHOC5: endogenous FDX2 deleted, rescued by Dox-inducible (Tet-On) HA-FDX2. DOX OFF = FDX2 "
    "loss (case), DOX On = FDX2 expressed (control). Confirmed in the paper (PMID 39151727, "
    "doi:10.1016/j.jbc.2024.107678); data agree (succinate/fumarate ratio up in DOX OFF, as reported).")
add("ST003417", ["AN005616"],
    [{"Drug Treatment": "5 uM TVB-2640"}], [{"Drug Treatment": "Control"}],
    "Pharmacological FASN inhibition; TVB-2640 (clinical compound) at highest dose (5 uM) vs control "
    "(TVB-3166 arms not used).")
add("ST003486", ["AN005724"],
    [{"treatment": "10 uM 680c91"}], [{"treatment": "DMSO vehicle"}],
    "Pharmacological; 680c91 is TDO2-selective (AT0174 is dual TDO2/IDO1) so 10 uM 680c91 vs DMSO.")
add("ST003487", ["AN005725"],
    [{"treatment": "10 uM 680c91"}], [{"treatment": "DMSO vehicle"}],
    "Pharmacological; 680c91 is TDO2-selective (AT0174 is dual TDO2/IDO1) so 10 uM 680c91 vs DMSO.")
add("ST003488", ["AN005726"],
    [{"treatment": "shTDO2-082"}, {"treatment": "shTDO2-098"}, {"treatment": "shTDO2-054"}],
    [{"treatment": "shSCR"}],
    "Genetic preferred: three TDO2 hairpins (082/098/054) pooled vs shSCR. Inhibitor, EV and TDO2-OE arms "
    "not used.")
add("ST003490", ["AN005729", "AN005730"],
    [{"factor": "suspension", "treatment": "10 uM AT0174"}], [{"factor": "suspension", "treatment": "DMSO"}],
    "Pharmacological only (AT0174, dual TDO2/IDO1 inhibitor); highest dose 10 uM vs DMSO in suspension "
    "culture (the TDO2-dependent condition studied; attached arm not used).")
add("ST003499", ["AN005743"],
    [{"Phenotype": "Mild"}], [{"Phenotype": "Control"}],
    "ADSL-deficient patient LCLs (two patients: T450S/D332H and R309H/c.1191+5G>C) pooled vs one healthy "
    "control line. NOTE: replicates are cultures of single individuals (pseudo-replication).")
add("ST003544", ["AN005821"],
    [{"Genotype": "KO"}], [{"Genotype": "WT"}],
    "Only design: NDUFS4 KO vs WT iPSCs.")
add("ST003551", ["AN005837"],
    [{"Genotype": "CRISPR/Cas9 KO (sgALDH7A1)", "Treatment": "PBS"}],
    [{"Genotype": "CRISPR/Cas9 Control (sgTomato)", "Treatment": "PBS"}],
    "Genetic: sgALDH7A1 vs sgTomato, PBS (no N-methyl-arginine) to isolate the gene effect; largest arms "
    "(6 v 6).")
add("ST003556", ["AN005845"],
    [{"Genotype": "CRISPR/Cas9 KO (sgALDH7A1)", "Treatment": "PBS"}],
    [{"Genotype": "CRISPR/Cas9 Control (sgTomato)", "Treatment": "PBS"}],
    "Genetic (conditioned media): sgALDH7A1 vs sgTomato, PBS (no N-methyl-arginine); 'Genotype:None' "
    "media-only arms not used.")
add("ST004125", ["AN006838"],
    [{"Genotype": "GK"}], [{"Genotype": "GFP"}],
    "Only GK contrast: GK OVEREXPRESSION vs GFP in HEK293T (ChREBP-MLX). NOTE: gain-of-function, sign is "
    "opposite to a loss contrast. ('Genotype:Genotype' arm = GPD1 per sample names; not used.)")
add("ST004410", ["AN007374"],
    [{"Genotype": "WT", "Treatment": "Antimycin_myxothiazol"}],
    [{"Genotype": "WT_ScURA", "Treatment": "Antimycin_myxothiazol"}],
    "No direct DHODH loss arm. Under complex III inhibition (antimycin/myxothiazol), endogenous DHODH "
    "activity is blocked; WT cells (DHODH-blocked) vs ScURA cells (cytosolic DHODH bypass) isolates the "
    "DHODH-dependent effect at matched treatment. Indirect - review.")
add("ST004441", ["AN007437", "AN007438"],
    [{"Treatment": "100 nM OT82"}], [{"Treatment": "Untreated"}],
    "Pharmacological NAMPT inhibitor OT82, highest dose (100 nM) vs untreated, 24 h. HHT arm unrelated.")
add("ST004490", ["AN007532"],
    [{"Genotype": "TIGAR-knockout"}], [{"Genotype": "Wild-type"}],
    "Only design: TIGAR KO vs WT HIEC-6.")
add("ST004493", ["AN007535"],
    [{"Genotype": "TIGAR-knockout"}], [{"Genotype": "Wild-type"}],
    "Only design: TIGAR KO vs WT HT29 CL.16E.")
add("ST004825", ["AN008164"],
    [{"Genotype": "WT", "Treatment": "HMPSNE"}], [{"Genotype": "WT", "Treatment": "Control"}],
    "Pharmacological 3-MST inhibitor HMPSNE vs untreated in parental (WT) HCT116 (n=5 v 5). CSC (4 v 4) and "
    "Rescue arms not used.")
add("ST004985", ["AN008452"], None, None,
    "All samples are VLCAD-deficient fibroblasts; Control vs Treated = untreated vs DODA (anaplerotic) "
    "treatment. No VLCAD-normal comparator.",
    not_computable="no gene-loss contrast (both groups VLCADD; only a drug-treatment contrast)")
add("ST005017", ["AN008524"],
    [{"Treatment": "LXG6403"}], [{"Treatment": "Control"}],
    "Only design: LOX inhibitor LXG6403 vs control (pharmacological).")
add("ST005225", ["AN008930"],
    [{"Sample source": "KO-ALDH1B1", "Treatment": "DMSO"}], [{"Sample source": "KO-EGFP", "Treatment": "DMSO"}],
    "Genetic preferred: ALDH1B1 KO vs EGFP-KO control, both DMSO (IGUANA-5 inhibitor arms not used).")

# ---- D26: units added under the inclusion rule of 2026-11-23 -----------------------------------
add("ST003033", ["AN004973", "AN004974"],
    [{"Genotype": "OGDH KO", "Treatment": "control"}], [{"Genotype": "wild type", "Treatment": "control"}],
    "OGDH KO vs WT A375, both unstarved (glucose-starvation arms not used; 2 QC injections excluded).")
add("ST002234", ["AN003644"],
    [{"Genotype": "PRDX1 KO", "Treatment": "Untreated"}], [{"Genotype": "WT", "Treatment": "Untreated"}],
    "PRDX1 KO vs WT U2OS, untreated (etoposide / nocodazole arms not used).")
add("ST004104", ["AN006805", "AN006806"],
    [{"Genotype": "KO"}], [{"Genotype": "Wild-type"}],
    "Only design: CRISPR PTEN KO vs WT iPSC-derived mesendoderm.")
add("ST002876", ["AN004713_highNSD2"],
    [{"Genotype": "High NSD2", "Treatment": "Knockdown 1"}, {"Genotype": "High NSD2", "Treatment": "Knockdown 2"}],
    [{"Genotype": "High NSD2", "Treatment": "Control"}],
    "shNSD2 (two hairpins pooled) vs control shRNA in the NSD2-high (t(4;14)) myeloma line.")
add("ST002876", ["AN004713_lowNSD2"],
    [{"Genotype": "Low NSD2", "Treatment": "Knockdown 1"}, {"Genotype": "Low NSD2", "Treatment": "Knockdown 2"}],
    [{"Genotype": "Low NSD2", "Treatment": "Control"}],
    "As highNSD2, in the NSD2-low isogenic line; lines kept apart.")
add("ST003787", ["AN006223"],
    [{"G6PD_status": "KO_clone1", "Treatment": "Vehicle"}, {"G6PD_status": "KO_clone2", "Treatment": "Vehicle"}],
    [{"G6PD_status": "WT", "Treatment": "Vehicle"}],
    "G6PD KO (two CRISPR clones pooled) vs WT HCT-15, vehicle. AN006223 holds only the Method:Polar_GSH "
    "injections; AN006222 (NAD method) not used. Asparaginase arms not used.")
add("ST003347", ["AN005482"],
    [{"Genotype": "A3V/R337X"}, {"Genotype": "M26L/R396H"}, {"Genotype": "R309H/c.1191+5G-&gt;C"}],
    [{"Genotype": "Wildtype"}],
    "ADSL-deficient patient fibroblasts (4 patients) vs 3 control donors, 5 cultures each (pseudo-replication, "
    "as ST003499; R309H/c.1191+5G>C is also an ST003499 patient).")
add("ST003121", ["AN005115"],
    [{"Genotype": "ALG13"}], [{"Genotype": "CTR"}],
    "Only design: ALG13-CDG vs control iPSC brain organoids (day 77).")
add("ST003691", ["AN006057"],
    [{"Class": "-/-PMM1"}], [{"Class": "+/+PMM1"}],
    "PMM1 KO vs PMM1 intact, both in the same PMM2-CDG (R141H/F119L) fibroblast line, so the contrast "
    "isolates PMM1 (NMR).")
add("ST001441", ["AN002407", "AN002408"],
    [{"Sample Source": "Patient", "Treatment Duration": "Day0"}],
    [{"Sample Source": "Control", "Treatment Duration": "Day0"}],
    "SUCLA2 patient fibroblasts vs control fibroblasts, Day0 (proliferative; Day5 low-serum not used). "
    "Disease controls: the controls are patients with OTHER mitochondrial diseases, not healthy donors.")
add("ST002323", ["AN003791_24h", "AN003792_24h"],
    [{"Condition": "dTAG-13_24h"}], [{"Condition": "DMSO_24h"}],
    "dTAG-13 degradation of FKBP12F36V-SETD1A vs DMSO, 24 h (MOLM-13): targeted protein loss.")
add("ST002323", ["AN003791_48h", "AN003792_48h"],
    [{"Condition": "dTAG-13_48h"}], [{"Condition": "DMSO_48h"}],
    "As 24h, 48 h vs time-matched DMSO.")
add("ST003365", ["AN005514", "AN005515"],
    [{"Drug Treatment": "1000 nM GSK2194069", SAMPLE_KEY: r"BT474 cells "}],
    [{"Drug Treatment": "Control", SAMPLE_KEY: r"BT474 cells "}],
    "Pharmacological FASN inhibitor GSK2194069, highest dose (1000 nM) vs control, intracellular samples "
    "only (no Sample Type factor; 'cells' vs 'med' in the sample id). Fasnall arms not used (Fasnall also "
    "inhibits complex I, PMID 38766222).")
add("ST003411", ["AN005602", "AN005603"],
    [{"Sample Type": "Intracellular", "Drug Treatment": "100 uM C75"}],
    [{"Sample Type": "Intracellular", "Drug Treatment": "Control"}],
    "Pharmacological FASN inhibitor C75, highest dose (100 uM) vs control, intracellular. ST003367 is the "
    "same samples deposited again (same sample ids, medium arms mislabelled) and is not registered.")
add("ST003412", ["AN005604", "AN005605"],
    [{"Sample Type": "Intracellular", "Drug Treatment": "5 uM TVB-2640"}],
    [{"Sample Type": "Intracellular", "Drug Treatment": "Control", SAMPLE_KEY: r"TVB-2640 0 nM"}],
    "Pharmacological FASN inhibitor TVB-2640, highest dose (5 uM) vs its own vehicle control (sample ids "
    "'TVB-2640 0 nM cells'), intracellular. Cerulenin and TVB-3166 arms not used (as ST003417).")
add("ST003430", ["AN005633", "AN005634"],
    [{"Sample Type": "Intracellular", "Drug Treatment": "500 nM GSK214069"}],
    [{"Sample Type": "Intracellular", "Drug Treatment": "Control"}],
    "Pharmacological FASN inhibitor GSK2194069 (deposited as GSK214069), highest dose (500 nM) vs control, "
    "intracellular.")
add("ST002231", ["AN003640_R132H", "AN003641_R132H"],
    [{"Genotype": "HOG R132H", "Treatment": "DMSO"}], [{"Genotype": "HOG EV", "Treatment": "DMSO"}],
    "Genetic gain of function: IDH1-R132H overexpression vs empty vector, both DMSO. NOTE: neomorphic "
    "mutant, not loss.")
add("ST002231", ["AN003640_BAY", "AN003641_BAY"],
    [{"Genotype": "HOG EV", "Treatment": "BAY2402234"}], [{"Genotype": "HOG EV", "Treatment": "DMSO"}],
    "Pharmacological: selective DHODH inhibitor BAY2402234 (10 nM, 24 h) vs DMSO in empty-vector HOG cells.")

QC_PAT = re.compile(r"\b(qc|pool(ed)?|blank)\b", re.I)


def parse_label(label: str) -> dict:
    out = {}
    for part in str(label).split("|"):
        if ":" in part:
            k, v = part.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def select(labels: pd.Series, sel: list[dict]) -> list[str]:
    cols = []
    for col, lab in labels.items():
        if QC_PAT.search(str(lab)):
            continue
        f = parse_label(lab)
        if any(all(re.search(v, str(col)) if k == SAMPLE_KEY else f.get(k) == v for k, v in d.items())
               for d in sel):
            cols.append(col)
    return cols


def clean_name(n: str) -> str:
    n = str(n).split(";")[0]
    n = re.sub(r"\s*\(CAS#[^)]*\)", "", n)
    return re.sub(r"\s+", " ", n).strip()


KEGG_RE = re.compile(r"\bC\d{5}\b")
HMDB_RE = re.compile(r"\bHMDB\d{5,7}\b", re.I)


def mwtab_ids(study: str, analysis: str) -> dict:
    """metabolite_name -> (HMDB, KEGG) from the METABOLITES block of this analysis."""
    path = MWTAB / f"{study}.mwtab.txt"
    if not path.exists():
        return {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    # locate this analysis' block
    starts = [i for i, l in enumerate(lines) if l.startswith("#METABOLOMICS WORKBENCH")]
    block = lines
    for j, s in enumerate(starts):
        e = starts[j + 1] if j + 1 < len(starts) else len(lines)
        if f"ANALYSIS_ID:{analysis}" in lines[s]:
            block = lines[s:e]
            break
    try:
        i0 = next(i for i, l in enumerate(block) if l.startswith("METABOLITES_START"))
        i1 = next(i for i, l in enumerate(block) if l.startswith("METABOLITES_END") and i > i0)
    except StopIteration:
        return {}
    header = block[i0 + 1].split("\t")
    hl = [h.strip().lower() for h in header]
    kcols = [i for i, h in enumerate(hl) if "kegg" in h]
    hcols = [i for i, h in enumerate(hl) if "hmdb" in h]
    ocol = hl.index("other_id") if "other_id" in hl else None
    out = {}
    for l in block[i0 + 2:i1]:
        p = l.split("\t")
        name = p[0].strip()
        kegg = hmdb = ""
        for i in kcols:
            m = KEGG_RE.search(p[i]) if i < len(p) else None
            if m:
                kegg = m.group(0); break
        for i in hcols + ([ocol] if ocol is not None else []):
            m = HMDB_RE.search(p[i]) if i < len(p) else None
            if m:
                h = m.group(0).upper()
                hmdb = "HMDB" + h[4:].zfill(7)
                break
        out[name] = (hmdb, kegg)
    return out


def run_one(study, an, spec):
    analysis = an.split("_")[0]     # sub_unit AN..._<suffix>: another contrast from the same analysis
    raw = pd.read_csv(PER_SAMPLE / f"{study}_{analysis}.csv", dtype=str)
    meta_mask = raw.iloc[:, 0].astype(str).str.startswith("#")
    labels = raw.loc[raw.iloc[:, 0] == "#GROUP"].iloc[0, 1:]
    body = raw.loc[~meta_mask].copy()
    case = select(labels, spec["case"])
    ctrl = select(labels, spec["control"])
    # drop per-sample factors (unique per sample) so the recorded group label is the condition
    def cond(lab):
        f = parse_label(lab)
        return " | ".join(f"{k}:{v}" for k, v in f.items() if k not in ("Sample Group", "Injection order"))
    case_groups = sorted({cond(l) for l in labels[case]})
    ctrl_groups = sorted({cond(l) for l in labels[ctrl]})
    if not case or not ctrl:
        raise ValueError(f"{study} {an}: empty group (case={len(case)}, control={len(ctrl)})")

    ids = mwtab_ids(study, analysis)
    vals = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    vals.index = body.index
    df = pd.DataFrame({"raw": body.iloc[:, 0].astype(str)})
    df["name"] = df["raw"].map(clean_name)
    df["HMDB"] = df["raw"].map(lambda r: ids.get(r.strip(), ("", ""))[0])
    df["KEGG"] = df["raw"].map(lambda r: ids.get(r.strip(), ("", ""))[1])
    df["key"] = df["name"].str.lower()
    pos = vals.where(vals > 0)
    df["median"] = pos.median(axis=1)
    # drop internal standards / unnamed rows
    df = df[~df["name"].str.fullmatch(r"(?i)istd|internal standard.*|")]
    n_rows = len(df)
    # merge duplicates: keep highest-median row, fill ids from siblings if missing
    keep = []
    for _, g in df.groupby("key", sort=False):
        best = g["median"].fillna(-np.inf).idxmax()
        row = g.loc[best].copy()
        for c in ("HMDB", "KEGG"):
            if not row[c]:
                nz = [x for x in g[c] if x]
                row[c] = nz[0] if nz else ""
        keep.append(row)
    dd = pd.DataFrame(keep)
    n_dup = n_rows - len(dd)

    x = np.log2(pos.loc[dd.index])
    a = x[case]; b = x[ctrl]
    na = a.notna().sum(axis=1); nb = b.notna().sum(axis=1)
    ok = (na >= 2) & (nb >= 2)
    t, p = stats.ttest_ind(a[ok], b[ok], axis=1, equal_var=False, nan_policy="omit")
    res = pd.DataFrame({
        "name": dd.loc[ok, "name"].values,
        "HMDB": dd.loc[ok, "HMDB"].values,
        "KEGG": dd.loc[ok, "KEGG"].values,
        "log2FC": (a[ok].mean(axis=1) - b[ok].mean(axis=1)).values,
        "p": np.asarray(p, dtype=float),
        "n_case": na[ok].astype(int).values,
        "n_control": nb[ok].astype(int).values,
    })
    res = res[np.isfinite(res["p"])]  # zero-variance in both groups -> nan p
    res = res.sort_values("p").reset_index(drop=True)
    res.to_csv(OUT / f"{study}_{an}.csv", index=False)
    return dict(case_group=" + ".join(case_groups), control_group=" + ".join(ctrl_groups),
                n_case=len(case), n_control=len(ctrl),
                n_metabolites_tested=len(res), n_p_lt_0_05=int((res["p"] < P_CUT).sum()),
                n_dup_merged=n_dup, n_dropped_lt2=int((~ok).sum()),
                n_kegg=int((res["KEGG"] != "").sum()), n_hmdb=int((res["HMDB"] != "").sum()))


def main():
    reg = pd.read_csv(REGISTRY, sep="\t", dtype=str)
    act = reg[(reg["collection"] == "C_workbench") & (reg["status"] == "active")]
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in act.iterrows():
        study, an, gene = r["study"], r["sub_unit"], r["causal_genes"]
        spec = C.get((study, an))
        base = dict(study=study, analysis=an, gene=gene)
        if spec is None:
            rows.append({**base, "case_group": "", "control_group": "", "n_case": 0, "n_control": 0,
                         "reason": "no contrast defined", "n_metabolites_tested": 0, "n_p_lt_0.05": 0,
                         "status": "not computable: no contrast defined"})
            continue
        if spec["not_computable"]:
            rows.append({**base, "case_group": "", "control_group": "", "n_case": 0, "n_control": 0,
                         "reason": spec["reason"], "n_metabolites_tested": 0, "n_p_lt_0.05": 0,
                         "status": f"not computable: {spec['not_computable']}"})
            continue
        out = run_one(study, an, spec)
        print(f"{study} {an}: case n={out['n_case']} ctrl n={out['n_control']} tested={out['n_metabolites_tested']} "
              f"sig={out['n_p_lt_0_05']} dup_merged={out['n_dup_merged']} dropped<2={out['n_dropped_lt2']} "
              f"KEGG={out['n_kegg']} HMDB={out['n_hmdb']}")
        rows.append({**base, "case_group": out["case_group"], "control_group": out["control_group"],
                     "n_case": out["n_case"], "n_control": out["n_control"], "reason": spec["reason"],
                     "n_metabolites_tested": out["n_metabolites_tested"],
                     "n_p_lt_0.05": out["n_p_lt_0_05"], "status": "ok"})
    pd.DataFrame(rows).to_csv(OUT / "contrasts.tsv", sep="\t", index=False)
    print(f"wrote {len(rows)} rows to contrasts.tsv")


if __name__ == "__main__":
    main()
