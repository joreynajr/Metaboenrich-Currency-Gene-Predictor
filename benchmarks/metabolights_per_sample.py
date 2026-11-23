"""Write the per-sample tables that benchmarks/metabolights_daa.py reads (D26).

For each (study, assay) in ASSAYS this script downloads the study's sample sheet
(s_*.txt), the assay sheet and the MAF (m_*_maf.tsv) from the MetaboLights FTP
(an existing copy under Metabolights Benchmark Data/<study>/ is reused) and writes
<study>/<study>_<assay>_per_sample.csv in the format of the MTBLS3628 tables:

    row 0   metabolite, CHEBI, <sample columns as in the MAF>
    row 1   #GROUP, (empty), <group label of each sample>
    then    one row per metabolite: metabolite_identification, its CHEBI id, values

The group label is the sample's Factor Value[...] columns, replicate factors
dropped, written "Factor:Value | Factor:Value" (the bare value when one factor is
left). MAF columns are matched to samples by Sample Name or MS Assay Name.
Dropped: QC / blank / pool / standard samples (by name or by an empty label),
rows without a metabolite name (or a bare number), rows without any value in the
kept samples. Values are parsed as numbers (text in a value cell -> missing).

    python benchmarks/metabolights_per_sample.py
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pandas as pd

DATA = Path.home() / "Desktop/Electrical Conductivity/Metabolights Benchmark Data"
FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{}/{}"
REPLICATE = re.compile(r"replicate", re.I)
NOT_SAMPLE = re.compile(r"(^|[^a-z])(qc|pool(ed)?|blank|std|standard)([^a-z]|$)", re.I)
CHEBI = re.compile(r"^CHEBI:\d+$")

# (study, assay) -> (sample sheet, assay sheet, MAF)
ASSAYS = {
    ("MTBLS4722", "LCMS"): ("s_MTBLS4722.txt", "a_MTBLS4722_LC-MS___metabolite_profiling.txt",
                            "m_MTBLS4722_LC-MS___metabolite_profiling_v2_maf.tsv"),
    ("MTBLS4223", "HILIC_neg"): ("s_MTBLS4223.txt", "a_MTBLS4223_LC-MS_negative_hilic_metabolite_profiling.txt",
                                 "m_MTBLS4223_LC-MS_negative_hilic_metabolite_profiling_v2_maf.tsv"),
    ("MTBLS7840", "RP_pos"): ("s_MTBLS7840.txt", "a_MTBLS7840_LC-MS_positive_reverse-phase_metabolite_profiling.txt",
                              "m_MTBLS7840_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv"),
    ("MTBLS7840", "RP_neg"): ("s_MTBLS7840.txt", "a_MTBLS7840_LC-MS_negative_reverse-phase_metabolite_profiling.txt",
                              "m_MTBLS7840_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv"),
    ("MTBLS11858", "RP_pos"): ("s_MTBLS11858.txt",
                               "a_MTBLS11858_LC-MS_positive_reverse-phase_metabolite_profiling.txt",
                               "m_MTBLS11858_LC-MS_positive_reverse-phase_metabolite_profiling_v2_maf.tsv"),
    ("MTBLS11858", "RP_neg"): ("s_MTBLS11858.txt",
                               "a_MTBLS11858_LC-MS_negative_reverse-phase_metabolite_profiling.txt",
                               "m_MTBLS11858_LC-MS_negative_reverse-phase_metabolite_profiling_v2_maf.tsv"),
    ("MTBLS405", "Cell_HILIC"): ("s_MTBLS405.txt", "a_MTBLS405_Cell-HILIC_mass_spectrometry.txt",
                                 "m_MTBLS405_Cell-HILIC_mass_spectrometry_v2_maf.tsv"),
    ("MTBLS405", "Cell_RP"): ("s_MTBLS405.txt", "a_MTBLS405_Cell-RP_mass_spectrometry.txt",
                              "m_MTBLS405_Cell-RP_mass_spectrometry_v2_maf.tsv"),
    ("MTBLS10469", "HILIC"): ("s_MTBLS10469.txt", "a_MTBLS10469_LC-MS_alternating_hilic.txt",
                              "m_MTBLS10469_LC-MS_alternating_hilic_v2_maf.tsv"),
    ("MTBLS10469", "RP"): ("s_MTBLS10469.txt", "a_MTBLS10469_LC-MS_alternating_reverse-phase-1.txt",
                           "m_MTBLS10469_LC-MS_alternating_reverse-phase-1_v2_maf.tsv"),
}


def fetch(study: str, name: str) -> Path:
    path = DATA / study / name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        # curl rather than urllib: it uses the Windows certificate store (proxy root CA)
        subprocess.run(["curl", "-sf", "-o", str(path), FTP.format(study, name)], check=True)
    return path


def per_sample(s_file: Path, a_file: Path, maf: Path) -> pd.DataFrame:
    s = pd.read_csv(s_file, sep="\t", dtype=str).fillna("")
    a = pd.read_csv(a_file, sep="\t", dtype=str).fillna("")
    m = pd.read_csv(maf, sep="\t", dtype=str).fillna("")
    factors = [c for c in s.columns if c.startswith("Factor Value[") and not REPLICATE.search(c)]
    s = s.drop_duplicates("Sample Name").set_index("Sample Name")

    def label(sample):
        vals = [(c[len("Factor Value["):-1], s.at[sample, c].strip()) for c in factors]
        if all(v == "" for _, v in vals):
            return ""
        return vals[0][1] if len(vals) == 1 else " | ".join(f"{k}:{v}" for k, v in vals)

    to_sample = {}
    for col in ("Sample Name", "MS Assay Name", "NMR Assay Name"):
        if col in a.columns:
            to_sample.update({x: y for x, y in zip(a[col], a["Sample Name"]) if x})
    first = list(m.columns).index("smallmolecule_abundance_std_error_sub") + 1
    cols, groups = [], []
    for c in m.columns[first:]:
        sample = to_sample.get(c)
        if sample is None or sample not in s.index:
            continue                                  # annotation column, not a sample
        g = label(sample)
        if not g or NOT_SAMPLE.search(c) or NOT_SAMPLE.search(sample) or NOT_SAMPLE.search(g):
            continue
        cols.append(c)
        groups.append(g)
    name = m["metabolite_identification"].str.strip()
    vals = m[cols].apply(lambda x: pd.to_numeric(x, errors="coerce"))
    keep = (name != "") & ~name.str.fullmatch(r"\d+") & vals.notna().any(axis=1)
    body = pd.DataFrame({"metabolite": name[keep],
                         "CHEBI": m.loc[keep, "database_identifier"].str.strip().where(
                             m.loc[keep, "database_identifier"].str.strip().str.match(CHEBI), "")})
    body = pd.concat([body, vals.loc[keep]], axis=1)
    head = pd.DataFrame([["#GROUP", ""] + groups], columns=body.columns)
    return pd.concat([head, body], ignore_index=True)


def main():
    for (study, assay), (s_name, a_name, m_name) in ASSAYS.items():
        df = per_sample(fetch(study, s_name), fetch(study, a_name), fetch(study, m_name))
        out = DATA / study / f"{study}_{assay}_per_sample.csv"
        df.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
        print(f"{out.name}: {df.shape[1] - 2} samples, {len(df) - 1} metabolites, groups "
              f"{df.iloc[0, 2:].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
