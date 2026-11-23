"""Write the per-sample tables that benchmarks/workbench_daa.py reads.

For each (study, analysis) this script takes the study's mwTab from the
Metabolomics Workbench REST API (rest/study/study_id/<ST>/mwtab/txt, saved
unchanged as <ST>.mwtab.txt; an existing copy is reused), cuts out the
MS_METABOLITE_DATA (or NMR_METABOLITE_DATA / NMR_BINNED_DATA) block of that
analysis and writes it as <ST>_<AN>.csv: a header row "metabolite,<sample ids>",
a "#GROUP" row with each sample's factor string, then one row per metabolite
(values parsed as numbers). This reproduces the per-sample tables of the
original units exactly.

    python benchmarks/workbench_per_sample.py ST003033:AN004974 ST002234:AN003644 ...
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workbench_daa import MWTAB, PER_SAMPLE  # noqa: E402

REST = "https://www.metabolomicsworkbench.org/rest/study/study_id/{}/mwtab/txt"
BLOCKS = ("MS_METABOLITE_DATA", "NMR_METABOLITE_DATA", "NMR_BINNED_DATA")


def fetch_mwtab(study: str) -> Path:
    path = MWTAB / f"{study}.mwtab.txt"
    if not path.exists():
        # curl rather than urllib: it uses the Windows certificate store (proxy root CA)
        subprocess.run(["curl", "-sf", "-o", str(path), REST.format(study)], check=True)
    return path


def decode(line: bytes) -> str:
    """Some deposits mix UTF-8 and Latin-1 lines."""
    try:
        return line.decode("utf-8")
    except UnicodeDecodeError:
        return line.decode("latin-1")


def data_block(study: str, analysis: str) -> list[list[str]]:
    lines = [decode(b) for b in fetch_mwtab(study).read_bytes().splitlines()]
    starts = [i for i, l in enumerate(lines) if l.startswith("#METABOLOMICS WORKBENCH")]
    for j, s in enumerate(starts):
        if f"ANALYSIS_ID:{analysis}" not in lines[s]:
            continue
        block = lines[s:starts[j + 1] if j + 1 < len(starts) else len(lines)]
        for name in BLOCKS:
            try:
                i0 = next(i for i, l in enumerate(block) if l.startswith(f"{name}_START"))
            except StopIteration:
                continue
            i1 = next(i for i, l in enumerate(block) if l.startswith(f"{name}_END") and i > i0)
            return [l.split("\t") for l in block[i0 + 1:i1]]
    raise ValueError(f"{study} {analysis}: no data block in the mwTab")


def per_sample(study: str, analysis: str) -> pd.DataFrame:
    rows = data_block(study, analysis)
    samples, factors = rows[0], rows[1]
    assert samples[0] == "Samples" and factors[0] == "Factors", (study, analysis)
    n = len(samples)
    body = pd.DataFrame([(r + [""] * n)[:n] for r in rows[2:] if "".join(r).strip()],
                        columns=["metabolite"] + samples[1:])
    for c in body.columns[1:]:
        body[c] = pd.to_numeric(body[c].str.strip().replace("", None), errors="coerce")
    group = pd.DataFrame([["#GROUP"] + [f.strip() for f in factors[1:n]]], columns=body.columns)
    return pd.concat([group, body], ignore_index=True)


def main(argv):
    for item in argv:
        study, analysis = item.split(":")
        df = per_sample(study, analysis)
        out = PER_SAMPLE / f"{study}_{analysis}.csv"
        df.to_csv(out, index=False, lineterminator="\n", encoding="utf-8")
        print(f"{out.name}: {df.shape[1] - 1} samples, {len(df) - 1} metabolites")


if __name__ == "__main__":
    main(sys.argv[1:])
