"""Blind predictions on the lab's collected published datasets.

Reads the standardised master table (sheet `metabolomics_master_standardize`
of published_test_predictions.xlsx), splits it into analysis units, runs the
frozen Metaboenrich command line on each, and writes the predictions:

    predictions/<run>/units.tsv            one row per unit: settings, mapping, top genes/pathways
    predictions/<run>/<unit>/              genes.tsv, pathways.tsv, transporters.tsv (top rows), summary.json
    predictions/<run>/PREDICTIONS.md       readable summary

The answer key (each paper's conclusions, and the sheet's "Expected tool
prediction" column) is NOT read here. Commit the predictions before
building or opening the answer key.

Units and significance rules (fixed before any results were seen):
  log2FC data   one unit per (study, comparison, matrix); significant if p < 0.05
  z-score data  one unit per (study, comparison, matrix, disease group);
                several subjects: mean z per metabolite, z-test on the mean
                (values are standardised), BH q < 0.05 and |mean z| >= 1;
                one subject: |z| >= 2
  Alaimo 2020   one unit per patient (each has a different disorder), |z| >= 2

    python benchmarks/published_predictions.py --out predictions/published_v1
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaboenrich import __version__                       # noqa: E402
from metaboenrich.__main__ import main as metaboenrich    # noqa: E402

MASTER = Path.home() / "Desktop/Electrical Conductivity/Published Test Data/published_test_predictions.xlsx"
ID_TRANSLATION = Path.home() / "Documents/Github/Metaboenrich/src/Metaboenrich/data/id_translation.csv"
MIN_METABOLITES = 3


def slug(*parts):
    s = "_".join(str(p) for p in parts if p and str(p) != "-")
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")[:90]


def normal_p(z):
    return np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])


def bh(p):
    p = np.asarray(p, float)
    if not len(p):
        return p
    order = np.argsort(p)
    q = np.empty(len(p))
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def read_tsv(path):
    """A results table, or an empty frame if the file is missing or empty."""
    try:
        return pd.read_csv(path, sep="\t")
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()


def clean_p(x):
    """Numeric p; 'NR (IPA filter p<0.05 only ...)' means 'passed p < 0.05' (Adam 2013)
    -> 0.049; '<0.001' -> 0.001; anything else non-numeric -> NaN."""
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    s = str(x).strip()
    if "p<0.05" in s.replace(" ", ""):
        return 0.049
    m = re.fullmatch(r"<\s*([0-9.eE-]+)", s)
    return float(m.group(1)) if m else np.nan


def units(master):
    m = master.copy()
    m["p_value"] = m["p_value"].map(clean_p)
    for c in ("log2FC", "Z_score"):
        m[c] = pd.to_numeric(m[c], errors="coerce")        # 'ND', stray header text -> NaN
    for c in ("Disease_Group", "Comparison", "Matrix", "Subject_ID", "Gene"):
        m[c] = m[c].fillna("-").astype(str)
    out = []
    for src, d in m.groupby("Source"):
        if src.startswith("Alaimo"):
            keys = ["Comparison", "Matrix", "Subject_ID"]
        elif d["log2FC"].notna().any():
            keys = ["Comparison", "Matrix"]
        else:
            keys = ["Comparison", "Matrix", "Disease_Group"]
        for k, g in d.groupby(keys):
            out.append((src, dict(zip(keys, k)), g))
    return out


def unit_daa(g):
    """DAA table + significance settings for one unit."""
    ids = g.groupby("Metabolite").agg(HMDB=("HMDB_ID", "first"), KEGG=("KEGG_ID", "first"))
    if g["log2FC"].notna().any():
        v = g.groupby("Metabolite").agg(effect=("log2FC", "mean"), p=("p_value", "min"))
        rule = "p < 0.05"
        return ids.join(v).reset_index(), 0.05, 0.0, rule
    z = g.pivot_table(index="Metabolite", columns="Subject_ID", values="Z_score", aggfunc="mean")
    n_subj = z.shape[1]
    if n_subj == 1:
        v = pd.DataFrame({"effect": z.iloc[:, 0], "p": normal_p(z.iloc[:, 0].values)}, index=z.index)
        return ids.join(v, how="inner").reset_index(), 0.0456, 0.0, "|z| >= 2 (one subject)"
    k = z.notna().sum(axis=1)
    mean = z.mean(axis=1)
    p = normal_p((mean * np.sqrt(k)).values)
    v = pd.DataFrame({"effect": mean, "p": bh(p)}, index=z.index)
    return ids.join(v, how="inner").reset_index(), 0.05, 1.0, f"mean z over {n_subj} subjects: BH q < 0.05 and |mean z| >= 1"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master", default=str(MASTER))
    ap.add_argument("--id-translation", default=str(ID_TRANSLATION))
    ap.add_argument("--out", default="predictions/published_v1")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    master = pd.read_excel(args.master, sheet_name="metabolomics_master_standardize")

    rows = []
    for n_unit, (src, key, g) in enumerate(units(master), start=1):
        # numbered so units stay distinct even when long names would collide
        subject = key.get("Subject_ID", "")
        uid = f"{n_unit:03d}_" + slug(src.split(" et al")[0] + src[-4:], subject, *[v for k, v in key.items() if k != "Subject_ID"])[:80]
        genes_given = sorted({x for x in g["Gene"].dropna().astype(str) if x not in ("-", "nan")})
        daa, alpha, min_fc, rule = unit_daa(g)
        daa = daa.dropna(subset=["effect", "p"])
        row = {"unit": uid, "source": src, **key, "species": "|".join(sorted(g.Species.dropna().astype(str).unique())),
               "diagnosed_gene_in_data": ";".join(genes_given), "metabolites": len(daa), "rule": rule}
        if len(daa) < MIN_METABOLITES:
            rows.append({**row, "status": f"skipped: fewer than {MIN_METABOLITES} metabolites"})
            continue
        udir = out / uid
        udir.mkdir(exist_ok=True)
        csv = udir / "_input.csv"
        daa.rename(columns={"Metabolite": "metabolite"}).to_csv(csv, index=False)
        try:
            metaboenrich(["--daa", str(csv), "--out", str(udir), "--id-col", "HMDB", "KEGG", "metabolite",
                          "--fc-col", "effect", "--p-col", "p", "--alpha", str(alpha),
                          "--min-log2fc", str(min_fc), "--id-translation", args.id_translation])
        except SystemExit as err:
            rows.append({**row, "status": f"error: {err}"})
            continue
        finally:
            csv.unlink(missing_ok=True)
        s = json.loads((udir / "summary.json").read_text())
        mapping = read_tsv(udir / "daa_mapping.tsv")
        genes, paths, trans = (read_tsv(udir / f) for f in ("genes.tsv", "pathways.tsv", "transporters.tsv"))
        rank = ""
        if genes_given and len(genes):
            syms = list(genes.symbol.fillna(""))
            hits = [syms.index(gn) + 1 for gn in genes_given if gn in syms]
            rank = min(hits) if hits else "not scored"
        rows.append({**row, "status": "ok", "mapped": len(mapping), "sources": s["sources"], "targets": s["targets"],
                     "one_sided": s["one_sided"], "pairs": s["pairs_retained"],
                     "rank_of_diagnosed_gene": rank,
                     "top_genes": ", ".join(genes.symbol.fillna(genes.ensembl_id).head(15)) if len(genes) else "",
                     "top_pathways": "; ".join(f"{r.subsystem} ({r.in_top}/{r.expected})" for r in paths.head(6).itertuples()) if len(paths) else "",
                     "top_transporters": ", ".join(trans.symbol.fillna("").head(10)) if len(trans) else ""})
        # keep the prediction outputs small: top rows only; drop bulky files
        for f, n in (("genes.tsv", 50), ("pathways.tsv", 30), ("transporters.tsv", 30), ("reactions.tsv", 50)):
            t = read_tsv(udir / f)
            if len(t):
                t.head(n).to_csv(udir / f, sep="\t", index=False)
        for f in ("currency_metabolites.tsv", "currency_edges.tsv", "intermediate_metabolites.tsv"):
            (udir / f).unlink(missing_ok=True)
        for f in (udir / "cytoscape").glob("full_network.graphml"):
            f.unlink()

    table = pd.DataFrame(rows)
    table.to_csv(out / "units.tsv", sep="\t", index=False)
    lines = [f"# Blind predictions: published datasets ({out.name})", "",
             f"Metaboenrich {__version__}; raw scoring; one-sided mode auto; id_translation name lookup.",
             "Answer key not consulted. Rules per unit are in `units.tsv`.", ""]
    for r in table.itertuples(index=False):
        lines += [f"## {r.unit}", f"- Source: {r.source}; status: {r.status}; rule: {r.rule}"]
        if r.status == "ok":
            lines += [f"- Mapped {r.mapped}/{r.metabolites} metabolites; {r.sources} up, {r.targets} down"
                      f"{' (one-sided)' if r.one_sided else ''}; {r.pairs} pairs",
                      f"- Diagnosed gene in data: {r.diagnosed_gene_in_data or '-'}; its rank: {r.rank_of_diagnosed_gene or '-'}",
                      f"- Top genes: {r.top_genes}", f"- Top pathways (in top 100 / expected): {r.top_pathways}",
                      f"- Top transporters: {r.top_transporters}"]
        lines.append("")
    (out / "PREDICTIONS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{(table.status == 'ok').sum()} units run, {(table.status != 'ok').sum()} skipped/errored -> {out}/")


if __name__ == "__main__":
    main()
