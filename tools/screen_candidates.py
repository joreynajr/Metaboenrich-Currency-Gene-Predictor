"""Screen repository candidates from tools/find_test_data.py for usable evaluation data.

For each Workbench or MetaboLights candidate with a network gene, a loss-of-function
perturbation, no tracing or lipid-only flag and no registry row, check what the D22
rules need before a unit can be registered:
  - named metabolites with per-sample values, per assay (Workbench: the study's
    named-metabolite list per analysis; MetaboLights: rows and filled value
    columns of each MAF file)
  - sample groups from the deposited factors, and a proposed case-vs-control
    contrast: case = groups naming the gene or a loss word (KO, deficient, patient,
    siRNA ...), control = groups naming wild type, control, vehicle, scramble ...,
    restricted to the untreated or vehicle arm when another factor varies
  - at least 3 samples per group and 20 named metabolites in the best assay
The verdict is a proposal: "usable" rows still need a person to confirm the contrast
before they go into benchmarks/data_registry.tsv and the DAA scripts.

    python tools/screen_candidates.py

Reads results/data_discovery/candidates.tsv; writes results/data_discovery/screen.tsv
and screen.md. Uses the same cache as find_test_data.py.
"""
import argparse
import io
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_test_data import FAILED, ML, MW, OUT, fetch, pmap, trust_windows_roots   # noqa: E402

ML_FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
MIN_PER_GROUP, MIN_METABOLITES = 3, 20
CASE = r"\bKO\b|knock-?out|knock-?down|\bKD\b|deficien|patient|mutant|mutation|\bnull\b|-/-|crispr|\bsg|\bsh|\bsi|" \
       r"loss|deleted|deletion|disease|CDG|affected"
CTRL = r"\bWT\b|wild[- ]?type|control|\bctrl\b|vehicle|scrambl|\bscr\b|non[- ]?target|\bNT\b|healthy|normal|parental|" \
       r"empty vector|\bEV\b|sgGFP|shGFP|siGFP|\bGFP\b|mock|untreated|unaffected|reference"
NEUTRAL = r"vehicle|untreated|DMSO|\bnone\b|\bno\b|baseline|control|basal|\b0\b"


def factor_dict(s):
    """'Genotype:KO | Treatment:Vehicle' -> {'Genotype': 'KO', 'Treatment': 'Vehicle'}."""
    out = {}
    for part in str(s).split("|"):
        k, _, v = part.partition(":")
        out[k.strip()] = v.strip() if _ else k.strip()
    return out


def propose(groups, gene):
    """groups: {factor string: n}. Returns case groups, control groups, n_case, n_control, note."""
    rows = [(g, n, factor_dict(g)) for g, n in groups.items() if g]
    if not rows:
        return [], [], 0, 0, "no sample factors"
    keys = sorted({k for _, _, d in rows for k in d})
    varying = [k for k in keys if len({d.get(k, "") for _, _, d in rows}) > 1]
    # the factor that carries the gene perturbation
    gk = next((k for k in varying if any(re.search(CASE, d.get(k, ""), re.I) or gene.lower() in d.get(k, "").lower()
                                         for _, _, d in rows) and any(re.search(CTRL, d.get(k, ""), re.I)
                                                                      for _, _, d in rows)), None)
    if gk is None:
        return [], [], 0, 0, "no factor with both a perturbed and a control level"
    others = [k for k in varying if k != gk]

    def neutral(d):                                  # untreated / vehicle arm of every other varying factor
        return all(re.search(NEUTRAL, d.get(k, ""), re.I) for k in others)

    keep = [r for r in rows if neutral(r[2])] if others else rows
    note = f"contrast on '{gk}'" + (f"; other factors held at untreated/vehicle: {', '.join(others)}" if others else "")
    if others and not keep:
        keep, note = rows, note + " (no untreated arm found: all arms pooled, check)"
    case = [g for g, n, d in keep if (re.search(CASE, d[gk], re.I) or gene.lower() in d[gk].lower())
            and not re.search(CTRL, d[gk], re.I)]
    ctrl = [g for g, n, d in keep if re.search(CTRL, d[gk], re.I)]
    n_case = sum(groups[g] for g in case)
    n_ctrl = sum(groups[g] for g in ctrl)
    return case, ctrl, n_case, n_ctrl, note


def workbench(sid, gene, cache):
    fac = fetch(f"{MW}/{sid}/factors", cache) or {}
    if isinstance(fac, dict) and "study_id" in fac:
        fac = {"1": fac}
    groups = Counter(v.get("factors", "") for v in (fac.values() if isinstance(fac, dict) else []))
    mets = fetch(f"{MW}/{sid}/metabolites", cache) or {}
    if isinstance(mets, dict) and "analysis_id" in mets:
        mets = {"1": mets}
    per = Counter(v.get("analysis_id", "") for v in (mets.values() if isinstance(mets, dict) else []))
    assays = ", ".join(f"{a}:{n}" for a, n in per.most_common())
    return groups, max(per.values(), default=0), assays


def metabolights(sid, gene, cache):
    listing = fetch(f"{ML_FTP}/{sid}/", cache, kind="txt") or ""
    files = sorted(set(re.findall(r'href="([^"?/]+)"', listing)))
    groups = Counter()
    s_file = next((f for f in files if f.startswith("s_")), None)
    if s_file:
        s = fetch(f"{ML_FTP}/{sid}/{s_file}", cache, kind="txt") or ""
        try:
            t = pd.read_csv(io.StringIO(s), sep="\t", dtype=str).fillna("")
            fcols = [c for c in t.columns if c.startswith("Factor Value[")]
            for _, r in t.iterrows():
                groups[" | ".join(f"{c[13:-1]}:{r[c]}" for c in fcols)] += 1
        except Exception:
            pass
    best, assays = 0, []
    for m in (f for f in files if f.startswith("m_") and f.endswith(".tsv")):
        txt = fetch(f"{ML_FTP}/{sid}/{m}", cache, kind="txt", max_bytes=3_000_000) or ""
        try:
            t = pd.read_csv(io.StringIO(txt), sep="\t", dtype=str)
        except Exception:
            continue
        named = t[t.get("metabolite_identification", pd.Series(dtype=str)).fillna("").str.strip() != ""]
        std = {"database_identifier", "chemical_formula", "smiles", "inchi", "metabolite_identification", "mass_to_charge",
               "fragmentation", "modifications", "charge", "retention_time", "taxid", "species", "database", "database_version",
               "reliability", "uri", "search_engine", "search_engine_score", "smallmolecule_abundance_sub",
               "smallmolecule_abundance_stdev_sub", "smallmolecule_abundance_std_error_sub", "chemical_shift", "multiplicity"}
        vals = [c for c in t.columns if c not in std]
        filled = named[vals].apply(pd.to_numeric, errors="coerce").notna().sum(axis=1).gt(0).sum() if vals else 0
        assays.append(f"{m[:60]}:{filled}")
        best = max(best, int(filled))
    return groups, best, "; ".join(assays)


def screen(row, cache):
    sid, gene = row.id, row.gene
    groups, n_met, assays = (workbench if row.source == "workbench" else metabolights)(sid, gene, cache)
    case, ctrl, nc, nk, note = propose(groups, gene)
    why = []
    if n_met < MIN_METABOLITES:
        why.append(f"best assay has {n_met} named metabolites with values (< {MIN_METABOLITES})")
    if not case or not ctrl:
        why.append(note)
    elif min(groups[g] for g in case + ctrl) < MIN_PER_GROUP and min(nc, nk) < MIN_PER_GROUP:
        why.append(f"fewer than {MIN_PER_GROUP} samples per group")
    verdict = "usable" if not why else ("check" if n_met >= MIN_METABOLITES and groups else "not usable")
    return {"verdict": verdict, "id": sid, "gene": gene, "source": row.source, "score": row.score,
            "best_assay_named_metabolites": n_met, "n_case": nc, "n_control": nk,
            "case_groups": " + ".join(case), "control_groups": " + ".join(ctrl), "contrast_note": note,
            "reason": "; ".join(why), "n_groups": len(groups),
            "groups": " || ".join(f"{g} ({n})" for g, n in groups.most_common(12)), "assays": assays,
            "title": row.title, "url": row.url}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    out = Path(args.out); cache = out / "cache"
    trust_windows_roots(cache)
    c = pd.read_csv(out / "candidates.tsv", sep="\t", dtype=str).fillna("")
    c = c[(c.source != "literature") & (c.in_registry == "") & (c.in_network == "True") & (c.perturbation == "loss")
          & (c.design_flags == "")].drop_duplicates(["id", "gene"])
    print(f"screening {len(c)} study x gene rows ({c.id.nunique()} studies)", flush=True)
    res = pd.DataFrame(pmap(lambda r: screen(r, cache), list(c.itertuples()), workers=6))
    order = {"usable": 0, "check": 1, "not usable": 2}
    res = res.sort_values(["verdict", "best_assay_named_metabolites"], key=lambda s: s.map(order) if s.name == "verdict"
                          else -pd.to_numeric(s), ascending=True)
    res.to_csv(out / "screen.tsv", sep="\t", index=False)
    L = ["# Screened repository candidates", "",
         f"{len(res)} study x gene rows from {res.id.nunique()} studies: " +
         ", ".join(f"{v} {n}" for v, n in res.verdict.value_counts().items()) + ".",
         f"Usable = >= {MIN_METABOLITES} named metabolites with values in one assay, a genotype factor with a perturbed "
         f"and a control level, >= {MIN_PER_GROUP} samples per side. Proposed contrasts need a person's check.", "",
         "| verdict | id | gene | metabolites | case (n) | control (n) | note / reason |", "|---|---|---|---|---|---|---|"]
    for r in res[res.verdict != "not usable"].itertuples():
        L.append(f"| {r.verdict} | [{r.id}]({r.url}) | {r.gene} | {r.best_assay_named_metabolites} | "
                 f"{r.case_groups[:70]} ({r.n_case}) | {r.control_groups[:70]} ({r.n_control}) | "
                 f"{(r.reason or r.contrast_note)[:110]} |")
    L += ["", "Not usable:", ""] + [f"- {r.id} {r.gene}: {r.reason}" for r in res[res.verdict == "not usable"].itertuples()]
    (out / "screen.md").write_text("\n".join(L), encoding="utf-8")
    print(res.verdict.value_counts().to_string(), f"\nfailed fetches: {len(FAILED)}")


if __name__ == "__main__":
    main()
