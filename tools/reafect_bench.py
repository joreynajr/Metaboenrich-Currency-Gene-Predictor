"""Head-to-head with Reafect (Bongaerts et al. 2022, Mol Genet Metab 136:199,
doi 10.1016/j.ymgme.2022.05.002) on its two z-score sets: 72 Rotterdam IEM
patients (Bongaerts) and 106 Miller 2015 patients as Reafect prepared them.

Reafect is "all rights reserved", so it is never copied here: it is run from
its own clone (git clone https://github.com/mbongaerts/Reafect) by the
`reafect-*` steps below, under a Python that has its dependencies
(pandas < 2 matters: its groupby.apply relies on the 1.x index behaviour).
The steps follow its reproduction notebook: the notebook cells holding the
KEGG module/pathway lists and custom reactions are executed verbatim, with
a = 0.85, b = 0.35, c = 0.75 and the Bongaerts z-score truncation.

Our side runs on the three_rule:three_rule network with the benchmark's own
rules (metaboenrich.benchmark): per patient |z| >= 2 (patient_daa, alpha
0.0456); per disorder the group z-test on the mean z, BH q < 0.05 and
|mean z| >= 1 (group_daa). Rankings: absorbing walk with sources = increased
and = decreased (raw), current flow (raw), one-hop neighbour.

    # Reafect's Python (e.g. a venv next to the clone)
    python tools/reafect_bench.py reafect-download --reafect R --work W
    python tools/reafect_bench.py reafect-update   --reafect R --work W
    python tools/reafect_bench.py reafect-prepare  --reafect R --work W --workers 14
    python tools/reafect_bench.py reafect-abc      --reafect R --work W
    python tools/reafect_bench.py reafect-patients --reafect R --work W --dataset Bongaerts --workers 12
    python tools/reafect_bench.py reafect-patients --reafect R --work W --dataset Miller --workers 12
    # the repo's Python, from the repo root
    python tools/reafect_bench.py ours    --reafect R
    python tools/reafect_bench.py compare --reafect R --work W

Writes results/reafect_bench/: ours_ranks.tsv.gz, ours_runs.tsv, per_patient.tsv,
per_disorder.tsv, summary.tsv, report.md.
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # metaboenrich, imported only by our steps

A_PARAM, B_PARAM, C_PARAM = 0.85, 0.35, 0.75           # the notebook's values
DATASETS = ("Bongaerts", "Miller")
OUT = Path("results/reafect_bench")
OURS = {"walk_up": ("walk", "increased"), "walk_down": ("walk", "decreased"), "flow": ("flow", "decreased")}
METHODS = ["reafect", "walk_up", "walk_down", "flow", "neighbour"]
LABEL = {"reafect": "Reafect", "walk_up": "Walk, sources = increased", "walk_down": "Walk, sources = decreased",
         "flow": "Current flow (raw)", "neighbour": "One-hop neighbour"}


# ---- the z-score files ------------------------------------------------------------
def causal_symbols(label):
    """The file's deficient_gene label as gene symbols. BCKDH(A)(B)/DBT is the
    BCKDH complex (any subunit counts); ASS / ASS_a / ASS_b are ASS1 patients;
    IVD_a / IVD_b are IVD patients."""
    label = str(label)
    if label.startswith("BCKDH"):
        return ["BCKDHA", "BCKDHB", "DBT"]
    base = label.split("$")[0]
    return [{"ASS": "ASS1"}.get(base, base)]


def read_zscores(reafect, dataset):
    """(z: metabolites x patients, ann: name/HMDB/KEGG per metabolite, gene label and EC per patient)."""
    d = pd.read_csv(Path(reafect) / "Databases" / f"Z_scores_IEM_patients_{dataset}.csv").set_index("network_ID")
    pats = [c for c in d.columns if c not in ("metabolite", "ion_mode")]
    genes, enz = d.loc["deficient_gene", pats], d.loc["deficient_enzyme", pats]
    d = d.drop(index=["deficient_gene", "deficient_enzyme"])
    z = d[pats].apply(pd.to_numeric, errors="coerce")
    kegg = [i if i[:1] in "CD" and i[1:].isdigit() else "" for i in d.index]
    ann = pd.DataFrame({"name": d["metabolite"].fillna("").values, "HMDB": "", "KEGG": kegg}, index=d.index)
    return z, ann, genes, enz


def groups(genes):
    """Patients per disorder (canonical causal gene set), disorders with >= 2 patients."""
    key = genes.map(lambda g: ";".join(causal_symbols(g)))
    return {k: list(v.index) for k, v in key.groupby(key) if len(v) >= 2}


# ---- Reafect, run from its clone ------------------------------------------------------
def _reafect_setup(reafect):
    sys.modules["_elementtree"] = None                 # its KGML parser calls Element.getchildren() (gone in 3.9):
    import xml.etree.ElementTree as ET                 # use the pure-Python Element and restore the old method
    ET.Element.getchildren = lambda self: list(self)
    sys.path.insert(0, str(Path(reafect).resolve()))
    from Reafect import Reafect
    nb = json.load(open(Path(reafect) / "Reafect_reproduce_results_publication.ipynb", encoding="utf-8"))
    ns = {"pd": pd, "np": np}
    for i in (4, 5, 7):                                # module list, pathway list, custom reactions
        exec("".join(nb["cells"][i]["source"]), ns)
    return Reafect, ns


def _reafect(Reafect, reafect, work, results="Results"):
    os.chdir(work)
    return Reafect(path_to_store_maps="Maps/", path_to_abc_E="Abc_E/", path_to_store_results=f"{results}/",
                   path_to_databases=str(Path(reafect).resolve() / "Databases") + "/")


def _truncate(z):
    """The notebook's Z_score_truncation (applied to the Bongaerts data only)."""
    alpha = 1 / (1 + np.exp(2 - abs(z)))
    return np.sign(z) * (abs(z) ** 0.75 * alpha + (1 - alpha) * abs(z))


def reafect_inputs(reafect, dataset):
    """Per patient, and per disorder (mean z over its patients), as Reafect's input tables."""
    z, _, genes, enz = read_zscores(reafect, dataset)
    runs = {}
    for p in z.columns:
        runs[p] = (z[p].dropna(), enz[p])
    for key, pats in groups(genes).items():
        sub = z[pats]
        need = max(2, math.ceil(len(pats) / 2))      # as group_daa: measured in at least half the patients
        runs[f"group_{key.replace(';', '_')}"] = (sub.mean(axis=1)[sub.notna().sum(axis=1) >= need], enz[pats[0]])
    out = {}
    for rid, (s, e) in runs.items():
        if dataset == "Bongaerts":
            s = s.apply(_truncate)
        df = s.reset_index()
        df.columns = ["network_ID", "Z_score"]
        out[rid] = (df, e)
    return out


def clean_maps(work):
    """KEGG's module text now carries tabs and stoichiometric coefficients ('2 C00080') that the 2021
    format did not; Reafect would make them node names, which sympy then splits. Strip both."""
    import pickle
    import re
    fixed = 0
    for f in (Path(work) / "Maps").glob("*.pickle"):
        edges = pickle.load(open(f, "rb"))
        new = [tuple(re.sub(r"^\d+\s+", "", x.strip()) if i < 2 else x for i, x in enumerate(e)) for e in edges]
        if new != edges:
            pickle.dump(new, open(f, "wb"), protocol=pickle.HIGHEST_PROTOCOL)
            fixed += 1
    return fixed


def _maps_to_prepare(work):
    names = [f[:-len(".pickle")] for f in os.listdir(Path(work) / "Maps") if f.endswith(".pickle")]
    ext = {n[:-3] for n in names if n.endswith("ext")}
    return [n for n in names if n not in ext and "hsa01100" not in n]


def _prepare_one(args):
    """prepare_pathways on a folder holding one map: the same as one iteration of its loop over maps."""
    import contextlib
    import io
    import shutil
    reafect, work, name = args
    Reafect, _ = _reafect_setup(reafect)
    d = Path(work) / "split" / name
    (d / "Maps").mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(work) / "Maps" / f"{name}.pickle", d / "Maps" / f"{name}.pickle")
    with contextlib.redirect_stdout(io.StringIO()):
        _reafect(Reafect, reafect, d).prepare_pathways()
    return name


def _patient_one(args):
    import contextlib
    import io
    reafect, work, dataset, rid = args
    Reafect, ns = _reafect_setup(reafect)
    df, e = reafect_inputs(reafect, dataset)[rid]
    r = _reafect(Reafect, reafect, work, f"Results_{dataset}")
    with contextlib.redirect_stdout(io.StringIO()):
        r.process_patient(patient_ID=rid, disease_enzyme=e, Z_score_data=df, type_of_input_Z_scores="Z_scores",
                          a_param=A_PARAM, b_param=B_PARAM, c_param=C_PARAM,
                          custom_reactions_with_enzyme=ns["custom_reactions_with_enzyme"])
    return rid


def run_reafect(step, args):
    import pickle
    import time
    reafect, work = str(Path(args.reafect).resolve()), str(Path(args.work).resolve())
    Path(work).mkdir(parents=True, exist_ok=True)
    Reafect, ns = _reafect_setup(reafect)
    if step == "download":
        r = _reafect(Reafect, reafect, work)
        r.get_modules_kegg(ns["hsa_modules"])
        r.get_pathways_kegg(ns["hsa_maps"])
    elif step == "update":
        print("maps cleaned:", clean_maps(work))
        _reafect(Reafect, reafect, work).update_kegg_pathways(ns["extra_reactions"], ns["custom_ID_to_KEGG"])
        print("maps cleaned after update:", clean_maps(work))
    elif step == "prepare":
        names, t0 = _maps_to_prepare(work), time.time()
        todo = [n for n in names if not (Path(work) / "split" / n / "Abc_E" / "all_m_R_E.pickle").exists()]
        todo.sort(key=lambda n: -os.path.getsize(Path(work) / "Maps" / f"{n}.pickle"))
        with ProcessPoolExecutor(args.workers) as ex:
            for i, n in enumerate(ex.map(_prepare_one, [(reafect, work, n) for n in todo])):
                print(f"{i + 1}/{len(todo)} {n} {time.time() - t0:.0f}s", flush=True)
        paths = []
        for n in names:
            with open(Path(work) / "split" / n / "Abc_E" / "all_m_R_E.pickle", "rb") as f:
                paths += pickle.load(f)
        (Path(work) / "Abc_E").mkdir(exist_ok=True)
        with open(Path(work) / "Abc_E" / "all_m_R_E.pickle", "wb") as f:
            pickle.dump(paths, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(len(names), "maps,", len(paths), "decay paths")
    elif step == "abc":
        _reafect(Reafect, reafect, work).determine_pathways_for_abc_params(
            a_params=[A_PARAM], b_params=[B_PARAM], c_params=[C_PARAM], n_cores=1)
    elif step == "patients":
        rids = list(reafect_inputs(reafect, args.dataset))
        t0 = time.time()
        with ProcessPoolExecutor(args.workers) as ex:
            for i, rid in enumerate(ex.map(_patient_one, [(reafect, work, args.dataset, r) for r in rids])):
                print(f"{i + 1}/{len(rids)} {rid} {time.time() - t0:.0f}s", flush=True)


# ---- ours ------------------------------------------------------------------------------
def _network():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metaboenrich import structure
    from metaboenrich.gem import load_sbml
    from metaboenrich.network import build_graph
    model = load_sbml("data/Human-GEM.xml")
    st = next(s for s in structure.stages(model, "three_rule", atom_dir="data") if s[0] == "three_rule")
    return model, build_graph(model, st[3], st[4], st[5], **st[6])


def _ours_one(args):
    """One of our rankings over every patient and disorder group of both datasets."""
    from metaboenrich import benchmark as B
    from metaboenrich.daa import load_name_bridge
    method_key, reafect, out = args
    model, graph = _network()
    method, orientation = OURS[method_key]
    ranker = B.Ranker(model, graph, method, None, load_name_bridge("data/id_translation.csv"), orientation)
    tmp = Path(out) / f"_daa_{method_key}.csv"
    ranks, runs = {}, []
    for dataset in DATASETS:
        z, ann, genes, _ = read_zscores(reafect, dataset)
        todo = [(p, "patient", B.patient_daa(z, ann, p), 0.0456, 0.0) for p in z.columns]
        todo += [(f"group_{k.replace(';', '_')}", "group", B.group_daa(z, ann, pats), 0.05, 1.0)
                 for k, pats in groups(genes).items()]
        for rid, level, daa, alpha, min_fc in todo:
            info, me, nb, _ = ranker.run(daa, tmp, alpha, min_fc)
            ranks[(dataset, rid, method_key)] = me["raw"]
            if method_key == "walk_up":
                ranks[(dataset, rid, "neighbour")] = nb
            runs.append({"dataset": dataset, "run": rid, "level": level, "method": method_key, **info})
    tmp.unlink(missing_ok=True)
    return ranks, runs, ranker.universe


def run_ours(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(len(OURS)) as ex:
        res = list(ex.map(_ours_one, [(m, args.reafect, str(out)) for m in OURS]))
    universe = res[0][2]
    cols, runs = {}, []
    for ranks, r, _ in res:
        runs += r
        for (dataset, rid, m), s in ranks.items():
            cols[f"{dataset}|{rid}|{m}"] = s.reindex(universe) if s is not None else pd.Series(np.nan, index=universe)
    pd.DataFrame(cols, index=universe).rename_axis("ensembl_id").to_csv(out / "ours_ranks.tsv.gz", sep="\t")
    pd.DataFrame(runs).to_csv(out / "ours_runs.tsv", sep="\t", index=False)
    # metabolite mapping rate: rows of each file that reach a model metabolite, and a network node
    from metaboenrich.daa import load_daa, load_name_bridge
    model, graph = _network()
    rates = {}
    for dataset in DATASETS:
        z, ann, _, _ = read_zscores(args.reafect, dataset)
        df = ann.assign(log2FC=1.0, p=0.0)
        daa, _, _ = load_daa(df, model, ["HMDB", "KEGG", "name"], "log2FC", "p", 1.0,
                             name_bridge=load_name_bridge("data/id_translation.csv"))
        rates[dataset] = {"rows": len(ann), "model": int(daa.met_id.notna().sum()),
                          "network": int(daa.met_id.map(graph.index).notna().sum())}
    (out / "mapping.json").write_text(json.dumps(rates, indent=2))
    print("wrote", out, rates)


# ---- compare ----------------------------------------------------------------------------
def _reafect_table(work, dataset, rid):
    f = Path(work) / f"Results_{dataset}" / rid / f"SR_scores_{A_PARAM}_{B_PARAM}_{C_PARAM}_Z_scores.csv"
    return pd.read_csv(f) if f.exists() else None


def run_compare(args):
    from scipy.stats import binomtest, rankdata
    out, work = Path(args.out), Path(args.work)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from metaboenrich.gem import load_sbml
    model = load_sbml("data/Human-GEM.xml")
    ours = pd.read_csv(out / "ours_ranks.tsv.gz", sep="\t", index_col=0)
    U = len(ours)
    sym = pd.Series({e: model.genes.get(e, e) for e in ours.index})
    ens_of = {}
    for e, s in sym.items():
        ens_of.setdefault(s, e)
    eg = pd.read_csv(Path(args.reafect) / "Databases" / "kegg_enzyme_with_gene_symbol.csv")
    ec_genes = eg.groupby("enzyme").gene_symbol.apply(lambda s: [x.split(" ")[0] for x in s]).to_dict()
    ec_genes.setdefault("6.2.1.n3", []).append("ACSF3")              # the notebook's extra_genes_with_enzyme
    ec_genes.setdefault("5.4.99.2", []).append("MUT")

    # Reafect per run: EC rank (0-based, as published) and gene best rank
    rf = {}
    scored_genes = set()
    for dataset in DATASETS:
        z, _, genes, enz = read_zscores(args.reafect, dataset)
        rids = list(z.columns) + [f"group_{k.replace(';', '_')}" for k in groups(genes)]
        for rid in rids:
            t = _reafect_table(work, dataset, rid)
            if t is None:
                continue
            t = t.dropna(subset=["enzyme"])
            ec_rank = t.groupby("enzyme")["rank"].min()
            g_rank = {}
            for ec, r in ec_rank.items():
                for g in ec_genes.get(ec, []):
                    g_rank[g] = min(g_rank.get(g, np.inf), r)
            scored_genes |= set(g_rank)
            rf[(dataset, rid)] = {"ec_rank": ec_rank, "max_rank": t["max_rank"].max(), "gene_rank": g_rank}
    common = sorted(ens_of[g] for g in scored_genes if g in ens_of)
    N = len(common)

    def rerank(values):
        """Ranks within the common universe; genes with no score share the tail mid-rank."""
        v = pd.Series(values, index=common, dtype=float).fillna(np.inf)
        return pd.Series(rankdata(v.values, method="average"), index=common)

    rows = []
    for dataset in DATASETS:
        z, _, genes, enz = read_zscores(args.reafect, dataset)
        runs = [(p, "patient", genes[p], enz[p]) for p in z.columns]
        runs += [(f"group_{k.replace(';', '_')}", "group", genes[p[0]], enz[p[0]]) for k, p in groups(genes).items()]
        for rid, level, glabel, ec in runs:
            cs = causal_symbols(glabel)
            c_ens = [ens_of[g] for g in cs if g in ens_of]
            c_common = [e for e in c_ens if e in set(common)]
            base = {"dataset": dataset, "run": rid, "level": level, "disorder": ";".join(cs), "enzyme": ec,
                    "causal_in_network": bool(c_ens), "causal_in_common": bool(c_common)}
            r = rf.get((dataset, rid))
            row = {**base}
            if r is not None:
                er = r["ec_rank"].get(ec)
                row["reafect_own_pct"] = 100 * er / r["max_rank"] if er is not None else np.nan
                row["reafect_n_enzymes"] = r["max_rank"]
                gr = rerank({ens_of[g]: v for g, v in r["gene_rank"].items() if g in ens_of and ens_of[g] in set(common)})
                row["reafect_common_rank"] = min(gr[e] for e in c_common) if c_common else np.nan
            for m in ["walk_up", "walk_down", "flow", "neighbour"]:
                col = f"{dataset}|{rid}|{m}"
                s = ours[col] if col in ours else pd.Series(np.nan, index=ours.index)
                row[f"{m}_own_pct"] = 100 * min(s[e] for e in c_ens) / U if c_ens and s.notna().any() else np.nan
                row[f"{m}_common_rank"] = (min(rerank(s.reindex(common))[e] for e in c_common)
                                           if c_common and s.notna().any() else np.nan)
            rows.append(row)
    pp = pd.DataFrame(rows)
    for m in METHODS:
        pp[f"{m}_common_pct"] = 100 * pp[f"{m}_common_rank"] / N
    pp.to_csv(out / "per_patient.tsv", sep="\t", index=False)

    def summarise(df, unit):
        """Per dataset and both together, on the rows where every method has a value in that universe."""
        rows = []
        for ds in list(DATASETS) + ["both"]:
            d = df if ds == "both" else df[df.dataset == ds]
            for space in ("common", "own"):
                cols = [f"{m}_{space}_pct" for m in METHODS]
                d2 = d.dropna(subset=cols)
                for m in METHODS:
                    x = d2[f"{m}_{space}_pct"]
                    if not len(x):
                        continue
                    rows.append({"unit": unit, "dataset": ds, "universe": space, "method": LABEL[m], "n": len(x),
                                 "median_pct": x.median(), "top1": (x <= 1).mean() * 100,
                                 "top5": (x <= 5).mean() * 100, "top10": (x <= 10).mean() * 100})
        return rows

    pat = pp[(pp.level == "patient") & pp.causal_in_common]
    dis = pat.groupby(["dataset", "disorder"])[[f"{m}_common_pct" for m in METHODS] +
                                               [f"{m}_own_pct" for m in METHODS]].median().reset_index()
    grp = pp[(pp.level == "group") & pp.causal_in_common]
    summary = pd.DataFrame(summarise(pat, "patient") + summarise(dis, "disorder (median of its patients)") +
                           summarise(grp, "disorder group (mean z, >= 2 patients)"))
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    dis.to_csv(out / "per_disorder.tsv", sep="\t", index=False)

    def paired(df, a, b):
        d = df[[f"{a}_common_pct", f"{b}_common_pct"]].dropna()
        w, l = int((d.iloc[:, 0] < d.iloc[:, 1]).sum()), int((d.iloc[:, 0] > d.iloc[:, 1]).sum())
        return w, l, (binomtest(w, w + l).pvalue if w + l else np.nan), len(d)

    pairs = {f"{name}, {ds}": paired(df if ds == "both" else df[df.dataset == ds], "reafect", "walk_up")
             for name, df in (("patients", pat), ("disorders (median of patients)", dis), ("disorder groups", grp))
             for ds in list(DATASETS) + ["both"]}
    mapping = json.loads((out / "mapping.json").read_text())
    write_report(out, summary, pairs, mapping, N, U, pp, args)
    print("wrote", out)


def write_report(out, summary, pairs, mapping, N, U, pp, args):
    def tab(df):
        cols = ["method", "n", "median_pct", "top1", "top5", "top10"]
        h = "| Method | n | Median percentile | Top 1% | Top 5% | Top 10% |\n|---|---|---|---|---|---|\n"
        return h + "".join(f"| {r.method} | {r.n} | {r.median_pct:.1f} | {r.top1:.0f}% | {r.top5:.0f}% | {r.top10:.0f}% |\n"
                           for r in df[cols].itertuples())
    pat = pp[pp.level == "patient"]
    rep = pat.dropna(subset=["reafect_own_pct"])
    bong = rep[rep.dataset == "Bongaerts"].reafect_own_pct
    mil = rep[rep.dataset == "Miller"].reafect_own_pct
    L = ["# Reafect head-to-head", "",
         "Reafect (Bongaerts et al. 2022, doi 10.1016/j.ymgme.2022.05.002) against our rankings, on Reafect's own "
         "z-score files: 72 Rotterdam IEM patients (Bongaerts, 40 genes) and 106 Miller 2015 patients (19 genes). "
         "Rank 1 / percentile 0 is best.", "",
         "## Reproduction of Reafect's published result", "",
         f"Bongaerts patients scored: {len(bong)} of 72. Disease enzyme within the top 2.5% / 5% / 10% of scored enzymes: "
         f"{(bong <= 2.5).mean() * 100:.0f}% / {(bong <= 5).mean() * 100:.0f}% / {(bong <= 10).mean() * 100:.0f}% "
         "(published, from the notebook's saved run: 62% / 81% / 83%). "
         f"Miller patients scored: {len(mil)} of 106; {(mil <= 2.5).mean() * 100:.0f}% / {(mil <= 5).mean() * 100:.0f}% / "
         f"{(mil <= 10).mean() * 100:.0f}% (published: 61% / 74% / 82%). KEGG content was downloaded now, not in "
         "January 2021; 5 of the notebook's 184 modules no longer exist (M00166, M00167, M00415, M00873, M00874).", "",
         "## Metabolite mapping to our network", ""]
    for d, m in mapping.items():
        L.append(f"- {d}: {m['rows']} metabolites, {m['model']} map to Human-GEM, {m['network']} to the three_rule network.")
    L += ["", "## Universes", "",
          f"- **Own**: each method's own candidate set. Reafect: the EC numbers it scored for that patient "
          f"(about 520-580; percentile = 0-based rank / number scored, as published). Ours: {U:,} network genes "
          "(percentile = rank / U; unscored genes share the tail mid-rank).",
          f"- **Common**: the {N} genes that Reafect scored in at least one run and that are in our network. "
          "Every method is re-ranked within it; genes a method left unscored share the tail mid-rank. "
          "Reafect gene rank = best rank of its EC numbers.", ""]
    L += ["Each table uses only the rows where every method has a value, so the methods are compared on the "
          "same patients or disorders.", ""]
    for unit in summary.unit.unique():
        for ds in summary.dataset.unique():
            s = summary[(summary.unit == unit) & (summary.dataset == ds)]
            for space in ("common", "own"):
                t = s[s.universe == space]
                if len(t):
                    L += [f"### {unit}: {ds}, {space} universe", "", tab(t), ""]
    L += ["## Paired: Reafect vs walk with sources = increased (common universe)", "",
          "| Level | Pairs | Reafect better | Walk better | Sign test p |", "|---|---|---|---|---|"]
    for name, (w, l, p, n) in pairs.items():
        L.append(f"| {name} | {n} | {w} | {l} | {p:.3g} |")
    L += ["", "## Not comparable / caveats", "",
          "- Reafect is per patient by design; ours is defined at group level (D21). Per-patient runs of ours use the "
          "benchmark's reference-only patient rule (|z| >= 2).",
          "- Reafect's Miller file keeps 349 of Miller's metabolites (those with KEGG ids), so our Miller numbers here "
          "are not our collection A numbers.",
          "- Bongaerts z-scores are truncated before Reafect, as in its notebook; ours use the raw z.",
          "- Causal gene for BCKDH(A)(B)/DBT is the best of BCKDHA, BCKDHB, DBT; ASS* = ASS1, IVD* = IVD.",
          f"- Reafect ran from its clone ({args.reafect}), unmodified; one shim restores ElementTree.getchildren "
          "for Python >= 3.9."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["reafect-download", "reafect-update", "reafect-prepare", "reafect-abc",
                                    "reafect-patients", "ours", "compare"])
    p.add_argument("--reafect", required=True, help="Reafect clone")
    p.add_argument("--work", default=None, help="Reafect working folder (maps, paths, results)")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    if args.step.startswith("reafect-"):
        run_reafect(args.step.split("-", 1)[1], args)
    elif args.step == "ours":
        run_ours(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
