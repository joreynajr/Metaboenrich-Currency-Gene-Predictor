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

Our test set (D26: results/data_figures/units.tsv, testable == True), Reafect against our
rankings on every unit; z per metabolite from each unit's data (units-input documents how):

    python tools/reafect_bench.py units-input   --reafect R --work W          # repo Python
    python tools/reafect_bench.py reafect-units --reafect R --work W --workers 6   # Reafect's Python
    python tools/reafect_bench.py ours-units    --reafect R --workers 3          # repo Python
    python tools/reafect_bench.py units-compare --reafect R --work W           # repo Python

Writes results/reafect_bench/units/: ours_ranks.tsv.gz, per_unit.tsv, coverage.tsv, summary.tsv,
paired.tsv, report.md.
"""
import argparse
import json
import math
import os
import re
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
    ec_genes = _ec_genes(args.reafect)

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


# ---- our test set: the D26 units --------------------------------------------------------
UNITS_TSV = Path("results/data_figures/units.tsv")
KEGG_ID = re.compile(r"[CD]\d{5}")
P_FLOOR = 1e-16                                          # z cap: |z| <= Phi^-1(1 - P_FLOOR / 2)
UNIT_DAA = {"C_workbench": (Path("results/workbench_daa"), ["KEGG", "HMDB", "name"]),
            "D_metabolights": (Path("results/metabolights_daa"), ["CHEBI", "name"])}   # as null_calibration
UNIT_RANKINGS = {"walk_up": ("results/correction_eval_d26_walk_increased", "M0"),
                 "walk_down": ("results/correction_eval_d26_walk_decreased", "M0"),
                 "flow": ("results/correction_eval_d26_flow_decreased_obs", "M0"),
                 "neighbour": ("results/correction_eval_d26_walk_increased", "neighbour")}


def test_units(path=UNITS_TSV):
    """The D26 test set: rows of units.tsv with testable == True (one row per comparison)."""
    u = pd.read_csv(path, sep="\t")
    return u[u.testable.astype(str).eq("True")].reset_index(drop=True)


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(label))


def z_from_p(log2fc, p):
    """z = sign(log2FC) * Phi^-1(1 - p/2), p floored at P_FLOOR (so |z| <= 8.29)."""
    from scipy.stats import norm
    p = np.clip(np.asarray(p, float), P_FLOOR, 1.0)
    return np.sign(np.asarray(log2fc, float)) * norm.isf(p / 2)


def _ec_genes(reafect):
    """EC -> gene symbols, from Reafect's table plus the notebook's extra_genes_with_enzyme."""
    eg = pd.read_csv(Path(reafect) / "Databases" / "kegg_enzyme_with_gene_symbol.csv")
    ec_genes = eg.groupby("enzyme").gene_symbol.apply(lambda s: [x.split(" ")[0] for x in s]).to_dict()
    ec_genes.setdefault("6.2.1.n3", []).append("ACSF3")
    ec_genes.setdefault("5.4.99.2", []).append("MUT")
    return ec_genes


def _symbol_ecs(ec_genes):
    out = {}
    for ec, gs in ec_genes.items():
        for g in gs:
            out.setdefault(g, set()).add(ec)
    out.setdefault("MMUT", set()).update(out.get("MUT", set()))          # MUT was renamed MMUT (HGNC 2019)
    return out


def _a_group(B, cases, data, sub_unit):
    """A_inborn_errors unit: (z, ann, samples) as registry_units reads it."""
    c = cases.loc[sub_unit]
    if c.dataset not in data:
        data[c.dataset] = B.load_t2020() if c.dataset == "t2020" else B.load_miller()
    z, ann, cohorts = data[c.dataset]
    return z, ann, [s for s in cohorts.get(c.cohort, []) if s in z.columns]


def unit_tables(units):
    """Per unit: every measured metabolite with its identifiers and a z-score.
    A: group_daa's group z (mean z * sqrt(n measured)), uncapped. B, C, D: z from log2FC and p (z_from_p)."""
    from metaboenrich import benchmark as B
    cases = pd.read_csv(B.CASES, sep="\t").fillna("").set_index("case_id")
    data, out = {}, {}
    for r in units.itertuples():
        if r.collection == "A_inborn_errors":
            z, ann, samples = _a_group(B, cases, data, r.sub_unit)
            g = B.group_daa(z, ann, samples)
            g["z"] = g.log2FC * np.sqrt(z[samples].notna().sum(axis=1).reindex(g.index))
            df = g[["name", "HMDB", "KEGG", "z"]].replace("", np.nan).reset_index(drop=True)
        elif r.collection == "B_published":
            d = Path("predictions/published_v2") / r.label
            t = pd.concat([pd.read_csv(d / f, sep="\t") for f in ("daa_mapping.tsv", "daa_unmapped.tsv")])
            t = t.dropna(subset=["log2fc", "p"])
            ids = t.input_id.astype(str).where(t.input_id.notna(), np.nan)
            df = pd.DataFrame({"name": ids.values, "KEGG": ids.where(ids.str.fullmatch(KEGG_ID.pattern) == True).values,
                               "met_id": t.met_id.replace("", np.nan).values, "z": z_from_p(t.log2fc, t.p)})
        else:
            folder, id_cols = UNIT_DAA[r.collection]
            t = pd.read_csv(folder / f"{r.study}_{r.sub_unit}.csv", dtype={c: str for c in id_cols}).replace("", np.nan)
            t = t.dropna(subset=["log2FC", "p"])
            df = t[id_cols].assign(z=z_from_p(t.log2FC, t.p)).reset_index(drop=True)
        out[r.label] = df
    return out


def kegg_ids(df, model, bridge, gem_kegg, lookup=None, loose=None):
    """KEGG compound per row and how it was found: the table's KEGG column; else the Human-GEM metabolite
    (by the published mapping's met_id, HMDB, ChEBI or name, as load_daa matches them) and its metKEGGID;
    else a KEGG id the name bridge (id_translation) gives for the name; else an unambiguous loose name match."""
    from metaboenrich.daa import build_loose_index, build_lookup, loose_key, map_identifier, name_variants
    lookup = lookup if lookup is not None else build_lookup(model)
    loose = loose if loose is not None else build_loose_index(model)

    def first(x):
        hits = KEGG_ID.findall(str(x)) if isinstance(x, str) else []
        return hits[0] if hits else None

    kegg, how = [], []
    for row in df.to_dict("records"):
        k, h = first(row.get("KEGG")), "table KEGG"
        if not k and isinstance(row.get("met_id"), str):
            k, h = first(gem_kegg.get(row["met_id"])), "Human-GEM (published mapping)"
        for col in ("HMDB", "CHEBI", "name"):
            if k or not isinstance(row.get(col), str):
                continue
            m = map_identifier(row[col], lookup)
            k, h = (first(gem_kegg.get(m)), f"Human-GEM via {col}") if m else (None, "")
        if not k and isinstance(row.get("name"), str):
            for v in name_variants(row["name"]):
                for x in bridge.get(v, []):
                    k = first(x) or first(gem_kegg.get(map_identifier(x, lookup)))
                    if k:
                        h = "id_translation name"
                        break
                if k:
                    break
        if not k and isinstance(row.get("name"), str):
            m = loose.get(loose_key(row["name"]))
            k, h = (first(gem_kegg.get(m)), "Human-GEM via loose name") if m else (None, "")
        kegg.append(k)
        how.append(h if k else "")
    return kegg, how


def run_units_input(args):
    """Reafect input per test unit: network_ID (KEGG), Z_score; all measured metabolites with a KEGG id."""
    from metaboenrich.daa import build_loose_index, build_lookup, load_name_bridge
    from metaboenrich.gem import load_sbml
    work = Path(args.work) / "units_input"
    work.mkdir(parents=True, exist_ok=True)
    units = test_units()
    model = load_sbml("data/Human-GEM.xml")
    bridge = load_name_bridge("data/id_translation.csv")
    gm = pd.read_csv("data/Human-GEM_metabolites.tsv", sep="\t", dtype=str)
    gem_kegg = gm.dropna(subset=["metKEGGID"]).drop_duplicates("metsNoComp").set_index("metsNoComp").metKEGGID.to_dict()
    lookup, loose = build_lookup(model), build_loose_index(model)
    sym_ecs = _symbol_ecs(_ec_genes(args.reafect))
    tables, rows = unit_tables(units), []
    for r in units.itertuples():
        df = tables[r.label]
        df["network_ID"], df["kegg_from"] = kegg_ids(df, model, bridge, gem_kegg, lookup, loose)
        k = df.dropna(subset=["network_ID", "z"])
        k = k.loc[k.z.abs().sort_values(ascending=False).index].drop_duplicates("network_ID")   # one z per compound
        k[["network_ID", "z"]].rename(columns={"z": "Z_score"}).to_csv(work / f"{safe_name(r.label)}.csv", index=False)
        causal = [s for s in str(r.genes_in_net).split(";") if s]
        ecs = sorted({e for s in causal for e in sym_ecs.get(s, ())})
        rows.append({"label": r.label, "file": safe_name(r.label), "n_measured": len(df),
                     "n_with_kegg": int(df.network_ID.notna().sum()), "n_kegg_unique": len(k),
                     "z_abs_max": float(k.z.abs().max()) if len(k) else np.nan,
                     **{f"kegg_from_{h}": int((df.kegg_from == h).sum()) for h in df.kegg_from.unique() if h},
                     "causal_genes": ";".join(causal), "causal_ecs": ";".join(ecs)})
    man = pd.DataFrame(rows).fillna({c: 0 for c in pd.DataFrame(rows).columns if c.startswith("kegg_from_")})
    man.to_csv(work / "manifest.tsv", sep="\t", index=False)
    print(man[["label", "n_measured", "n_with_kegg", "n_kegg_unique"]].to_string())


def _unit_one(args):
    import contextlib
    import io
    reafect, work, file, enzyme = args
    Reafect, ns = _reafect_setup(reafect)
    df = pd.read_csv(Path(work) / "units_input" / f"{file}.csv")
    r = _reafect(Reafect, reafect, work, "Results_units")
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            r.process_patient(patient_ID=file, disease_enzyme=enzyme, Z_score_data=df, type_of_input_Z_scores="Z_scores",
                              a_param=A_PARAM, b_param=B_PARAM, c_param=C_PARAM,
                              custom_reactions_with_enzyme=ns["custom_reactions_with_enzyme"])
        return file, ""
    except Exception as e:                                # e.g. no input compound in any KEGG module
        return file, f"{type(e).__name__}: {e}"


def run_reafect_units(reafect, work, workers):
    """Reafect on every test unit's input; also the compounds and EC numbers its prepared KEGG modules contain."""
    import pickle
    import time
    man = pd.read_csv(Path(work) / "units_input" / "manifest.tsv", sep="\t", dtype=str).fillna("")
    with open(Path(work) / "Abc_E" / f"all_m_R_E_{A_PARAM}_{B_PARAM}_{C_PARAM}.pickle", "rb") as f:
        paths = pickle.load(f)
    comps = set(paths["m"].unique())
    rxns = set(paths["R"].unique())
    del paths
    ns = _reafect_setup(reafect)[1]
    er = pd.concat([pd.read_csv(Path(reafect) / "Databases" / "kegg_enzyme_with_reaction.csv")[["enzyme", "R"]],
                    ns["custom_reactions_with_enzyme"][["enzyme", "R"]]])
    ecs = sorted(set(er[er.R.isin(rxns)].enzyme.astype(str)))
    n_in = {r.file: int(pd.read_csv(Path(work) / "units_input" / f"{r.file}.csv").network_ID.isin(comps).sum())
            for r in man.itertuples()}
    (Path(work) / "units_input" / "reafect_network.json").write_text(
        json.dumps({"compounds": sorted(comps), "ecs": ecs, "n_input_in_network": n_in}, indent=1))
    todo = sorted(man.itertuples(), key=lambda r: -n_in[r.file])
    t0, failed = time.time(), {}
    with ProcessPoolExecutor(workers) as ex:
        jobs = [(reafect, work, r.file, r.causal_ecs.split(";")[0] if r.causal_ecs else None) for r in todo]
        for i, (file, err) in enumerate(ex.map(_unit_one, jobs)):
            if err:
                failed[file] = err
            print(f"{i + 1}/{len(jobs)} {file} {time.time() - t0:.0f}s {err}", flush=True)
    (Path(work) / "units_input" / "reafect_failed.json").write_text(json.dumps(failed, indent=1))


def _ours_units_one(args):
    """Full gene ranks of one of our methods on every test unit, as correction_eval's reference
    (Ranker.run for A, its gene step on the unit's sources and targets otherwise)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from correction_eval import Evaluator
    from metaboenrich import benchmark as B
    from metaboenrich.daa import load_name_bridge
    from null_calibration import Calibrator, registry_units
    method_key, out = args
    method, orientation = OURS[method_key]
    model, graph = _network()
    bridge = load_name_bridge("data/id_translation.csv")
    ev = Evaluator(model, graph, bridge, Calibrator(graph, None, "flow"), method, orientation)
    U = len(ev.universe)
    units = {u["label"]: u for u in registry_units(model, graph, bridge, orientation)[0]}
    cases, data = pd.read_csv(B.CASES, sep="\t").fillna("").set_index("case_id"), {}
    tmp = Path(out) / f"_daa_{method_key}.csv"
    cols = {}
    for r in test_units().itertuples():
        u = units[r.label]
        tail = pd.Series((1 + U) / 2.0, index=ev.universe)
        if not u["sources"] or not (u["targets"] or u["ground"]):
            cols[f"{r.label}|{method_key}"] = tail
            continue
        _, ranks, nb = ev.reference(u)
        if r.collection == "A_inborn_errors":
            z, ann, samples = _a_group(B, cases, data, r.sub_unit)
            _, me, _, _ = ev.ranker.run(B.group_daa(z, ann, samples), tmp, 0.05, 1.0)
            if me["raw"] is not None:
                ranks = me["raw"]
        cols[f"{r.label}|{method_key}"] = ranks.reindex(ev.universe)
        cols[f"{r.label}|neighbour_{method_key}"] = nb.reindex(ev.universe)
    tmp.unlink(missing_ok=True)
    return cols, ev.universe, {e: model.genes.get(e, e) for e in ev.universe}


def run_ours_units(args):
    out = Path(args.out) / "units"
    out.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(min(args.workers, len(OURS))) as ex:
        res = list(ex.map(_ours_units_one, [(m, str(out)) for m in OURS]))
    universe, sym = res[0][1], res[0][2]
    cols = {}
    for c, _, _ in res:
        cols.update(c)
    df = pd.DataFrame(cols, index=universe)
    for label in test_units().label:                     # the neighbour ranking ignores orientation: one copy
        nb = [f"{label}|neighbour_{m}" for m in ("walk_up", "walk_down") if f"{label}|neighbour_{m}" in df]
        df[f"{label}|neighbour"] = df[nb[0]] if nb else (1 + len(universe)) / 2.0
    df = df[[c for c in df.columns if "|neighbour_" not in c]]
    df.insert(0, "symbol", pd.Series(sym))
    df.rename_axis("ensembl_id").to_csv(out / "ours_ranks.tsv.gz", sep="\t")
    print("wrote", out / "ours_ranks.tsv.gz", df.shape)


def run_units_compare(args):
    from scipy.stats import binomtest, rankdata
    out, work = Path(args.out) / "units", Path(args.work)
    units = test_units()
    man = pd.read_csv(work / "units_input" / "manifest.tsv", sep="\t", dtype={"causal_ecs": str}).set_index("label")
    net = json.loads((work / "units_input" / "reafect_network.json").read_text())
    ff = work / "units_input" / "reafect_failed.json"
    failed = json.loads(ff.read_text()) if ff.exists() else {}
    net_ecs = set(net["ecs"])
    ours = pd.read_csv(out / "ours_ranks.tsv.gz", sep="\t", index_col=0)
    sym = ours.pop("symbol")
    U = len(ours)
    ens_of = {}
    for e, s in sym.items():
        ens_of.setdefault(s, e)
    ens_of.setdefault("MUT", ens_of.get("MMUT"))
    ec_genes = _ec_genes(args.reafect)
    sym_ecs = _symbol_ecs(ec_genes)
    ref = {}                                              # causal ranks the correction_eval runs report
    for m, (d, ranking) in UNIT_RANKINGS.items():
        f = Path(d) / "rankings.tsv"
        rk = pd.read_csv(f, sep="\t") if f.exists() else pd.DataFrame(columns=["unit", "ranking", "rank"])
        ref[m] = rk[rk.ranking == ranking].set_index("unit")["rank"]

    rf, scored = {}, set()
    for r in units.itertuples():
        f = work / "Results_units" / safe_name(r.label) / f"SR_scores_{A_PARAM}_{B_PARAM}_{C_PARAM}_Z_scores.csv"
        if not f.exists():
            continue
        t = pd.read_csv(f).dropna(subset=["enzyme"])
        ec_rank = t.groupby("enzyme")["rank"].min()
        g_rank = {}
        for ec, v in ec_rank.items():
            for g in ec_genes.get(ec, []):
                e = ens_of.get(g)
                if e is not None:
                    g_rank[e] = min(g_rank.get(e, np.inf), v)
        scored |= set(g_rank)
        rf[r.label] = {"ec_rank": ec_rank, "max_rank": t["max_rank"].max() if len(t) else np.nan, "gene_rank": g_rank}
    causal_ens = {r.label: [ens_of[s] for s in str(r.genes_in_net).split(";") if s in ens_of] for r in units.itertuples()}
    common = sorted((scored | {e for v in causal_ens.values() for e in v}) & set(ours.index))
    N = len(common)

    def rerank(values):
        v = pd.Series(values, dtype=float).reindex(common).fillna(np.inf)
        return pd.Series(rankdata(v.values, method="average"), index=common)

    rows, cov = [], []
    for r in units.itertuples():
        mm = man.loc[r.label]
        cs = [s for s in str(r.genes_in_net).split(";") if s]
        ce = causal_ens[r.label]
        ecs = sorted({e for s in cs for e in sym_ecs.get(s, ())})
        x = rf.get(r.label)
        unit_ecs = set(x["ec_rank"].index) if x else set()
        row = {"label": r.label, "collection": r.collection, "study": r.study, "cluster": r.cluster,
               "primary": bool(r.primary), "perturbation": r.perturbation, "causal_genes": ";".join(cs),
               "causal_ecs": ";".join(ecs), "n_measured": int(mm.n_measured), "n_with_kegg": int(mm.n_with_kegg),
               "n_kegg_unique": int(mm.n_kegg_unique),
               "n_kegg_in_reafect_network": net["n_input_in_network"].get(safe_name(r.label), 0),
               "reafect_ran": x is not None, "reafect_error": failed.get(safe_name(r.label), ""),
               "causal_ec_in_reafect_modules": bool(set(ecs) & net_ecs), "causal_scored_by_reafect": bool(set(ecs) & unit_ecs),
               "reafect_n_enzymes": x["max_rank"] if x else np.nan}
        er = min((x["ec_rank"][e] for e in ecs if e in unit_ecs), default=None) if x else None
        row["reafect_causal_ec_rank"] = er
        row["reafect_own_pct"] = 100 * er / x["max_rank"] if er is not None else 100.0
        gr = rerank(x["gene_rank"] if x else {})
        row["reafect_common_rank"] = min(gr[e] for e in ce) if ce else np.nan
        for m in ["walk_up", "walk_down", "flow", "neighbour"]:
            s = ours[f"{r.label}|{m}"]
            own = min(s[e] for e in ce)
            row[f"{m}_rank"] = own
            row[f"{m}_rank_reported"] = ref[m].get(r.label, np.nan)
            row[f"{m}_own_pct"] = 100 * own / U
            row[f"{m}_common_rank"] = min(rerank(s.reindex(common))[e] for e in ce)
        for m in METHODS:
            row[f"{m}_common_pct"] = 100 * row[f"{m}_common_rank"] / N
        rows.append(row)
        for s in cs:
            se = sorted(sym_ecs.get(s, ()))
            cov.append({"label": r.label, "cluster": r.cluster, "collection": r.collection, "gene": s,
                        "in_our_network": s in ens_of, "reafect_ecs": ";".join(se),
                        "ec_in_reafect_modules": bool(set(se) & net_ecs), "scored_in_unit": bool(set(se) & unit_ecs),
                        "reafect_ec_rank": min((x["ec_rank"][e] for e in se if e in unit_ecs), default=np.nan) if x else np.nan,
                        "reafect_n_enzymes": row["reafect_n_enzymes"],
                        "n_kegg_in_reafect_network": row["n_kegg_in_reafect_network"]})
    pu = pd.DataFrame(rows)
    pu.to_csv(out / "per_unit.tsv", sep="\t", index=False)
    pd.DataFrame(cov).to_csv(out / "coverage.tsv", sep="\t", index=False)

    subsets = {"all": lambda d: d, "primary": lambda d: d[d.primary], "non-primary": lambda d: d[~d.primary],
               "A (inborn errors)": lambda d: d[d.collection == "A_inborn_errors"],
               "B-D (rest)": lambda d: d[d.collection != "A_inborn_errors"],
               **{c: (lambda d, c=c: d[d.collection == c]) for c in sorted(pu.collection.unique()) if c != "A_inborn_errors"}}
    rowsets = {"all units": lambda d: d, "causal scored by Reafect": lambda d: d[d.causal_scored_by_reafect]}
    pct_cols = [f"{m}_{s}_pct" for m in METHODS for s in ("own", "common")]

    def levels(d):
        return {"unit": d, "cluster": d.groupby("cluster")[pct_cols].median().reset_index()}

    srows, prow = [], []
    for rs, fr in rowsets.items():
        for sub, fs in subsets.items():
            for level, d in levels(fs(fr(pu))).items():
                for space in ("own", "common"):
                    for m in METHODS:
                        v = d[f"{m}_{space}_pct"]
                        if len(v):
                            srows.append({"rows": rs, "subset": sub, "level": level, "universe": space, "method": LABEL[m],
                                          "n": len(v), "median_pct": v.median(), "mean_pct": v.mean(),
                                          "top1": (v <= 1).mean() * 100, "top5": (v <= 5).mean() * 100,
                                          "top10": (v <= 10).mean() * 100})
                for m in METHODS[1:]:
                    a, b = d["reafect_common_pct"], d[f"{m}_common_pct"]
                    w, l = int((a < b).sum()), int((a > b).sum())
                    prow.append({"rows": rs, "subset": sub, "level": level, "versus": LABEL[m], "pairs": len(d),
                                 "reafect_better": w, "ours_better": l, "ties": len(d) - w - l,
                                 "sign_test_p": binomtest(w, w + l).pvalue if w + l else np.nan})
    summary, paired = pd.DataFrame(srows), pd.DataFrame(prow)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    paired.to_csv(out / "paired.tsv", sep="\t", index=False)
    write_units_report(out, pu, summary, paired, N, U, args)
    print("wrote", out)


def write_units_report(out, pu, summary, paired, N, U, args):
    def tab(rows, space, level):
        """Subsets as rows, methods as columns: median percentile (share in the top 10%)."""
        h = "| Subset | n | " + " | ".join(LABEL[m] for m in METHODS) + " |\n|---|---|" + "---|" * len(METHODS) + "\n"
        for sub in ("all", "primary", "non-primary", "A (inborn errors)", "B-D (rest)"):
            s = sel(rows, sub, level, space).set_index("method")
            if len(s):
                h += f"| {sub} | {int(s.n.iloc[0])} | " + " | ".join(
                    f"{s.loc[LABEL[m], 'median_pct']:.1f} ({s.loc[LABEL[m], 'top10']:.0f}%)" for m in METHODS) + " |\n"
        return h

    def sel(rows, subset, level, space):
        return summary[(summary.rows == rows) & (summary.subset == subset) & (summary.level == level) &
                       (summary.universe == space)]

    nu, nc = len(pu), pu.cluster.nunique()
    cov_mod, cov_sc = pu.causal_ec_in_reafect_modules, pu.causal_scored_by_reafect
    cl = pu.groupby("cluster")[["causal_ec_in_reafect_modules", "causal_scored_by_reafect"]].any()
    pa = pu.collection == "A_inborn_errors"
    chk = {m: int((pu[f"{m}_rank"] != pu[f"{m}_rank_reported"]).sum() - pu[f"{m}_rank_reported"].isna().sum())
           for m in ["walk_up", "walk_down", "flow", "neighbour"]}
    miss = {m: int(pu[f"{m}_rank_reported"].isna().sum()) for m in chk}
    L = ["# Reafect on our test set (D26)", "",
         f"Reafect (Bongaerts et al. 2022, doi 10.1016/j.ymgme.2022.05.002; a, b, c = {A_PARAM}, {B_PARAM}, {C_PARAM}) "
         f"against our rankings on the D26 test set: {nu} testable comparisons (units) in {nc} clusters "
         "(results/data_figures/units.tsv, testable == True). Percentile 0 is best. Cluster = median percentile of "
         "its units.", "",
         "## Reafect input", "",
         "- One z per metabolite, keyed by KEGG compound. A (inborn errors): the group z of group_daa "
         "(mean patient z x sqrt(patients measured)), uncapped. B, C, D: z = sign(log2FC) x Phi^-1(1 - p/2), "
         f"p floored at {P_FLOOR:g} (|z| <= 8.29). This z is a test statistic, not a per-patient deviation: it grows "
         "with sample size and, where the table's p is from a t-test on few samples, is not the z-score scale Reafect "
         "was tuned on.",
         "- All measured metabolites with a KEGG id, significant or not; one z per compound (largest |z|). "
         "No Bongaerts truncation: the notebook truncates only the Bongaerts per-patient file, not Miller's.",
         "- KEGG id: the table's KEGG column; else the Human-GEM metabolite (published mapping met_id, HMDB, ChEBI, "
         "name) and its metKEGGID (data/Human-GEM_metabolites.tsv); else a KEGG id from the name bridge "
         "(data/id_translation.csv); else an unambiguous loose name match.",
         f"- Measured metabolites per unit (median {pu.n_measured.median():.0f}); with a KEGG id: median "
         f"{(100 * pu.n_with_kegg / pu.n_measured).median():.0f}% (range {(100 * pu.n_with_kegg / pu.n_measured).min():.0f}-"
         f"{(100 * pu.n_with_kegg / pu.n_measured).max():.0f}%); distinct KEGG compounds in Reafect's KEGG modules: "
         f"median {pu.n_kegg_in_reafect_network.median():.0f} (range {pu.n_kegg_in_reafect_network.min()}-"
         f"{pu.n_kegg_in_reafect_network.max()}). Per unit: per_unit.tsv.", "",
         "## Coverage of our causal genes by Reafect", "",
         f"- Causal gene's EC in Reafect's KEGG modules: {int(cov_mod.sum())} of {nu} units "
         f"({int(cl.causal_ec_in_reafect_modules.sum())} of {nc} clusters); A: {int(cov_mod[pa].sum())} of {int(pa.sum())}, "
         f"B-D: {int(cov_mod[~pa].sum())} of {int((~pa).sum())}.",
         f"- Causal EC scored by Reafect in that unit: {int(cov_sc.sum())} of {nu} units "
         f"({int(cl.causal_scored_by_reafect.sum())} of {nc} clusters); A: {int(cov_sc[pa].sum())} of {int(pa.sum())}, "
         f"B-D: {int(cov_sc[~pa].sum())} of {int((~pa).sum())}.",
         "- Unscored causal genes count as Reafect's tail: own-universe percentile 100, common-universe tail mid-rank. "
         "Tables are given on all units and on the units where Reafect scores the causal gene. Per gene: coverage.tsv.",
         f"- Reafect runs that failed or produced no table: {int((~pu.reafect_ran).sum())} units.", "",
         "## Universes", "",
         "- **Own**: Reafect = 0-based rank of the causal EC among the ECs it scored for the unit / number scored "
         f"(as published); ours = causal gene's best rank / {U:,} network genes.",
         f"- **Common**: {N:,} genes = genes Reafect scored (via EC -> gene, Reafect's kegg_enzyme_with_gene_symbol) in "
         "at least one unit and in our network, plus every unit's causal genes in our network (so a causal gene "
         "Reafect cannot score is still ranked, in its tail). Every method re-ranked within it; unscored genes share "
         "the tail mid-rank.", ""]
    L += ["## Results", "", "Cells: median percentile (share of units or clusters with the causal gene in the top 10%). "
          "Mean, top 1% and top 5%, and per collection: summary.tsv.", ""]
    for rs in ("all units", "causal scored by Reafect"):
        for space in ("common", "own"):
            for level in ("unit", "cluster"):
                L += [f"### {space.capitalize()} universe, {level}s ({rs})", "", tab(rs, space, level), ""]
    L += ["## Paired: Reafect vs walk, sources = increased (common universe)", "",
          "| Rows | Subset | Level | Pairs | Reafect better | Walk better | Ties | Sign test p |", "|---|---|---|---|---|---|---|---|"]
    p = paired[paired.versus == LABEL["walk_up"]]
    for r in p[p.subset.isin(["all", "A (inborn errors)", "B-D (rest)", "primary", "non-primary"])].itertuples():
        L.append(f"| {r.rows} | {r.subset} | {r.level} | {r.pairs} | {r.reafect_better} | {r.ours_better} | {r.ties} | "
                 f"{r.sign_test_p:.3g} |")
    L += ["", "Two-sided sign test, ties dropped. Against the other methods: paired.tsv.", "",
          "## Our ranks: reproduction", "",
          "Full gene ranks recomputed (ours_ranks.tsv.gz; correction_eval's reference path: Ranker.run for A, the "
          "gene step on the unit's sources and targets otherwise). Causal ranks that differ from the "
          "results/correction_eval_d26_* rankings.tsv: " +
          ", ".join(f"{LABEL[m]} {chk[m]} (absent there: {miss[m]})" for m in chk) +
          ". A unit with no sources in an orientation has no ranking there: all genes share the mid-rank.", "",
          "## Not comparable / caveats", "",
          "- Reafect was built for per-patient z-scores against a reference population; our units are group "
          "comparisons (patients vs controls, knockout vs wild type, inhibitor vs vehicle) in plasma, CSF, urine, "
          "cells and tissue. KEGG modules cover core metabolism only.",
          "- Reafect ranks EC numbers; gene ranks take the best EC of the gene. Several genes share an EC.",
          f"- Reafect ran unmodified from its clone ({args.reafect}), KEGG downloaded in 2026, a shim restores "
          "ElementTree.getchildren for Python >= 3.9."]
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["reafect-download", "reafect-update", "reafect-prepare", "reafect-abc",
                                    "reafect-patients", "ours", "compare",
                                    "units-input", "reafect-units", "ours-units", "units-compare"])
    p.add_argument("--reafect", required=True, help="Reafect clone")
    p.add_argument("--work", default=None, help="Reafect working folder (maps, paths, results)")
    p.add_argument("--dataset", choices=DATASETS)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", default=str(OUT))
    args = p.parse_args(argv)
    if args.step == "reafect-units":
        run_reafect_units(str(Path(args.reafect).resolve()), str(Path(args.work).resolve()), args.workers)
    elif args.step.startswith("reafect-"):
        run_reafect(args.step.split("-", 1)[1], args)
    elif args.step == "ours":
        run_ours(args)
    elif args.step == "units-input":
        run_units_input(args)
    elif args.step == "ours-units":
        run_ours_units(args)
    elif args.step == "units-compare":
        run_units_compare(args)
    else:
        run_compare(args)


if __name__ == "__main__":
    main()
