"""Benchmark: can Metaboenrich find the known defective enzyme in inborn
errors of metabolism from untargeted plasma metabolomics?

Data: per-sample metabolite z-scores (vs reference populations) from
Thistlethwaite et al. 2020 and Miller et al. 2015 (Baylor), fetched with
benchmarks/get_benchmark_data.py. Cases and causal genes: benchmarks/cases.tsv.

Two levels:
  group    all patients with a disorder: mean z per metabolite, z-test on the
           mean (the values are already standardised), BH-adjusted;
           significant if q < 0.05 and |mean z| >= 1
  patient  each patient alone: significant if |z| >= 2

For each run the causal gene's rank is recorded among all genes on network
reactions, for Metaboenrich and for a neighbour baseline that ranks each
enzyme by how many significant metabolites its own reactions touch.

    python -m metaboenrich.benchmark --out results/benchmark
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .daa import load_daa
from .flow import current_flow
from .gem import load_sbml
from .network import DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits, resolve_currency
from .scoring import gene_table, node_table, reaction_table

DATA = Path("data/benchmark")
CASES = Path("benchmarks/cases.tsv")
T2020_ANN = ["SUPER_PATHWAY", "SUB_PATHWAY", "PLATFORM", "KEGG", "CAS", "RI", "MASS", "HMDB_ID"]


# ---- data ---------------------------------------------------------------------
def load_t2020():
    t = pd.read_csv(DATA / "thistlethwaite2020.csv", index_col=0, low_memory=False)
    samples = [c for c in t.columns if c not in T2020_ANN]
    z = t[samples].apply(pd.to_numeric, errors="coerce")
    ann = pd.DataFrame({"name": t.index, "HMDB": t["HMDB_ID"].fillna(""), "KEGG": t["KEGG"].fillna("")},
                       index=t.index)
    cohorts = pd.read_csv(DATA / "cohorts_coded.tsv", sep="\t").groupby("cohort")["sample"].apply(list).to_dict()
    return z, ann, cohorts


def load_miller():
    m = pd.read_csv(DATA / "miller2015.csv", index_col=0, low_memory=False)
    samples = [c for c in m.columns if c.startswith("IEM_")]
    dx = m.loc["diagnosis", samples]
    m = m.drop(index="diagnosis")
    z = m[samples].apply(pd.to_numeric, errors="coerce")
    hmdb_col = next(c for c in m.columns if "HMDB" in c)
    ann = pd.DataFrame({"name": m.index, "HMDB": m[hmdb_col].fillna(""), "KEGG": m["KEGG"].fillna("")},
                       index=m.index)
    cohorts = dx.groupby(dx).apply(lambda s: list(s.index)).to_dict()
    return z, ann, cohorts


def bh(p):
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    q[order] = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1.0)


def normal_p(z):
    return np.array([math.erfc(abs(v) / math.sqrt(2)) for v in z])


def group_daa(z, ann, samples):
    sub = z[samples]
    k = sub.notna().sum(axis=1)
    need = max(2, math.ceil(len(samples) / 2)) if len(samples) > 1 else 1
    mean = sub.mean(axis=1)
    keep = (k >= need) & mean.notna()
    zg = mean[keep] * np.sqrt(k[keep])
    p = normal_p(zg.values)
    return pd.DataFrame({"name": ann.loc[keep[keep].index, "name"], "HMDB": ann.loc[keep[keep].index, "HMDB"],
                         "KEGG": ann.loc[keep[keep].index, "KEGG"], "log2FC": mean[keep].values,
                         "p": bh(p) if len(p) else p})


def patient_daa(z, ann, sample):
    v = z[sample].dropna()
    return pd.DataFrame({"name": ann.loc[v.index, "name"], "HMDB": ann.loc[v.index, "HMDB"],
                         "KEGG": ann.loc[v.index, "KEGG"], "log2FC": v.values, "p": normal_p(v.values)})


# ---- one run ------------------------------------------------------------------
class Ranker:
    def __init__(self, model, graph):
        self.model, self.graph = model, graph
        self.universe = sorted(set().union(*graph.reaction_genes.values()))
        self._universe_set = set(self.universe)
        self.symbol_to_ens = {}
        for ens, sym in model.genes.items():
            self.symbol_to_ens.setdefault(sym, ens)
        self.rxn_nodes = np.flatnonzero(graph.is_reaction)
        self.rxn_mets = {r: [] for r in self.rxn_nodes}
        for m, r in graph.edges:
            self.rxn_mets[r].append(m)

    def run(self, daa_df, tmp_csv, alpha, min_fc):
        daa_df.to_csv(tmp_csv, index=False)
        daa, _, _ = load_daa(tmp_csv, self.model, ["HMDB", "KEGG", "name"], "log2FC", "p", alpha, False, min_fc)
        daa["node"] = daa.met_id.map(self.graph.index)
        on = daa.dropna(subset=["node"]).astype({"node": int})
        sources = on[on.role == "source"].node.tolist()
        targets = on[on.role == "target"].node.tolist()
        info = {"mapped": len(daa), "sig_mapped": int(daa.role.isin(["source", "target"]).sum()),
                "sources": len(sources), "targets": len(targets), "pairs": 0}

        # Neighbour baseline: significant metabolites touched by each reaction.
        sig = dict(zip(on[on.role.isin(["source", "target"])].node, on[on.role.isin(["source", "target"])].log2fc.abs()))
        nb = {}
        for r, mets in self.rxn_mets.items():
            hits = [sig[m] for m in mets if m in sig]
            if hits:
                s = len(hits) + sum(hits) / 1000.0
                for g in self.graph.reaction_genes[r]:
                    nb[g] = max(nb.get(g, 0.0), s)
        nb_rank = pd.Series({g: nb.get(g, 0.0) for g in self.universe}).rank(ascending=False, method="average")

        ranks = {"loo": None, "raw": None}
        genes = pd.DataFrame()
        ground = None
        if bool(sources) != bool(targets):          # one-sided: exchange current with unchanged metabolites
            ground = on[on.role == ""].node.tolist()
        info["one_sided"] = bool(ground)
        if (sources and targets) or ground:
            flow = current_flow(self.graph, sources, targets, ground=ground)
            info["pairs"] = len(flow.pairs)
            if flow.pairs:
                nodes = node_table(self.graph, flow)
                for key, cols in (("loo", ["conductivity_loo", "bottleneck_loo"]),
                                  ("raw", ["conductivity", "bottleneck"])):
                    rxn = reaction_table(nodes, self.graph, self.model, set(sources) | set(targets), cols)
                    g_df = gene_table(rxn, self.graph, self.model)
                    if key == "loo":
                        genes = g_df
                    scored = [g for g in g_df.ensembl_id if g in self._universe_set]
                    rank = {g: i + 1 for i, g in enumerate(scored)}
                    tail = (len(scored) + 1 + len(self.universe)) / 2.0
                    ranks[key] = pd.Series({g: rank.get(g, tail) for g in self.universe})
        return info, ranks, nb_rank, genes

    def causal(self, symbols):
        ens = [self.symbol_to_ens.get(s) for s in symbols]
        in_model = [e for e in ens if e]
        in_net = [e for e in in_model if e in self._universe_set]
        return in_model, in_net


def best_rank(rank_series, genes):
    if rank_series is None or not genes:
        return None
    return float(min(rank_series[g] for g in genes))


# ---- main ---------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(prog="metaboenrich.benchmark", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="data/Human-GEM.xml")
    p.add_argument("--out", default="results/benchmark")
    p.add_argument("--levels", nargs="+", default=["group", "patient"], choices=["group", "patient"])
    p.add_argument("--include", nargs="+", default=["primary", "secondary"],
                   help="case statuses to run (default: primary secondary)")
    args = p.parse_args(argv)
    t0 = time.time()
    out = Path(args.out)
    (out / "group_genes").mkdir(parents=True, exist_ok=True)
    tmp = out / "_daa_tmp.csv"

    model = load_sbml(args.model)
    rules, _ = resolve_currency(model, "role")
    rules.edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, model)
    graph = build_graph(model, rules)
    ranker = Ranker(model, graph)
    U = len(ranker.universe)
    print(f"Network: {graph.n} nodes; {U} genes on network reactions (the ranking universe)")

    data = {"t2020": load_t2020(), "miller": load_miller()}
    cases = pd.read_csv(CASES, sep="\t").fillna("")
    cases = cases[cases.status.isin(args.include)]

    rows_g, rows_p = [], []
    for _, c in cases.iterrows():
        z, ann, cohorts = data[c.dataset]
        samples = [s for s in cohorts.get(c.cohort, []) if s in z.columns]
        symbols = c.causal_genes.split(";")
        in_model, in_net = ranker.causal(symbols)
        base = {"case": c.case_id, "disorder": c.disorder, "status": c.status, "category": c.category,
                "causal_genes": c.causal_genes, "patients": len(samples),
                "causal_in_model": ";".join(model.genes[e] for e in in_model),
                "causal_in_network": ";".join(model.genes[e] for e in in_net), "universe": U}

        fmt = lambda v: "-" if v is None else f"{v:.0f}"
        if "group" in args.levels and samples:
            info, me, nb, genes = ranker.run(group_daa(z, ann, samples), tmp, 0.05, 1.0)
            r = {"rank_metaboenrich": best_rank(me["loo"], in_net), "rank_metaboenrich_raw": best_rank(me["raw"], in_net),
                 "rank_neighbour": best_rank(nb, in_net)}
            rows_g.append({**base, **info, **r})
            if len(genes):
                genes.head(50).to_csv(out / "group_genes" / f"{c.case_id}.tsv", sep="\t", index=False)
            print(f"[group]   {c.case_id:6s} n={len(samples):3d} src={info['sources']:3d} tgt={info['targets']:3d} "
                  f"pairs={info['pairs']:5d}  rank ME(loo)={fmt(r['rank_metaboenrich']):>5s} "
                  f"ME(raw)={fmt(r['rank_metaboenrich_raw']):>5s}  NB={fmt(r['rank_neighbour']):>6s}")

        if "patient" in args.levels:
            for s in samples:
                info, me, nb, _ = ranker.run(patient_daa(z, ann, s), tmp, 0.0456, 0.0)
                rows_p.append({**base, "sample": s, **info,
                               "rank_metaboenrich": best_rank(me["loo"], in_net),
                               "rank_metaboenrich_raw": best_rank(me["raw"], in_net),
                               "rank_neighbour": best_rank(nb, in_net)})
            if samples:
                done = [r for r in rows_p if r["case"] == c.case_id]
                med = lambda k: np.median([r[k] for r in done if r[k] is not None]) if any(r[k] is not None for r in done) else None
                n_run = sum(r["rank_metaboenrich"] is not None for r in done)
                print(f"[patient] {c.case_id:6s} {len(done):3d} patients, {n_run:3d} runnable, median rank "
                      f"ME(loo)={fmt(med('rank_metaboenrich')):>5s} ME(raw)={fmt(med('rank_metaboenrich_raw')):>5s} "
                      f"NB={fmt(med('rank_neighbour')):>6s}")
    tmp.unlink(missing_ok=True)

    for name, rows in (("group", rows_g), ("patient", rows_p)):
        if rows:
            df = pd.DataFrame(rows)
            for m in ("metaboenrich", "metaboenrich_raw", "neighbour"):
                df[f"pct_{m}"] = df[f"rank_{m}"] / U
            df.to_csv(out / f"{name}_results.tsv", sep="\t", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "network_nodes": graph.n, "gene_universe": U, "runtime_s": round(time.time() - t0, 1),
        "group_rule": "BH q < 0.05 on z-test of mean z, and |mean z| >= 1",
        "patient_rule": "|z| >= 2", "scoring": "leave-one-out conductivity + bottleneck"}, indent=2))
    print(f"\nWrote {out}/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
