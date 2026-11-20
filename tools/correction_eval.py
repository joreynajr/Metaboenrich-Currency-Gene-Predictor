"""Do published degree-bias corrections move known causal genes up the ranking?

Compared with no correction (M0, the negative control), on the null calibration
run's units (its method, walk or current flow, and orientation) that have a known causal gene in the network. Every correction acts
on `conductivity` and `bottleneck` separately, the two are combined with the
same rank_score, and each gene takes its best reaction as in Ranker.run. The set
of scored reactions (conductivity > 0) and the tail rank for unscored genes are
the same for every method, so corrections only reorder.

  M0       no correction
  M1-D22   seed-degree reference model, z from the run's null (DADA alpha_SD);
           undefined z keeps its raw score (D22 item 4)
  M1-last  as M1, undefined z ranked below every defined z
  M2       candidate-degree reference model: z against the 1,000 reaction nodes
           of nearest hub_sum in the same run (DADA alpha_CD); M2-hub_max as a sensitivity run
  M3       empirical p = (b + 1) / (perms + 1) from the same permutations (Phipson and Smyth 2010)
  UNI-C, UNI-O, UNI-S   DADA's uniform prioritization of M0 with each of M1-D22, M1-last, M2, M3
  M0-LOO   current flow only: the benchmark's leave-one-out scores, uncorrected (context)

No method is adopted here (handoff_correction_methods_evaluation.md).

    python tools/correction_eval.py
Writes results/correction_eval/: report.md, rankings.tsv, summary.tsv, paired.tsv, hub_genes.tsv, bias.tsv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from metaboenrich import benchmark as B, structure              # noqa: E402
from metaboenrich.daa import load_name_bridge                    # noqa: E402
from metaboenrich.flow import current_flow                       # noqa: E402
from metaboenrich.gem import load_sbml                           # noqa: E402
from metaboenrich.network import build_graph                     # noqa: E402
from metaboenrich.scoring import gene_table, node_table, rank_score, reaction_table   # noqa: E402
from metaboenrich.walk import absorbing_walk                     # noqa: E402
from null_calibration import KEYS, SCORES, Calibrator, registry_units   # noqa: E402

RUN = Path("results/null_calibration/three_rule_decreased_measurable_kNone_n1000_exceed")
UNIT_MASTER = Path.home() / "Desktop/Electrical Conductivity/unit_master_v2.csv"
PRIMARY = {"inherited defect", "gene removed or lost"}
# One experiment registered as two studies.
MERGED = {"ST004490": "TIGAR (ST004490/ST004493)", "ST004493": "TIGAR (ST004490/ST004493)",
          "ST003551": "ALDH7A1 (ST003551/ST003556)", "ST003556": "ALDH7A1 (ST003551/ST003556)"}
N_NEIGHBOURS = 1000                 # DADA's n for the candidate-degree model
ADJUSTED = ["M1-D22", "M1-last", "M2", "M3"]
UNI = ["UNI-C", "UNI-O", "UNI-S"]
RANKINGS = ["M0"] + ADJUSTED + [f"{u}/{a}" for a in ADJUSTED for u in UNI]
SENSITIVITY = ["M2-hub_max"]
CONTEXT = ["M0-LOO", "neighbour", "random"]


# ---- corrections (pure functions, unit-tested) ---------------------------------
def pct(x):
    """Percentile rank in (0, 1], ties averaged, as Series.rank(pct=True)."""
    x = np.asarray(x, float)
    return rankdata(x) / len(x) if len(x) else x


def lex_pct(primary, secondary):
    """Percentile rank by primary, ties broken by secondary."""
    key = rankdata(primary, method="dense") * (len(primary) + 1) + rankdata(secondary, method="dense")
    return pct(key)


def m1_d22(z, raw):
    """D22 item 4 as a percentile over scored nodes: defined z -> percentile of z among
    defined z, times the share of defined z; undefined z -> percentile of the raw score
    among all scored nodes."""
    d = ~np.isnan(z)
    out = pct(raw)
    if d.any():
        out[d] = pct(z[d]) * d.mean()
    return out


def m1_last(z, raw):
    """Undefined z below every defined z; each group ordered by its own score."""
    d = ~np.isnan(z)
    out = np.empty(len(z))
    n_u = (~d).sum()
    out[~d] = rankdata(raw[~d]) / len(z)
    out[d] = (n_u + rankdata(z[d])) / len(z)
    return out


def nearest(degree, n=N_NEIGHBOURS):
    """Per node, the n nodes of nearest degree (itself included; ties at the edge by node order)."""
    degree = np.asarray(degree, float)
    n = min(n, len(degree))
    return np.stack([np.argsort(np.abs(degree - d), kind="stable")[:n] for d in degree])


def m2_z(obs, nbrs):
    """DADA alpha_CD: score against the mean and SD (n - 1) of the nodes of similar degree."""
    pop = obs[nbrs]
    sd = pop.std(axis=1, ddof=1)
    return np.where(sd > 0, (obs - pop.mean(axis=1)) / np.where(sd > 0, sd, 1.0), 0.0)


def phipson_smyth(b, perms):
    return (np.asarray(b, float) + 1) / (perms + 1)


def ordinal(score, tiebreak):
    """Rank 1 = best: by score descending, then tiebreak descending (reaction_table's order)."""
    order = np.lexsort((-np.asarray(tiebreak, float), -np.asarray(score, float)))
    r = np.empty(len(order), dtype=int)
    r[order] = np.arange(1, len(order) + 1)
    return r


def uniform(r_raw, r_adj, rule, cand_degree=None, lam=None, seed_degree=None, net_degree=None):
    """DADA's merged ranking mapped back to 1..|C|. UNI-C: raw rank for candidates of
    degree > lam, else adjusted; UNI-O: the better of the two; UNI-S: raw for all if the
    seeds' mean degree > the network's, else adjusted. Ties broken by the unused ranking."""
    r_raw, r_adj = np.asarray(r_raw), np.asarray(r_adj)
    if rule == "UNI-S":
        return (r_raw if seed_degree > net_degree else r_adj).copy()
    if rule == "UNI-C":
        use_raw = np.asarray(cand_degree) > lam
    elif rule == "UNI-O":
        use_raw = r_raw < r_adj
    else:
        raise ValueError(rule)
    used, unused = np.where(use_raw, r_raw, r_adj), np.where(use_raw, r_adj, r_raw)
    order = np.lexsort((unused, used))
    out = np.empty(len(order), dtype=int)
    out[order] = np.arange(1, len(order) + 1)
    return out


def rank_mismatch(a, b):
    """Genes whose rank differs between two rankings (the M0 reproduction check)."""
    a, b = pd.Series(a), pd.Series(b)
    idx = a.index.union(b.index)
    a, b = a.reindex(idx), b.reindex(idx)
    return sorted(idx[~((a == b) | (a.isna() & b.isna()))])


def p_auc(ranks, universe, k_max=100):
    """DADA's ROC with ranks 1..k_max as predicted positives, one causal gene per unit:
    sensitivity = share of units with the causal gene at rank <= k, 1 - specificity =
    share of the other genes at rank <= k. Area up to k_max, divided by its width."""
    r = np.asarray(ranks, float)
    ks = np.arange(0, k_max + 1)
    tpr = np.array([(r <= k).mean() for k in ks])
    fpr = np.array([np.mean((k - (r <= k)) / (universe - 1)) for k in ks])
    return float(np.trapezoid(tpr, fpr) / fpr[-1])


# ---- per unit ------------------------------------------------------------------
class Evaluator:
    def __init__(self, model, graph, bridge, cal, method="walk", orientation="decreased"):
        self.model, self.graph, self.cal, self.method = model, graph, cal, method
        self.ranker = B.Ranker(model, graph, method, None, bridge, orientation)
        self.universe = self.ranker.universe
        self.hub_sum = cal.base.hub_sum.to_numpy(float)
        self.nbrs = {"hub_sum": nearest(self.hub_sum), "hub_max": nearest(cal.base.hub_max.to_numpy(float))}
        self.lam = self.hub_sum.mean()
        self.net_degree = cal.degree[cal.all_mets].mean()
        pos = {r: i for i, r in enumerate(cal.rxn)}
        g_rxn = {}
        for r in cal.rxn:
            for g in graph.reaction_genes[r]:
                g_rxn.setdefault(g, []).append(pos[r])
        self.gene_rxn = g_rxn
        self.gene_degree = pd.Series({g: self.hub_sum[g_rxn[g]].mean() for g in self.universe})
        self.gene_nodes = pd.Series({g: len(g_rxn[g]) for g in self.universe})
        met_deg = cal.degree[cal.all_mets]
        self.hub_mets = set(cal.all_mets[met_deg >= np.quantile(met_deg, 0.99)])

    def gene_ranks(self, rxn, score):
        """Ranker.run's gene step for a given reaction final score."""
        t = rxn.assign(final_score=score).sort_values(["final_score", "conductivity"], ascending=False)
        g = gene_table(t, self.graph, self.model)
        scored = [x for x in g.ensembl_id if x in self.ranker._universe_set] if len(g) else []
        rank = {x: i + 1 for i, x in enumerate(scored)}
        tail = (len(scored) + 1 + len(self.universe)) / 2.0
        return pd.Series({x: rank.get(x, tail) for x in self.universe}), g

    def reference(self, u):
        """Ranker.run for this unit's sources and targets: the walk or flow, the gene ranks, the neighbour baseline."""
        src, tgt, phi, ground = u["sources"], u["targets"], u["phi"], u["ground"]
        if self.method == "flow":
            flow = current_flow(self.graph, src, tgt, ground=ground)
        else:
            flow = absorbing_walk(self.graph, src, tgt, phi, kappa=None, ground=ground)
        nodes = node_table(self.graph, flow)
        rxn = reaction_table(nodes, self.graph, self.model, set(src) | set(tgt), ["conductivity", "bottleneck"])
        ranks, _ = self.gene_ranks(rxn, rxn.final_score)
        sig = {m: phi[m] for m in set(src) | set(tgt)}
        nb = {}
        for r, mets in self.ranker.rxn_mets.items():
            hits = [sig[m] for m in mets if m in sig]
            if hits:
                s = len(hits) + sum(hits) / 1000.0
                for g in self.graph.reaction_genes[r]:
                    nb[g] = max(nb.get(g, 0.0), s)
        nb_rank = pd.Series({g: nb.get(g, 0.0) for g in self.universe}).rank(ascending=False, method="average")
        return nodes, ranks, nb_rank

    def loo_ranks(self, nodes, u):
        """Current flow's leave-one-out gene ranks, as the benchmark's default flow ranking."""
        rxn = reaction_table(nodes, self.graph, self.model, set(u["sources"]) | set(u["targets"]),
                             ["conductivity_loo", "bottleneck_loo"])
        return self.gene_ranks(rxn, rxn.final_score)[0]

    def m0_ranks(self, nodes, cond, bott, decimals=None):
        """M0 gene ranks with the given reaction scores (optionally rounded) in place of the walk's."""
        t = nodes.copy()
        t.loc[self.cal.rxn, "conductivity"] = cond if decimals is None else np.round(cond, decimals)
        t.loc[self.cal.rxn, "bottleneck"] = bott if decimals is None else np.round(bott, decimals)
        rxn = reaction_table(t, self.graph, self.model, set(), ["conductivity", "bottleneck"])
        return self.gene_ranks(rxn, rxn.final_score)[0]

    def rankings(self, u, un, perms, nodes, has_exceed):
        """Reaction final scores per ranking over the scored reactions, then gene ranks."""
        obs = {s: un[f"observed_{s}"].to_numpy() for s in SCORES}
        S = obs["conductivity"] > 0
        rxn_idx = self.cal.rxn[S]
        rxn = nodes.loc[rxn_idx].copy()
        rxn["genes"] = ""
        for s in SCORES:                                   # the run's stored scores, not the fresh walk's
            rxn[s] = obs[s][S]
        raw = {s: obs[s][S] for s in SCORES}
        col = {"M0": {s: pct(raw[s]) for s in SCORES}}
        z = {s: un[f"z_{s}"].to_numpy()[S] for s in SCORES}
        col["M1-D22"] = {s: m1_d22(z[s], raw[s]) for s in SCORES}
        col["M1-last"] = {s: m1_last(z[s], raw[s]) for s in SCORES}
        for hub, name in (("hub_sum", "M2"), ("hub_max", "M2-hub_max")):
            col[name] = {s: pct(m2_z(obs[s], self.nbrs[hub])[S]) for s in SCORES}
        if has_exceed:
            col["M3"] = {s: lex_pct(1 - phipson_smyth(un[f"n_null_ge_observed_{s}"].to_numpy()[S], perms), raw[s])
                         for s in SCORES}
        final = {m: rank_score(pd.DataFrame(c), list(SCORES)).to_numpy() for m, c in col.items()}
        r_raw = ordinal(final["M0"], raw["conductivity"])
        seed_deg = self.cal.degree[u["sources"]].mean()
        for a in [a for a in ADJUSTED if a in final]:
            r_adj = ordinal(final[a], raw["conductivity"])
            for rule in UNI:
                r = uniform(r_raw, r_adj, rule, self.hub_sum[S], self.lam, seed_deg, self.net_degree)
                final[f"{rule}/{a}"] = (len(r) - r + 1) / len(r)
        out = {}
        for m, f in final.items():
            ranks, g = self.gene_ranks(rxn, f)
            out[m] = {"ranks": ranks, "genes": g,
                      "rho_hub": spearmanr(f, self.hub_sum[S]).correlation,
                      "rho_nodes": spearmanr(ranks.to_numpy(), self.gene_nodes[ranks.index].to_numpy()).correlation}
        return out, S, z


def best(ranks, genes):
    return float(min(ranks[g] for g in genes))


def cluster_of(r):
    if r.collection == "A_inborn_errors":
        return r.label
    return MERGED.get(r.study, r.study)


# ---- main ----------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=str(RUN))
    p.add_argument("--unit-master", default=str(UNIT_MASTER))
    p.add_argument("--out", default="results/correction_eval")
    p.add_argument("--check-only", action="store_true", help="run the M0 reproduction check only")
    args = p.parse_args(argv)
    t0 = time.time()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((run / "run_info.json").read_text())
    perms = info["perms"]
    units_tsv = pd.read_csv(run / "units.tsv", sep="\t")
    nodes_tsv = pd.read_csv(run / "nodes.tsv", sep="\t", low_memory=False)
    has_exceed = "n_null_ge_observed_conductivity" in nodes_tsv.columns
    reg = pd.read_csv(Path("benchmarks/data_registry.tsv"), sep="\t", dtype=str).fillna("")
    um = pd.read_csv(args.unit_master, dtype=str).fillna("")

    model = load_sbml("data/Human-GEM.xml")
    sset, key = info["stage"].split(":", 1)
    st = next(s for s in structure.stages(model, sset, atom_dir="data") if s[0] == key)
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    bridge = load_name_bridge("data/id_translation.csv")
    method = info.get("method", "walk")
    cal = Calibrator(graph, None, method)
    ev = Evaluator(model, graph, bridge, cal, method, info["orientation"])
    scorer = "current_flow" if method == "flow" else "absorbing_walk"
    U = len(ev.universe)
    units, _ = registry_units(model, graph, bridge, info["orientation"])
    units = {u["label"]: u for u in units}

    ran = units_tsv[units_tsv.status == "run"].merge(
        reg[["collection", "study", "sub_unit", "causal_genes"]], on=["collection", "study", "sub_unit"], how="left")
    ran = ran.merge(um[["label", "perturbation"]], on="label", how="left")
    ran["perturbation"] = ran.perturbation.fillna("")
    ran.loc[ran.study == "MTBLS3628", "perturbation"] = "gene removed or lost"     # registered after unit_master_v2's run
    ran["causal_genes"] = ran.causal_genes.fillna("")
    ran["causal_in_net"] = [ev.ranker.causal([s for s in c.split(";") if s])[1] for c in ran.causal_genes]
    ran["cluster"] = [cluster_of(r) for r in ran.itertuples()]
    ran["primary"] = ran.perturbation.isin(PRIMARY)
    ran["viable"] = ran.viable.astype(str).eq("True")
    skipped = ran[(ran.causal_genes != "") & (ran.causal_in_net.str.len() == 0)]
    ev_units = ran[ran.causal_in_net.str.len() > 0].reset_index(drop=True)

    cases, data = pd.read_csv(B.CASES, sep="\t").fillna("").set_index("case_id"), {}
    rank_rows, unit_rows, bias_rows, hub_rows, m0_rows = [], [], [], [], []
    for r in ev_units.itertuples():
        u = units[r.label]
        un = nodes_tsv[nodes_tsv.unit == r.label].reset_index(drop=True)
        assert (un.node_id.to_numpy() == cal.base.node_id.to_numpy()).all()
        nodes, ref_ranks, nb_rank = ev.reference(u)
        ref_nodes, path = nodes, "Ranker.run's gene step on the unit's sources and targets"
        if r.collection == "A_inborn_errors":           # the benchmark's own entry point; keep its walk
            c = cases.loc[r.sub_unit]
            if c.dataset not in data:
                data[c.dataset] = B.load_t2020() if c.dataset == "t2020" else B.load_miller()
            z_, ann, cohorts = data[c.dataset]
            samples = [s for s in cohorts.get(c.cohort, []) if s in z_.columns]
            fn, walks = getattr(B, scorer), []
            setattr(B, scorer, lambda *a, **k: walks.append(fn(*a, **k)) or walks[-1])
            try:
                _, me, nb, _ = ev.ranker.run(B.group_daa(z_, ann, samples), out / "_daa_tmp.csv", 0.05, 1.0)
            finally:
                setattr(B, scorer, fn)
            assert not rank_mismatch(nb_rank, nb)
            if me["raw"] is None:
                path = f"Ranker.run returns no ranking (walk pairs: {len(walks[0].pairs) if walks else 'no walk'}); " + path
            else:
                ref_ranks, ref_nodes, path = me["raw"], node_table(graph, walks[0]), "Ranker.run"
        obs = [un[f"observed_{s}"].to_numpy() for s in SCORES]
        ref = [ref_nodes.loc[cal.rxn, s].to_numpy() for s in SCORES]
        m0_tsv = ev.m0_ranks(nodes, *obs)
        exact = rank_mismatch(m0_tsv, ref_ranks)
        rounded = rank_mismatch(ev.m0_ranks(nodes, *obs, decimals=12), ev.m0_ranks(ref_nodes, *ref, decimals=12))
        genes = r.causal_in_net
        m0_rows.append({"unit": r.label, "reference": path,
                        "max_abs_score_diff": max(float(np.abs(a - b).max()) for a, b in zip(obs, ref)),
                        "scored_differs": int(sum(((a > 0) != (b > 0)).sum() for a, b in zip(obs, ref))),
                        "genes_rank_differs_exact": len(exact), "genes_rank_differs_12_decimals": len(rounded),
                        "causal_rank_nodes_tsv": best(m0_tsv, genes), "causal_rank_reference": best(ref_ranks, genes)})
        print(f"{r.label[:50]:50s} exact {len(exact):4d}  12 decimals {len(rounded):3d}  causal "
              f"{best(m0_tsv, genes):7.1f} vs {best(ref_ranks, genes):7.1f}", flush=True)
        if args.check_only:
            continue
        res, S, z = ev.rankings(u, un, perms, nodes, has_exceed)
        m0_genes = res["M0"]["genes"].set_index("ensembl_id") if len(res["M0"]["genes"]) else pd.DataFrame()
        for g in genes:                                  # hub-metabolite causal genes, via the M0 best reaction
            if g in m0_genes.index:
                node = ev.graph.index[m0_genes.loc[g, "best_reaction"].split(" ")[0]]
                rxns = [node]
            else:
                rxns = [cal.rxn[i] for i in ev.gene_rxn[g]]
            mets = {m for rr in rxns for m in ev.ranker.rxn_mets[rr]} & ev.hub_mets
            if mets:
                hub_rows.append({"unit": r.label, "gene": model.genes[g],
                                 "hub_metabolites": ";".join(sorted(graph.node_names[m] for m in mets)),
                                 **{m: res[m]["ranks"][g] for m in res}})
        ranks = {m: best(res[m]["ranks"], genes) for m in res}
        if method == "flow":
            ranks["M0-LOO"] = best(ev.loo_ranks(nodes, u), genes)
        ranks["neighbour"] = best(nb_rank, genes)
        ranks["random"] = (U + 1) / 2
        rank_rows += [{"unit": r.label, "ranking": m, "rank": v} for m, v in ranks.items()]
        bias_rows += [{"unit": r.label, "ranking": m, "rho_final_vs_hub_sum": res[m]["rho_hub"],
                       "rho_gene_rank_vs_n_nodes": res[m]["rho_nodes"]} for m in res]
        unit_rows.append({**{k: getattr(r, k) for k in KEYS}, "label": r.label, "viable": r.viable,
                          "primary": r.primary, "perturbation": r.perturbation, "cluster": r.cluster,
                          "causal_genes": r.causal_genes, "causal_in_network": ";".join(model.genes[g] for g in genes),
                          "gene_degree": ev.gene_degree[genes].mean(), "n_sources": r.n_sources,
                          "scored_reactions": int(S.sum()),
                          "undefined_z_scored_conductivity": float(np.isnan(z["conductivity"]).mean()),
                          "undefined_z_scored_bottleneck": float(np.isnan(z["bottleneck"]).mean()),
                          "uni_s_uses_raw": bool(cal.degree[u["sources"]].mean() > ev.net_degree)})
    (out / "_daa_tmp.csv").unlink(missing_ok=True)
    m0 = pd.DataFrame(m0_rows)
    m0.to_csv(out / "m0_check.tsv", sep="\t", index=False)
    if args.check_only:
        print(f"M0 check: {(m0.genes_rank_differs_exact > 0).sum()} of {len(m0)} units differ exactly, "
              f"{(m0.genes_rank_differs_12_decimals > 0).sum()} at 12 decimals; wrote {out / 'm0_check.tsv'}")
        return

    uinfo = pd.DataFrame(unit_rows)
    rk = pd.DataFrame(rank_rows).merge(uinfo, left_on="unit", right_on="label")
    rk.to_csv(out / "rankings.tsv", sep="\t", index=False)
    bias = pd.DataFrame(bias_rows)
    bias.to_csv(out / "bias.tsv", sep="\t", index=False)
    hub = pd.DataFrame(hub_rows)
    hub.to_csv(out / "hub_genes.tsv", sep="\t", index=False)
    run_info = {"run": run.as_posix(), "method": method, "orientation": info["orientation"], "perms": perms, "has_exceedance": has_exceed, "universe": U,
                "reaction_nodes": len(cal.rxn), "lambda_hub_sum": ev.lam, "network_metabolite_mean_degree": ev.net_degree,
                "hub_metabolite_degree_min": float(min(cal.degree[m] for m in ev.hub_mets)),
                "n_hub_metabolites": len(ev.hub_mets), "gene_degree_mean": float(ev.gene_degree.mean()),
                "runtime_s": round(time.time() - t0, 1)}
    (out / "run_info.json").write_text(json.dumps(run_info, indent=2))
    write_report(out, rk, uinfo, bias, hub, m0, skipped, run_info, ran)
    print(f"wrote {out} in {time.time() - t0:.0f}s")


# ---- report --------------------------------------------------------------------
def order(rk):
    present = set(rk.ranking)
    return [m for m in RANKINGS + SENSITIVITY + CONTEXT if m in present]


def sets(rk):
    return [("primary", rk[rk.primary]), ("primary, viable", rk[rk.primary & rk.viable]),
            ("secondary", rk)]


def metric_rows(rk, U):
    rows = []
    for set_name, g in sets(rk):
        for m in order(g):
            x = g[g.ranking == m]["rank"].to_numpy()
            rows.append({"set": set_name, "ranking": m, "units": len(x),
                         "clusters": g[g.ranking == m].cluster.nunique(), "median_rank": np.median(x),
                         "top10": int((x <= 10).sum()), "top100": int((x <= 100).sum()),
                         "pAUC_1_100": p_auc(x, U) if m != "random" else 50 / (U - 1)})   # random: the diagonal, FPR(100) / 2
    return pd.DataFrame(rows)


def paired_rows(rk):
    rows = []
    for set_name, g in sets(rk)[:2]:
        m0 = g[g.ranking == "M0"].set_index("unit")["rank"]
        for m in [m for m in order(g) if m not in ("M0", "random")]:
            x = g[g.ranking == m].set_index("unit")
            d = x["rank"] - m0[x.index]
            cl = d.groupby(x.cluster).median()
            try:
                pw = wilcoxon(cl).pvalue if (cl != 0).any() else np.nan
            except ValueError:
                pw = np.nan
            rows.append({"set": set_name, "ranking": m, "improved": int((d < 0).sum()), "worse": int((d > 0).sum()),
                         "unchanged": int((d == 0).sum()), "median_change": d.median(),
                         "clusters": len(cl), "cluster_median_change": cl.median(), "wilcoxon_p_clusters": pw})
    return pd.DataFrame(rows)


def md(df, floatfmt=None):
    floatfmt = floatfmt or {}
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = []
        for c, v in zip(cols, r):
            if isinstance(v, float):
                cells.append("-" if pd.isna(v) else format(v, floatfmt.get(c, ".1f")))
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def write_report(out, rk, uinfo, bias, hub, m0, skipped, info, ran):
    U = info["universe"]
    summary = metric_rows(rk, U)
    summary.to_csv(out / "summary.tsv", sep="\t", index=False)
    paired = paired_rows(rk)
    paired.to_csv(out / "paired.tsv", sep="\t", index=False)
    pv = rk[rk.primary & rk.viable]
    L = ["# Degree-bias corrections vs no correction: causal-gene ranks", "",
         f"Null calibration run `{info['run']}` ({info['perms']} permutations, seed 0). Gene universe {U}; "
         f"rank 1 = best; unscored genes take the tail rank. Multi-gene units: the best-ranked causal gene in the "
         f"network (the benchmark's `best_rank`). No method is adopted by this report.", ""]
    L += ["## 0. M0 reproduction check", ""]
    fail = m0[m0.genes_rank_differs_exact > 0]
    if len(fail):
        causal = (m0.causal_rank_nodes_tsv != m0.causal_rank_reference).sum()
        L += [f"**Not exact.** Gene ranks rebuilt from `nodes.tsv` differ from the reference in {len(fail)} of "
              f"{len(m0)} units ({fail.genes_rank_differs_exact.min()}-{fail.genes_rank_differs_exact.max()} genes, "
              f"by a few places). Reaction scores agree to {m0.max_abs_score_diff.max():.1g} and the scored set is "
              f"identical in every unit; the causal-gene rank differs in {causal} units. Cause: many scored reactions "
              "tie exactly in theory, and float noise at 1e-16 breaks those ties differently in the runner's "
              "prepared walk and Ranker.run's unprepared one (Ranker.run itself is deterministic). With scores "
              f"rounded to 12 decimals, {(m0.genes_rank_differs_12_decimals == 0).sum()} of {len(m0)} units match "
              "exactly. Accepted as the negative control by DM (2026-11-17, option 1): every ranking below is built "
              "from `nodes.tsv`, so methods are compared on the same scores. Per unit: `m0_check.tsv`.", ""]
        no_rank = m0[m0.reference.str.startswith("Ranker.run returns no ranking")]
        for x in no_rank.itertuples():
            L += [f"- {x.unit}: Ranker.run finds no walk pairs and returns no ranking, but the calibration ran it; "
                  "compared with Ranker.run's gene step on the calibration's walk instead.", ""]
    else:
        L += [f"Passed for all {rk.unit.nunique()} units: gene ranks rebuilt from `nodes.tsv` (`observed_conductivity`, "
              "`observed_bottleneck`) equal Ranker.run's ranks for the same sources and targets; for the "
              f"{(uinfo.collection == 'A_inborn_errors').sum()} inborn-error units also Ranker.run itself on the "
              "group DAA, including the neighbour baseline."]
    L += ["", "## 1. Main table", "",
          "pAUC_1_100: DADA's ROC with ranks 1-100 as predicted positives (sensitivity = share of units with the "
          "causal gene in the top k; 1 - specificity = share of the other genes in the top k), area over k = 0..100 "
          "divided by its width, so random placement = FPR(100) / 2 = 50 / (U - 1), about 0.023.", ""]
    for set_name in ["primary, viable", "primary"]:
        L += [f"### {set_name}", ""] + md(summary[summary.set == set_name].drop(columns="set"),
                                          {"pAUC_1_100": ".3f"}) + [""]
    L += ["## 2. Paired comparison with M0", "",
          "Rank change = method - M0 (negative = better). Wilcoxon signed-rank on one value per independence cluster "
          "(median change within the cluster). With this few clusters only a large effect is detectable; units that "
          "share samples are never counted twice.", ""]
    for set_name in ["primary, viable", "primary"]:
        L += [f"### {set_name}", ""] + md(paired[paired.set == set_name].drop(columns="set"),
                                          {"wilcoxon_p_clusters": ".3f"}) + [""]
    L += ["## 3. Degree strata", "",
          f"Gene degree = mean hub_sum of the gene's reaction nodes (multi-gene units: mean over the causal genes in "
          f"the network); split at the mean over all genes ({info['gene_degree_mean']:.0f}).", ""]
    rk = rk.assign(stratum=np.where(rk.gene_degree > info["gene_degree_mean"], "high degree", "low degree"))
    for stratum in ["low degree", "high degree"]:
        for set_name, g in sets(rk[rk.stratum == stratum])[:2]:
            s = metric_rows(g.assign(), U)
            s = s[s.set == set_name].drop(columns="set")
            L += [f"### {stratum}, {set_name}", ""] + (md(s, {"pAUC_1_100": ".3f"}) if len(s) else ["(none)"]) + [""]
    L += ["## 4. Causal genes at hub metabolites", "",
          f"Causal genes whose M0 best reaction (or, if unscored, any reaction) touches a metabolite in the top 1% of "
          f"metabolite degree (degree >= {info['hub_metabolite_degree_min']:.0f}, {info['n_hub_metabolites']} "
          "metabolites). Rank per ranking; the long form is in `hub_genes.tsv`.", ""]
    if len(hub):
        cols = ["unit", "gene", "hub_metabolites"] + [m for m in RANKINGS + SENSITIVITY if m in hub.columns]
        L += md(hub[cols], {m: ".0f" for m in cols}) + [""]
        hub.melt(id_vars=["unit", "gene", "hub_metabolites"], var_name="ranking", value_name="rank").to_csv(
            out / "hub_genes.tsv", sep="\t", index=False)
    else:
        L += ["(none)", ""]
    bs = bias.groupby("ranking")[["rho_final_vs_hub_sum", "rho_gene_rank_vs_n_nodes"]].median()
    bs = bs.reindex([m for m in RANKINGS + SENSITIVITY if m in bs.index]).reset_index()
    L += ["## 5-6. Bias removed? Gene node-count bias", "",
          "Medians over evaluated units. rho_final_vs_hub_sum: Spearman of reaction final score vs hub_sum over the "
          "scored reactions (a correction should move it toward 0). rho_gene_rank_vs_n_nodes: Spearman of gene rank vs "
          "the gene's number of reaction nodes over the gene universe (negative = more reactions, better rank); "
          "reported, not corrected.", ""] + md(bs, {"rho_final_vs_hub_sum": ".3f", "rho_gene_rank_vs_n_nodes": ".3f"})
    L += ["", "## 7. Per-unit appendix", "", "Causal-gene rank per unit and ranking. Long form: `rankings.tsv`.", ""]
    wide = rk.pivot_table(index="unit", columns="ranking", values="rank").reindex(columns=order(rk))
    meta = uinfo.set_index("label")[["collection", "study", "sub_unit", "n_metabolites", "small_flag", "viable",
                                     "primary", "cluster", "causal_in_network", "undefined_z_scored_conductivity"]]
    app = meta.join(wide).reset_index().rename(columns={"label": "unit", "undefined_z_scored_conductivity": "undef_z"})
    app["small_flag"] = app.small_flag.fillna("").astype(str)
    L += md(app, {**{m: ".0f" for m in wide.columns}, "undef_z": ".1%", "n_metabolites": ".0f"})
    L += ["", "## 8. Rules as implemented and adaptations", "",
          "- **M1-D22** (interpretation of D22 item 4), per column over scored reactions: a node with defined z gets "
          "the percentile of its z among nodes with defined z, times the share of scored nodes with defined z; a node "
          "with undefined z gets the percentile of its raw score among all scored nodes.",
          "- **M1-last**: undefined-z nodes take percentiles below every defined-z node, ordered by raw score; defined-z "
          "nodes are ordered by z.",
          f"- Undefined z among scored reactions: median {uinfo.undefined_z_scored_conductivity.median():.1%} per unit "
          f"(conductivity), max {uinfo.undefined_z_scored_conductivity.max():.1%}; per unit in the appendix (undef_z).",
          "- **M1 adaptation**: DADA's random seed sets come from nearest-seed buckets; ours swap each real source for "
          "a metabolite within |log2 degree difference| <= 0.5 (widened until >= 10 candidates), from the measurable "
          "pool, no metabolite twice. Seeds are metabolites; candidates are reaction nodes.",
          "- **M1 SD**: population SD (divide by n) as stored by the runner; DADA divides by n - 1 (no effect on ranks "
          "at n = 1000 beyond a constant factor).",
          f"- **M2 adaptation**: candidates are reaction nodes; degree = hub_sum; the population is all "
          f"{info['reaction_nodes']} reaction nodes in the same run including zero-score ones and the node itself; "
          "ties at the 1,000th neighbour broken by node order; SD with n - 1 as DADA. M2-hub_max: the same with hub_max.",
          "- **M3**: p = (b + 1) / (perms + 1), b = permutations with null >= observed (Phipson and Smyth 2010); "
          "ranked by 1 - p, ties by raw score. A scored node never reached by the null has b = 0 and the smallest p, "
          "so this rule puts scored nodes with no null traffic at the top: a direct consequence of the estimator.",
          f"- **UNI adaptations**: rules applied to reaction nodes before the gene step; candidate degree = hub_sum; "
          f"lambda (UNI-C) = mean hub_sum over all reaction nodes ({info['lambda_hub_sum']:.1f}); UNI-S compares the "
          f"mean degree of the unit's source metabolites with the mean degree of all network metabolites "
          f"({info['network_metabolite_mean_degree']:.2f}); R_RAW and R_ADJ are ordinal ranks in reaction_table's "
          "order (final score, then conductivity); the merged rank is mapped to 1..|C| and passed to the gene step as "
          f"(|C| - rank + 1) / |C|. UNI-S used R_RAW in {int(uinfo.uni_s_uses_raw.sum())} of {len(uinfo)} units.",
          "- **Gene step**: unchanged (gene_table; ties by total accumulated current from the observed walk, which "
          "`nodes.tsv` does not store, so the walk is rerun per unit and its scores checked against `nodes.tsv`).",
          "- **Hub-metabolite genes** use the M0 best reaction for every method, so the list is the same across rows.",
          "- **pAUC**: DADA scored 100 candidates per case; here the candidates are the whole universe, so the ROC is "
          "the partial curve up to rank 100, rescaled.", "",
          "## 9. Not run and deviations", ""]
    if not info["has_exceedance"]:
        L.append("- M3 and its UNI rows: not run (this run has no exceedance counts).")
    for r in skipped.itertuples():
        L.append(f"- {r.label}: causal gene {r.causal_genes} not in the network; not evaluated.")
    for r in uinfo[uinfo.scored_reactions == 0].itertuples():
        L.append(f"- {r.label}: no reaction carries traffic, so every ranking gives the causal gene the tail rank "
                 f"({(U + 1) / 2:.0f}); kept in the tables, counted as unchanged.")
    np_ = ran[(ran.causal_genes == "")]
    L.append(f"- {len(np_)} run units have no causal gene in the registry and are not evaluated.")
    L.append(f"- Gene universe here is {U} (genes on reaction nodes of this stage); the handoff quoted 2,302 from "
             f"another network. Secondary set: {rk.unit.nunique()} units evaluated (handoff: 53).")
    L.append("- Out of scope per the handoff: rewired-network normalization, DADA's eigenvector-centrality ratio, "
             "sources = increased, gene-level node-count correction.")
    L.append("")
    (out / "report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
