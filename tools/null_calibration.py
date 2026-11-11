"""Label-free degree-bias calibration of the directed absorbing walk.

For each evaluation unit (one row of benchmarks/data_registry.tsv, one
orientation), the real targets and their phi are held fixed and the real
sources are replaced by degree-matched random metabolites carrying the real
sources' |log2FC|. The walk's reaction-node scores over many such draws are
the structural null: score a reaction earns from where it sits, with no real
data in the sources. Its slope against hub-adjacency (the degree of the
metabolites a reaction touches) is the structural bias; the same statistics on
the observed scores show how much of it reaches a real result. No causal gene
is used.

Every registry row appears in the output: run, or not run with the reason
(D22). Units come from
  A_inborn_errors  the benchmark's group DAA (benchmarks/cases.tsv)
  B_published      predictions/published_v2/<unit>/daa_mapping.tsv, group-level units only (D21, D22)
  C_workbench      results/workbench_daa/<study>_<analysis>.csv (Welch t, raw p < 0.05, D22)

    python tools/null_calibration.py --perms 1000
    python tools/null_calibration.py --perms 200 --threshold cit arg aadc     # source-count experiment

Writes results/null_calibration/<run>/: units.tsv (every registry row), nodes.tsv,
slope.tsv, perms.tsv, draws.tsv, [threshold.tsv], run_info.json, report.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from metaboenrich import benchmark as B, structure          # noqa: E402
from metaboenrich.daa import load_daa, load_name_bridge     # noqa: E402
from metaboenrich.gem import load_sbml                      # noqa: E402
from metaboenrich.network import build_graph                # noqa: E402
from metaboenrich.walk import _EPS, absorbing_walk, prepare_walk, walk_states   # noqa: E402
from published_predictions import rule_settings             # noqa: E402

REGISTRY = Path("benchmarks/data_registry.tsv")
PUBLISHED_RUN = Path("predictions/published_v2")
WORKBENCH_DAA = Path("results/workbench_daa")
KEYS = ["collection", "study", "sub_unit", "n_metabolites", "small_flag"]
SCORES = ("conductivity", "bottleneck")
HUBS = ("hub_max", "hub_sum")
MIN_SOURCES, MIN_METABOLITES = 3, 20      # viable: enough sources for a null, a broad enough assay


def loose(s):
    """The registry cuts comparison names at 60 characters and differs in spacing and punctuation."""
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def registry_units(model, graph, bridge, orientation, published_run=PUBLISHED_RUN, workbench=WORKBENCH_DAA):
    """Every registry row as runnable units, or as a not-run row with the reason.
    Unit: dict of the registry KEYS plus label, sources, targets, phi, ground, measured."""
    reg = pd.read_csv(REGISTRY, sep="\t", dtype=str).fillna("")
    cases = pd.read_csv(B.CASES, sep="\t").fillna("").set_index("case_id")
    pub = pd.read_csv(published_run / "units.tsv", sep="\t", dtype=str).fillna("")
    data = {}
    units, not_run = [], []

    def config(df, id_cols, alpha, min_fc):
        daa, _, _ = load_daa(df, model, id_cols, "log2FC", "p", alpha, False, min_fc,
                             name_bridge=bridge, sources=orientation)
        daa["node"] = daa.met_id.map(graph.index)
        on = daa.dropna(subset=["node"]).astype({"node": int})
        src = on[on.role == "source"].node.tolist()
        tgt = on[on.role == "target"].node.tolist()
        ground = on[on.role == ""].node.tolist() if bool(src) != bool(tgt) else None
        return dict(sources=src, targets=tgt, phi=dict(zip(on.node, on.phi)), ground=ground,
                    measured=set(on.node), n_in_network=len(on))

    for _, r in reg.iterrows():
        keys = {k: r[k] for k in KEYS}
        if r.status != "active":
            not_run.append({**keys, "label": r.sub_unit, "reason": f"{r.status}: {r.drop_reason}"})
            continue
        if r.collection == "A_inborn_errors":
            c = cases.loc[r.sub_unit]
            if c.dataset not in data:
                data[c.dataset] = B.load_t2020() if c.dataset == "t2020" else B.load_miller()
            z, ann, cohorts = data[c.dataset]
            samples = [s for s in cohorts.get(c.cohort, []) if s in z.columns]
            if not samples:
                not_run.append({**keys, "label": r.sub_unit, "reason": "no cohort samples"})
                continue
            units.append({**keys, "label": r.sub_unit, **config(B.group_daa(z, ann, samples),
                                                                  ["HMDB", "KEGG", "name"], 0.05, 1.0)})
        elif r.collection == "B_published":
            comp = pub.Comparison.map(loose)
            g = pub[(pub.source == r.study) & (comp == loose(r.sub_unit))]
            if g.empty:
                g = pub[(pub.source == r.study) & comp.str.startswith(loose(r.sub_unit))]
            if g.empty:
                not_run.append({**keys, "label": r.sub_unit, "reason": f"no unit in {published_run.as_posix()}"})
                continue
            single = g[g.Subject_ID != ""]
            if len(single):                                # one row for all of a study's per-person units
                not_run.append({**keys, "label": f"{len(single)} single-subject units",
                                "reason": "single subject: each person has a different disorder, no group (D21, D22)"})
            for u in g[g.Subject_ID == ""].itertuples(index=False):
                if u.status != "ok":
                    reason = u.status
                else:
                    reason = ""
                if reason:
                    not_run.append({**keys, "label": u.unit, "reason": reason})
                    continue
                m = pd.read_csv(published_run / u.unit / "daa_mapping.tsv", sep="\t")
                df = pd.DataFrame({"name": m.met_id, "log2FC": m.log2fc, "p": m.p})
                units.append({**keys, "label": u.unit, **config(df, ["name"], *rule_settings(u.rule))})
        else:
            f = workbench / f"{r.study}_{r.sub_unit}.csv"
            if not f.exists():
                why = "no DAA table"
                c = workbench / "contrasts.tsv"
                if c.exists():
                    ct = pd.read_csv(c, sep="\t", dtype=str).fillna("")
                    hit = ct[(ct.study == r.study) & (ct.analysis == r.sub_unit)]
                    if len(hit) and hit.status.iloc[0] != "ok":
                        why = hit.status.iloc[0]
                not_run.append({**keys, "label": r.sub_unit, "reason": why})
                continue
            df = pd.read_csv(f, dtype={"HMDB": str, "KEGG": str}).replace("", np.nan)
            units.append({**keys, "label": f"{r.study}_{r.sub_unit}",
                          **config(df, ["KEGG", "HMDB", "name"], 0.05, 0.0)})
    return units, not_run


def candidate_pools(sources, degree, pool, exclude, rng_min=10):
    """Per real source, the metabolites of similar degree it may be swapped for:
    |log2 degree difference| <= 0.5, widened by 0.5 until there are rng_min."""
    pool = np.array(sorted(set(pool) - set(exclude)))
    ld = np.log2(degree[pool])
    out = []
    for s in sources:
        width = 0.5
        while True:
            c = pool[np.abs(ld - np.log2(degree[s])) <= width]
            if len(c) >= rng_min or width > 8:
                break
            width += 0.5
        out.append(c)
    return out


def draw(cands, rng):
    """One random source per real source, no metabolite twice."""
    used, picks = set(), []
    for c in cands:
        for _ in range(50):
            x = int(c[rng.integers(len(c))])
            if x not in used:
                break
        used.add(x)
        picks.append(x)
    return picks


def effect_ratio(hub, y):
    """Mean of y over the top 10% by hub, divided by that over the bottom 50%."""
    lo = y[hub <= np.quantile(hub, 0.5)].mean()
    return y[hub >= np.quantile(hub, 0.9)].mean() / lo if lo > 0 else np.nan


def slope_rows(df, label, rng, n_boot=500):
    rows = []
    for score in SCORES:
        y = df[f"null_mean_{score}"].to_numpy()
        for hub in HUBS:
            x = df[hub].to_numpy(float)
            coef = np.polyfit(x, y, 1)[0] if np.ptp(x) > 0 else np.nan
            boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(x), len(x))
                if np.ptp(x[i]) > 0:
                    boots.append(np.polyfit(x[i], y[i], 1)[0])
            rho = spearmanr(x, y).correlation
            rows.append({"unit": label, "score": score, "hub": hub, "slope": coef,
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "spearman_rho": rho, "n_nodes": len(x)})
    return rows


class Calibrator:
    """The graph-side pieces every unit shares."""

    def __init__(self, graph, kappa):
        self.graph, self.kappa = graph, kappa
        self.ws = walk_states(graph)
        isr = graph.is_reaction
        self.degree = np.bincount(graph.edges.ravel(), minlength=graph.n).astype(float)
        adj = sp.coo_matrix((np.ones(len(graph.edges)), (graph.edges[:, 0], graph.edges[:, 1])), shape=(graph.n,) * 2)
        _, comp = connected_components(adj, directed=False)
        self.main_comp = comp == np.bincount(comp).argmax()
        self.rxn = np.flatnonzero(isr)
        mets_of = {r: [] for r in self.rxn}
        for m, r in graph.edges:
            mets_of[r].append(m)
        self.base = pd.DataFrame({"node_id": [graph.node_ids[r] for r in self.rxn],
                                  "n_metabolites": [len(mets_of[r]) for r in self.rxn],
                                  "hub_max": [max(self.degree[mets_of[r]]) for r in self.rxn],
                                  "hub_sum": [sum(self.degree[mets_of[r]]) for r in self.rxn]})
        self.all_mets = np.flatnonzero(~isr)

    def prepare(self, src, tgt, phi, ground):
        if tgt:
            eff_t = tgt
            absorb_p = {t: 1.0 if self.kappa is None else 1.0 - float(np.exp(-self.kappa * phi.get(t, 0.0))) for t in tgt}
        else:                                              # one-sided: the measured, unchanged absorb
            eff_t = [g for g in ground if g not in set(src)]
            absorb_p = {g: 1.0 for g in eff_t}
        return eff_t, prepare_walk(self.graph, eff_t, absorb_p, ws=self.ws)

    def null(self, src, tgt, phi, ground, prep, exclude, pool, perms, rng):
        """Null moments per reaction node, plus the per-permutation rows and the draws."""
        cands = candidate_pools(src, self.degree, pool, exclude)
        n = len(self.rxn)
        s1, s2 = {s: np.zeros(n) for s in SCORES}, {s: np.zeros(n) for s in SCORES}
        traffic, rows, drawn = np.zeros(n, dtype=int), [], []
        for k in range(perms):
            rs = draw(cands, rng)
            drawn.append(rs)
            phi_k = {**{t: phi.get(t, 0.0) for t in tgt}, **{r: phi[s] for r, s in zip(rs, src)}}
            res = absorbing_walk(self.graph, rs, tgt, phi_k, kappa=self.kappa, ground=ground, prep=prep)
            for s in SCORES:
                v = getattr(res, s)[self.rxn]
                s1[s] += v
                s2[s] += v * v
            traffic += res.conductivity[self.rxn] > _EPS
            rows.append({"perm": k, "n_sources": len(rs), "success_fraction": res.walk_info["walk_success_fraction"],
                         "sources_unreached": int(sum(prep.h[r] <= _EPS for r in rs)),
                         "sources_off_main": int(sum(not self.main_comp[r] for r in rs))})
        mean = {s: s1[s] / perms for s in SCORES}
        sd = {s: np.sqrt(np.maximum(s2[s] / perms - mean[s] ** 2, 0.0)) for s in SCORES}
        return mean, sd, traffic, rows, drawn, cands


def run_units(cal, units, pool, perms, rng):
    node_frames, perm_rows, draw_rows, slopes, summary = [], [], [], [], []
    hub = cal.base.hub_sum.to_numpy(float)
    t0 = time.time()
    for u in units:
        keys = {k: u[k] for k in KEYS} | {"label": u["label"]}
        src, tgt, phi, ground = u["sources"], u["targets"], u["phi"], u["ground"]
        row = {**keys, "n_in_network": u["n_in_network"], "n_sources": len(src), "n_targets": len(tgt)}
        if not src or not (tgt or ground):
            summary.append({**row, "status": "not run: " + ("no sources" if not src else "no targets")})
            continue
        eff_t, prep = cal.prepare(src, tgt, phi, ground)
        obs = absorbing_walk(cal.graph, src, tgt, phi, kappa=cal.kappa, ground=ground, prep=prep)
        exclude = set(eff_t) | set(tgt) | set(ground or [])
        mean, sd, traffic, rows, drawn, cands = cal.null(src, tgt, phi, ground, prep, exclude, pool, perms, rng)
        df = cal.base.copy()
        df.insert(0, "unit", u["label"])
        df["n_sources"], df["n_targets"] = len(src), len(eff_t)
        df["n_perms_with_traffic"] = traffic
        for s in SCORES:
            o = getattr(obs, s)[cal.rxn]
            df[f"null_mean_{s}"], df[f"null_sd_{s}"], df[f"observed_{s}"] = mean[s], sd[s], o
            df[f"z_{s}"] = np.where(sd[s] > _EPS, (o - mean[s]) / np.where(sd[s] > _EPS, sd[s], 1.0), np.nan)
        node_frames.append(df)
        perm_rows += [{"unit": u["label"], **r} for r in rows]
        perm_rows.append({"unit": u["label"], "perm": "observed", "n_sources": len(src),
                          "success_fraction": obs.walk_info["walk_success_fraction"],
                          "sources_unreached": obs.walk_info["walk_sources_unreached"],
                          "sources_off_main": int(sum(not cal.main_comp[s] for s in src))})
        # How much of each degree bucket's shortlist the draws actually used.
        flat = np.concatenate(drawn)
        bucket = np.floor(np.log2(cal.degree)).astype(int)
        short = np.unique(np.concatenate(cands))
        for b in sorted(set(bucket[src])):
            in_b = short[bucket[short] == b]
            got = flat[bucket[flat] == b]
            draw_rows.append({"unit": u["label"], "degree_bucket": f"{2 ** b}-{2 ** (b + 1) - 1}",
                              "real_sources": int((bucket[src] == b).sum()), "shortlist": len(in_b),
                              "draws": len(got), "distinct_drawn": len(np.unique(got))})
        slopes += slope_rows(df, u["label"], rng)
        null_p = pd.DataFrame(rows)
        y_null, y_obs = df.null_mean_conductivity.to_numpy(), df.observed_conductivity.to_numpy()
        summary.append({**row, "status": "run",
                        "viable": len(src) >= MIN_SOURCES and float(u["n_metabolites"] or 0) >= MIN_METABOLITES,
                        "rho_null": spearmanr(hub, y_null).correlation, "rho_observed": spearmanr(hub, y_obs).correlation,
                        "ratio_null": effect_ratio(hub, y_null), "ratio_observed": effect_ratio(hub, y_obs),
                        "undefined_z": float(np.isnan(df.z_conductivity).mean()),
                        "success_observed": obs.walk_info["walk_success_fraction"],
                        "success_null": null_p.success_fraction.median(),
                        "distinct_source_sets": len({tuple(sorted(d)) for d in drawn})})
        print(f"{u['label'][:50]:50s} {len(src):3d} sources {len(eff_t):3d} targets {time.time() - t0:5.0f}s", flush=True)
    return node_frames, perm_rows, draw_rows, slopes, summary


def threshold_experiment(cal, units, labels, sizes, reps, perms, pool, rng):
    """Subsample each unit's real sources to k and rerun the null: where does the
    bias estimate settle onto the full source set's?"""
    hub = cal.base.hub_sum.to_numpy(float)
    rows = []
    for u in (u for u in units if u["label"] in labels):
        src, tgt, phi, ground = u["sources"], u["targets"], u["phi"], u["ground"]
        for k in [s for s in sizes if s < len(src)] + [len(src)]:
            for rep in range(reps if k < len(src) else 1):
                sub = sorted(rng.choice(src, k, replace=False).tolist()) if k < len(src) else src
                eff_t, prep = cal.prepare(sub, tgt, phi, ground)
                exclude = set(eff_t) | set(tgt) | set(ground or [])
                mean, _, _, rows_k, drawn, _ = cal.null(sub, tgt, phi, ground, prep, exclude, pool, perms, rng)
                y = mean["conductivity"]
                rows.append({"unit": u["label"], "k": k, "full": k == len(src), "rep": rep,
                             "rho_null": spearmanr(hub, y).correlation, "ratio_null": effect_ratio(hub, y),
                             "distinct_source_sets": len({tuple(sorted(d)) for d in drawn})})
            print(f"threshold {u['label']} k={k}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="three_rule:three_rule", metavar="SET:KEY")
    p.add_argument("--orientation", nargs="+", default=["decreased"], choices=B.ORIENTATIONS,
                   help="'increased' runs, but its null is not comparable (success fraction) until matched on h(s)")
    p.add_argument("--pool", nargs="+", default=["measurable"], choices=["all", "measurable"],
                   help="random sources from all graph metabolites, or only from metabolites measured in any "
                        "unit's data (default)")
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--kappa", type=float, default=None, help="partial absorption (default: none, as the benchmark)")
    p.add_argument("--collections", nargs="+", default=["A_inborn_errors", "B_published", "C_workbench"])
    p.add_argument("--threshold", nargs="*", default=[], metavar="UNIT",
                   help="units for the source-count experiment (e.g. cit arg aadc)")
    p.add_argument("--sizes", nargs="+", type=int, default=[1, 2, 3, 5, 8, 12, 20])
    p.add_argument("--reps", type=int, default=20, help="random source subsets per size (threshold experiment)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/null_calibration")
    args = p.parse_args(argv)

    model = load_sbml("data/Human-GEM.xml")
    sset, key = args.stage.split(":", 1)
    st = next(s for s in structure.stages(model, sset, atom_dir="data") if s[0] == key)
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    bridge = load_name_bridge("data/id_translation.csv")
    cal = Calibrator(graph, args.kappa)

    for orientation in args.orientation:
        units, not_run = registry_units(model, graph, bridge, orientation)
        not_run += [{**{k: u[k] for k in KEYS}, "label": u["label"], "reason": "collection not selected"}
                    for u in units if u["collection"] not in args.collections]
        units = [u for u in units if u["collection"] in args.collections]
        measured = set().union(*(u["measured"] for u in units))
        for pool_name in args.pool:
            t0 = time.time()
            rng = np.random.default_rng(args.seed)
            pool = cal.all_mets if pool_name == "all" else np.array(sorted(measured))
            run = Path(args.out) / f"{key}_{orientation}_{pool_name}_k{args.kappa}_n{args.perms}"
            run.mkdir(parents=True, exist_ok=True)
            node_frames, perm_rows, draw_rows, slopes, summary = run_units(cal, units, pool, args.perms, rng)
            nodes = pd.concat(node_frames, ignore_index=True)
            meta = pd.DataFrame(summary)[KEYS + ["label"]]
            nodes = meta.rename(columns={"n_metabolites": "unit_n_metabolites", "label": "unit"}).merge(nodes, on="unit")
            nodes.to_csv(run / "nodes.tsv", sep="\t", index=False)
            pooled = nodes.copy()                                  # each unit scaled to mean null 1
            for s in SCORES:
                scale = pooled.groupby("unit")[f"null_mean_{s}"].transform("mean")
                pooled[f"null_mean_{s}"] = pooled[f"null_mean_{s}"] / scale.where(scale > 0, 1.0)
            table = pd.concat([pd.DataFrame(summary), pd.DataFrame(not_run).assign(status=lambda d: "not run: " + d.reason)
                               .drop(columns="reason")], ignore_index=True)
            viable = set(table.loc[table.viable.eq(True), "label"])
            slopes += slope_rows(pooled, "pooled: all units", rng)
            if viable:
                slopes += slope_rows(pooled[pooled.unit.isin(viable)], "pooled: viable units", rng)
            slope = pd.DataFrame(slopes)
            table.to_csv(run / "units.tsv", sep="\t", index=False)
            slope.to_csv(run / "slope.tsv", sep="\t", index=False)
            perms = pd.DataFrame(perm_rows)
            perms.to_csv(run / "perms.tsv", sep="\t", index=False)
            draws = pd.DataFrame(draw_rows)
            draws.to_csv(run / "draws.tsv", sep="\t", index=False)
            thr = None
            if args.threshold:
                thr = threshold_experiment(cal, units, set(args.threshold), args.sizes, args.reps, args.perms, pool, rng)
                thr.to_csv(run / "threshold.tsv", sep="\t", index=False)
            info = {"stage": args.stage, "metabolites": int((~graph.is_reaction).sum()),
                    "reaction_nodes": int(graph.is_reaction.sum()), "edges": len(graph.edges),
                    "walk_states": int(cal.ws.n_states), "method": "walk", "orientation": orientation,
                    "kappa": args.kappa, "kappa_mode": "every walker absorbs at the first target (target phi unused)"
                    if args.kappa is None else "partial absorption, real target phi",
                    "source_pool": pool_name, "pool_size": int(len(pool)), "perms": args.perms, "seed": args.seed,
                    "degree_matching": "|log2 degree difference| <= 0.5, widened by 0.5 until >= 10 candidates",
                    "viable": f"n_sources >= {MIN_SOURCES} and n_metabolites >= {MIN_METABOLITES}",
                    "registry": REGISTRY.as_posix(), "registry_rows": int(len(pd.read_csv(REGISTRY, sep='\t'))),
                    "runtime_s": round(time.time() - t0, 1)}
            (run / "run_info.json").write_text(json.dumps(info, indent=2))
            write_report(run, info, table, nodes, slope, perms, draws, thr)
            print(f"wrote {run} in {time.time() - t0:.0f}s", flush=True)


def fmt(x, f=".2f"):
    return "-" if x is None or pd.isna(x) else format(x, f)


def group_lines(table, slope):
    ran = table[table.status == "run"]
    groups = [("all run units", ran), ("viable", ran[ran.viable.astype(bool)]),
              ("not viable", ran[~ran.viable.astype(bool)]), ("single source", ran[ran.n_sources == 1]),
              ("2+ sources", ran[ran.n_sources >= 2])]
    lines = ["| units | n | rho null | rho observed | ratio null | ratio observed | undefined z |",
             "|---|---|---|---|---|---|---|"]
    for name, g in groups:
        if len(g):
            lines.append(f"| {name} | {len(g)} | {fmt(g.rho_null.median())} | {fmt(g.rho_observed.median())} | "
                         f"{fmt(g.ratio_null.median())} | {fmt(g.ratio_observed.median())} | "
                         f"{fmt(g.undefined_z.median(), '.0%')} |")
    lines += ["", "Medians over units; rho = Spearman of conductivity vs hub_sum; ratio = mean conductivity of the "
              "top 10% of reaction nodes by hub_sum over that of the bottom 50%.", "",
              "Pooled (each unit's null scaled to mean 1), conductivity x hub_sum:", ""]
    for _, r in slope[slope.unit.str.startswith("pooled") & (slope.score == "conductivity") & (slope.hub == "hub_sum")].iterrows():
        lines.append(f"- {r.unit}: rho {r.spearman_rho:.3f}, slope {r.slope:.3g} (95% CI {r.ci_low:.3g} to {r.ci_high:.3g})")
    return lines


def write_report(run, info, table, nodes, slope, perms, draws, thr):
    ran = table[table.status == "run"]
    lines = [f"# Null calibration: {info['stage']}, sources = {info['orientation']}, pool = {info['source_pool']}", ""]
    if info["orientation"] == "increased":
        lines += ["**Not usable:** under sources = increased the null is not comparable (real sources reach a target "
                  "far more often than degree-matched random ones). Kept for reference only.", ""]
    lines += [f"{info['perms']} permutations per unit, kappa = {info['kappa']} ({info['kappa_mode']}). "
              f"Pool: {info['pool_size']} metabolites. Viable: {info['viable']}.", "",
              f"Registry: {info['registry_rows']} rows -> {len(table)} output rows "
              f"(published rows split by matrix / disease group); {len(ran)} run, {len(table) - len(ran)} not run.", "",
              "## Summary", ""] + group_lines(table, slope)
    und = nodes.z_conductivity.isna().mean()
    null_p, obs_p = perms[perms.perm != "observed"], perms[perms.perm == "observed"]
    lines += ["", "## Pitfalls", "",
              f"- Reaction nodes with undefined z (no null traffic): {und:.1%} of node x unit rows; they keep their "
              "raw score (D22).",
              f"- Walk success fraction: real sources median {obs_p.success_fraction.median():.3f}, "
              f"random median {null_p.success_fraction.median():.3f}.",
              f"- Random sources off the main network: {null_p.sources_off_main.sum() / null_p.n_sources.sum():.1%}; "
              f"random sources that reach no target: {null_p.sources_unreached.sum() / null_p.n_sources.sum():.1%} "
              f"(real: {obs_p.sources_unreached.sum() / obs_p.n_sources.sum():.1%}).",
              f"- Degree buckets where the real sources fill at least half the shortlist (drawn without "
              f"replacement, so the null barely varies there): {(draws.real_sources >= draws.shortlist / 2).sum()} "
              f"of {len(draws)} (unit x bucket); "
              f"units with fewer than 100 distinct null source sets: "
              f"{(ran.distinct_source_sets < 100).sum()} of {len(ran)}."]
    if thr is not None and len(thr):
        lines += ["", "## Source-count threshold", "",
                  "Real sources subsampled to k (random subsets), null rerun; median and 2.5-97.5 percentile over "
                  "subsets. The last row per unit is the full source set.", "",
                  "| unit | k | subsets | rho null | rho interval | ratio null | ratio interval | distinct null sets |",
                  "|---|---|---|---|---|---|---|---|"]
        for (unit, k), g in thr.groupby(["unit", "k"], sort=False):
            lines.append(f"| {unit} | {k} | {len(g)} | {fmt(g.rho_null.median())} | {fmt(g.rho_null.quantile(.025))} to "
                         f"{fmt(g.rho_null.quantile(.975))} | {fmt(g.ratio_null.median())} | "
                         f"{fmt(g.ratio_null.quantile(.025))} to {fmt(g.ratio_null.quantile(.975))} | "
                         f"{int(g.distinct_source_sets.median())} |")
    lines += ["", "## Per unit", "",
              "| collection | study | unit | n_metabolites | small | in network | sources | targets | viable | "
              "rho null | rho obs | ratio null | ratio obs | undefined z | status |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples(index=False):
        lines.append(f"| {r.collection} | {r.study} | {str(r.label)[:60]} | {r.n_metabolites} | "
                     f"{'FLAG' if r.small_flag else ''} | {fmt(r.n_in_network, '.0f')} | {fmt(r.n_sources, '.0f')} | "
                     f"{fmt(r.n_targets, '.0f')} | {'yes' if r.viable == True else ''} | {fmt(r.rho_null)} | "
                     f"{fmt(r.rho_observed)} | {fmt(r.ratio_null)} | {fmt(r.ratio_observed)} | "
                     f"{fmt(r.undefined_z, '.0%')} | {r.status} |")
    lines.append("")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
