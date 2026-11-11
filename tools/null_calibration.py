"""Label-free degree-bias calibration of the directed absorbing walk.

For each dataset configuration (one benchmark case, one orientation), the real
targets and their phi are held fixed and the real sources are replaced by
degree-matched random metabolites carrying the real sources' |log2FC|. The
walk's reaction-node scores over many such draws are the structural null:
score a reaction earns from where it sits, with no real data in the sources.
Its slope against hub-adjacency (the degree of the metabolites a reaction
touches) is the structural bias. No causal gene is used.

    python tools/null_calibration.py --perms 200                 # pilot
    python tools/null_calibration.py --perms 1000 --pool all measurable

Writes results/null_calibration/<run>/ per (orientation, pool): nodes.tsv,
slope.tsv, perms.tsv, run_info.json, report.md.
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
from metaboenrich import benchmark as B, structure          # noqa: E402
from metaboenrich.daa import load_daa, load_name_bridge     # noqa: E402
from metaboenrich.gem import load_sbml                      # noqa: E402
from metaboenrich.network import build_graph                # noqa: E402
from metaboenrich.walk import _EPS, absorbing_walk, prepare_walk, walk_states   # noqa: E402

SCORES = ("conductivity", "bottleneck")
HUBS = ("hub_max", "hub_sum")


def configurations(model, graph, bridge, include, orientation, tmp):
    """[(case id, sources, targets, phi, ground, measured nodes)] from the benchmark's group DAA."""
    data = {"t2020": B.load_t2020(), "miller": B.load_miller()}
    cases = pd.read_csv(B.CASES, sep="\t").fillna("")
    out = []
    for _, c in cases[cases.status.isin(include)].iterrows():
        z, ann, cohorts = data[c.dataset]
        samples = [s for s in cohorts.get(c.cohort, []) if s in z.columns]
        if not samples:
            continue
        B.group_daa(z, ann, samples).to_csv(tmp, index=False)
        daa, _, _ = load_daa(tmp, model, ["HMDB", "KEGG", "name"], "log2FC", "p", 0.05, False, 1.0,
                             name_bridge=bridge, sources=orientation)
        daa["node"] = daa.met_id.map(graph.index)
        on = daa.dropna(subset=["node"]).astype({"node": int})
        src = on[on.role == "source"].node.tolist()
        tgt = on[on.role == "target"].node.tolist()
        ground = on[on.role == ""].node.tolist() if bool(src) != bool(tgt) else None
        out.append((c.case_id, src, tgt, dict(zip(on.node, on.phi)), ground, set(on.node)))
    return out


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
            rows.append({"configuration": label, "score": score, "hub": hub, "slope": coef,
                         "ci_low": np.percentile(boots, 2.5), "ci_high": np.percentile(boots, 97.5),
                         "spearman_rho": rho, "n_nodes": len(x)})
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", default="three_rule:three_rule", metavar="SET:KEY")
    p.add_argument("--orientation", nargs="+", default=["decreased", "increased"], choices=B.ORIENTATIONS)
    p.add_argument("--pool", nargs="+", default=["all", "measurable"], choices=["all", "measurable"],
                   help="random sources from all graph metabolites, or only from metabolites measured in any "
                        "configuration's data")
    p.add_argument("--perms", type=int, default=200)
    p.add_argument("--kappa", type=float, default=None, help="partial absorption (default: none, as the benchmark)")
    p.add_argument("--include", nargs="+", default=["primary", "secondary"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/null_calibration")
    args = p.parse_args(argv)

    model = load_sbml("data/Human-GEM.xml")
    sset, key = args.stage.split(":", 1)
    st = next(s for s in structure.stages(model, sset, atom_dir="data") if s[0] == key)
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    bridge = load_name_bridge("data/id_translation.csv")
    ws = walk_states(graph)

    isr = graph.is_reaction
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n).astype(float)
    adj = sp.coo_matrix((np.ones(len(graph.edges)), (graph.edges[:, 0], graph.edges[:, 1])), shape=(graph.n,) * 2)
    _, comp = connected_components(adj, directed=False)
    main_comp = comp == np.bincount(comp).argmax()
    rxn = np.flatnonzero(isr)
    mets_of = {r: [] for r in rxn}
    for m, r in graph.edges:
        mets_of[r].append(m)
    base = pd.DataFrame({"node_id": [graph.node_ids[r] for r in rxn],
                         "n_metabolites": [len(mets_of[r]) for r in rxn],
                         "hub_max": [max(degree[mets_of[r]]) for r in rxn],
                         "hub_sum": [sum(degree[mets_of[r]]) for r in rxn]})
    all_mets = np.flatnonzero(~isr)
    tmp = Path(args.out) / "_daa_tmp.csv"
    tmp.parent.mkdir(parents=True, exist_ok=True)

    for orientation in args.orientation:
        configs = configurations(model, graph, bridge, args.include, orientation, tmp)
        measured = set().union(*(c[5] for c in configs))
        for pool_name in args.pool:
            t0 = time.time()
            rng = np.random.default_rng(args.seed)
            pool = all_mets if pool_name == "all" else np.array(sorted(measured))
            run = Path(args.out) / f"{key}_{orientation}_{pool_name}_k{args.kappa}_n{args.perms}"
            run.mkdir(parents=True, exist_ok=True)
            node_frames, perm_rows, skipped, slopes = [], [], [], []
            for case, src, tgt, phi, ground, _ in configs:
                if not src or not (tgt or ground):
                    skipped.append({"configuration": case, "reason": "no sources" if not src else "no targets"})
                    continue
                if tgt:
                    eff_t, absorb_p = tgt, {t: 1.0 if args.kappa is None else
                                           1.0 - float(np.exp(-args.kappa * phi.get(t, 0.0))) for t in tgt}
                else:                                              # one-sided: the measured, unchanged absorb
                    eff_t = [g for g in ground if g not in set(src)]
                    absorb_p = {g: 1.0 for g in eff_t}
                prep = prepare_walk(graph, eff_t, absorb_p, ws=ws)
                obs = absorbing_walk(graph, src, tgt, phi, kappa=args.kappa, ground=ground, prep=prep)
                exclude = set(eff_t) | set(tgt) | set(ground or [])
                cands = candidate_pools(src, degree, pool, exclude)
                s1 = {s: np.zeros(len(rxn)) for s in SCORES}
                s2 = {s: np.zeros(len(rxn)) for s in SCORES}
                traffic = np.zeros(len(rxn), dtype=int)
                for k in range(args.perms):
                    rs = draw(cands, rng)
                    phi_k = {**{t: phi.get(t, 0.0) for t in tgt}, **{r: phi[s] for r, s in zip(rs, src)}}
                    res = absorbing_walk(graph, rs, tgt, phi_k, kappa=args.kappa, ground=ground, prep=prep)
                    for s in SCORES:
                        v = getattr(res, s)[rxn]
                        s1[s] += v
                        s2[s] += v * v
                    traffic += res.conductivity[rxn] > _EPS
                    perm_rows.append({"configuration": case, "perm": k, "n_sources": len(rs),
                                      "success_fraction": res.walk_info["walk_success_fraction"],
                                      "sources_unreached": int(sum(prep.h[r] <= _EPS for r in rs)),
                                      "sources_off_main": int(sum(not main_comp[r] for r in rs))})
                df = base.copy()
                df.insert(0, "configuration", case)
                df["n_sources"], df["n_targets"] = len(src), len(eff_t)
                df["n_perms_with_traffic"] = traffic
                for s in SCORES:
                    mean = s1[s] / args.perms
                    sd = np.sqrt(np.maximum(s2[s] / args.perms - mean ** 2, 0.0))
                    o = getattr(obs, s)[rxn]
                    df[f"null_mean_{s}"], df[f"null_sd_{s}"], df[f"observed_{s}"] = mean, sd, o
                    df[f"z_{s}"] = np.where(sd > _EPS, (o - mean) / np.where(sd > _EPS, sd, 1.0), np.nan)
                perm_rows.append({"configuration": case, "perm": "observed", "n_sources": len(src),
                                  "success_fraction": obs.walk_info["walk_success_fraction"],
                                  "sources_unreached": obs.walk_info["walk_sources_unreached"],
                                  "sources_off_main": int(sum(not main_comp[s] for s in src))})
                node_frames.append(df)
                slopes += slope_rows(df, case, rng)
                print(f"{orientation} {pool_name} {case}: {len(src)} sources, {len(eff_t)} targets, "
                      f"{time.time() - t0:.0f}s", flush=True)

            nodes = pd.concat(node_frames, ignore_index=True)
            nodes.to_csv(run / "nodes.tsv", sep="\t", index=False)
            pooled = nodes.copy()                                  # each configuration scaled to mean null 1
            for s in SCORES:
                scale = pooled.groupby("configuration")[f"null_mean_{s}"].transform("mean")
                pooled[f"null_mean_{s}"] = pooled[f"null_mean_{s}"] / scale.where(scale > 0, 1.0)
            slopes += slope_rows(pooled, "pooled (each configuration scaled to mean 1)", rng)
            slope = pd.DataFrame(slopes)
            slope.to_csv(run / "slope.tsv", sep="\t", index=False)
            perms = pd.DataFrame(perm_rows)
            perms.to_csv(run / "perms.tsv", sep="\t", index=False)
            info = {"stage": args.stage, "metabolites": int((~isr).sum()), "reaction_nodes": int(isr.sum()),
                    "edges": len(graph.edges), "walk_states": int(ws.n_states), "method": "walk",
                    "orientation": orientation, "kappa": args.kappa,
                    "kappa_mode": "every walker absorbs at the first target (target phi unused)"
                    if args.kappa is None else "partial absorption, real target phi",
                    "source_pool": pool_name, "pool_size": int(len(pool)), "perms": args.perms, "seed": args.seed,
                    "degree_matching": "|log2 degree difference| <= 0.5, widened by 0.5 until >= 10 candidates",
                    "configurations": [f[0] for f in configs], "skipped": skipped,
                    "runtime_s": round(time.time() - t0, 1)}
            (run / "run_info.json").write_text(json.dumps(info, indent=2))
            write_report(run, info, nodes, slope, perms)
            print(f"wrote {run} in {time.time() - t0:.0f}s", flush=True)


def write_report(run, info, nodes, slope, perms):
    pooled = slope[slope.configuration.str.startswith("pooled")]
    per = slope[~slope.configuration.str.startswith("pooled")]
    lines = [f"# Null calibration: {info['stage']}, sources = {info['orientation']}, pool = {info['source_pool']}",
             "", f"{info['perms']} permutations per configuration, kappa = {info['kappa']} ({info['kappa_mode']}); "
             f"{len(info['configurations']) - len(info['skipped'])} configurations run, "
             f"{len(info['skipped'])} skipped.", "", "## Pooled slope of null mean on hub-adjacency", "",
             "| score | hub | slope | 95% CI | Spearman rho |", "|---|---|---|---|---|"]
    for _, r in pooled.iterrows():
        lines.append(f"| {r.score} | {r.hub} | {r.slope:.3g} | {r.ci_low:.3g} to {r.ci_high:.3g} | {r.spearman_rho:.3f} |")
    lines += ["", "## Spread across configurations (Spearman rho)", "", "| score | hub | median | min | max |",
              "|---|---|---|---|---|"]
    for (s, h), g in per.groupby(["score", "hub"]):
        lines.append(f"| {s} | {h} | {g.spearman_rho.median():.3f} | {g.spearman_rho.min():.3f} | "
                     f"{g.spearman_rho.max():.3f} |")
    und = nodes.z_conductivity.isna().mean()
    null_p = perms[perms.perm != "observed"]
    obs_p = perms[perms.perm == "observed"].set_index("configuration")
    lines += ["", "## Pitfalls", "",
              f"- Reaction nodes with undefined z (no null traffic): {und:.1%} of node x configuration rows.",
              f"- Walk success fraction: real sources median {obs_p.success_fraction.median():.3f}, "
              f"random sources median {null_p.success_fraction.median():.3f}.",
              f"- Random sources off the main network: {null_p.sources_off_main.sum() / null_p.n_sources.sum():.1%}; "
              f"random sources that reach no target: {null_p.sources_unreached.sum() / null_p.n_sources.sum():.1%} "
              f"(real sources: {obs_p.sources_unreached.sum() / obs_p.n_sources.sum():.1%}).",
              "", "## Size of the effect", "",
              "Null mean of the top 10% of reaction nodes by hub_sum, divided by that of the bottom 50% "
              "(median over configurations):", ""]
    for s in SCORES:
        ratios = []
        for _, g in nodes.groupby("configuration"):
            hi = g[g.hub_sum >= g.hub_sum.quantile(0.9)][f"null_mean_{s}"].mean()
            lo = g[g.hub_sum <= g.hub_sum.quantile(0.5)][f"null_mean_{s}"].mean()
            if lo > 0:
                ratios.append(hi / lo)
        lines.append(f"- {s}: {np.median(ratios):.2f} (range {min(ratios):.2f} to {max(ratios):.2f})")
    lines.append("")
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
