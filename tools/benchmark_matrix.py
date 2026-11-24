"""One table of every ranking we have against every subset, with Reafect alongside.

Rows: each correction_eval ranking (walk and current flow, both orientations, M0..M3, UNI-*,
M0-LOO), the one-hop neighbour, random, and Reafect. Columns per subset: median causal-gene
percentile over units and over clusters, share of clusters with the gene in the top 1% / 10%,
and two-sided cluster-level sign tests against Reafect and against the one-hop neighbour.

Percentiles are rank / universe (own universe, as in reafect_bench units-compare). A unit a
ranking could not score (no sources in that orientation) is put at the mid-rank, 50%, and a
causal gene Reafect did not score at 100%, so every row covers the same units.

    python tools/benchmark_matrix.py --eval walk_down=results/correction_eval_d26_walk_decreased \
        walk_up=... flow_down=... flow_up=... --reafect results/reafect_bench/units \
        --out results/benchmark_matrix

Writes matrix.tsv (long), matrix.md, matrix.tex, matrix_heatmap.png/pdf and unit_percentiles.tsv
(units x rankings, read by tools/report/build_data_figures.py --ranks).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

LABEL = {"walk_up": "Walk, sources = increased", "walk_down": "Walk, sources = decreased",
         "flow_up": "Current flow, sources = increased", "flow_down": "Current flow, sources = decreased"}
SUBSETS = [("all", lambda u: u.index == u.index),
           ("primary", lambda u: u.primary),
           ("viable primary", lambda u: u.primary & u.viable),
           ("non-primary", lambda u: ~u.primary),
           ("A inborn errors", lambda u: u.collection.str.startswith("A")),
           ("B-D", lambda u: ~u.collection.str.startswith("A"))]


def load(evals, reafect):
    """Unit table (index = unit) and percentile table (units x rankings)."""
    p = pd.read_csv(Path(reafect) / "per_unit.tsv", sep="\t").set_index("label")
    units = p[["collection", "study", "cluster", "primary", "perturbation", "causal_genes"]].copy()
    units["primary"] = units.primary.astype(bool)
    pct = {"Reafect": p.reafect_own_pct.fillna(100.0)}
    viable = None
    for key, d in evals.items():
        d = Path(d)
        info = json.loads((d / "run_info.json").read_text())
        rk = pd.read_csv(d / "rankings.tsv", sep="\t")
        if key == "walk_down":                # viability as in the summary table: >= 3 decreased sources
            viable = rk.drop_duplicates("unit").set_index("unit").viable.astype(bool)
        for m, g in rk.groupby("ranking"):
            name = m if m in ("neighbour", "random") else f"{LABEL[key]} | {m}"
            if name in pct:
                continue
            s = g.set_index("unit")["rank"] / info["universe"] * 100
            pct[name] = s.reindex(units.index).fillna(50.0)
    units["viable"] = viable.reindex(units.index, fill_value=False) if viable is not None else False
    return units, pd.DataFrame(pct).loc[units.index]


def sign(a, b):
    d = a - b
    w, l = int((d < 0).sum()), int((d > 0).sum())
    return w, l, (binomtest(w, w + l).pvalue if w + l else float("nan"))


def matrix(units, pct):
    rows = []
    for sname, f in SUBSETS:
        us = units[f(units)]
        clu = pct.loc[us.index].groupby(us.cluster).median()
        for r in pct.columns:
            w, l, p = sign(clu[r], clu["Reafect"])
            wn, ln, pn = sign(clu[r], clu["neighbour"])
            rows.append({"subset": sname, "ranking": r, "units": len(us), "clusters": len(clu),
                         "median_unit_pct": pct.loc[us.index, r].median(), "median_cluster_pct": clu[r].median(),
                         "top1_clusters": (clu[r] <= 1).mean() * 100, "top10_clusters": (clu[r] <= 10).mean() * 100,
                         "vs_reafect": f"{w}-{l}", "p_vs_reafect": p, "vs_neighbour": f"{wn}-{ln}", "p_vs_neighbour": pn})
    return pd.DataFrame(rows)


def fmt_p(w_l, p):
    return f"{w_l}{'*' if p < 0.05 else ''}" if p == p else w_l


def write(m, out):
    out.mkdir(parents=True, exist_ok=True)
    m.to_csv(out / "matrix.tsv", sep="\t", index=False)
    L = ["# Benchmark matrix", "",
         "Median causal-gene percentile over clusters (0 = top; lower is better), share of clusters with the "
         "gene in the top 10%, and cluster wins-losses against Reafect and the one-hop neighbour "
         "(* two-sided sign test p < 0.05). Own universe; unscored units at 50% (ours) or 100% (Reafect).", ""]
    for s, g in m.groupby("subset", sort=False):
        g = g.sort_values("median_cluster_pct")
        L += [f"## {s}: {g.units.iloc[0]} units, {g.clusters.iloc[0]} clusters", "",
              "| ranking | cluster median % | unit median % | top 10% | vs Reafect | vs neighbour |", "|---|---|---|---|---|---|"]
        L += [f"| {r.ranking} | {r.median_cluster_pct:.1f} | {r.median_unit_pct:.1f} | {r.top10_clusters:.0f}% | "
              f"{fmt_p(r.vs_reafect, r.p_vs_reafect)} | {fmt_p(r.vs_neighbour, r.p_vs_neighbour)} |" for r in g.itertuples()]
        L.append("")
    (out / "matrix.md").write_text("\n".join(L), encoding="utf-8")
    keep = lambda r: "UNI-" not in r and "M1-last" not in r and "hub_max" not in r
    T = [r"\begin{table}[ht]\centering\small", r"\begin{tabular}{lrrrr}", r"\toprule",
         r"Ranking & Cluster med.\ \% & Top 10\% & vs Reafect & vs one-hop \\", r"\midrule"]
    for s in ("primary", "non-primary", "A inborn errors", "B-D"):
        g = m[(m.subset == s) & m.ranking.map(keep)].sort_values("median_cluster_pct")
        T.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{s}: {g.units.iloc[0]} units, {g.clusters.iloc[0]} clusters}}}} \\")
        for r in g.itertuples():
            T.append(f"{r.ranking.replace('|', '--')} & {r.median_cluster_pct:.1f} & {r.top10_clusters:.0f} & "
                     f"{fmt_p(r.vs_reafect, r.p_vs_reafect).replace('*', '$^*$')} & "
                     f"{fmt_p(r.vs_neighbour, r.p_vs_neighbour).replace('*', '$^*$')} \\\\")
        T.append(r"\midrule")
    T[-1] = r"\bottomrule"
    T += [r"\end{tabular}",
          r"\caption{Causal-gene percentile (0 = top) for every scoring and correction, with Reafect. Cluster med.: "
          r"median of per-cluster medians. Top 10\%: share of clusters. vs: clusters where the row ranks the gene "
          r"higher--lower; $^*$ two-sided sign test $p<0.05$.}", r"\end{table}"]
    (out / "matrix.tex").write_text("\n".join(T), encoding="utf-8")
    heatmap(m[m.ranking.map(keep)], out)


def heatmap(m, out):
    """Rankings x subsets, cell = cluster-median percentile, one sequential hue (darker = better)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    subsets = [s for s, _ in SUBSETS]
    t = m.pivot(index="ranking", columns="subset", values="median_cluster_pct")[subsets]
    t = t.loc[t["primary"].sort_values().index]
    n = m.drop_duplicates("subset").set_index("subset")
    fig, ax = plt.subplots(figsize=(8.6, 0.32 * len(t) + 1.6))
    ax.imshow(np.log10(t.clip(lower=0.5)), cmap="Blues_r", vmin=np.log10(0.5), vmax=np.log10(60), aspect="auto")
    for i, r in enumerate(t.index):
        for j, s in enumerate(subsets):
            v = t.loc[r, s]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5, color="white" if v < 6 else "#222")
    ax.set_xticks(range(len(subsets)), [f"{s}\n{n.loc[s, 'clusters']} clusters" for s in subsets], fontsize=8)
    ax.set_yticks(range(len(t)), [r.replace(" | ", "  ·  ") for r in t.index], fontsize=7.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.xaxis.tick_top()
    ax.set_title("Causal-gene percentile, median over clusters (0 = top; darker = better)", fontsize=9, pad=34)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"matrix_heatmap.{ext}", dpi=200)
    plt.close(fig)


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--eval", nargs="+", required=True, help="key=correction_eval dir; keys " + ", ".join(LABEL))
    a.add_argument("--reafect", default="results/reafect_bench/units")
    a.add_argument("--out", default="results/benchmark_matrix")
    args = a.parse_args(argv)
    units, pct = load(dict(e.split("=", 1) for e in args.eval), args.reafect)
    m = matrix(units, pct)
    write(m, Path(args.out))
    pct.rename_axis("label").to_csv(Path(args.out) / "unit_percentiles.tsv", sep="\t", float_format="%.3f")
    print(m[m.subset == "primary"].sort_values("median_cluster_pct").head(12).to_string(index=False))


if __name__ == "__main__":
    main()
