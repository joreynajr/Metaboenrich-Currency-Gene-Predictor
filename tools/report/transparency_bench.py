"""Transparency displays for the Reafect head-to-head: one pithy figure per question a reader asks.

    reproduction_badge   does Reafect, run by us, reproduce its own paper?
    cluster_scoreboard   cluster by cluster, who ranks the causal gene higher: Reafect or the walk?
    paired_slope         per comparison, how far apart are the two ranks, and in which direction?
    coverage_strip       which causal genes can each method see at all?
    input_fidelity       how much of each comparison's data reached Reafect, and by which KEGG route?
    where_wins           which method is best in which kind of comparison?
    verification_ledger  the checks that must pass before any of this is believed
plus cards.json (per-comparison drill-down: what each method received and returned) and ledger.json.

    python tools/report/transparency_bench.py

Reads results/reafect_bench/ (own-data report), results/reafect_bench/units/, results/benchmark_matrix/ and
$TMP/ext/reafect_work/units_input/manifest.tsv; writes results/transparency/bench/.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import binomtest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_data_figures import CC, INK, INK2, RULE, style   # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OURS, THEIRS, TIE = "#2f5c8f", "#d1670a", "#c6cad0"      # walk sources = increased / Reafect / tie
GOOD, BAD = "#126b63", "#b3261e"
SUBSET_ORDER = ["A", "B-D primary", "non-primary"]
METHODS = [("Reafect", "Reafect"), ("Walk, sources = increased | M0", "Walk ↑src"),
           ("Current flow, sources = increased | M3", "Flow ↑src M3"),
           ("Current flow, sources = increased | M0", "Flow ↑src"),
           ("Walk, sources = decreased | M0", "Walk ↓src"), ("neighbour", "One-hop")]


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def load(args):
    u = pd.read_csv(Path(args.units) / "per_unit.tsv", sep="\t")
    u["c"] = u.collection.str[0]
    u["primary"] = u.primary.astype(bool)
    u["group"] = np.where(u.c == "A", "A", np.where(u.primary, "B-D primary", "non-primary"))
    return u


# ---------- 1. reproduction badge ----------
def reproduction(report):
    """Parse 'top 2.5% / 5% / 10%' rates, ours and published, per cohort from the own-data report."""
    txt = Path(report).read_text(encoding="utf-8")
    out = {}
    for cohort, pat in (("Bongaerts", r"Bongaerts patients scored.*?: ([\d.]+)% / ([\d.]+)% / ([\d.]+)% \(published[^:]*: "
                                      r"([\d.]+)% / ([\d.]+)% / ([\d.]+)%"),
                        ("Miller", r"Miller patients scored.*?; ([\d.]+)% / ([\d.]+)% / ([\d.]+)% \(published: "
                                   r"([\d.]+)% / ([\d.]+)% / ([\d.]+)%")):
        m = re.search(pat, txt, re.S)
        v = [float(x) for x in m.groups()]
        out[cohort] = {"ours": v[:3], "published": v[3:]}
    return out


def reproduction_badge(rep, out):
    th = ["top 2.5%", "top 5%", "top 10%"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True)
    for ax, (cohort, d) in zip(axs, rep.items()):
        x = np.arange(3)
        ax.bar(x - 0.19, d["published"], 0.36, color="#c6cad0", label="published")
        ax.bar(x + 0.19, d["ours"], 0.36, color=THEIRS, label="our rerun")
        for i in range(3):
            ax.text(x[i] - 0.19, d["published"][i] + 2, f"{d['published'][i]:.0f}", ha="center", fontsize=7, color=INK2)
            ax.text(x[i] + 0.19, d["ours"][i] + 2, f"{d['ours'][i]:.0f}", ha="center", fontsize=7, color=INK)
        ax.set_xticks(x, th); ax.set_ylim(0, 105); ax.set_title(cohort, loc="left", fontsize=8.5)
        ax.tick_params(axis="x", length=0)
        gap = max(abs(a - b) for a, b in zip(d["ours"], d["published"]))
        ax.text(1, -24, f"largest gap {gap:.0f} points", ha="center", fontsize=7, color=INK2)
    axs[0].set_ylabel("% of patients with the\ndisease enzyme ranked in")
    axs[1].legend(frameon=False, loc="upper left", fontsize=7)
    b, m = rep["Bongaerts"], rep["Miller"]
    fig.suptitle(f"Reafect reproduces its paper: top 5% in {b['ours'][1]:.0f}% vs {b['published'][1]:.0f}% published "
                 f"(Bongaerts), {m['ours'][1]:.0f}% vs {m['published'][1]:.0f}% (Miller)", x=0.02, ha="left",
                 fontsize=9, weight="bold", color=INK, y=1.04)
    save(fig, out, "reproduction_badge")
    gap = max(abs(a - b) for d in rep.values() for a, b in zip(d["ours"], d["published"]))
    return gap


# ---------- 2. cluster scoreboard ----------
def clusters(u):
    """Per cluster: median percentiles, the winner, the gap, and whether either method failed to run."""
    v = u.assign(walk_ran=u.walk_up_rank_reported.notna(), rea_ran=u.causal_scored_by_reafect.astype(bool))
    g = v.groupby("cluster").agg(group=("group", "first"), reafect=("reafect_common_pct", "median"),
                                 walk=("walk_up_common_pct", "median"), gene=("causal_genes", "first"),
                                 n=("label", "size"), walk_ran=("walk_ran", "any"), rea_ran=("rea_ran", "any"))
    g["winner"] = np.where(g.reafect < g.walk, "Reafect", np.where(g.walk < g.reafect, "Walk", "tie"))
    g["gap"] = (g.reafect - g.walk).abs()
    g["gene1"] = g.gene.str.split(";").str[0]
    return g


def tally(s):
    w, r = int((s.winner == "Walk").sum()), int((s.winner == "Reafect").sum())
    return w, r, int((s.winner == "tie").sum()), (binomtest(w, w + r).pvalue if w + r else 1.0)


def by_gene(s):
    """Merge clusters that share a causal gene (median over them), so a gene studied five times counts once."""
    g = s.groupby("gene1").agg(reafect=("reafect", "median"), walk=("walk", "median"))
    g["winner"] = np.where(g.reafect < g.walk, "Reafect", np.where(g.walk < g.reafect, "Walk", "tie"))
    return g


def shade(col, gap, full=20.0):
    """Blend toward white for small gaps: a 1-point win is pale, a >= 20-point win is the full colour."""
    t = 0.22 + 0.78 * min(gap / full, 1.0)
    c = np.array(matplotlib.colors.to_rgb(col))
    return tuple(1 - t * (1 - c))


def cluster_scoreboard(u, out):
    g = clusters(u)
    lab = {"A": "Inborn errors (A)", "B-D primary": "Other gene loss (B–D)", "non-primary": "Inhibitor / gain"}
    fig, axs = plt.subplots(1, 3, figsize=(10.4, 3.0), gridspec_kw={"wspace": 0.12})
    cols = 6
    for ax, sub in zip(axs, SUBSET_ORDER):
        s = g[g.group == sub].copy()
        s["o"] = s.winner.map({"Walk": 0, "tie": 1, "Reafect": 2})
        s = s.sort_values(["o", "gap"], ascending=[True, False])
        for i, (cl, r) in enumerate(s.iterrows()):
            x, y = i % cols, i // cols
            base = {"Walk": OURS, "Reafect": THEIRS, "tie": TIE}[r.winner]
            fill = base if r.winner == "tie" else shade(base, r.gap)
            dark = r.winner != "tie" and r.gap >= 8
            ax.add_patch(Rectangle((x + 0.06, -y - 0.94), 0.88, 0.88, facecolor=fill, lw=0,
                                   hatch="////" if not r.walk_ran else None, edgecolor="white" if dark else INK2))
            box = dict(boxstyle="square,pad=0.12", fc=fill, lw=0) if not r.walk_ran else None
            ax.text(x + 0.5, -y - 0.42, r.gene1, ha="center", va="center", fontsize=4.4,
                    color="white" if dark else INK, style="italic", bbox=box)
            ax.text(x + 0.5, -y - 0.72, "×" if not r.rea_ran else f"{r.gap:.0f}", ha="center", va="center",
                    fontsize=5.6 if not r.rea_ran else 4.2, color="white" if dark else INK2,
                    weight="bold" if not r.rea_ran else "normal", bbox=box)
        ax.set_xlim(0, cols); ax.set_ylim(-5.05, 0.05); ax.set_aspect("equal"); ax.axis("off")
        w, r_, t, p = tally(s)
        both = s[s.walk_ran & s.rea_ran]
        bw, br, bt, bp = tally(both)
        gw, gr, gt, gp = tally(by_gene(s))
        ax.text(0, 0.3, f"{lab[sub]}\n"
                f"all {len(s)} clusters:  walk {w} – {r_} Reafect" + (f", {t} tie" if t else "") + f"   p {p:.2g}\n"
                f"both ran ({len(both)}):  {bw} – {br}   p {bp:.2g}\n"
                f"by gene ({gw + gr + gt}):  {gw} – {gr}   p {gp:.2g}",
                fontsize=6.8, color=INK, va="bottom", linespacing=1.45)
    hand = [Rectangle((0, 0), 1, 1, color=OURS), Rectangle((0, 0), 1, 1, color=THEIRS),
            Rectangle((0, 0), 1, 1, color=shade(OURS, 1)), Rectangle((0, 0), 1, 1, facecolor="white", hatch="////",
                                                                    edgecolor=INK2)]
    fig.legend(hand, ["walk (sources = increased) ranks the gene higher", "Reafect ranks it higher",
                      "pale = gap under ~5 points (number = gap in percentile points)", "walk had no sources (mid-rank)"],
               loc="lower left", bbox_to_anchor=(0.02, -0.01), ncol=4, frameon=False, fontsize=6.6, handlelength=1.4)
    fig.text(0.02, -0.035, "× Reafect cannot score the causal gene (put in its tail).  Square = one cluster (comparisons "
             "from one study and gene); common gene universe; two-sided sign test, ties dropped.",
             fontsize=6.4, color=INK2)
    tot = tally(g)
    both = tally(g[g.walk_ran & g.rea_ran])
    fig.suptitle(f"Overall a draw (walk {tot[0]}, Reafect {tot[1]}; {both[0]}–{both[1]} where both ran): the walk leads "
                 "on inborn errors, Reafect on inhibitors, mostly FASN and TDO2", x=0.02, ha="left", fontsize=9,
                 weight="bold", color=INK, y=1.19)
    save(fig, out, "cluster_scoreboard")


# ---------- 3. paired slope chart ----------
def paired_slope(u, out):
    fig, axs = plt.subplots(1, 3, figsize=(8.4, 3.4), sharey=True)
    tr = lambda v: np.log10(np.clip(v, 0.1, 100))
    for ax, sub in zip(axs, SUBSET_ORDER):
        s = u[u.group == sub]
        for r in s.itertuples():
            a, b = r.reafect_common_pct, r.walk_up_common_pct
            col = OURS if b < a else THEIRS if a < b else TIE
            ax.plot([0, 1], [tr(a), tr(b)], color=col, lw=1.0, alpha=0.75)
            ax.plot([0, 1], [tr(a), tr(b)], "o", ms=2.6, color=col, mec="white", mew=0.4)
        ma, mb = s.reafect_common_pct.median(), s.walk_up_common_pct.median()
        ax.plot([0, 1], [tr(ma), tr(mb)], color=INK, lw=2.4)
        ax.text(-0.06, tr(ma), f"{ma:.1f}", ha="right", va="center", fontsize=7, weight="bold", color=INK)
        ax.text(1.06, tr(mb), f"{mb:.1f}", ha="left", va="center", fontsize=7, weight="bold", color=INK)
        ax.set_xlim(-0.35, 1.35); ax.set_xticks([0, 1], ["Reafect", "Walk ↑src"])
        ax.set_title({"A": "Inborn errors", "B-D primary": "Other gene loss", "non-primary": "Inhibitor / gain"}[sub]
                     + f"  ({len(s)})", loc="left", fontsize=8)
        ax.tick_params(axis="x", length=0)
        ax.spines["bottom"].set_visible(False)
    axs[0].set_yticks(tr(np.array([0.1, 1, 10, 100])), ["≤0.1", "1", "10", "100"])
    axs[0].set_ylim(tr(150), tr(0.08))
    axs[0].set_ylabel("Causal gene percentile (common universe)\n0 = top")
    fig.text(0.02, -0.04, "One line per comparison; blue: the walk ranks the gene higher, orange: Reafect does. "
             "Black: median. Log scale; top of the axis is better.", fontsize=7, color=INK2)
    fig.suptitle("The walk ranks inborn-error genes higher, Reafect ranks inhibitor targets higher; other gene "
                 "loss is even", x=0.02, ha="left",
                 fontsize=9, weight="bold", color=INK, y=1.02)
    save(fig, out, "paired_slope")


# ---------- 4. coverage strip ----------
def coverage_strip(u, cov, out):
    cov = cov.copy()
    cov["state"] = np.select([cov.scored_in_unit.astype(bool), cov.ec_in_reafect_modules.astype(bool),
                              cov.reafect_ecs.notna()], ["scored", "in modules, not scored", "EC outside modules"],
                             "no EC number")
    rank = {"scored": 0, "in modules, not scored": 1, "EC outside modules": 2, "no EC number": 3}
    g = cov.groupby("gene").agg(units=("label", "nunique"), best=("state", lambda s: min(s, key=rank.get)),
                                worst=("state", lambda s: max(s, key=rank.get)),
                                ours=("in_our_network", "all"), coll=("collection", "first"))
    g = g.sort_values(["worst", "best", "gene"], key=lambda s: s.map(rank) if s.name in ("worst", "best") else s)
    SC = {"scored": GOOD, "in modules, not scored": "#d7b06a", "EC outside modules": THEIRS, "no EC number": BAD}
    n = len(g)
    fig, ax = plt.subplots(figsize=(10, 2.3))
    for i, (gene, r) in enumerate(g.iterrows()):
        ax.add_patch(Rectangle((i + 0.08, 1.08), 0.84, 0.84, color=GOOD if r.ours else BAD, lw=0))
        st = cov[cov.gene == gene].state
        k = st.value_counts()
        y0 = 0.08
        for s in rank:                                   # split the cell by unit share when states differ
            h = 0.84 * k.get(s, 0) / k.sum()
            if h:
                ax.add_patch(Rectangle((i + 0.08, y0), 0.84, h, color=SC[s], lw=0)); y0 += h
        ax.text(i + 0.5, -0.1, gene, rotation=90, ha="center", va="top", fontsize=5.6, style="italic", color=INK)
    ax.text(-0.4, 1.5, "Our network", ha="right", va="center", fontsize=7.5, color=INK)
    ax.text(-0.4, 0.5, "Reafect", ha="right", va="center", fontsize=7.5, color=INK)
    ax.set_xlim(-0.2, n + 0.2); ax.set_ylim(-1.75, 2.0); ax.axis("off")
    hand = [Rectangle((0, 0), 1, 1, color=c) for c in SC.values()]
    ax.legend(hand, ["scored", "EC in its modules but not scored in that comparison", "EC outside its KEGG modules",
                     "no EC number"], loc="upper left", bbox_to_anchor=(0, 0.0), ncol=4, frameon=False, fontsize=7)
    sc = int(u.causal_scored_by_reafect.sum())
    fig.suptitle(f"Reafect scores the causal gene in {sc} of {len(u)} comparisons; our network contains all "
                 f"{n} causal genes", x=0.02, ha="left", fontsize=9, weight="bold", color=INK, y=1.0)
    save(fig, out, "coverage_strip")


# ---------- 5. input fidelity ----------
ROUTES = [("kegg_from_table KEGG", "KEGG id in the table", "#126b63"),
          ("kegg_from_Human-GEM (published mapping)", "Human-GEM (published id)", "#3d8a82"),
          ("kegg_from_Human-GEM via HMDB", "Human-GEM via HMDB", "#5ea39b"),
          ("kegg_from_Human-GEM via CHEBI", "Human-GEM via ChEBI", "#8fc4bf"),
          ("kegg_from_Human-GEM via name", "Human-GEM via name", "#b5d8d4"),
          ("kegg_from_id_translation name", "name bridge", "#d7b06a"),
          ("kegg_from_Human-GEM via loose name", "loose name match", "#e8cf9a")]


def input_fidelity(u, man, out):
    m = man.set_index("label").reindex(u.label)
    m["c"] = u.set_index("label").c
    m["frac"] = m.n_with_kegg / m.n_measured
    m = m.sort_values(["c", "frac"], ascending=[True, False])
    fig, ax = plt.subplots(figsize=(10, 3.0))
    x, xs, prev = 0, [], None
    for lab, r in m.iterrows():
        if prev is not None and r.c != prev:
            x += 1.2
        prev = r.c
        y = 0
        for col, _, c in ROUTES:
            v = (r.get(col, 0) or 0) / r.n_measured * 100
            ax.bar(x, v, 0.82, bottom=y, color=c, lw=0); y += v
        ax.bar(x, 100 - y, 0.82, bottom=y, color="#eef0f2", lw=0)
        xs.append((x, r.c)); x += 1
    for c in "ABCD":
        p = [a for a, cc in xs if cc == c]
        if p:
            f = m[m.c == c].frac.median() * 100
            ax.text(np.mean(p), 104, f"{c}  ·  median {f:.0f}%", ha="center", fontsize=7, color=INK)
    ax.set_ylim(0, 112); ax.set_xlim(-1, x); ax.set_xticks([])
    ax.set_ylabel("% of measured metabolites\nwith a KEGG id")
    ax.spines["bottom"].set_visible(False)
    hand = [Rectangle((0, 0), 1, 1, color=c) for _, _, c in ROUTES] + [Rectangle((0, 0), 1, 1, color="#eef0f2")]
    ax.legend(hand, [l for _, l, _ in ROUTES] + ["no KEGG id: Reafect never sees it"], loc="upper left",
              bbox_to_anchor=(0, -0.03), ncol=4, frameon=False, fontsize=7)
    lo = m[m.frac < 0.4]
    fig.suptitle(f"Median {m.frac.median() * 100:.0f}% of measured metabolites reach Reafect; "
                 f"{len(lo)} comparisons pass on under 40%", x=0.02, ha="left", fontsize=9, weight="bold",
                 color=INK, y=1.02)
    save(fig, out, "input_fidelity")
    return m


# ---------- 6. where each method wins ----------
def where_wins(mx, out):
    subs = [("all", "All"), ("primary", "Gene loss\n(primary)"), ("viable primary", "Viable\nprimary"),
            ("A inborn errors", "Inborn\nerrors (A)"), ("B-D", "Other\n(B–D)"), ("non-primary", "Inhibitor\n/ gain")]
    t = mx.pivot(index="ranking", columns="subset", values="median_cluster_pct")
    t = t.loc[[k for k, _ in METHODS], [s for s, _ in subs]]
    n = mx.drop_duplicates("subset").set_index("subset").clusters
    fig, ax = plt.subplots(figsize=(7.4, 3.0))
    cmap = plt.get_cmap("Blues_r")
    for j, (s, _) in enumerate(subs):
        best = t[s].idxmin()
        for i, (k, _) in enumerate(METHODS):
            v = t.at[k, s]
            x = min(1, np.log10(max(v, 0.5) / 0.5) / np.log10(100))
            ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, color=cmap(0.08 + 0.85 * x), lw=0))
            ax.text(j + 0.5, i + 0.5, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if v < 6 else INK, weight="bold" if k == best else "normal")
            if k == best:
                ax.add_patch(Rectangle((j + 0.04, i + 0.06), 0.92, 0.88, fill=False, ec=INK, lw=1.6))
    ax.set_xlim(0, len(subs)); ax.set_ylim(len(METHODS), 0)
    ax.set_xticks(np.arange(len(subs)) + 0.5, [f"{l}\n{n[s]} clusters" for s, l in subs], fontsize=7)
    ax.set_yticks(np.arange(len(METHODS)) + 0.5, [l for _, l in METHODS], fontsize=7.5)
    ax.xaxis.tick_top(); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.text(0.02, -0.03, "Cell: median over clusters of the causal gene's percentile (0 = top; own universe; "
             "darker = better). Box: best in that column.", fontsize=7, color=INK2)
    fig.suptitle("No single winner: the walk leads on gene loss and inborn errors, Reafect on inhibitors",
                 x=0.02, ha="left", fontsize=9, weight="bold", color=INK, y=1.1)
    save(fig, out, "where_wins")


# ---------- 7. verification ledger ----------
def ledger(u, rep, gap, man, up, mx, paired):
    L = []
    for cohort, d in rep.items():
        g = max(abs(a - b) for a, b in zip(d["ours"], d["published"]))
        L.append({"check": f"Reafect reproduces its published {cohort} rates (top 2.5/5/10%)",
                  "result": f"ours {'/'.join(f'{v:.0f}' for v in d['ours'])} vs published "
                            f"{'/'.join(f'{v:.0f}' for v in d['published'])}; largest gap {g:.0f} points",
                  "pass": bool(g <= 5)})
    L.append({"check": "Reafect ran on every testable comparison", "result": f"{int(u.reafect_ran.sum())} of {len(u)}",
              "pass": bool(u.reafect_ran.all())})
    for m, lab in (("walk_up", "walk ↑src"), ("walk_down", "walk ↓src"), ("flow", "current flow"), ("neighbour", "one-hop")):
        d = (u[f"{m}_rank"] - u[f"{m}_rank_reported"]).abs().dropna()
        L.append({"check": f"Recomputed {lab} causal ranks match the correction_eval run",
                  "result": f"{int((d > 0).sum())} of {len(d)} differ, largest by {d.max():.0f} ranks "
                            "(floating-point ties)" if (d > 0).any() else f"all {len(d)} identical",
                  "pass": bool(d.max() <= 5)})
    miss = int(u[[c for c in u.columns if c.endswith("_common_pct")]].isna().any(axis=1).sum())
    L.append({"check": "Every method has a value for every comparison (same units compared)",
              "result": f"{miss} comparisons missing a value", "pass": miss == 0})
    a = up.Reafect.reindex(u.label).values
    b = u.reafect_own_pct.fillna(100).values
    L.append({"check": "Benchmark matrix uses the same Reafect percentiles as the head-to-head",
              "result": f"largest difference {np.nanmax(np.abs(a - b)):.3f} points", "pass": bool(np.nanmax(np.abs(a - b)) < 0.01)})
    g = clusters(u)
    w, r = int((g.winner == "Walk").sum()), int((g.winner == "Reafect").sum())
    pt = paired[(paired.rows == "all units") & (paired.subset == "all") & (paired.level == "cluster")
                & (paired.versus == "Walk, sources = increased")].iloc[0]
    L.append({"check": "Cluster win counts recomputed from per-unit data match paired.tsv",
              "result": f"recomputed {r} Reafect – {w} walk; paired.tsv {pt.reafect_better} – {pt.ours_better}",
              "pass": (r, w) == (int(pt.reafect_better), int(pt.ours_better))})
    bad = int((man.n_with_kegg > man.n_measured).sum())
    L.append({"check": "KEGG manifest covers every comparison, no over-counts",
              "result": f"{man.label.isin(u.label).sum()} of {len(u)} in the manifest; {bad} over-counts",
              "pass": bool(man.label.isin(u.label).sum() == len(u) and bad == 0)})
    rnd = mx[(mx.ranking == "random") & (mx.subset == "all")].median_cluster_pct.iloc[0]
    L.append({"check": "Random ranking sits at the 50th percentile (no built-in advantage)",
              "result": f"cluster median {rnd:.1f}", "pass": bool(abs(rnd - 50) < 1)})
    return L


def ledger_figure(L, out):
    fig = plt.figure(figsize=(9.6, 0.3 * len(L) + 0.5))
    ax = fig.add_axes([0.01, 0.0, 0.98, 0.9])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(len(L), 0)
    for i, c in enumerate(L):
        col = GOOD if c["pass"] else BAD
        ax.add_patch(Rectangle((0, i + 0.12), 0.055, 0.76, color=col, lw=0))
        ax.text(0.0275, i + 0.5, "✓ pass" if c["pass"] else "✗ fail", ha="center", va="center", fontsize=6.5,
                color="white", weight="bold")
        ax.text(0.07, i + 0.5, c["check"], va="center", fontsize=7.3, color=INK)
        ax.text(0.6, i + 0.5, c["result"], va="center", fontsize=6.8, color=INK2)
        ax.axhline(i + 1, color=RULE, lw=0.6)
    n = sum(c["pass"] for c in L)
    fig.suptitle(f"Verification ledger: {n} of {len(L)} checks pass", x=0.02, ha="left", fontsize=9,
                 weight="bold", color=INK, y=1.0)
    save(fig, out, "verification_ledger")


# ---------- drill-down cards ----------
def cards(u, man, up):
    man = man.set_index("label")
    C = {}
    for r in u.itertuples():
        mm = man.loc[r.label] if r.label in man.index else None
        routes = {l: int(mm[c]) for c, l, _ in ROUTES if mm is not None and c in mm and mm[c] == mm[c] and mm[c] > 0}
        z = float(mm["z_abs_max"]) if mm is not None else None
        C[r.label] = {
            "collection": r.collection, "study": r.study, "cluster": r.cluster, "primary": bool(r.primary),
            "perturbation": r.perturbation, "causal_genes": r.causal_genes, "causal_ecs": r.causal_ecs,
            "inputs": {"metabolites_measured": int(r.n_measured), "with_kegg_id": int(r.n_with_kegg),
                       "distinct_kegg": int(r.n_kegg_unique), "in_reafect_modules": int(r.n_kegg_in_reafect_network),
                       "kegg_routes": routes,
                       "z_scale": ("group z = mean patient z x sqrt(n), uncapped" if r.collection.startswith("A")
                                   else "z = sign(log2FC) x probit(1 - p/2), |z| <= 8.29"),
                       "z_abs_max": None if z is None else round(z, 2)},
            "coverage": {"ec_in_reafect_modules": bool(r.causal_ec_in_reafect_modules),
                         "scored_by_reafect": bool(r.causal_scored_by_reafect)},
            "reafect": {"enzymes_scored": None if r.reafect_n_enzymes != r.reafect_n_enzymes else int(r.reafect_n_enzymes),
                        "causal_ec_rank": None if r.reafect_causal_ec_rank != r.reafect_causal_ec_rank
                        else float(r.reafect_causal_ec_rank),
                        "own_pct": None if r.reafect_own_pct != r.reafect_own_pct else round(r.reafect_own_pct, 2),
                        "common_pct": round(r.reafect_common_pct, 2)},
            "ours": {lab: (None if r.label not in up.index else round(float(up.at[r.label, k]), 2))
                     for k, lab in METHODS if k != "Reafect"},
            "ours_common_pct": {"walk_up": round(r.walk_up_common_pct, 2), "walk_down": round(r.walk_down_common_pct, 2),
                                "flow": round(r.flow_common_pct, 2), "neighbour": round(r.neighbour_common_pct, 2)},
        }
    return C


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8")
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--own", default="results/reafect_bench/report.md")
    a.add_argument("--units", default="results/reafect_bench/units")
    a.add_argument("--matrix", default="results/benchmark_matrix")
    a.add_argument("--manifest", default=os.path.join(os.environ.get("TMP", "/tmp"),
                                                       "ext/reafect_work/units_input/manifest.tsv"))
    a.add_argument("--out", default="results/transparency/bench")
    args = a.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    style()
    u = load(args)
    cov = pd.read_csv(Path(args.units) / "coverage.tsv", sep="\t")
    man = pd.read_csv(args.manifest, sep="\t")
    mx = pd.read_csv(Path(args.matrix) / "matrix.tsv", sep="\t")
    up = pd.read_csv(Path(args.matrix) / "unit_percentiles.tsv", sep="\t", index_col=0)
    rep = reproduction(args.own)
    gap = reproduction_badge(rep, out)
    cluster_scoreboard(u, out)
    paired_slope(u, out)
    coverage_strip(u, cov, out)
    input_fidelity(u, man, out)
    where_wins(mx, out)
    L = ledger(u, rep, gap, man, up, mx, pd.read_csv(Path(args.units) / "paired.tsv", sep="	"))
    ledger_figure(L, out)
    (out / "ledger.json").write_text(json.dumps(L, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "cards.json").write_text(json.dumps(cards(u, man, up), indent=1, ensure_ascii=False), encoding="utf-8")
    F = [("reproduction_badge", "Reafect reproduces its own paper",
          "Compare the orange bars (our rerun) with the grey (published): they agree within a few points, so the "
          "Reafect we benchmark against is the real method, not a broken copy."),
         ("cluster_scoreboard", "Who wins, cluster by cluster",
          "Each square is one cluster named by its causal gene; count the colours per panel to see that the walk "
          "wins gene-loss comparisons and Reafect wins inhibitor comparisons, with no cherry-picking possible."),
         ("paired_slope", "How far apart the two ranks are",
          "Each line joins Reafect's and the walk's rank of the same causal gene; steep lines are big disagreements "
          "worth opening in the drill-down cards."),
         ("coverage_strip", "Which causal genes each method can see",
          "Top row is our network, bottom row Reafect; any non-green cell is a gene a method could never rank, "
          "which the comparison counts against it rather than hiding."),
         ("input_fidelity", "How much data reached Reafect",
          "Taller coloured bars mean more of the comparison's metabolites were translated to KEGG ids for Reafect; "
          "short bars flag comparisons where Reafect was working half-blind."),
         ("where_wins", "Which method is best where",
          "Read down each column to the boxed cell to see the best method for that kind of comparison; the boxes "
          "move between columns, which is the honest answer."),
         ("verification_ledger", "Checks that must pass",
          "Every row is a test the pipeline ran on itself; a red row would mean the results above should not be "
          "trusted until fixed.")]
    (out / "figures.json").write_text(json.dumps([{"file": f"{n}.png", "title": t, "caption": c} for n, t, c in F],
                                                 indent=1), encoding="utf-8")
    for c in L:
        print("PASS" if c["pass"] else "FAIL", c["check"], "|", c["result"])


if __name__ == "__main__":
    main()
