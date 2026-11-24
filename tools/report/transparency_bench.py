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


def reproduction_badge(rep, pp, out):
    """Top block: published rate vs our rerun, gaps in patients. Bottom block: Reafect vs the walk on the same patients,
    ranked within the ~1,060 genes both can score."""
    th = [2.5, 5, 10]
    pat = pp[pp.level == "patient"]
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 2.9), sharey=True, gridspec_kw={"wspace": 0.22})
    for ax, (cohort, d) in zip(axs, rep.items()):
        own = pat[(pat.dataset == cohort)].reafect_own_pct.dropna()
        n = len(own)
        com = pat[(pat.dataset == cohort) & pat.causal_in_common].dropna(
            subset=["reafect_common_pct", "walk_up_common_pct"])
        for i, t in enumerate(th):
            y = i
            k = int((own <= t).sum())
            pub = d["published"][i]
            dk = k - round(pub * n / 100)
            ours = 100 * k / n
            ax.plot([pub, ours], [y, y], color=RULE, lw=2, zorder=1)
            ax.scatter(pub, y, s=46, facecolor="white", edgecolor=INK2, lw=1.4, zorder=3)
            ax.scatter(ours, y, s=46, color=THEIRS, edgecolor="white", lw=1, zorder=4)
            ax.text(103, y, f"{dk:+d}" if dk else "0",
                    va="center", fontsize=6.6, color=INK2)
            y = i + 4
            r = 100 * (com.reafect_common_pct <= t).mean()
            w = 100 * (com.walk_up_common_pct <= t).mean()
            dr = int((com.reafect_common_pct <= t).sum() - (com.walk_up_common_pct <= t).sum())
            ax.plot([w, r], [y, y], color=RULE, lw=2, zorder=1)
            ax.scatter(w, y, s=46, color=OURS, edgecolor="white", lw=1, zorder=3)
            ax.scatter(r, y, s=46, color=THEIRS, edgecolor="white", lw=1, zorder=4)
            ax.text(103, y, f"{dr:+d}", va="center", fontsize=6.6, color=INK2)
        ax.text(30, -0.85, f"Paper vs our rerun · {n} patients", fontsize=6.8, color=INK, weight="bold")
        ax.text(30, 3.15, f"Reafect vs walk · same {len(com)} patients", fontsize=6.8, color=INK, weight="bold")
        ax.text(103, -0.85, "Δ patients", fontsize=6.2, color=INK2)
        ax.set_title(cohort, loc="left", fontsize=8.5, pad=14)
        ax.set_xlim(30, 100); ax.set_ylim(6.6, -1.4)
        ax.set_xticks([40, 60, 80, 100]); ax.grid(axis="x", color=RULE, lw=0.5); ax.set_axisbelow(True)
        ax.spines[["left"]].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.set_xlabel("% of patients whose disease gene is in the top x%", fontsize=7)
    axs[0].set_yticks([0, 1, 2, 4, 5, 6], ["top 2.5%", "top 5%", "top 10%"] * 2, fontsize=7)
    h = [plt.Line2D([], [], ls="", marker="o", mfc="white", mec=INK2, ms=6, label="paper (notebook's saved run)"),
         plt.Line2D([], [], ls="", marker="o", color=THEIRS, ms=6, label="Reafect, our rerun"),
         plt.Line2D([], [], ls="", marker="o", color=OURS, ms=6, label="walk ↑src")]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.06, -0.13), ncol=3, frameon=False, fontsize=7)
    gap = max(abs(a - b) for d in rep.values() for a, b in zip(d["ours"], d["published"]))
    fig.suptitle(f"Our Reafect run matches the paper to within {gap:.0f} points; on its own data it beats the walk",
                 x=0.02, ha="left", fontsize=9, weight="bold", color=INK, y=1.07)
    fig.text(0.02, -0.25, "Only totals can be matched: the paper's notebook saved figures, not per-patient ranks. "
             "Likely cause of the small gaps: KEGG downloaded now, not Jan 2021\n(5 of 184 modules gone). "
             "Upper block ranks among Reafect's ~570 scored enzymes, as published; lower block among the ~1,060 genes "
             "both methods can score.\nPatients whose gene is outside our network (HGD, PAH, MCCC1) are left out of "
             "the lower block.", fontsize=6.3, color=INK2)
    save(fig, out, "reproduction_badge")
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
TAN = "#d7b06a"
# Comparisons that ran but were dropped as "no causal gene in the network" (fate nogene), with the real reason.
# Read from the papers; the registry rows simply had no causal gene filled in.
NOGENE_REASON = {
    "Burrage et al 2019": ("recoverable", "urea-cycle genes (OTC, ASS1, ASL, ARG1 …) not filled in"),
    "Hou et al 2017": ("recoverable", "FH (fumarase siRNA) not filled in"),
    "Reisz et al 2023": ("recoverable", "HPRT1 (Lesch-Nyhan, carrier) not filled in"),
    "Sebaa et al 2023": ("recoverable", "ACADVL (VLCAD deficiency) not filled in"),
    "Du et al 2023|MNGIE vs Healthy Control": ("recoverable", "TYMP (MNGIE) not filled in"),
    "Du et al 2023|MELAS vs Healthy Control": ("not testable", "MELAS: mitochondrial tRNA, no nuclear gene"),
    "Cappuccio et al 2017": ("not testable", "SLC2A1: glucose transporter, not in the network"),
    "Sidorina et al 2024": ("not testable", "disease vs disease (MMA, PA, CblC): no single gene"),
}


def walk_state(r):
    if pd.isna(r.walk_up_rank_reported):
        return "no sources"
    return "never reached" if r.walk_up_rank == 1096 else "ranked"


def reafect_state(r):
    if r.causal_scored_by_reafect:
        return "ranked"
    if r.causal_ec_in_reafect_modules:
        return "in modules, not scored"
    return "EC outside modules" if isinstance(r.causal_ecs, str) and r.causal_ecs else "not an enzyme"


def coverage_strip(u, cov, out):
    v = u.copy()
    v["w"] = v.apply(walk_state, axis=1)
    v["r"] = v.apply(reafect_state, axis=1)
    v["grp"] = np.select([(v.w == "ranked") & (v.r == "ranked"), v.w == "ranked", v.r == "ranked"],
                         ["both", "walk only", "Reafect only"], "neither")
    order = {"both": 0, "walk only": 1, "Reafect only": 2, "neither": 3}
    v["g1"] = v.causal_genes.str.split(";").str[0]
    v = v.sort_values(["grp", "r", "w", "g1"], key=lambda s: s.map(order) if s.name == "grp" else s).reset_index(drop=True)
    gap = 1.2
    xs, x, prev = [], 0.0, None
    for g in v.grp:
        if prev is not None and g != prev:
            x += gap
        xs.append(x); x += 1; prev = g
    v["x"] = xs
    WC = {"ranked": GOOD, "never reached": TAN, "no sources": "white"}
    RC = {"ranked": GOOD, "in modules, not scored": TAN, "EC outside modules": THEIRS, "not an enzyme": BAD}
    fig = plt.figure(figsize=(11.5, 4.3))
    ax = fig.add_axes([0.075, 0.40, 0.915, 0.50])
    for r in v.itertuples():
        ax.add_patch(Rectangle((r.x + 0.06, 1.08), 0.88, 0.84, facecolor=WC[r.w], lw=0.6 if r.w == "no sources" else 0,
                               edgecolor=INK2, hatch="////" if r.w == "no sources" else None))
        ax.add_patch(Rectangle((r.x + 0.06, 0.08), 0.88, 0.84, facecolor=RC[r.r], lw=0))
        n_g = len(str(r.causal_genes).split(";"))
        ax.text(r.x + 0.5, -0.08, r.g1 + (f" +{n_g - 1}" if n_g > 1 else ""), rotation=90, ha="center", va="top",
                fontsize=4.6, style="italic", color=INK)
    lab = {"both": "both rank the gene", "walk only": "only the walk", "Reafect only": "only Reafect", "neither": "neither"}
    for g, d in v.groupby("grp", sort=False):
        x0, x1 = d.x.min(), d.x.max() + 1
        ax.plot([x0 + 0.06, x1 - 0.06], [2.12] * 2, color=INK2, lw=0.8)
        txt = f"{lab[g]}  {len(d)}" if len(d) > 8 else f"{lab[g]}\n{len(d)}".replace("only ", "only\n")
        ax.text((x0 + x1) / 2, 2.22, txt, ha="center", va="bottom", fontsize=7.2 if len(d) > 8 else 6.4, color=INK,
                weight="bold" if g == "both" else "normal")
    ax.text(-0.6, 1.5, "Walk ↑src", ha="right", va="center", fontsize=7.8, color=INK)
    ax.text(-0.6, 0.5, "Reafect", ha="right", va="center", fontsize=7.8, color=INK)
    ax.set_xlim(-0.2, v.x.max() + 1.2); ax.set_ylim(-1.0, 3.0); ax.axis("off")
    hw = [Rectangle((0, 0), 1, 1, color=GOOD), Rectangle((0, 0), 1, 1, color=TAN),
          Rectangle((0, 0), 1, 1, facecolor="white", hatch="////", edgecolor=INK2),
          Rectangle((0, 0), 1, 1, color=THEIRS), Rectangle((0, 0), 1, 1, color=BAD)]
    fig.legend(hw, ["gives the causal gene a real rank",
                    "data-limited: walk never reaches the reaction / Reafect's module has no measured neighbour",
                    "walk has no sources (no increased metabolites)",
                    "method limit: EC number outside Reafect's KEGG modules",
                    "method limit: not an enzyme (no EC number), so Reafect cannot represent it"],
               loc="upper left", bbox_to_anchor=(0.075, 0.33), ncol=2, frameon=False, fontsize=6.6, handlelength=1.3,
               columnspacing=1.6)
    # comparisons dropped before the head-to-head: the network filter, with the real reasons
    reg = pd.read_csv(ROOT / "results/data_figures/units.tsv", sep="\t", dtype=str).fillna("")
    ng = reg[reg.fate == "nogene"]
    rows = []
    for r in ng.itertuples():
        k = f"{r.study}|{r.sub_unit}"
        rows.append(NOGENE_REASON.get(k) or NOGENE_REASON.get(r.study) or ("unknown", r.study))
    t = pd.DataFrame(rows, columns=["kind", "why"]).groupby(["kind", "why"]).size().reset_index(name="n")
    rec = t[t.kind == "recoverable"]; nt = t[t.kind != "recoverable"]
    fig.text(0.075, 0.115, f"Not in this figure: {len(ng)} comparisons dropped earlier as \"no causal gene in the network\"."
             f"  {int(rec.n.sum())} are recoverable (the gene is in the network but was never filled in): " +
             "; ".join(f"{w} ×{n}" if n > 1 else w for w, n in zip(rec.why, rec.n)) + ".",
             fontsize=6.4, color=INK2, wrap=True)
    fig.text(0.075, 0.06, f"{int(nt.n.sum())} are genuinely untestable: " +
             "; ".join(f"{w} ×{n}" if n > 1 else w for w, n in zip(nt.why, nt.n)) +
             ".  One column per comparison (+n: further causal genes; Reafect counts as ranking if it scores any of them).",
             fontsize=6.4, color=INK2)
    nw, nr = int((v.w == "ranked").sum()), int((v.r == "ranked").sum())
    nm = int(v.r.isin(["EC outside modules", "not an enzyme"]).sum())
    fig.suptitle(f"A real rank for the causal gene: walk {nw} of {len(v)} comparisons, Reafect {nr}. Both are sometimes "
                 f"data-limited; {nm} of Reafect's gaps are built into an EC/KEGG-module method", x=0.02, ha="left",
                 fontsize=9, weight="bold", color=INK, y=0.985)
    save(fig, out, "coverage_strip")


ROUTES = [("kegg_from_table KEGG", "KEGG id in the table", "#126b63"),
          ("kegg_from_Human-GEM (published mapping)", "Human-GEM (published id)", "#3d8a82"),
          ("kegg_from_Human-GEM via HMDB", "Human-GEM via HMDB", "#5ea39b"),
          ("kegg_from_Human-GEM via CHEBI", "Human-GEM via ChEBI", "#8fc4bf"),
          ("kegg_from_Human-GEM via name", "Human-GEM via name", "#b5d8d4"),
          ("kegg_from_id_translation name", "name bridge", "#d7b06a"),
          ("kegg_from_Human-GEM via loose name", "loose name match", "#e8cf9a")]


def input_fidelity(u, man, out, mdir):
    """Both methods' inputs side by side: share of measured metabolites each can use, and signal each receives."""
    from scipy.stats import spearmanr
    t = pd.read_csv(ROOT / "results/data_figures/units.tsv", sep="\t")
    t = t[t.testable.astype(str) == "True"].drop_duplicates("label").set_index("label")
    m = man.set_index("label").reindex(u.label)
    d = u.set_index("label")[["c", "n_measured", "n_with_kegg", "n_kegg_in_reafect_network", "reafect_common_pct"]].copy()
    d["ours"] = (t.n_in_network / t.n_metabolites).reindex(d.index) * 100
    d["rea_mod"] = d.n_kegg_in_reafect_network / d.n_measured * 100
    d["rea_kegg"] = d.n_with_kegg / d.n_measured * 100
    d["ours_sig"] = (t.n_dec + t.n_inc).reindex(d.index)
    d["rea_sig"] = [(pd.read_csv(Path(mdir) / f"{m.loc[k, 'file']}.csv").Z_score.abs() >= 1.96).sum() for k in d.index]
    d = d.sort_values(["c", "ours"], ascending=[True, False])
    x, xs, prev = 0.0, [], None
    for c in d.c:
        if prev is not None and c != prev:
            x += 2.0
        xs.append(x); x += 1; prev = c
    d["x"] = xs
    fig = plt.figure(figsize=(11.5, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1], height_ratios=[1.35, 1], hspace=0.62, wspace=0.16,
                          left=0.07, right=0.985, top=0.86, bottom=0.17)
    a1, a2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    a3, a4 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    w = 0.42
    a1.bar(d.x - w / 2, d.ours, w, color=OURS, lw=0, label="ours: in our network")
    a1.bar(d.x + w / 2, d.rea_kegg, w, color="#f3d9c2", lw=0, label="Reafect: KEGG id, but outside its modules")
    a1.bar(d.x + w / 2, d.rea_mod, w, color=THEIRS, lw=0, label="Reafect: inside its KEGG modules (used)")
    for c in "ABCD":
        s = d[d.c == c]
        if len(s):
            a1.text(s.x.mean(), 103, c if len(s) < 5 else
                    f"{c}  ·  median ours {s.ours.median():.0f}%, Reafect {s.rea_mod.median():.0f}%",
                    ha="center", fontsize=6.6, color=INK)
    a1.set_ylim(0, 112); a1.set_xlim(-1, x); a1.set_xticks([]); a1.spines["bottom"].set_visible(False)
    a1.set_ylabel("% of measured\nmetabolites usable", fontsize=7.5)
    a1.legend(loc="upper left", bbox_to_anchor=(0, -0.01), ncol=3, frameon=False, fontsize=6.6, handlelength=1.2)
    a1.set_title("a  How much of each comparison's data each method can use", loc="left", fontsize=8, color=INK)
    a2.vlines(d.x, d.ours_sig.clip(lower=0.8), d.rea_sig.clip(lower=0.8), color=RULE, lw=0.8)
    a2.scatter(d.x, d.ours_sig.clip(lower=0.8), s=9, color=OURS, zorder=3, label="ours: significant sources + targets")
    a2.scatter(d.x, d.rea_sig.clip(lower=0.8), s=9, color=THEIRS, zorder=3, label="Reafect: |z| ≥ 1.96 with a KEGG id")
    a2.set_yscale("log"); a2.set_xlim(-1, x); a2.set_xticks([]); a2.spines["bottom"].set_visible(False)
    a2.set_ylabel("signal metabolites\n(log; 0 drawn at 1)", fontsize=7.5)
    for c in "ABCD":
        s = d[d.c == c]
        if len(s):
            a2.text(s.x.mean(), 0.42, f"{c}: {s.ours_sig.median():.0f} vs {s.rea_sig.median():.0f}", ha="center",
                    fontsize=6.6, color=INK, transform=a2.get_xaxis_transform() if False else a2.transData)
    a2.set_ylim(0.35, max(d.rea_sig.max(), d.ours_sig.max()) * 1.6)
    a2.legend(loc="upper right", ncol=2, frameon=False, fontsize=6.6, handlelength=1.0)
    a2.set_title("b  How many signal metabolites each method is given (median per collection: ours vs Reafect)",
                 loc="left", fontsize=8, color=INK)
    rho, p = spearmanr(d.rea_mod, d.reafect_common_pct)
    CCOL = {"A": "#2f5c8f", "B": "#8a8f96", "C": "#8fc4bf", "D": "#1b3550"}
    for c in "ABCD":
        s = d[d.c == c]
        a3.scatter(s.rea_mod, s.reafect_common_pct.clip(lower=0.05), s=12, color=CCOL[c], label=c, lw=0)
    a3.set_yscale("log"); a3.invert_yaxis()
    a3.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
    a3.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a3.set_xlabel("% of metabolites inside Reafect's modules", fontsize=7)
    a3.set_ylabel("Reafect causal-gene percentile\n(log; top = better)", fontsize=7)
    a3.legend(fontsize=6.2, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.22), ncol=4,
              handletextpad=0.1, columnspacing=0.8, title="collection", title_fontsize=6.2)
    a3.set_title(f"c  Mapping share does not predict Reafect's result\n    (Spearman ρ = {rho:.2f}, p = {p:.2f})",
                 loc="left", fontsize=8, color=INK)
    tot = m.groupby(u.set_index("label").c.reindex(m.index).values)[[r for r, _, _ in ROUTES]].sum()
    tot = tot.div(tot.sum(axis=1), axis=0) * 100
    for i, c in enumerate(tot.index):
        left = 0
        for col, _, colr in ROUTES:
            v = tot.loc[c, col]
            a4.barh(i, v, left=left, color=colr, lw=0, height=0.62); left += v
        name = tot.loc[c, ["kegg_from_id_translation name", "kegg_from_Human-GEM via loose name"]].sum()
        a4.text(101, i, f"{name:.0f}%", va="center", fontsize=6.6, color=INK)
    a4.set_yticks(range(len(tot)), tot.index, fontsize=7); a4.invert_yaxis(); a4.set_xlim(0, 112)
    a4.set_xlabel("% of KEGG ids, by how they were found", fontsize=7)
    a4.set_title("d  Name-based ids (tan, right: share) are unverified", loc="left", fontsize=8, color=INK)
    a4.legend([Rectangle((0, 0), 1, 1, color=c) for _, _, c in ROUTES], [l for _, l, _ in ROUTES],
              loc="upper left", bbox_to_anchor=(-0.05, -0.32), ncol=2, frameon=False, fontsize=5.9, handlelength=1.0)
    A = d[d.c == "A"]
    fig.suptitle(f"Both methods can use about the same share of the data (ours {d.ours.median():.0f}%, Reafect "
                 f"{d.rea_mod.median():.0f}%), Reafect's result does not track its mapping, and on inborn errors\n"
                 f"Reafect is handed {A.rea_sig.median() / A.ours_sig.median():.0f}× the signal metabolites the walk "
                 "uses (group z grows with patient count)", x=0.02, ha="left", fontsize=9, weight="bold", color=INK,
                 y=0.985)
    save(fig, out, "input_fidelity")
    return d, tot


# ---------- 6. where each method wins ----------
WIN_METHODS = [("reafect", "Reafect", THEIRS, "o"), ("walk_up", "Walk ↑src", OURS, "o"),
               ("flow", "Current flow ↓src", "#7a9cc0", "s"), ("walk_down", "Walk ↓src", "#9aa3ad", "D"),
               ("neighbour", "One-hop", "#c6cad0", "^")]


def where_wins(u, own, out):
    """One row per kind of comparison: each method's median causal-gene percentile among the ~1,060 genes both
    methods can score (clusters, then median), and the paired cluster sign test, walk vs Reafect."""
    rows = [("All comparisons", np.ones(len(u), bool), None),
            ("Gene loss", u.primary.values, "By perturbation"), ("Inhibitor / gain", ~u.primary.values, None),
            ("Inborn errors (A)", (u.c == "A").values, "By collection"), ("Other studies (B–D)", (u.c != "A").values, None)]
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    tr = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    y, ylab, yt = 0.0, [], []
    for name, m, head in rows:
        if head:
            y += 0.55
            ax.text(-0.3, y - 0.62, head, fontsize=7, color=INK2, style="italic", transform=tr)
        g = u[m].groupby("cluster")[[f"{k}_common_pct" for k, *_ in WIN_METHODS]].median()
        med = {k: g[f"{k}_common_pct"].median() for k, *_ in WIN_METHODS}
        ax.plot([min(med.values()), max(med.values())], [y, y], color=RULE, lw=1.6, zorder=1)
        for k, lab, col, mk in WIN_METHODS[::-1]:
            ax.scatter(max(med[k], 0.4), y, s=58 if k in ("reafect", "walk_up") else 34, color=col, marker=mk,
                       edgecolor="white", lw=0.9, zorder=4 if k in ("reafect", "walk_up") else 3)
        w = int((g.walk_up_common_pct < g.reafect_common_pct).sum())
        r = int((g.walk_up_common_pct > g.reafect_common_pct).sum())
        p = binomtest(w, w + r).pvalue
        lead = "walk" if w > r else "Reafect"
        verdict = f"{lead} wins" if p < 0.05 else f"{lead} ahead, not significant" if w != r else "even"
        col = (OURS if w > r else THEIRS) if p < 0.05 else INK2
        ax.text(1.03, y - 0.08, verdict, va="center", fontsize=7.4, color=col, weight="bold" if p < 0.05 else "normal",
                transform=tr)
        ax.text(1.03, y + 0.26, f"walk {w} – Reafect {r} clusters, p = {p:.2g}", va="center", fontsize=6, color=INK2,
                transform=tr)
        ylab.append(f"{name}\n{len(g)} clusters"); yt.append(y)
        y += 1
    ax.set_xscale("log"); ax.set_xlim(0.35, 75)
    ticks = [0.5, 1, 3, 10, 30]
    ax.set_xticks(ticks, [f"{t:g}%\n≈ rank {max(1, round(t * 10.6))}" for t in ticks], fontsize=6.6)
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.invert_xaxis()
    ax.set_yticks(yt, ylab, fontsize=7.2); ax.set_ylim(y - 0.4, -0.7)
    ax.grid(axis="x", color=RULE, lw=0.5); ax.set_axisbelow(True)
    ax.spines[["left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("median percentile of the causal gene among ~1,060 shared genes  (right = better)", fontsize=7)
    ax.text(1.03, -0.55, "walk ↑src vs Reafect", fontsize=6.6, color=INK, weight="bold", transform=tr)
    h = [plt.Line2D([], [], ls="", marker=mk, color=col, ms=6 if k in ("reafect", "walk_up") else 5, label=lab)
         for k, lab, col, mk in WIN_METHODS]
    fig.legend(handles=h, loc="lower left", bbox_to_anchor=(0.12, -0.2), ncol=5, frameon=False, fontsize=6.8)
    ow = own[(own.ranking.isin(["Reafect", "Walk, sources = increased | M0"])) & (own.subset == "all")]
    ow = ow.set_index("ranking").median_cluster_pct
    fig.text(0.02, -0.27, "Why the shared list: on each method's own list (Reafect ~570 enzymes, ours 2,191 genes, many "
             f"unreachable) the walk looks better overall ({ow['Walk, sources = increased | M0']:.1f}% vs "
             f"{ow['Reafect']:.1f}%) only because its list is padded.\nCurrent flow ↑src and the degree corrections "
             "(M3) have no shared-list ranks yet and are left out. Sign test on cluster medians, ties dropped.",
             fontsize=6.3, color=INK2)
    fig.suptitle("Reafect wins on inhibitors and on studies B–D; the walk leads on inborn errors, not significantly",
                 x=0.02, ha="left", fontsize=9, weight="bold", color=INK, y=1.02)
    fig.subplots_adjust(right=0.72)
    save(fig, out, "where_wins")


# ---------- 7. verification ledger ----------
# Thresholds are recorded in docs/decisions.md D27. state: pass / caveat / fail; "caveat" rows are known limits that
# no threshold can clear, listed so the reader weighs them.
def rank_check(u, m, ties):
    d = (u[f"{m}_rank"] - u[f"{m}_rank_reported"]).abs().dropna()
    moved = u.loc[d[d > 0].index, "label"]
    if not len(moved):
        return f"all {len(d)} identical", "pass"
    t = ties[ties.method == m].set_index("label").reindex(moved) if ties is not None else None
    if t is None or t.tie.isna().any():
        return f"{len(moved)} of {len(d)} differ, by up to {d.max():.0f} ranks; tie test not run", "caveat"
    nt = int(t.tie.astype(bool).sum())
    if nt == len(moved):
        return f"{len(moved)} of {len(d)} differ, by up to {d.max():.0f} ranks; all exact score ties", "pass"
    rel = t[~t.tie.astype(bool)].max_rel_score_diff.max()
    return (f"{len(moved)} of {len(d)} differ, by up to {d.max():.0f} ranks; {nt} exact ties, {len(moved) - nt} "
            f"scores differ by up to {100 * rel:.2f}% (unexplained)", "caveat" if d.max() <= 5 else "fail")


def ledger(u, rep, man, up, mx, paired, fid, ties):
    d, tot = fid
    L = []

    def add(group, check, threshold, result, state):
        L.append({"group": group, "check": check, "threshold": threshold, "result": result, "state": state,
                  "pass": state == "pass"})

    G1, G2, G3 = "Matches outside evidence", "Internal consistency", "Known caveats"
    for cohort, r in rep.items():
        g = max(abs(a - b) for a, b in zip(r["ours"], r["published"]))
        add(G1, f"Reafect reproduces its published {cohort} rates (top 2.5/5/10%)", "gap ≤ 5 points",
            f"ours {'/'.join(f'{v:.0f}' for v in r['ours'])} vs paper "
            f"{'/'.join(f'{v:.0f}' for v in r['published'])}; largest gap {g:.0f}", "pass" if g <= 5 else "fail")
    for m, lab in (("walk_up", "walk ↑src"), ("walk_down", "walk ↓src"), ("flow", "current flow"),
                   ("neighbour", "one-hop")):
        res, st = rank_check(u, m, ties)
        n = int(u[f"{m}_rank_reported"].notna().sum())
        add(G2, f"Recomputed {lab} ranks match the correction_eval run ({n} with a reported rank)",
            "differences are exact ties", res, st)
    a = up.Reafect.reindex(u.label).values
    b = u.reafect_own_pct.fillna(100).values
    dm = np.nanmax(np.abs(a - b))
    add(G2, "Benchmark matrix uses the same Reafect percentiles as the head-to-head", "< 0.01 points",
        f"largest difference {dm:.3f}", "pass" if dm < 0.01 else "fail")
    g = clusters(u)
    w, r = int((g.winner == "Walk").sum()), int((g.winner == "Reafect").sum())
    pt = paired[(paired.rows == "all units") & (paired.subset == "all") & (paired.level == "cluster")
                & (paired.versus == "Walk, sources = increased")].iloc[0]
    ok = (r, w) == (int(pt.reafect_better), int(pt.ours_better))
    add(G2, "Cluster win counts recomputed from per-comparison data match paired.tsv", "equal",
        f"{r} Reafect – {w} walk both ways", "pass" if ok else "fail")
    bad = int((man.n_with_kegg > man.n_measured).sum())
    inm = int(man.label.isin(u.label).sum())
    add(G2, "KEGG manifest covers every comparison, no over-counts", "all, 0 over-counts",
        f"{inm} of {len(u)}; {bad} over-counts", "pass" if inm == len(u) and bad == 0 else "fail")
    add(G2, "No method is given the causal gene", "absent from every input",
        "ours: metabolites only; Reafect: EC only labels its output (Reafect.py:684; code read)", "pass")
    rnd = mx[(mx.ranking == "random") & (mx.subset == "all")].median_cluster_pct.iloc[0]
    add(G2, "Random ranking scores 50% (sanity check of the scoring code)", "|median − 50| < 1",
        f"cluster median {rnd:.1f}", "pass" if abs(rnd - 50) < 1 else "fail")
    ws, rs = u.apply(walk_state, axis=1), u.apply(reafect_state, axis=1)
    add(G3, "Causal gene given a real rank (others are filled in: walk 50%, Reafect end of list)", "—",
        f"walk {int((ws == 'ranked').sum())} of {len(u)}, Reafect {int((rs == 'ranked').sum())} of {len(u)}", "caveat")
    A = d[d.c == "A"]
    add(G3, "Signal metabolites handed to each method, inborn errors (A)", "—",
        f"median: Reafect {A.rea_sig.median():.0f}, walk {A.ours_sig.median():.0f} (group z grows with patient count)",
        "caveat")
    nm = tot[["kegg_from_id_translation name", "kegg_from_Human-GEM via loose name"]].sum(axis=1)
    add(G3, "KEGG ids found by name matching, not verified by hand", "—",
        ", ".join(f"{c} {v:.0f}%" for c, v in nm.items()), "caveat")
    add(G3, "Test-set labels changed after first results", "—",
        "2 comparisons relabelled inherited defect (Adam 2013 FH, Kennedy 2019 ABAT; D26 item 5); ranks unchanged",
        "caveat")
    reg = pd.read_csv(ROOT / "results/data_figures/units.tsv", sep="\t", dtype=str).fillna("")
    ng = reg[reg.fate == "nogene"]
    kinds = [(NOGENE_REASON.get(f"{x.study}|{x.sub_unit}") or NOGENE_REASON.get(x.study) or ("unknown",))[0]
             for x in ng.itertuples()]
    add(G3, "Comparisons dropped as \"no causal gene in the network\"", "—",
        f"{len(ng)} dropped; {kinds.count('recoverable')} recoverable (gene annotation never filled in)", "caveat")
    return L


def ledger_figure(L, out):
    groups = list(dict.fromkeys(c["group"] for c in L))
    rows = []
    for gname in groups:
        rows.append(("head", gname)); rows += [("row", c) for c in L if c["group"] == gname]
    fig = plt.figure(figsize=(11.2, 0.29 * len(rows) + 0.5))
    ax = fig.add_axes([0.01, 0.0, 0.98, 0.95])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(len(rows), 0)
    COL = {"pass": (GOOD, "✓ pass"), "caveat": ("#b7791f", "! caveat"), "fail": (BAD, "✗ fail")}
    for i, (kind, c) in enumerate(rows):
        if kind == "head":
            ax.text(0, i + 0.62, c, fontsize=7.6, weight="bold", color=INK)
            if i == 0:
                ax.text(0.47, i + 0.62, "threshold", fontsize=6.6, color=INK2)
                ax.text(0.6, i + 0.62, "result", fontsize=6.6, color=INK2)
            continue
        col, txt = COL[c["state"]]
        ax.add_patch(Rectangle((0, i + 0.14), 0.058, 0.72, color=col, lw=0))
        ax.text(0.029, i + 0.5, txt, ha="center", va="center", fontsize=6.2, color="white", weight="bold")
        ax.text(0.068, i + 0.5, c["check"], va="center", fontsize=6.7, color=INK)
        ax.text(0.47, i + 0.5, c["threshold"], va="center", fontsize=6.3, color=INK2)
        ax.text(0.6, i + 0.5, c["result"], va="center", fontsize=6.3, color=INK2 if c["state"] == "pass" else INK)
        ax.axhline(i + 1, color=RULE, lw=0.6)
    chk = [c for c in L if c["threshold"] != "—"]
    n = {k: sum(c["state"] == k for c in chk) for k in COL}
    known = len(L) - len(chk)
    fig.suptitle(f"Verification ledger: {n['pass']} of {len(chk)} checks pass, {n['caveat']} with a caveat, "
                 f"{n['fail']} fail; plus {known} known caveats to weigh", x=0.01, ha="left", fontsize=9,
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
    gap = reproduction_badge(rep, pd.read_csv(Path(args.own).parent / "per_patient.tsv", sep="\t"), out)
    cluster_scoreboard(u, out)
    paired_slope(u, out)
    coverage_strip(u, cov, out)
    fid = input_fidelity(u, man, out, Path(args.manifest).parent)
    where_wins(u, mx, out)
    tf = out / "tie_check.tsv"
    L = ledger(u, rep, man, up, mx, pd.read_csv(Path(args.units) / "paired.tsv", sep="\t"), fid,
               pd.read_csv(tf, sep="\t") if tf.exists() else None)
    ledger_figure(L, out)
    (out / "ledger.json").write_text(json.dumps(L, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "cards.json").write_text(json.dumps(cards(u, man, up), indent=1, ensure_ascii=False), encoding="utf-8")
    F = [("reproduction_badge", "Reafect reproduces its own paper",
          "Top: hollow dot is the paper, orange our rerun; they sit within a few patients, so the Reafect we "
          "benchmark is the real method. Bottom: on the same patients, Reafect (orange) ranks the disease gene "
          "higher than the walk (blue), its home ground."),
         ("cluster_scoreboard", "Who wins, cluster by cluster",
          "Each square is one cluster; deeper colour means a bigger gap, hatching or an x means one method could not "
          "run, and each panel is re-scored with only clusters where both ran and with repeated genes merged."),
         ("paired_slope", "How far apart the two ranks are",
          "Each line joins Reafect's and the walk's rank of the same causal gene; steep lines are big disagreements "
          "worth opening in the drill-down cards."),
         ("coverage_strip", "Which comparisons each method can actually rank",
          "Both rows ask the same question, did the method give the causal gene a real rank; tan gaps are data "
          "limits either method can hit, orange and red are limits built into an EC/KEGG-module method."),
         ("input_fidelity", "How much data each method can use",
          "Blue and orange bars side by side show both methods lose a similar share of metabolites; the scatter "
          "shows Reafect's result does not depend on our mapping, and panel b shows the extra signal it gets on A."),
         ("where_wins", "Which method is best where",
          "Each row is a kind of comparison; dots further right rank the causal gene higher, all on the same "
          "~1,060-gene list. The right-hand verdict is the paired cluster test of walk vs Reafect."),
         ("verification_ledger", "Checks that must pass, and caveats to weigh",
          "Green rows met a threshold fixed in decisions.md D27; amber rows are differences the checks cannot "
          "explain, or known limits no threshold can clear; a red row would mean the results above should not be "
          "trusted until fixed.")]
    (out / "figures.json").write_text(json.dumps([{"file": f"{n}.png", "title": t, "caption": c} for n, t, c in F],
                                                 indent=1), encoding="utf-8")
    for c in L:
        print(c["state"].upper(), c["check"], "|", c["result"])


if __name__ == "__main__":
    main()
