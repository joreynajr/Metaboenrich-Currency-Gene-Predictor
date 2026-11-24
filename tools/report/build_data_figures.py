"""Figures describing the evaluation data: where it came from, what survives each
criterion, and what the curated test set is made of.

  data_map.png           a  the discovery sweep (tools/find_test_data.py, screen, curation)
                         b  the registry funnel to testable, primary units, by collection
                         c  every registered comparison as a square, coloured by its fate
                         d  what the usable comparisons are (perturbation, material, data form)
                         e  decreased vs increased metabolites in the network, per comparison
  test_set_overview.png  the testable set: composition, material, metabolic area of the
                         causal gene, size, and viability for the bias check per orientation
  test_set_map.png       one row per testable unit

Testable (D26): human, data usable, runs in at least one orientation, and the causal gene is
in the three_rule network, so its rank is a prediction that can be scored. Primary: the gene
was lost (inherited defect, gene removed or lost). Viable for the bias check (D22):
>= 3 sources and >= 20 metabolites, per orientation.

Inputs: benchmarks/data_registry.tsv; the null calibration units.tsv for each orientation
(walk; status and source counts); results/data_discovery/ (sweep, screen, curation).

    python tools/report/build_data_figures.py --down <run dir> --up <run dir> --out results/data_figures
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

# palette shared with the earlier data map
CC = {"A": "#2f5c8f", "B": "#5b8bbf", "C": "#a9c4de", "D": "#1b3550"}
CT = {"A": "#2f5c8f", "B": "#4a79ad", "C": "#5b7fa6", "D": "#1b3550"}
COLL = {"A": "A — patient-cohort papers", "B": "B — other publications",
        "C": "C — Metabolomics Workbench", "D": "D — MetaboLights"}
PERT = {"inherited defect": "#126b63", "gene removed or lost": "#8fc4bf", "chemically blocked": "#d1670a",
        "gene added or altered": "#d7b06a", "not recorded": "#c6cad0"}
DEC, INC = "#2f5c8f", "#d1670a"
INK, INK2, RULE = "#1d1f22", "#5a5c61", "#e2e5e9"
FATE = [("primary", "testable, gene lost (primary)", "#126b63"),
        ("nonprimary", "testable, inhibitor or gain of function", "#8fc4bf"),
        ("nogene", "runs, but no causal gene in the network", "#d7b06a"),
        ("norun", "no significant metabolites in the network", "#d1670a"),
        ("single", "single subject, no group", "#e8cf9a"),
        ("mouse", "mouse (human only, D25)", "#8c7b6b"),
        ("lowsignal", "under 3 changed metabolites in the network", "#d1670a"),
        ("panel", "targeted panel, not processed (recheck, D29)", "#6f7276"),
        ("fewsamples", "under 3 samples per side", "#7d8087"),
        ("lt3", "under 3 metabolites in the network", "#9a9da1"),
        ("raw", "raw data only / table not loaded", "#c6cad0"),
        ("novalues", "deposit has no values", "#a99a8a"),
        ("tracing", "isotope tracing", "#6a5190"),
        ("lipid", "lipid panel only", "#3f2f5c"),
        ("caselist", "excluded in the benchmark case list", "#4a4d53"),
        ("other", "set aside by hand (weak design, other)", "#c2b6a6")]
FC = {k: c for k, _, c in FATE}
FL = {k: l for k, l, _ in FATE}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9aa0a6",
                         "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "axes.titlecolor": INK})


def letter(ax, s, dx=-0.08, dy=1.04):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=11, weight="bold", va="bottom", ha="left", color=INK)


def open_y(ax):
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ---- the unit table ----------------------------------------------------------------
def gene_subsystems():
    """Most common Human-GEM subsystem of each network gene's reactions."""
    from metaboenrich import structure
    from metaboenrich.gem import load_sbml, load_subsystems
    from metaboenrich.network import build_graph
    model = load_sbml(str(ROOT / "data/Human-GEM.xml"))
    st = next(s for s in structure.stages(model, "three_rule", atom_dir=str(ROOT / "data")) if s[0] == "three_rule")
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    subs = load_subsystems(str(ROOT / "data/Human-GEM.yml"))
    per = {}
    for r, genes in graph.reaction_genes.items():
        ss = [s for rid in graph.reactions_of(r) for s in subs.get(rid, subs.get(f"R_{rid}", []))]
        for g in genes:
            per.setdefault(model.genes[g], Counter()).update(ss)
    return {g: c.most_common(1)[0][0] for g, c in per.items() if c}, {model.genes[g] for g in
                                                                      set().union(*graph.reaction_genes.values())}


def unit_table(down, up):
    reg = pd.read_csv(ROOT / "benchmarks/data_registry.tsv", sep="\t", dtype=str).fillna("")
    keys = ["collection", "study", "sub_unit"]
    d = pd.read_csv(down / "units.tsv", sep="\t", dtype=str).fillna("")
    u = pd.read_csv(up / "units.tsv", sep="\t", dtype=str).fillna("")
    u = u.drop_duplicates(keys + ["label"])[keys + ["label", "status", "n_sources", "viable"]]
    d = d.drop_duplicates(keys + ["label"])
    t = d.merge(u.rename(columns={"status": "status_up", "n_sources": "n_sources_up", "viable": "viable_up_s"}),
                on=keys + ["label"], how="left")
    t = t.merge(reg[keys + ["causal_genes", "perturbation", "sample_matrix", "data_format", "sample_detail",
                            "status", "drop_reason", "n_samples"]].rename(columns={"status": "reg_status"}),
                on=keys, how="left").fillna("")
    t["status_down"] = t.status
    t["status_up"] = t.status_up.where(t.status_up != "", t.status)
    num = lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    t["n_dec"] = num(t.n_sources)                          # sources with sources = decreased
    t["n_inc"] = num(t.n_sources_up)                       # sources with sources = increased
    t["viable_down"] = t.viable.eq("True")
    t["viable_up"] = t.viable_up_s.eq("True")
    # the registry decides membership (D28): a unit dropped there is out even if an older null-calibration run has it
    t["runs"] = (t.status_down.eq("run") | t.status_up.eq("run")) & t.reg_status.eq("active")
    subsys, net = gene_subsystems()
    t["genes_in_net"] = [";".join(g for g in c.split(";") if g in net) for c in t.causal_genes]
    t["primary"] = t.perturbation.isin(["inherited defect", "gene removed or lost"])
    t["testable"] = t.runs & t.genes_in_net.ne("")
    t["subsystem"] = [subsys.get(g.split(";")[0], "") if g else "" for g in t.genes_in_net]
    sys.path.insert(0, str(ROOT / "tools"))
    from correction_eval import cluster_of
    t["cluster"] = [cluster_of(r) for r in t.itertuples()]

    def fate(r):
        s, why = r.status_down, f"{r.drop_reason} {r.status_down}".lower()
        if r.testable:
            return "primary" if r.primary else "nonprimary"
        if r.runs:
            return "nogene"
        if "single subject" in why:
            return "single"
        if "mouse" in why:
            return "mouse"
        if why.startswith("d29:"):
            return "lowsignal"
        if "minimum of 3 per side" in why:
            return "fewsamples"
        if ("<15" in why or "panel" in why) and "lipid" not in why:
            return "panel"
        if "fewer than 3" in why or "lt3" in why:
            return "lt3"
        if "raw only" in why or "processed" in why or "usable table" in why or "not loaded" in why:
            return "raw"
        if "no values" in why:
            return "novalues"
        if "tracing" in why:
            return "tracing"
        if "lipid" in why:
            return "lipid"
        if "cases.tsv" in why:
            return "caselist"
        if "no sources" in s or "no targets" in s:
            return "norun"
        return "other"
    t["fate"] = [fate(r) for r in t.itertuples()]
    t["c"] = t.collection.str[0]
    return t


# ---- figure 1: data map ---------------------------------------------------------------
def sweep_counts():
    dd = ROOT / "results/data_discovery"
    cand = pd.read_csv(dd / "candidates.tsv", sep="\t", dtype=str).fillna("")
    screen = pd.read_csv(dd / "screen.tsv", sep="\t", dtype=str).fillna("")
    cur = pd.read_csv(dd / "curated.tsv", sep="\t", dtype=str).fillna("")
    rep = (dd / "report.md").read_text(encoding="utf-8")
    m = dict(re.findall(r"(workbench|metabolights|literature) (\d+)", rep.split("\n")[2]))
    reg = pd.read_csv(ROOT / "benchmarks/data_registry.tsv", sep="\t", dtype=str).fillna("")
    old = {"ST", "MTBLS"}
    hits = cand[(cand.in_network == "True") & (cand.perturbation == "loss") & (cand.design_flags == "")
                & (cand.source != "literature")]
    d26 = reg[reg.sample_detail.ne("")]                 # rows the sweep added (D26) carry a sample description
    return {"swept": {"workbench": 4721, "metabolights": 3705},
            "human": {"workbench": int(m.get("workbench", 0)), "metabolights": int(m.get("metabolights", 0))},
            "hits": hits.groupby("source").id.nunique().to_dict(),
            "screened": screen.groupby("source").id.nunique().to_dict(),
            "registered": {"workbench": d26[d26.collection == "C_workbench"].study.nunique(),
                           "metabolights": d26[d26.collection == "D_metabolights"].study.nunique()},
            "registered_units": len(d26), "literature": int(m.get("literature", 0)),
            "literature_hits": cand[(cand.source == "literature") & (cand.in_network == "True")
                                    & (cand.perturbation == "loss") & (cand.in_registry == "")].id.nunique()}


def data_map(t, out):
    s = sweep_counts()
    fig = plt.figure(figsize=(11.2, 15.2))
    gs = fig.add_gridspec(4, 2, height_ratios=[0.62, 0.78, 1.55, 1.45], width_ratios=[1.12, 1.0], hspace=0.42,
                          wspace=0.5, left=0.2, right=0.75, top=0.965, bottom=0.04)
    # a: discovery sweep
    ax = fig.add_subplot(gs[0, :])
    stages = ["studies in the repository", "human", "name a perturbed network gene",
              "screened for usable data", "passed curation and registered"]
    keys = ["swept", "human", "hits", "screened", "registered"]
    src = [("workbench", "Metabolomics Workbench", CC["C"]), ("metabolights", "MetaboLights", CC["D"])]
    left = np.zeros(len(stages))
    for k, lab, col in src:
        v = np.array([s[key].get(k, 0) for key in keys], float)
        ax.barh(range(len(stages)), v, left=left, color=col, edgecolor="white", lw=0.8, height=0.62, label=lab)
        left += v
    for i, v in enumerate(left):
        ax.text(v * 1.08 if v > 0 else 1.2, i, f"{int(v):,}", va="center", fontsize=7.5, color=INK)
    ax.set_xscale("log"); ax.set_xlim(1, 30000)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages); ax.invert_yaxis(); open_y(ax)
    ax.set_xlabel("number of studies (log scale)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_title(f"Systematic sweep of both repositories added {s['registered_units']} comparisons "
                 f"from {sum(s['registered'].values())} studies", loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=7)
    ax.text(1.0, -0.42, f"Literature (Europe PMC): {s['literature']:,} papers swept, {s['literature_hits']} name a "
            "perturbed network gene; not yet screened for deposited data.", transform=ax.transAxes, ha="right",
            fontsize=6.8, color=INK2)
    letter(ax, "a", dx=-0.25)

    # b: registry funnel
    ax = fig.add_subplot(gs[1, :])
    stages = ["comparisons registered", "human, data in a usable form", "runs in at least one orientation",
              "causal gene in the network\n(testable prediction)", "gene lost (primary)"]
    masks = [np.ones(len(t), bool), t.fate.isin(["primary", "nonprimary", "nogene", "norun"]), t.runs,
             t.testable, t.testable & t.primary]
    left = np.zeros(len(stages))
    for c in "ABCD":
        v = np.array([((t.c == c) & m).sum() for m in masks], float)
        ax.barh(range(len(stages)), v, left=left, color=CC[c], edgecolor="white", lw=0.8, height=0.62,
                label=COLL[c])
        left += v
    for i, (v, m) in enumerate(zip(left, masks)):
        k = t[m].cluster.nunique() if i >= 3 else None
        ax.text(v + 2, i, f"{int(v)}" + (f"  ({k} clusters)" if k else ""), va="center", fontsize=7.5)
    ax.set_yticks(range(len(stages))); ax.set_yticklabels(stages); ax.invert_yaxis(); open_y(ax)
    ax.set_xlim(0, len(t) * 1.25); ax.set_xlabel("number of comparisons")
    ax.set_title(f"{int(masks[3].sum())} of {len(t)} registered comparisons make a testable prediction",
                 loc="left")
    ax.legend(frameon=False, loc="lower right", fontsize=7, title="source material", title_fontsize=7)
    letter(ax, "b", dx=-0.25)

    # c: waffle by fate
    ax = fig.add_subplot(gs[2, :])
    NC, y, ylab = 24, 0, []
    order = [k for k, _, _ in FATE]
    for c in "ABCD":
        g = t[t.c == c].copy()
        g["o"] = g.fate.map(order.index)
        g = g.sort_values("o")
        nr = int(np.ceil(len(g) / NC))
        ylab.append((y + nr / 2 - 0.07, c, len(g)))
        for i, f in enumerate(g.fate):
            ax.add_patch(Rectangle((i % NC, y + i // NC), 0.86, 0.86, facecolor=FC[f], lw=0))
        y += nr + 0.75
    ax.set_xlim(-0.4, NC + 0.2); ax.set_ylim(y - 0.6, -0.5); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    for yy, c, n in ylab:
        ax.text(-0.7, yy, f"{COLL[c]}\n{n} comparisons", ha="right", va="center", fontsize=7.5)
    present = [k for k in order if (t.fate == k).any()]
    ax.legend([Rectangle((0, 0), 1, 1, color=FC[k]) for k in present],
              [f"{FL[k]} ({(t.fate == k).sum()})" for k in present], frameon=False, loc="upper left",
              bbox_to_anchor=(1.01, 1.0), handlelength=0.95, labelspacing=0.42, fontsize=6.6)
    ax.set_title("Each square is one registered comparison, coloured by what happens to it", loc="left")
    letter(ax, "c", dx=-0.25)

    # d: attributes of the usable comparisons
    ax = fig.add_subplot(gs[3, 0])
    use = t[t.fate.isin(["primary", "nonprimary", "nogene", "norun"])]
    ATTR = [("What was done to the gene", "perturbation", list(PERT)),
            ("Material measured", "sample_matrix", ["patient body fluid", "patient cells", "tissue", "cultured cells",
                                                     "not recorded"]),
            ("Form of the data", "data_format", ["per-sample values", "group-mean z", "z-score", "fold change",
                                                  "unknown"]),
            ("Causal gene", "gene_class", ["in the network", "known, not in the network", "none"])]
    use = use.assign(gene_class=np.where(use.genes_in_net != "", "in the network",
                                         np.where(use.causal_genes != "", "known, not in the network", "none")))
    ypos, yt, ytl, heads = 0, [], [], []
    for name, col, cats in ATTR:
        heads.append((ypos - 0.74, name))
        for cat in cats:
            sub = use[use[col] == cat]
            if not len(sub):
                continue
            l = 0
            for c in "ABCD":
                v = (sub.c == c).sum()
                if v:
                    ax.barh(ypos, v, left=l, color=CC[c], edgecolor="white", lw=0.6, height=0.68); l += v
            ax.text(l + 1.5, ypos, str(int(l)), va="center", fontsize=6.5)
            yt.append(ypos); ytl.append(cat); ypos += 1
        ypos += 1.2
    ax.set_yticks(yt); ax.set_yticklabels(ytl, fontsize=7); ax.invert_yaxis(); open_y(ax)
    ax.set_xlabel("number of comparisons")
    for yy, name in heads:
        ax.text(-0.02, yy, name, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=7.5,
                style="italic", color="#33363b")
    ax.set_title(f"What the {len(use)} usable comparisons are", loc="left")
    letter(ax, "d", dx=-0.75)

    # e: decreased vs increased
    ax = fig.add_subplot(gs[3, 1])
    r = t[t.runs].copy()
    ZY = 0.7
    for lab, sub, col in [("testable, primary", r[r.testable & r.primary], FC["primary"]),
                          ("testable, non-primary", r[r.testable & ~r.primary], FC["nonprimary"]),
                          ("no causal gene in network", r[~r.testable], FC["nogene"])]:
        x = sub.n_dec.where(sub.n_dec > 0, ZY); yv = sub.n_inc.where(sub.n_inc > 0, ZY)
        ax.scatter(x, yv, s=22, c=col, edgecolor="white", lw=0.5, label=f"{lab} ({len(sub)})", zorder=3)
    for v in (3,):
        ax.axvline(v, color="#126b63", lw=0.9, ls=(0, (4, 2.5))); ax.axhline(v, color="#126b63", lw=0.9, ls=(0, (4, 2.5)))
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.6, 300); ax.set_ylim(0.6, 300)
    ticks = [ZY, 1, 3, 10, 30, 100]
    ax.set_xticks(ticks); ax.set_xticklabels(["0", "1", "3", "10", "30", "100"])
    ax.set_yticks(ticks); ax.set_yticklabels(["0", "1", "3", "10", "30", "100"])
    ax.minorticks_off()
    ax.set_xlabel("decreased metabolites in the network"); ax.set_ylabel("increased metabolites in the network")
    ax.text(3.3, 220, "viable: ≥ 3 sources\n(dashed, either side)", fontsize=6.6, color="#126b63", va="top")
    ax.set_title("Changed metabolites on each side", loc="left")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.2), fontsize=6.6, ncol=1)
    letter(ax, "e", dx=-0.3)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"data_map.{ext}", dpi=250)
    plt.close(fig)


# ---- figure 2: test-set overview ---------------------------------------------------------
def overview(t, out):
    ts = t[t.testable].copy()
    fig = plt.figure(figsize=(11.2, 9.6))
    gs = fig.add_gridspec(2, 3, wspace=0.75, hspace=0.45, left=0.1, right=0.98, top=0.88, bottom=0.08)
    n, k = len(ts), ts.cluster.nunique()
    pr = ts[ts.primary]
    fig.text(0.02, 0.975, f"The test set: {n} comparisons in {k} clusters, each with a causal gene in the network",
             fontsize=11.5, weight="bold", color=INK, va="top")
    fig.text(0.02, 0.945, f"Gene lost (primary): {len(pr)} comparisons / {pr.cluster.nunique()} clusters.   "
             f"Inhibitor or gain of function: {n - len(pr)} / {ts[~ts.primary].cluster.nunique()}.   "
             f"{ts.genes_in_net.str.split(';').str[0].nunique()} distinct causal genes.   All human.",
             fontsize=8, color=INK2, va="top")

    # a: comparisons and clusters by collection and perturbation
    ax = fig.add_subplot(gs[0, 0])
    cols = [c for c in "ABCD" if (ts.c == c).any()]
    y = np.arange(len(cols))
    left = np.zeros(len(cols))
    for p, colr in PERT.items():
        v = np.array([((ts.c == c) & (ts.perturbation == p)).sum() for c in cols], float)
        if v.sum():
            ax.barh(y, v, left=left, color=colr, edgecolor="white", lw=0.6, height=0.62, label=p)
            left += v
    for i, c in enumerate(cols):
        kc = ts[ts.c == c].cluster.nunique()
        ax.text(left[i] + 1, i, f"{int(left[i])}  ({kc} cluster{'s' if kc > 1 else ''})", va="center", fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([c + "\n" + COLL[c].split(" — ")[1].replace(" ", "\n", 1) for c in cols])
    ax.invert_yaxis()
    open_y(ax); ax.set_xlim(0, left.max() * 1.6); ax.set_xlabel("comparisons")
    ax.set_title("Source and what was done to the gene", loc="left")
    ax.legend(frameon=False, fontsize=6.4, loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=2)
    letter(ax, "a", dx=-0.45)

    # b: material
    ax = fig.add_subplot(gs[0, 1])
    mats = ts.sample_matrix.value_counts()
    ax.barh(range(len(mats)), mats.values, color="#5a5c61", height=0.62)
    for i, (m, v) in enumerate(mats.items()):
        ax.text(v + 0.8, i, f"{v}  ({ts[ts.sample_matrix == m].cluster.nunique()} cl.)", va="center", fontsize=7)
    ax.set_yticks(range(len(mats))); ax.set_yticklabels(mats.index); ax.invert_yaxis(); open_y(ax)
    ax.set_xlim(0, mats.max() * 1.5); ax.set_xlabel("comparisons"); ax.set_title("Material measured", loc="left")
    letter(ax, "b", dx=-0.42)

    # c: metabolic area of the causal gene
    ax = fig.add_subplot(gs[0, 2])
    ss = ts.drop_duplicates("cluster").subsystem.replace("", "not assigned").value_counts()
    top = ss.head(14)
    if len(ss) > 14:
        top = pd.concat([top, pd.Series({f"{len(ss) - 14} other subsystems": ss.iloc[14:].sum()})])
    ax.barh(range(len(top)), top.values, color="#126b63", height=0.62)
    for i, v in enumerate(top.values):
        ax.text(v + 0.2, i, str(v), va="center", fontsize=7)
    ax.set_yticks(range(len(top))); ax.set_yticklabels([s if len(s) <= 34 else s[:33] + "…" for s in top.index],
                                                       fontsize=6.6)
    ax.invert_yaxis(); open_y(ax); ax.set_xlabel("clusters")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_title("Causal gene's metabolic area\n(Human-GEM subsystem)", loc="left")
    letter(ax, "c", dx=-0.85)

    # d: size distributions
    ax = fig.add_subplot(gs[1, 0])
    met = pd.to_numeric(ts.n_in_network, errors="coerce")
    smp = pd.to_numeric(ts.n_samples, errors="coerce")
    data = [smp.dropna().clip(lower=1), met.dropna().clip(lower=1)]
    for i, (d, lab) in enumerate(zip(data, ["samples per comparison", "metabolites in the network"])):
        jit = (np.random.default_rng(i).random(len(d)) - 0.5) * 0.35
        ax.scatter(np.full(len(d), i) + jit, d, s=10, color=CC["A"], alpha=0.75, lw=0)
        ax.plot([i - 0.25, i + 0.25], [d.median()] * 2, color=INK, lw=1.6)
        ax.text(i + 0.3, d.median(), f"median {d.median():.0f}", fontsize=6.8, va="center")
    ax.set_yscale("log"); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([0, 1]); ax.set_xticklabels(["samples\n(both groups)", "metabolites\nin network"])
    ax.set_xlim(-0.5, 1.9); ax.set_title("Size of each comparison", loc="left")
    nr = int(smp.isna().sum() + (smp == 0).sum())
    ax.text(0, -0.2, f"samples not recorded for {nr} comparison{'s' if nr != 1 else ''}",
            transform=ax.transAxes, fontsize=6.4, color=INK2)
    letter(ax, "d", dx=-0.2)

    # e: changed metabolites per side
    ax = fig.add_subplot(gs[1, 1])
    for i, (col, lab, colr) in enumerate([("n_dec", "decreased", DEC), ("n_inc", "increased", INC)]):
        d = ts[col].clip(lower=0) + 0.7
        jit = (np.random.default_rng(10 + i).random(len(d)) - 0.5) * 0.35
        ax.scatter(np.full(len(d), i) + jit, d, s=10, color=colr, alpha=0.75, lw=0)
        ax.plot([i - 0.25, i + 0.25], [ts[col].median() + 0.7] * 2, color=INK, lw=1.6)
        ax.text(i + 0.3, ts[col].median() + 0.7, f"median {ts[col].median():.0f}\nzero in {(ts[col] == 0).sum()}",
                fontsize=6.8, va="center")
    ax.set_yscale("log"); ax.set_yticks([0.7, 1.7, 3.7, 10.7, 30.7, 100.7])
    ax.set_yticklabels(["0", "1", "3", "10", "30", "100"]); ax.minorticks_off()
    ax.set_xticks([0, 1]); ax.set_xticklabels(["decreased", "increased"]); ax.set_xlim(-0.5, 1.9)
    ax.set_title("Significant metabolites in the network", loc="left")
    letter(ax, "e", dx=-0.2)

    # f: viability per orientation
    ax = fig.add_subplot(gs[1, 2])
    cats = [("both orientations", ts.viable_down & ts.viable_up), ("decreased only", ts.viable_down & ~ts.viable_up),
            ("increased only", ~ts.viable_down & ts.viable_up), ("neither", ~ts.viable_down & ~ts.viable_up)]
    for i, (lab, m) in enumerate(cats):
        l = 0
        for p in (True, False):
            v = (m & (ts.primary == p)).sum()
            ax.barh(i, v, left=l, color=FC["primary" if p else "nonprimary"], height=0.62, edgecolor="white", lw=0.6)
            l += v
        ax.text(l + 0.8, i, f"{int(l)}", va="center", fontsize=7)
    ax.set_yticks(range(4)); ax.set_yticklabels([c[0] for c in cats]); ax.invert_yaxis(); open_y(ax)
    ax.set_xlabel("comparisons"); ax.set_xlim(0, max(m.sum() for _, m in cats) * 1.3)
    ax.set_title("Viable for the bias check,\nby orientation", loc="left")
    ax.legend([Rectangle((0, 0), 1, 1, color=FC["primary"]), Rectangle((0, 0), 1, 1, color=FC["nonprimary"])],
              ["gene lost", "inhibitor / gain"], frameon=False, fontsize=6.6, loc="lower right")
    letter(ax, "f", dx=-0.42)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"test_set_overview.{ext}", dpi=250)
    plt.close(fig)


# ---- figure 3: one row per testable unit ---------------------------------------------------
def unit_map(t, out, ranks=None):
    ts = t[t.testable].copy()
    ts["o"] = ts.c.map("ABCD".index)
    ts = ts.sort_values(["o", "primary", "cluster", "label"], ascending=[True, False, True, True]).reset_index(drop=True)
    y, ys, heads = 0.0, [], []
    for c in "ABCD":
        g = ts[ts.c == c]
        if not len(g):
            continue
        heads.append((y, c, len(g), g.cluster.nunique()))
        y += 1.25
        for _ in range(len(g)):
            ys.append(y); y += 1
        y += 0.6
    ts["y"] = ys
    H = y
    fig = plt.figure(figsize=(12.4, 0.155 * H + 2.4))
    FH = fig.get_figheight()
    if ranks is not None:
        fig.set_figwidth(14.2)
    wr = [7.0, 1.3, 1.9, 1.25] + ([1.55] if ranks is not None else [])
    gs = fig.add_gridspec(1, len(wr), width_ratios=wr, wspace=0.07, left=0.012, right=0.995,
                          top=1 - 1.05 / FH, bottom=1.15 / FH)
    axs = [fig.add_subplot(gs[0, i]) for i in range(len(wr))]
    axT, axM, axC, axF = axs[:4]
    for ax in axs:
        ax.set_ylim(H, -1.4); ax.set_yticks([])
        for sp in ("left", "bottom"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(axis="x", length=0)
    X = {"cond": 1.3, "gene": 50, "mat": 60, "pert": 74, "n": 100}
    axT.set_xlim(0, 101); axT.set_xticks([])
    for name, x, ha in [("Comparison", X["cond"], "left"), ("Gene", X["gene"], "left"), ("Material", X["mat"], "left"),
                        ("What was done to the gene", X["pert"], "left"), ("n", X["n"], "right")]:
        axT.text(x, -0.9, name, ha=ha, va="bottom", fontsize=7.2, weight="bold", color=INK)
    for y0, c, n, k in heads:
        for ax in axs:
            ax.axhline(y0 + 0.55, color=RULE, lw=0.7, zorder=0)
        axT.text(X["cond"] - 1.0, y0 + 0.15, f"{COLL[c]}  ·  {n} comparisons, {k} cluster{'s' if k > 1 else ''}",
                 fontsize=7.6, weight="bold", color=CT[c], va="center")
    disorder = pd.read_csv(ROOT / "benchmarks/cases.tsv", sep="\t", dtype=str).set_index("case_id").disorder
    for r in ts.itertuples():
        axT.add_patch(Rectangle((0, r.y - 0.42), 0.75, 0.84, color=CC[r.c], lw=0))
        name = (disorder.get(r.label, r.label) if r.c == "A" else r.label if not r.sample_detail
                else f"{r.study} {r.sub_unit} · {r.sample_detail}")
        axT.text(X["cond"], r.y, name if len(name) <= 70 else name[:68] + "…", va="center", fontsize=6.2, color=INK)
        g = r.genes_in_net.split(";")
        axT.text(X["gene"], r.y, g[0] + (f" +{len(g) - 1}" if len(g) > 1 else ""), va="center", fontsize=6.2,
                 style="italic", color=INK)
        axT.text(X["mat"], r.y, r.sample_matrix, va="center", fontsize=6.2, color=INK2)
        axT.add_patch(Rectangle((X["pert"], r.y - 0.3), 1.4, 0.6, color=PERT.get(r.perturbation, "#c6cad0"), lw=0))
        axT.text(X["pert"] + 2.2, r.y, r.perturbation, va="center", fontsize=6.2, color=INK2)
        ns = pd.to_numeric(r.n_samples, errors="coerce")
        axT.text(X["n"], r.y, "–" if pd.isna(ns) or ns == 0 else f"{int(ns)}", va="center", ha="right", fontsize=6.2,
                 color=INK2)
    for c, g in ts.groupby("cluster"):
        if len(g) > 1:
            axT.plot([X["cond"] - 0.55] * 2, [g.y.min() - 0.3, g.y.max() + 0.3], color=INK2, lw=0.9)
    met = pd.to_numeric(ts.n_metabolites, errors="coerce").fillna(0)
    net = pd.to_numeric(ts.n_in_network, errors="coerce").fillna(0)
    axM.set_xlim(0, max(1700, met.max() * 1.35))
    for r, m, nn in zip(ts.itertuples(), met, net):
        axM.barh(r.y, m, height=0.62, color="#c6cad0", lw=0)
        axM.barh(r.y, nn, height=0.62, color="#5a5c61", lw=0)
        axM.text(m + 20, r.y, f"{int(m)} / {int(nn)}", va="center", fontsize=5.6, color=INK2)
    axM.text(0, -0.9, "Metabolites\nmeasured / in network", fontsize=7.2, weight="bold", color=INK, va="bottom")
    axM.tick_params(axis="x", labelsize=6, colors=INK2, pad=1)
    lim = 90
    axC.set_xlim(-lim, lim); axC.axvline(0, color=INK2, lw=0.6)
    for r in ts.itertuples():
        for v, sgn, col in ((r.n_dec, -1, DEC), (r.n_inc, 1, INC)):
            w = min(v, lim - 14)
            axC.barh(r.y, sgn * w, height=0.62, color=col, lw=0)
            axC.text(sgn * (w + 3.5), r.y, str(v), va="center", ha="right" if sgn < 0 else "left", fontsize=5.8,
                     color=INK2)
    axC.text(0, -0.9, "Significant metabolites in network\n← decreased   increased →", fontsize=7.2, weight="bold",
             color=INK, va="bottom", ha="center")
    axC.set_xticks([-60, -30, 0, 30, 60]); axC.set_xticklabels(["60", "30", "0", "30", "60"])
    axC.tick_params(axis="x", labelsize=6, colors=INK2, pad=1)
    FLG = [("primary", "Gene\nlost"), ("viable_down", "Viable\n↓src"), ("viable_up", "Viable\n↑src")]
    axF.set_xlim(-0.6, len(FLG) - 0.4); axF.set_xticks([])
    for i, (kf, lab) in enumerate(FLG):
        axF.text(i, -0.9, lab, ha="center", va="bottom", fontsize=6.6, weight="bold", color=INK)
        for r in ts.itertuples():
            on = bool(getattr(r, kf))
            axF.plot(i, r.y, marker="o", ms=4.0, mfc="#126b63" if on else "white", mec="#126b63" if on else "#9aa0a6",
                     mew=0.8)
    if ranks is not None:
        rank_strip(axs[4], ts, ranks)
    fig.text(0.012, 1 - 0.3 / FH, f"Every testable comparison: {len(ts)} in {ts.cluster.nunique()} clusters",
             fontsize=10.5, weight="bold", color=INK, va="top")
    hand = [Rectangle((0, 0), 1, 1, color=PERT[k]) for k in PERT if (ts.perturbation == k).any()]
    labs = [k for k in PERT if (ts.perturbation == k).any()]
    hand += [Rectangle((0, 0), 1, 1, color="#c6cad0"), Rectangle((0, 0), 1, 1, color="#5a5c61"),
             Rectangle((0, 0), 1, 1, color=DEC), Rectangle((0, 0), 1, 1, color=INC)]
    labs += ["measured", "in the network", "decreased (p < 0.05)", "increased (p < 0.05)"]
    fig.legend(hand, labs, loc="lower left", bbox_to_anchor=(0.012, 0.4 / FH), ncol=9, frameon=False, fontsize=6.6,
               handlelength=1.0, columnspacing=1.3)
    fig.text(0.012, 0.08 / FH, "Viable: ≥ 3 sources and ≥ 20 metabolites in that orientation (↓src: decreased "
             "metabolites are the sources; ↑src: increased). Brackets join comparisons in one cluster. n: samples in "
             "both groups as registered. Workbench and MetaboLights significance: Welch t-test, raw p < 0.05." +
             ("\nCausal gene percentile: rank among 2,191 network genes (0 = top; Reafect: among the enzymes it "
              "scored); a ranking with no sources is at the mid-rank, 50; × Reafect cannot score the gene."
              if ranks is not None else ""),
             fontsize=6.0, color=INK2, va="bottom")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"test_set_map.{ext}", dpi=220)
    plt.close(fig)


RANK_COLS = [("Walk, sources = increased | M0", "Walk\n↑src"), ("Current flow, sources = increased | M3", "Flow\n↑src M3"),
             ("neighbour", "One-\nhop"), ("Reafect", "Rea-\nfect")]


def rank_strip(ax, ts, ranks):
    """Causal-gene percentile per method, one cell per comparison; darker = ranked nearer the top."""
    cmap = plt.get_cmap("Blues_r")
    ax.set_xlim(-0.55, len(RANK_COLS) - 0.45); ax.set_xticks([])
    for i, (k, lab) in enumerate(RANK_COLS):
        ax.text(i, -0.9, lab, ha="center", va="bottom", fontsize=6.4, weight="bold", color=INK)
        for r in ts.itertuples():
            v = ranks.at[r.label, k] if r.label in ranks.index else float("nan")
            if pd.isna(v) or (k == "Reafect" and v >= 100):
                ax.text(i, r.y, "×", ha="center", va="center", fontsize=6, color="#9aa0a6")
                continue
            x = min(1.0, max(0.0, np.log10(max(v, 0.5) / 0.5) / np.log10(100)))
            ax.add_patch(Rectangle((i - 0.42, r.y - 0.4), 0.84, 0.8, color=cmap(0.08 + 0.85 * x), lw=0))
            ax.text(i, r.y, f"{v:.0f}" if v >= 1 else "<1", ha="center", va="center", fontsize=5.2,
                    color="white" if v < 5 else INK)
    ax.text((len(RANK_COLS) - 1) / 2, -3.2, "Causal gene percentile", ha="center", va="bottom", fontsize=7.2,
            weight="bold", color=INK)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--down", required=True, help="null calibration run, sources = decreased (walk)")
    p.add_argument("--up", required=True, help="null calibration run, sources = increased (walk)")
    p.add_argument("--out", default="results/data_figures")
    p.add_argument("--ranks", help="unit_percentiles.tsv from tools/benchmark_matrix.py: adds a results strip "
                                   "to test_set_map")
    args = p.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    style()
    t = unit_table(Path(args.down), Path(args.up))
    t.drop(columns=["c"]).to_csv(out / "units.tsv", sep="\t", index=False)
    data_map(t, out)
    overview(t, out)
    unit_map(t, out, pd.read_csv(args.ranks, sep="	", index_col=0) if args.ranks else None)
    ts = t[t.testable]
    info = {"comparisons": len(t), "testable": int(t.testable.sum()), "testable_clusters": int(ts.cluster.nunique()),
            "primary": int((ts.primary).sum()), "primary_clusters": int(ts[ts.primary].cluster.nunique()),
            "fate": t.fate.value_counts().to_dict()}
    (out / "summary.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
