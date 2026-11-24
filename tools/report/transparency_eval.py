"""Transparency figures for the causal-rank evaluation: one finding per picture.

  badge       M0 reproduces the benchmark's causal rank in every unit; random pAUC is 50/(U-1)
  shuffle     causal genes shuffled across units: every method should fall towards 50%
  ladders     where each method puts the causal gene, every comparison side by side
  winloss     cluster win/loss strips for key method pairs, with the sign-test p and its
              value after dropping the three biggest wins
  nulls       for the five audit units, a fresh null for the causal gene's reaction
              (sources = increased walk), the observed score marked, and the stored
              exceedance count beside the fresh one
  audit       the same five units followed from significant metabolites to sources and
              targets, a path from a source to the causal reaction, and the top 10 genes

The five audit units are random.Random(26).sample(sorted(testable labels), 5), the same
pick as the data-side sampler. Per-unit details go to cards.json; figures.json lists each
figure with a one-sentence caption.

    python tools/report/transparency_eval.py [--skip-model]

--skip-model leaves out nulls and audit (they load Human-GEM and rerun the walk, ~2 min);
--replot redraws them from the audit data already in cards.json.
Reads results/correction_eval_d26_*, results/null_calibration/*_d26, results/benchmark_matrix,
results/reafect_bench/units/ours_ranks.tsv.gz and results/data_figures/units.tsv.
"""
import argparse
import json
import random
import sys
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools" / "report"))
from build_data_figures import DEC, INC, INK, INK2, RULE, style   # noqa: E402

R = ROOT / "results"
EVALS = {"walk_up": "correction_eval_d26_walk_increased", "walk_down": "correction_eval_d26_walk_decreased",
         "flow_up": "correction_eval_d26_flow_increased", "flow_down": "correction_eval_d26_flow_decreased"}
SHORT = {"walk_up": "Walk ↑src", "walk_down": "Walk ↓src", "flow_up": "Flow ↑src", "flow_down": "Flow ↓src"}
RUN_LABEL = {"walk_up": "Walk, sources = increased", "walk_down": "Walk, sources = decreased",
             "flow_up": "Current flow, sources = increased", "flow_down": "Current flow, sources = decreased"}
NULL_RUN = R / "null_calibration/three_rule_increased_measurable_kNone_n1000_exceed_d26"
WALK_UP, FLOW_UP_M3 = "Walk, sources = increased | M0", "Current flow, sources = increased | M3"
# the four methods drawn side by side: (column in unit_percentiles, short name, colour, marker)
METHODS = [(WALK_UP, "Walk ↑src", DEC, "o"), (FLOW_UP_M3, "Flow ↑src M3", INC, "s"),
           ("Reafect", "Reafect", "#126b63", "D"), ("neighbour", "One-hop", "#8a8f96", "^")]
GOOD, BAD, TIE = DEC, INC, "#c6cad0"
SEED, N_AUDIT = 26, 5


def save(fig, out, name, title, caption, figs):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    figs.append({"file": f"{name}.png", "title": title, "caption": caption})


def load():
    units = pd.read_csv(R / "data_figures/units.tsv", sep="\t")
    units = units[units.testable].set_index("label")
    pct = pd.read_csv(R / "benchmark_matrix/unit_percentiles.tsv", sep="\t", index_col=0).loc[units.index]
    rk = {k: pd.read_csv(R / d / "rankings.tsv", sep="\t") for k, d in EVALS.items()}
    return units, pct, rk


AUDIT_STRATA = (("inborn errors (A)", 2), ("other gene loss", 2), ("inhibitor or gain", 1))


def audit_stratum(r):
    if str(r.collection).startswith("A"):
        return "inborn errors (A)"
    return "other gene loss" if bool(r.primary) else "inhibitor or gain"


def audit_labels(units):
    """One comparison per cluster, clusters drawn by a fixed seed within each stratum (2 + 2 + 1)."""
    rng = random.Random(SEED)
    st = units.apply(audit_stratum, axis=1)
    out = []
    for name, k in AUDIT_STRATA:
        cl = sorted(units[st == name].cluster.unique())
        for c in rng.sample(cl, min(k, len(cl))):
            out.append(rng.choice(sorted(units.index[units.cluster == c])))
    return out


# ---------------------------------------------------------------- reproduction badge
def badge(out, figs, cards):
    """Scores rebuilt from scratch must equal the stored ones to float precision; causal ranks may move
    only where reactions tie exactly in theory and 1e-16 noise breaks the tie (accepted 2026-11-17)."""
    tiles, tol = [], 1e-12
    for k, d in EVALS.items():
        m = pd.read_csv(R / d / "m0_check.tsv", sep="\t")
        same = int((m.causal_rank_nodes_tsv == m.causal_rank_reference).sum())
        moved = m[m.causal_rank_nodes_tsv != m.causal_rank_reference]
        gap = (moved.causal_rank_nodes_tsv - moved.causal_rank_reference).abs()
        dmax = m.max_abs_score_diff.max()
        note = (f"scores agree to {dmax:.0e}" +
                (f"; {len(moved)} move {span(gap)} places on ties" if len(moved) else ""))
        tiles.append((SHORT[k], same, len(m), note, dmax < tol and int(m.scored_differs.sum()) == 0,
                      int(gap.max()) if len(moved) else 0))
    s = pd.read_csv(R / EVALS["walk_up"] / "summary.tsv", sep="\t")
    U = json.loads((R / EVALS["walk_up"] / "run_info.json").read_text())["universe"]
    rnd = s[s.ranking == "random"].pAUC_1_100.unique()
    want = 50 / (U - 1)
    fig, axs = plt.subplots(1, 5, figsize=(11.5, 1.9))
    for ax, (name, a, b, note, ok, _) in zip(axs, tiles):
        tile(ax, ok, f"{a} / {b}", "units with the identical\ncausal rank", name, note)
    ok = len(rnd) and np.allclose(rnd, want)
    tile(axs[-1], ok, f"{rnd[0]:.4f}", f"random pAUC\n(expected 50/(U−1) = {want:.4f})", "Random baseline",
         f"U = {U:,} genes")
    a = sum(t[1] for t in tiles); b = sum(t[2] for t in tiles); mv = max(t[5] for t in tiles)
    title = (f"Rebuilt from scratch, every score matches the stored run to 1e-16" + "\n" +
             f"{a} of {b} causal ranks are identical, the rest move at most {mv} places on exact ties; random is where "
             "theory puts it")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, weight="bold", color=INK)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.72, bottom=0.04, wspace=0.08)
    save(fig, out, "badge", title, "Each tile reruns the uncorrected ranking from the raw inputs and compares it with the "
         "stored benchmark; blue tiles mean the scores behind every other figure reproduce, and the few rank moves are "
         "reactions tied exactly in theory whose order float noise decides.", figs)
    cards["_reproduction"] = [{"run": t[0], "identical": t[1], "units": t[2], "note": t[3]} for t in tiles]


def span(gap):
    return f"{gap.min():.0f}" if gap.min() == gap.max() else f"{gap.min():.0f}–{gap.max():.0f}"


def tile(ax, ok, big, small, head, note):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    col = GOOD if ok else BAD
    ax.add_patch(Rectangle((0.02, 0.02), 0.96, 0.96, facecolor="white", edgecolor=col, lw=1.6))
    ax.add_patch(Rectangle((0.02, 0.80), 0.96, 0.18, facecolor=col, lw=0))
    ax.text(0.06, 0.89, ("✓ " if ok else "✗ ") + head, color="white", fontsize=7, weight="bold", va="center")
    ax.text(0.5, 0.56, big, ha="center", va="center", fontsize=15, weight="bold", color=INK)
    ax.text(0.5, 0.27, small, ha="center", va="center", fontsize=6.5, color=INK2)
    ax.text(0.5, 0.1, note, ha="center", va="center", fontsize=5.6, color=INK2, linespacing=1.3)


# ---------------------------------------------------------------- shuffle control
def shuffle(units, out, figs, cards, perms=1000):
    """Every method on the gene list both ours and Reafect can score (reafect_bench common_ranks.tsv.gz)."""
    o = pd.read_csv(R / "reafect_bench/units/common_ranks.tsv.gz", sep="\t", index_col=0)
    U = len(o)
    sym = pd.read_csv(R / "reafect_bench/units/ours_ranks.tsv.gz", sep="\t", index_col=0,
                      usecols=["ensembl_id", "symbol"]).symbol.reindex(o.index)
    genes = {u: [g for g in str(units.at[u, "genes_in_net"]).split(";") if g] for u in units.index}
    rows = {u: sym.index[sym.isin(genes[u])] for u in units.index}
    meths = [("walk_up", "Walk ↑src (M0)"), ("reafect", "Reafect"), ("flow", "Current flow (M0)"),
             ("walk_down", "Walk ↓src (M0)"), ("neighbour", "One-hop")]
    labels = [u for u in units.index if len(rows[u])]
    rng = np.random.default_rng(SEED)
    res = []
    for key, name in meths:
        M = np.stack([o[f"{u}|{key}"].to_numpy() for u in labels], axis=1)      # genes x units
        pos = {g: i for i, g in enumerate(o.index)}
        idx = [[pos[g] for g in rows[u]] for u in labels]

        def med(assign):
            return np.median([M[idx[a], j].min() / U * 100 for j, a in enumerate(assign)])
        obs = med(range(len(labels)))
        null = []
        for _ in range(perms):
            while True:
                p = rng.permutation(len(labels))
                if not (p == np.arange(len(labels))).any():
                    break
            null.append(med(p))
        null = np.array(null)
        res.append((name, obs, null, (null <= obs).mean()))
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    for i, (name, obs, null, p) in enumerate(res):
        lo, hi = np.percentile(null, [2.5, 97.5])
        ax.plot([lo, hi], [i, i], color=TIE, lw=7, solid_capstyle="round", zorder=1)
        ax.plot(np.median(null), i, "|", color=INK2, ms=12, mew=1.5, zorder=2)
        ax.plot(obs, i, "o", color=DEC, ms=8, mec="white", mew=1.5, zorder=3)
        ax.text(hi + 1.2, i, f"shuffled {np.median(null):.0f}%", va="center", fontsize=7, color=INK2)
        ax.text(obs - 1.2, i, f"real {obs:.1f}%", va="center", ha="right", fontsize=7, color=INK, weight="bold")
    ax.axvline(50, color=RULE, lw=1, zorder=0)
    ax.text(50, len(res) - 0.4, "random = 50%", ha="center", fontsize=6.5, color=INK2)
    ax.set_yticks(range(len(res)), [r[0] for r in res])
    ax.set_ylim(len(res) - 0.3, -0.7)
    ax.set_xlim(-8, 75)
    ax.set_xlabel(f"median causal-gene percentile over comparisons, among the {U:,} shared genes (0 = top)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    w, r, d = res[0], res[1], res[3]
    title = (f"Negative control, shared gene list: walk ↑src {w[1]:.0f}% real vs {np.median(w[2]):.0f}% shuffled, "
             f"Reafect {r[1]:.0f}% vs {np.median(r[2]):.0f}%" + "\n" +
             ("Both find the specific gene" if max(w[3], r[3]) < 0.05 else "Not both beat their shuffle") +
             f"; walk ↓src {'does' if d[3] < 0.05 else 'does not'} ({d[1]:.0f}% vs {np.median(d[2]):.0f}%, "
             f"p = {d[3]:.2g})")
    ax.set_title(title, loc="left", fontsize=9, weight="bold", pad=10)
    fig.tight_layout()
    save(fig, out, "shuffle", title, "Grey bands are 95% of 1,000 shuffles of causal genes across comparisons, dots are "
         "the real assignment; a method that only found 'popular' genes would score as well on the shuffles, so the "
         "gap is the signal (shuffles stay below 50% where the same genes recur, e.g. FASN, or are well connected).",
         figs)
    cards["_shuffle"] = {name: {"real_median_pct": round(obs, 2), "shuffled_median_pct": round(float(np.median(null)), 2),
                                "shuffled_95": [round(float(x), 2) for x in np.percentile(null, [2.5, 97.5])],
                                "p_shuffle_le_real": float(p), "perms": perms, "units": len(labels)}
                         for name, obs, null, p in res}
    return res


# ---------------------------------------------------------------- rank ladders
# drawn on the gene list both our methods and Reafect can score (reafect_bench units-compare, about 1,060 genes)
SHARED = [("walk_up_common_pct", "Walk ↑src", DEC, "o"), ("reafect_common_pct", "Reafect", "#126b63", "D"),
          ("neighbour_common_pct", "One-hop", "#8a8f96", "^")]


def ladders(units, pct, out, figs):
    pu = pd.read_csv(R / "reafect_bench/units/per_unit.tsv", sep="\t").set_index("label").reindex(units.index)
    order = units.assign(o=units.collection.str[0].map("ABCD".index), w=pu.walk_up_common_pct).sort_values(["o", "w"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13, 4.0))
    P = pu.loc[order.index]
    cols = [m[0] for m in SHARED]
    lo = P[cols].min(axis=1).clip(lower=0.05)
    hi = P[cols].max(axis=1).clip(lower=0.05)
    ax.vlines(x, lo, hi, color=RULE, lw=1.2, zorder=1)
    for col, name, c, mk in SHARED:
        v = P[col].clip(lower=0.05).to_numpy()
        miss = (col == "reafect_common_pct") & ~P.causal_scored_by_reafect.astype(bool)
        ax.scatter(x[~miss], v[~miss], s=16, color=c, marker=mk, lw=0.4, edgecolor="white", zorder=3, label=name)
        if miss.any():
            ax.scatter(x[miss], v[miss], s=16, marker="x", color=c, lw=0.9, zorder=3)
    ax.set_yscale("log"); ax.set_ylim(130, 0.04)
    ax.set_yticks([0.1, 1, 10, 50, 100], ["0.1%", "1%", "10%", "50%", "100%"])
    ax.axhline(10, color=INK2, lw=0.6, ls=(0, (3, 3)), zorder=0)
    ax.text(len(x) - 0.5, 9, "top 10%", ha="right", va="bottom", fontsize=6.5, color=INK2)
    ax.set_xlim(-1, len(x))
    ax.set_xticks([])
    for c in "ABCD":
        s = np.flatnonzero(order.collection.str[0].to_numpy() == c)
        if len(s):
            ax.text(s.mean(), 160, {"A": "A · inborn errors", "B": "B", "C": "C · Workbench", "D": "D · MetaboLights"}[c],
                    ha="center", va="top", fontsize=7, color=INK2)
            if s[0] > 0:
                ax.axvline(s[0] - 0.5, color=INK2, lw=0.5)
    ax.set_ylabel("causal-gene percentile among\nshared genes (top = better)")
    top = {name: int((P[col] <= 10).sum()) for col, name, _, _ in SHARED}
    title = (f"Causal gene in the top 10% of the shared gene list: walk ↑src {top['Walk ↑src']} of {len(P)} "
             f"comparisons, Reafect {top['Reafect']}, one-hop {top['One-hop']}")
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    ax.legend(loc="lower left", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0, -0.2), handletextpad=0.3)
    fig.text(0.99, 0.02, "All three on the ~1,060 genes both methods can score (flow ↑src M3 has no shared-list ranks "
             "yet). × Reafect cannot score the gene (placed at the bottom of its list).\nComparisons sorted by walk "
             "↑src within each collection; the grey line spans best to worst method.", ha="right", fontsize=6.3,
             color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.2)
    save(fig, out, "ladders", title, "Each column is one comparison and each mark one method; read across a collection to "
         "see whether a method's wins are broad or rest on a few comparisons, and down a column to see how far apart "
         "the methods are on the same data.", figs)


# ---------------------------------------------------------------- win / loss strips
# (first, second, label, list): "own" = our 2,191-gene list (fair between our methods); "shared" = the ~1,060 genes
# both ours and Reafect can score (reafect_bench per_unit.tsv), required whenever Reafect is one of the pair
PAIRS = [(WALK_UP, "neighbour", "Walk ↑src vs one-hop", "own"),
         ("walk_up_common_pct", "reafect_common_pct", "Walk ↑src vs Reafect", "shared"),
         (WALK_UP, "Walk, sources = decreased | M0", "Walk ↑src vs walk ↓src", "own"),
         ("Walk, sources = increased | M3", WALK_UP, "Walk ↑src: M3 vs none", "own"),
         (FLOW_UP_M3, "Current flow, sources = increased | M0", "Flow ↑src: M3 vs none", "own"),
         ("flow_common_pct", "reafect_common_pct", "Current flow vs Reafect", "shared")]


def winloss(units, pct, out, figs, cards):
    pu = pd.read_csv(R / "reafect_bench/units/per_unit.tsv", sep="\t").set_index("label").reindex(units.index)
    clus = {"own": pct.groupby(units.cluster).median(),
            "shared": pu.select_dtypes("number").groupby(units.cluster).median()}
    clu = clus["own"]
    fig, axs = plt.subplots(len(PAIRS), 1, figsize=(10.5, 0.52 * len(PAIRS) + 1.0))
    rows = []
    for ax, (a, b, name, lst) in zip(axs, PAIRS):
        c_ = clus[lst]
        d = (c_[a] - c_[b]).sort_values()             # negative: a ranks the gene higher
        w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        p3 = binomtest(max(w - 3, 0), max(w - 3, 0) + l).pvalue if w > l else binomtest(w, w + max(l - 3, 0)).pvalue
        cols = [GOOD if v < 0 else BAD if v > 0 else TIE for v in d]
        for i, c in enumerate(cols):
            ax.add_patch(Rectangle((i + 0.06, 0.1), 0.88, 0.8, color=c, lw=0))
        ax.set_xlim(-0.2, len(d) + 0.2); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(-0.8, 0.5, name + (" (shared list)" if lst == "shared" else ""), ha="right", va="center",
                fontsize=7.5, color=INK)
        star = "*" if p < 0.05 else ""
        ax.text(len(d) + 0.8, 0.5, f"{w}–{l}" + (f" · {t} tied" if t else "") + f" · p {p:.2g}{star}"
                f" · drop 3 best {'wins' if w > l else 'losses'}: {p3:.2g}", va="center", fontsize=7, color=INK2)
        rows.append({"pair": name, "wins": w, "losses": l, "ties": t, "p": p, "p_drop3": p3})
    fig.legend([Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, TIE, BAD)],
               ["first method ranks the gene higher", "tie", "second method ranks it higher"],
               loc="lower left", bbox_to_anchor=(0.22, 0.035), ncol=3, frameon=False, fontsize=7)
    fig.text(0.225, 0.005, "Our methods against each other: our 2,191-gene list. Pairs with Reafect: the ~1,060 genes "
             "both can score.", ha="left", fontsize=6.3, color=INK2)
    r0 = rows[0]
    title = (f"One square per cluster ({len(clu)}): walk ↑src beats one-hop in {r0['wins']} of "
             f"{r0['wins'] + r0['losses'] + r0['ties']} clusters, " +
             ("and the result does not hinge on a few" if r0["p_drop3"] < 0.05 else "but it rests on a few"))
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.22, right=0.66, top=0.88, bottom=0.16, hspace=0.25)
    save(fig, out, "winloss", title, "Each square is one cluster, sorted by margin; a long unbroken blue run with p still "
         "small after the three biggest wins are removed means the advantage is broad, not three lucky studies "
         "(our methods against each other on our 2,191-gene list; any pair with Reafect on the ~1,060 shared genes).",
         figs)
    cards["_winloss"] = rows


# ---------------------------------------------------------------- model-based: nulls and audit
def build_model():
    from metaboenrich import structure
    from metaboenrich.daa import load_name_bridge
    from metaboenrich.gem import load_sbml
    from metaboenrich.network import build_graph
    from null_calibration import Calibrator, registry_units
    from correction_eval import Evaluator
    info = json.loads((NULL_RUN / "run_info.json").read_text())
    model = load_sbml(str(ROOT / "data/Human-GEM.xml"))
    sset, key = info["stage"].split(":", 1)
    st = next(s for s in structure.stages(model, sset, atom_dir=str(ROOT / "data")) if s[0] == key)
    graph = build_graph(model, st[3], st[4], st[5], **st[6])
    bridge = load_name_bridge(str(ROOT / "data/id_translation.csv"))
    cal = Calibrator(graph, None, "walk")
    ev = Evaluator(model, graph, bridge, cal, "walk", "increased")
    units, _ = registry_units(model, graph, bridge, "increased")
    return model, graph, cal, ev, {u["label"]: u for u in units}, info


def causal_reactions(graph, model, genes):
    sym2ens = {}
    for e, s in model.genes.items():
        sym2ens.setdefault(s, set()).add(e)
    ens = set().union(*(sym2ens.get(g, set()) for g in genes)) if genes else set()
    return [r for r, gs in graph.reaction_genes.items() if ens & set(gs)], ens


def path_to(graph, sources, target):
    """Shortest directed path (metabolite -> reaction -> metabolite ...) from any source to target."""
    succ = [[] for _ in range(graph.n)]
    for u, v in graph.arcs:
        succ[u].append(v)
    prev = {s: None for s in sources}
    q = deque(sources)
    while q:
        x = q.popleft()
        if x == target:
            break
        for y in succ[x]:
            if y not in prev:
                prev[y] = x
                q.append(y)
    if target not in prev:
        return []
    p, x = [], target
    while x is not None:
        p.append(x); x = prev[x]
    return p[::-1]


def model_figs(units, rk, perms=1000):
    """Recompute the five audit units: walk ranks, a path to the causal reaction, and a fresh null."""
    from null_calibration import candidate_pools, draw
    model, graph, cal, ev, U, info = build_model()
    labels = audit_labels(units)
    measured = np.array(sorted(set().union(*(u["measured"] for u in U.values()))))
    stored = pd.read_csv(NULL_RUN / "nodes.tsv", sep="\t", usecols=["unit", "node_id", "n_null_ge_observed_conductivity"])
    stored = stored[stored.unit.isin(labels)]
    rng = np.random.default_rng(SEED)
    walk_rk = rk["walk_up"][rk["walk_up"].ranking == "M0"].set_index("unit")["rank"]
    audits = {}
    for lab in labels:
        u = U[lab]
        genes = [g for g in str(units.at[lab, "genes_in_net"]).split(";") if g]
        crx, ens = causal_reactions(graph, model, genes)
        src, tgt, phi, ground = u["sources"], u["targets"], u["phi"], u["ground"]
        nodes, ranks, _ = ev.reference(u)
        cond = nodes.conductivity.to_numpy()
        best = max(crx, key=lambda r: cond[r])
        rxn_idx = np.array(sorted(graph.reaction_genes))
        rc = cond[rxn_idx]
        order = rxn_idx[np.argsort(-rc, kind="stable")]
        gene_rx = {}
        for r_, gs in graph.reaction_genes.items():
            for g_ in gs:
                gene_rx.setdefault(g_, []).append(r_)
        rank = float(ranks[list(ens & set(ranks.index))].min())
        eff_t, prep = cal.prepare(src, tgt, phi, ground)
        obs = float(cal.score(src, tgt, phi, ground, prep).conductivity[best])
        cands = candidate_pools(src, cal.degree, measured, set(eff_t) | set(tgt) | set(ground or []))
        null = []
        for _ in range(perms):
            rs = draw(cands, rng)
            phi_k = {**{t: phi.get(t, 0.0) for t in tgt}, **{r: phi[s] for r, s in zip(rs, src)}}
            null.append(float(cal.score(rs, tgt, phi_k, ground, prep).conductivity[best]))
        st = stored[(stored.unit == lab) & (stored.node_id == graph.node_ids[best])]
        rid = graph.node_ids[best]
        audits[lab] = {
            "genes": genes, "n_decreased": int(units.at[lab, "n_dec"]), "n_increased": int(units.at[lab, "n_inc"]),
            "sources": [graph.node_names[s] for s in src], "targets": [graph.node_names[t] for t in tgt][:40],
            "n_sources": len(src), "n_targets": len(tgt), "n_absorbing": len(eff_t), "causal_reaction": rid,
            "causal_reaction_name": model.reactions[rid].name if rid in model.reactions else graph.node_names[best],
            "path": [("R " if graph.is_reaction[x] else "M ") + graph.node_names[x]
                     for x in path_to(graph, list(src), best)],
            "top10": [{"rank": float(r), "gene": model.genes.get(e, e), "causal": e in ens,
                       "best_reaction": graph.node_ids[max(gene_rx.get(e, [best]), key=lambda x: cond[x])]}
                      for e, r in ranks.sort_values().head(10).items()],
            "causal_rank_recomputed": rank, "causal_rank_benchmark": float(walk_rk.get(lab, float("nan"))),
            "stratum": audit_stratum(units.loc[lab]),
            "why": {"reaction_pct": float(100 * (rc < cond[best]).mean()), "reaction_rank": int((rc > cond[best]).sum() + 1),
                    "n_reactions": int(len(rc)), "causal_reaction_genes": len(graph.reaction_genes[best]),
                    "carries_flow": bool(cond[best] > 0),
                    "top_reactions": [{"name": graph.node_names[x], "genes": len(graph.reaction_genes[x])}
                                      for x in order if graph.reaction_genes[x]][:3]},
            "null": {"reaction": rid, "observed": obs, "values": null, "fresh_exceed": int(sum(v >= obs for v in null)),
                     "fresh_perms": perms, "stored_perms": 1000,
                     "stored_exceed": int(st.n_null_ge_observed_conductivity.iloc[0]) if len(st) else None}}
    return audits


def nulls_fig(audits, labels, out, figs):
    fig, axs = plt.subplots(1, len(labels), figsize=(12.5, 2.7))
    ex = []
    for ax, lab in zip(axs, labels):
        a, n = audits[lab], audits[lab]["null"]
        null, obs, k, k0 = np.array(n["values"]), n["observed"], n["fresh_exceed"], n["stored_exceed"]
        ex.append(k / n["fresh_perms"])
        nz = null[null > 0]
        ref = [x for x in (obs, *nz) if x > 0]
        lo, hi = (min(ref) / 2, max(ref) * 2) if ref else (1e-6, 1)
        ax.hist(nz, bins=np.logspace(np.log10(lo), np.log10(hi), 30), color=TIE, lw=0)
        ax.set_xscale("log")
        if obs > 0:
            ax.axvline(obs, color=DEC, lw=2)
        else:
            ax.text(0.03, 0.5, "real flow = 0:\nno walker reaches\nthis reaction", transform=ax.transAxes, color=DEC,
                    fontsize=6.5, weight="bold", va="center")
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_title(f"{lab} · {', '.join(a['genes'])}\n{k}/{n['fresh_perms']} nulls ≥ real · stored run {k0}", fontsize=7, loc="left", color=INK, linespacing=1.4)
        ax.set_xlabel(f"flow through {n['reaction']}", fontsize=6.5)
        ax.tick_params(axis="x", labelsize=6)
    title = (f"What M3 sees in five random comparisons: the causal reaction's real flow (blue) is ordinary, "
             f"{min(ex):.0%}–{max(ex):.0%} of random source sets match it\n"
             "so the exceedance test has little to work with, which is why M3 rarely lifts the walk; "
             "a fresh rerun reproduces the stored counts")
    fig.suptitle(title, x=0.01, ha="left", fontsize=8.8, weight="bold", linespacing=1.35)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.7, bottom=0.17, wspace=0.08)
    save(fig, out, "nulls", title, "Grey: flow through the causal gene's reaction for 1,000 degree-matched random source "
         "sets; blue: the real sources. M3 scores how rarely random sources do as well; a fresh rerun landing on the "
         "stored count shows the null is stable, not a lucky draw.", figs)


def audit_fig(audits, labels, out, figs):
    fig, axs = plt.subplots(len(labels), 1, figsize=(12.5, 1.5 * len(labels) + 0.6))
    for ax, lab in zip(axs, labels):
        a = audits[lab]
        ax.set_xlim(0, 100); ax.set_ylim(0, 10); ax.axis("off")
        ax.add_patch(Rectangle((0, 0.2), 100, 9.6, facecolor="#f6f7f9", lw=0))
        ax.text(0.8, 9.0, f"{lab}  ·  causal gene {', '.join(a['genes'])}", fontsize=8, weight="bold", va="top",
                color=INK)
        ax.text(99.2, 9.0, a.get("stratum", ""), fontsize=6.5, va="top", ha="right", color=INK2, style="italic")
        ax.text(0.8, 6.6, "① significant metabolites", fontsize=6.5, color=INK2)
        for i, (n, c, s) in enumerate(((a["n_decreased"], DEC, "decreased"), (a["n_increased"], INC, "increased"))):
            w = min(n, 60) / 60 * 10
            ax.add_patch(Rectangle((0.8, 4.9 - i * 1.6), max(w, 0.15), 1.1, color=c, lw=0))
            ax.text(1.3 + w, 5.45 - i * 1.6, f"{n} {s}", fontsize=6.3, va="center", color=INK2)
        ax.text(15, 6.6, "② walk: increased are the sources", fontsize=6.5, color=INK2)
        tg = f"{a['n_targets']} targets" if a["n_targets"] else f"{a['n_absorbing']} unchanged absorb"
        ax.text(15, 5.4, f"{a['n_sources']} source{'s' if a['n_sources'] != 1 else ''} → {tg}", fontsize=7, color=INK, va="center")
        exs = ", ".join(a["sources"][:3])
        ax.text(15, 3.9, (exs[:44] + "…") if len(exs) > 44 else exs, fontsize=5.8, color=INK2, va="center",
                style="italic")
        w = a.get("why")
        ax.text(34, 6.6, "③ why the causal gene ranks where it does", fontsize=6.5, color=INK2)
        if w:
            def genes(n):
                return f"{n} gene{'s' if n != 1 else ''}"
            ax.add_patch(Rectangle((34, 5.0), 40, 0.55, facecolor="#e6e8eb", lw=0))
            ax.add_patch(Rectangle((34 + 40 * w["reaction_pct"] / 100 - 0.25, 4.7), 0.5, 1.15, color="#126b63", lw=0))
            ax.text(34, 4.5, "lowest reaction score", fontsize=5, color=INK2, va="top")
            ax.text(74, 4.5, "highest", fontsize=5, color=INK2, va="top", ha="right")
            nm = a["causal_reaction_name"]
            nm = nm if len(nm) <= 60 else nm[:59] + "…"
            msg = (f"its best reaction is #{w['reaction_rank']:,} of {w['n_reactions']:,} by score "
                   f"(top {100 * w['reaction_rank'] / w['n_reactions']:.1f}%)" if w["carries_flow"]
                   else "the walk sends no flow through its reactions")
            ax.text(34, 3.2, msg, fontsize=5.9, color=INK, va="center")
            ax.text(34, 2.15, f"{nm} · {genes(w['causal_reaction_genes'])}"
                    + (" · generic protein reaction" if "protein" in a["causal_reaction_name"].lower() else ""),
                    fontsize=5.4, color=INK2, va="center")
            tops = "; ".join(f"{t['name'][:28]} ({genes(t['genes'])})" for t in w["top_reactions"][:2])
            ax.text(34, 1.1, "highest-scoring: " + tops, fontsize=5.4, color=INK2, va="center")
        fam = {}
        for t in a["top10"]:
            fam.setdefault(t.get("best_reaction"), []).append(t["gene"])
        tag = {r: chr(97 + i) for i, r in enumerate(r for r, g in fam.items() if len(g) > 1)}
        ax.text(77, 6.6, f"④ top 10 genes · causal at #{a['causal_rank_recomputed']:.0f} · "
                f"{len(fam)} distinct reaction{'s' if len(fam) != 1 else ''}", fontsize=6.5, color=INK2)
        if len({t["rank"] for t in a["top10"]}) == 1:
            ax.text(77, 4.6, "no gene scored: every gene tied at the mid-rank", fontsize=6.3, color=INK2)
            continue
        for j, t in enumerate(a["top10"]):
            c, r = j // 5, j % 5
            lt = tag.get(t.get("best_reaction"))
            ax.text(77 + c * 11, 5.3 - r * 1.15, f"{t['rank']:>3.0f}  {t['gene']}" + (f"  [{lt}]" if lt else ""),
                    fontsize=6, va="center", color="#126b63" if t["causal"] else INK,
                    weight="bold" if t["causal"] else "normal")
    title = ("Audit (seed 26, one comparison per cluster: 2 inborn errors, 2 other gene loss, 1 inhibitor): "
             "from the data to the causal gene's rank")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.035, hspace=0.12)
    fig.text(0.01, 0.003, "④ genes with the same letter share their best-scoring reaction, so they tie: one strong "
             "reaction can fill several places in the top 10.", fontsize=6, color=INK2)
    save(fig, out, "audit", title, "Five comparisons drawn by a fixed seed, one per cluster across three kinds of "
         "comparison; each row shows what the walk starts from, where the causal gene's best reaction falls among all "
         "reaction scores and why, and which reactions fill the top 10.", figs)


# ---------------------------------------------------------------- cards
def write_cards(units, pct, rk, audits, cards, out):
    ranks = {}
    for k, df in rk.items():
        for m, g in df.groupby("ranking"):
            name = m if m in ("neighbour", "random") else f"{RUN_LABEL[k]} | {m}"
            for u, r in g.set_index("unit")["rank"].items():
                ranks.setdefault(u, {})[name] = float(r)
    keep = ["collection", "study", "sub_unit", "cluster", "causal_genes", "genes_in_net", "perturbation",
            "sample_matrix", "sample_detail", "n_samples", "n_metabolites", "n_in_network", "n_dec", "n_inc",
            "n_sources", "n_sources_up", "viable_down", "viable_up", "primary"]
    for lab, r in units.iterrows():
        c = {k: (None if pd.isna(r[k]) else r[k].item() if hasattr(r[k], "item") else r[k]) for k in keep}
        c["rank"] = ranks.get(lab, {})
        c["percentile"] = {k: round(float(v), 3) for k, v in pct.loc[lab].items()}
        c["ladder"] = {name: round(float(pct.at[lab, col]), 3) for col, name, _, _ in METHODS}
        if lab in audits:
            c["audit"] = audits[lab]
        cards[lab] = c
    (out / "cards.json").write_text(json.dumps(cards, indent=1, default=str), encoding="utf-8")


def main(argv=None):
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--out", default=str(R / "transparency/eval"))
    a.add_argument("--skip-model", action="store_true")
    a.add_argument("--replot", action="store_true", help="redraw nulls and audit from the audit data in cards.json")
    args = a.parse_args(argv)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    style()
    units, pct, rk = load()
    figs, cards = [], {"_audit_units": audit_labels(units), "_seed": SEED}
    badge(out, figs, cards)
    shuffle(units, out, figs, cards)
    ladders(units, pct, out, figs)
    winloss(units, pct, out, figs, cards)
    audits = {}
    if args.replot:
        audits = {k: v["audit"] for k, v in json.loads((out / "cards.json").read_text(encoding="utf-8")).items()
                  if isinstance(v, dict) and "audit" in v}
    elif not args.skip_model:
        audits = model_figs(units, rk)
    if audits:
        nulls_fig(audits, audit_labels(units), out, figs)
        audit_fig(audits, audit_labels(units), out, figs)
    write_cards(units, pct, rk, audits, cards, out)
    (out / "figures.json").write_text(json.dumps(figs, indent=1), encoding="utf-8")
    for f in figs:
        print(f["file"], "|", f["title"].encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
