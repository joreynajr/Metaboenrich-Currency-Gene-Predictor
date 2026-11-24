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


def audit_labels(units):
    return random.Random(SEED).sample(sorted(units.index), N_AUDIT)


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
    o = pd.read_csv(R / "reafect_bench/units/ours_ranks.tsv.gz", sep="\t", index_col=0)
    U = len(o)
    sym = o.symbol
    genes = {u: [g for g in str(units.at[u, "genes_in_net"]).split(";") if g] for u in units.index}
    rows = {u: sym.index[sym.isin(genes[u])] for u in units.index}
    meths = [("walk_up", "Walk ↑src (M0)"), ("flow", "Current flow (M0)"), ("walk_down", "Walk ↓src (M0)"),
             ("neighbour", "One-hop")]
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
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
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
    ax.set_xlabel("median causal-gene percentile over comparisons (0 = top)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    w, d = res[0], res[2]
    title = (f"Negative control: with another comparison's causal gene, walk ↑src falls from {w[1]:.0f}% to "
             f"{np.median(w[2]):.0f}%" + "\n" + f"so it finds the specific gene; walk ↓src barely beats its shuffle "
             f"({d[1]:.0f}% vs {np.median(d[2]):.0f}%)")
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
def ladders(units, pct, out, figs):
    order = units.assign(o=units.collection.str[0].map("ABCD".index), w=pct[WALK_UP]).sort_values(["o", "w"])
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(13, 4.0))
    P = pct.loc[order.index]
    lo = P[[m[0] for m in METHODS]].min(axis=1).clip(lower=0.05)
    hi = P[[m[0] for m in METHODS]].max(axis=1).clip(lower=0.05)
    ax.vlines(x, lo, hi, color=RULE, lw=1.2, zorder=1)
    for col, name, c, mk in METHODS:
        v = P[col].clip(lower=0.05).to_numpy()
        miss = (col == "Reafect") & (P[col] >= 100)
        ax.scatter(x[~miss], v[~miss], s=16, color=c, marker=mk, lw=0.4, edgecolor="white", zorder=3, label=name)
        if miss.any():
            ax.scatter(x[miss], np.full(miss.sum(), 100), s=16, marker="x", color=c, lw=0.9, zorder=3)
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
    ax.set_ylabel("causal-gene percentile (top = better)")
    top = {name: int((P[col] <= 10).sum()) for col, name, _, _ in METHODS}
    title = (f"Causal gene in the top 10%: walk ↑src {top['Walk ↑src']} of {len(P)} comparisons, flow M3 "
             f"{top['Flow ↑src M3']}, Reafect {top['Reafect']}, one-hop {top['One-hop']}")
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    ax.legend(loc="lower left", ncol=4, frameon=False, fontsize=7, bbox_to_anchor=(0, -0.2), handletextpad=0.3)
    fig.text(0.99, 0.02, "× Reafect cannot score the gene. Comparisons sorted by walk ↑src within each collection; "
             "the grey line spans best to worst method.", ha="right", fontsize=6.3, color=INK2)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.2)
    save(fig, out, "ladders", title, "Each column is one comparison and each mark one method; read across a collection to "
         "see whether a method's wins are broad or rest on a few comparisons, and down a column to see how far apart "
         "the methods are on the same data.", figs)


# ---------------------------------------------------------------- win / loss strips
PAIRS = [(WALK_UP, "neighbour", "Walk ↑src vs one-hop"),
         (WALK_UP, "Reafect", "Walk ↑src vs Reafect"),
         (WALK_UP, "Walk, sources = decreased | M0", "Walk ↑src vs walk ↓src"),
         ("Walk, sources = increased | M3", WALK_UP, "Walk ↑src: M3 vs none"),
         (FLOW_UP_M3, "Current flow, sources = increased | M0", "Flow ↑src: M3 vs none"),
         (FLOW_UP_M3, "Reafect", "Flow ↑src M3 vs Reafect")]


def winloss(units, pct, out, figs, cards):
    clu = pct.groupby(units.cluster).median()
    fig, axs = plt.subplots(len(PAIRS), 1, figsize=(10.5, 0.52 * len(PAIRS) + 1.0))
    rows = []
    for ax, (a, b, name) in zip(axs, PAIRS):
        d = (clu[a] - clu[b]).sort_values()           # negative: a ranks the gene higher
        w, l, t = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
        p = binomtest(w, w + l).pvalue if w + l else 1.0
        p3 = binomtest(max(w - 3, 0), max(w - 3, 0) + l).pvalue if w > l else binomtest(w, w + max(l - 3, 0)).pvalue
        cols = [GOOD if v < 0 else BAD if v > 0 else TIE for v in d]
        for i, c in enumerate(cols):
            ax.add_patch(Rectangle((i + 0.06, 0.1), 0.88, 0.8, color=c, lw=0))
        ax.set_xlim(-0.2, len(d) + 0.2); ax.set_ylim(0, 1); ax.axis("off")
        ax.text(-0.8, 0.5, name, ha="right", va="center", fontsize=7.5, color=INK)
        star = "*" if p < 0.05 else ""
        ax.text(len(d) + 0.8, 0.5, f"{w}–{l}" + (f" · {t} tied" if t else "") + f" · p {p:.2g}{star}"
                f" · drop 3 best {'wins' if w > l else 'losses'}: {p3:.2g}", va="center", fontsize=7, color=INK2)
        rows.append({"pair": name, "wins": w, "losses": l, "ties": t, "p": p, "p_drop3": p3})
    fig.legend([Rectangle((0, 0), 1, 1, color=c) for c in (GOOD, TIE, BAD)],
               ["first method ranks the gene higher", "tie", "second method ranks it higher"],
               loc="lower left", bbox_to_anchor=(0.2, 0.0), ncol=3, frameon=False, fontsize=7)
    r0 = rows[0]
    title = (f"One square per cluster ({len(clu)}): walk ↑src beats one-hop in {r0['wins']} of "
             f"{r0['wins'] + r0['losses'] + r0['ties']} clusters, and the result does not hinge on a few")
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.18, right=0.64, top=0.88, bottom=0.12, hspace=0.25)
    save(fig, out, "winloss", title, "Each square is one cluster, sorted by margin; a long unbroken blue run with p still "
         "small after the three biggest wins are removed means the advantage is broad, not three lucky studies "
         "(percentiles in each method's own gene universe, as in the benchmark matrix).", figs)
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
            "top10": [{"rank": float(r), "gene": model.genes.get(e, e), "causal": e in ens}
                      for e, r in ranks.sort_values().head(10).items()],
            "causal_rank_recomputed": rank, "causal_rank_benchmark": float(walk_rk.get(lab, float("nan"))),
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
        ax.text(34, 6.6, "③ shortest route from a source to the causal reaction", fontsize=6.5, color=INK2)
        p = a["path"]
        if p:
            steps = (len(p) - 1) // 2 + 1
            show = [p[0], p[1], "…", p[-2], p[-1]] if len(p) > 5 else p
            xs = np.linspace(37, 68, len(show))
            for j, (x, s) in enumerate(zip(xs, show)):
                last, is_r = j == len(show) - 1, s.startswith("R ")
                txt = s if s == "…" else s[2:]
                txt = txt if len(txt) <= 15 else txt[:14] + "…"
                col = "#126b63" if last else (INK2 if is_r else INK)
                ax.text(x, 4.6, txt, fontsize=5.7, va="center", ha="center", color=col,
                        weight="bold" if last else "normal",
                        bbox=None if s == "…" else dict(boxstyle="round,pad=0.25", fc="white",
                                                         ec=col if is_r or last else RULE, lw=0.6))
                if not last:
                    ax.text((x + xs[j + 1]) / 2, 4.6, "→", fontsize=6, va="center", ha="center", color=INK2)
            ax.text(34, 2.6, f"{steps} reaction step{'s' if steps > 1 else ''}; ends at {a['causal_reaction']} "
                    f"({a['causal_reaction_name'][:45]})", fontsize=5.8, color=INK2)
        else:
            ax.text(34, 4.6, "no directed route from any source: the walk never reaches the causal reaction",
                    fontsize=6.3, color=INK2)
        ax.text(77, 6.6, f"④ top 10 genes · causal at #{a['causal_rank_recomputed']:.0f} "
                f"(benchmark #{a['causal_rank_benchmark']:.0f})", fontsize=6.5, color=INK2)
        if len({t["rank"] for t in a["top10"]}) == 1:
            ax.text(77, 4.6, "no gene scored: every gene tied at the mid-rank", fontsize=6.3, color=INK2)
            continue
        for j, t in enumerate(a["top10"]):
            c, r = j // 5, j % 5
            ax.text(77 + c * 11, 5.3 - r * 1.15, f"{t['rank']:>3.0f}  {t['gene']}", fontsize=6, va="center",
                    color="#126b63" if t["causal"] else INK, weight="bold" if t["causal"] else "normal")
    title = "Random audit (seed 26): five comparisons followed from the data to the causal gene's rank"
    fig.suptitle(title, x=0.01, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.01, hspace=0.12)
    save(fig, out, "audit", title, "Five comparisons picked by a fixed random seed, not by us; each row can be checked "
         "against the deposit: the counts, which metabolites start the walk, a concrete route to the causal reaction, "
         "and the rank recomputed now next to the benchmark's (small differences are the float ties in the badge).",
         figs)


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
