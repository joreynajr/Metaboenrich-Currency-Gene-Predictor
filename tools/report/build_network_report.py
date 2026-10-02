"""Build the Network Explorer: an interactive cytoscape.js view of the current
each method sends through the network for every group-level benchmark case
(one self-contained HTML page).

Everything is recomputed from the model and the benchmark data (data/ and
results/ are not committed):
    data/Human-GEM.xml, data/Human-GEM.yml         network and subsystems (pathways)
    data/benchmark/{thistlethwaite2020.csv, miller2015.csv, cohorts_coded.tsv}
    benchmarks/cases.tsv                            cases (status primary + secondary) and causal genes
    results/bench_rw/{flow,walk_full}/group_results.tsv
                                                    causal-gene ranks, checked against ours on every build
    tools/report/network_report_template.html       the page (HTML/CSS/JS) with two placeholders
    data/vendor/cytoscape-<version>.min.js          downloaded once from cdnjs, SHA-256 checked, inlined

Produce the benchmark results on branch directed-rw with
    python -m metaboenrich.benchmark --levels group --out results/bench_rw/flow
    python -m metaboenrich.benchmark --levels group --method walk --out results/bench_rw/walk_full

Then:
    python tools/report/build_network_report.py --out docs/network_report.html
    python tools/report/build_network_report.py --json-only          # data only, to results/network_report/data.json
    python tools/report/build_network_report.py --cases pku msud     # quick test on a few cases
    python tools/report/build_network_report.py --from-json results/network_report/data.json
                                                    # re-render the page from saved data (no recomputation)

Each case is one disorder at group level (mean z-score per metabolite, BH
q < 0.05 and |mean z| >= 1), run exactly as the benchmark runs it, with both
current flow and the absorbing walk (kappa = None). Views per case:
    net      the current-carrying subnetwork: every edge whose max current over
             pairs is >= 0.1 under either method, plus every source and target
             (threshold raised until the view has <= 500 nodes)
    pw-...   one view per Human-GEM subsystem picked by either method's
             over-representation test among its top 100 reactions (a subsystem
             with more than 150 reactions on the network shows only the reactions
             carrying current under either method)
    causal   the causal gene's reactions, their metabolites and the reactions
             next to those metabolites that carry current (max >= 0.1), so the
             causal gene can always be found in the graph
Both methods share one layout per view, so nodes stay put when switching. Small
views (<= 200 nodes) get a label-aware pass that pushes apart nodes whose labels
would overlap at 100% zoom.
"""
import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metaboenrich.benchmark import CASES, group_daa, load_miller, load_t2020   # noqa: E402
from metaboenrich.daa import load_daa                                            # noqa: E402
from metaboenrich.flow import current_flow                                       # noqa: E402
from metaboenrich.gem import load_sbml, load_subsystems                          # noqa: E402
from metaboenrich.network import (DEFAULT_REACTION_EDITS, build_graph,           # noqa: E402
                                  load_reaction_edits, resolve_currency)
from metaboenrich.scoring import gene_table, node_table, pathway_table, reaction_table  # noqa: E402
from metaboenrich.stage_diagrams import fruchterman_reingold                     # noqa: E402
from metaboenrich.walk import absorbing_walk                                     # noqa: E402

TEMPLATE = ROOT / "tools" / "report" / "network_report_template.html"
BENCH = ROOT / "results" / "bench_rw"
BENCH_DIRS = {"flow": "flow", "walk": "walk_full"}
CYTOSCAPE_VERSION = "3.30.2"
CYTOSCAPE_SHA256 = "83e8c54a6bec655bfd81df07df605649c268af69aeca67a5ea2da54ea42dac81"
CYTOSCAPE_URL = f"https://cdnjs.cloudflare.com/ajax/libs/cytoscape/{CYTOSCAPE_VERSION}/cytoscape.min.js"
CYTOSCAPE_FILE = ROOT / "data" / "vendor" / f"cytoscape-{CYTOSCAPE_VERSION}.min.js"

ALPHA, MIN_Z = 0.05, 1.0                 # group-level significance, as in the benchmark
METHODS = ("flow", "walk")
NET_THRESHOLDS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
NET_MAX_NODES = 500
PW_TOP, PW_MIN, PW_TOP_N = 6, 3, 100     # pathways per method: top 6 with q < 0.05, else top 3 by p
PW_MAX_REACTIONS = 150                   # larger subsystems: only reactions carrying current
CAUSAL_MAX_NODES = 200
SPREAD_MAX_NODES = 200                   # label-aware overlap removal up to this view size
ZERO = 1e-6                              # currents below this are stored as 0 (numerical noise)


# ---- formatting -----------------------------------------------------------------
def sig(x, digits=4):
    """Round to `digits` significant figures; NaN/inf become None (JSON null)."""
    if x is None:
        return None
    x = float(x)
    if not math.isfinite(x):
        return None
    return 0 if x == 0 else float(f"{x:.{digits}g}")


def cur(x):
    return 0 if abs(x) < ZERO else sig(x)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def rel(path):
    """A path relative to the repository when it lies inside it (for messages)."""
    path = Path(path).resolve()
    return path.relative_to(ROOT) if path.is_relative_to(ROOT) else path


def git_label():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = " + uncommitted changes" if run("status", "--porcelain", "--untracked-files=no") else ""
        return f"branch {run('branch', '--show-current')} · commit {run('rev-parse', '--short', 'HEAD')}{dirty}"
    except OSError:
        return ""


# ---- network and cases ----------------------------------------------------------
class Network:
    """The benchmark's network (role-based currency rules + reaction edits)
    and the lookups every case needs."""

    def __init__(self, model_path, yml_path):
        self.model = load_sbml(model_path)
        rules, _ = resolve_currency(self.model, "role")
        rules.edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, self.model)
        g = self.graph = build_graph(self.model, rules)
        self.subsystems = load_subsystems(yml_path)
        self.rxn_nodes = np.flatnonzero(g.is_reaction)
        base = lambda i: g.base_reaction.get(i, g.node_ids[i])
        # First subsystem per reaction node: the rule pathway_table uses.
        self.sub_of = {int(i): (self.subsystems.get(base(i)) or ["(none)"])[0] for i in self.rxn_nodes}
        self.universe = sorted(set().union(*g.reaction_genes.values()))
        self.universe_set = set(self.universe)
        self.symbol_to_ens = {}
        for ens, sym in self.model.genes.items():
            self.symbol_to_ens.setdefault(sym, ens)
        self.symbols = {i: [self.model.genes.get(e) or e for e in g.reaction_genes[i]] for i in self.rxn_nodes}
        # Edge direction from the model's arcs: +1 metabolite -> reaction,
        # -1 reaction -> metabolite, 0 both ways (reversible reaction).
        arcset = set(map(tuple, np.asarray(g.arcs).reshape(-1, 2).tolist()))
        self.edge_dir = np.array([int((m, r) in arcset) - int((r, m) in arcset)
                                  for m, r in g.edges.tolist()], dtype=int)
        self.edges_of = {}
        for e, (m, r) in enumerate(g.edges.tolist()):
            self.edges_of.setdefault(m, []).append(e)
            self.edges_of.setdefault(r, []).append(e)


def load_case_daa(net, daa_df, tmp_csv):
    """DAA -> mapped metabolites, exactly as benchmark.Ranker.run does it
    (through a CSV round trip, so empty identifiers read back as missing)."""
    daa_df.to_csv(tmp_csv, index=False)
    daa, _, _ = load_daa(tmp_csv, net.model, ["HMDB", "KEGG", "name"], "log2FC", "p", ALPHA, False, MIN_Z)
    daa["node"] = daa.met_id.map(net.graph.index)
    return daa


def run_method(net, method, on, sources, targets, ground, causal_ens):
    """One method on one case: FlowResult, scored reactions, pathway table and
    the causal gene's rank (computed exactly as the benchmark computes it)."""
    g = net.graph
    if method == "walk":
        flow = absorbing_walk(g, sources, targets, dict(zip(on.node, on.phi)), kappa=None, ground=ground)
    else:
        flow = current_flow(g, sources, targets, ground=ground)
    out = {"flow": flow, "rxn": None, "pathways": pd.DataFrame(), "causal_rank": None, "gene_ranks": {}}
    if not flow.pairs:
        return out
    nodes = node_table(g, flow)
    rxn = reaction_table(nodes, g, net.model, set(sources) | set(targets), ["conductivity", "bottleneck"])
    genes = gene_table(rxn, g, net.model)
    scored = [e for e in genes.ensembl_id if e in net.universe_set]
    rank = {e: i + 1 for i, e in enumerate(scored)}
    tail = (len(scored) + 1 + len(net.universe)) / 2.0       # unscored genes share the remaining ranks
    gene_rank = {e: float(rank.get(e, tail)) for e in causal_ens}
    out.update(rxn=rxn, pathways=pathway_table(rxn, g, net.subsystems, top_n=PW_TOP_N),
               gene_ranks=gene_rank, causal_rank=min(gene_rank.values()) if gene_rank else None)
    return out


def pick_pathways(pw):
    """Rows (by p) a method contributes views for: the top 6 with q < 0.05,
    or the top 3 by p when fewer than 3 are significant."""
    if not len(pw):
        return []
    sig_rows = pw[pw.q < 0.05].head(PW_TOP)
    rows = sig_rows if len(sig_rows) >= PW_MIN else pw.head(PW_MIN)
    return rows.subsystem.tolist()


def induced_edges(net, nodes):
    """All graph edges between the given nodes (sorted edge indices)."""
    ns = set(nodes)
    return sorted({e for v in ns if net.graph.is_reaction[v] for e in net.edges_of.get(v, [])
                   if int(net.graph.edges[e, 0]) in ns})


def label_text(net, v):
    """The label the page draws for a node (same rule as rxnLabel/short in the template)."""
    g = net.graph
    if g.is_reaction[v]:
        s = net.symbols[v]
        if not s:
            return g.node_ids[v].removeprefix("R_")
        return f"{s[0]}, {s[1]} +{len(s) - 2}" if len(s) > 2 else ", ".join(s)
    name = g.node_names[v]
    return name[:25] + "…" if len(name) > 26 else name


def spread(pos, boxes, iters=300):
    """Push apart nodes whose boxes overlap (each box: half-width, top, bottom
    relative to the node centre; the label hangs below the node). Deterministic:
    each overlapping pair moves apart along the axis with the smaller overlap."""
    n = len(pos)
    hw, top, bot = boxes[:, 0], boxes[:, 1], boxes[:, 2]
    iu = np.triu_indices(n, 1)
    for _ in range(iters):
        dx = pos[:, None, 0] - pos[None, :, 0]
        dy = pos[:, None, 1] - pos[None, :, 1]
        ox = (hw[:, None] + hw[None, :]) - np.abs(dx)
        # vertical overlap of [y+top, y+bot] intervals
        oy = np.minimum(pos[:, None, 1] + bot[:, None], pos[None, :, 1] + bot[None, :]) -              np.maximum(pos[:, None, 1] + top[:, None], pos[None, :, 1] + top[None, :])
        hit = (ox > 0) & (oy > 0)
        hit[np.diag_indices(n)] = False
        if not hit[iu].any():
            break
        move = np.zeros_like(pos)
        horiz = hit & (ox <= oy)
        vert = hit & (ox > oy)
        sx = np.where(dx >= 0, 1.0, -1.0)
        sy = np.where(dy >= 0, 1.0, -1.0)
        move[:, 0] = (np.where(horiz, sx * ox, 0) * 0.5).sum(axis=1)
        move[:, 1] = (np.where(vert, sy * oy, 0) * 0.5).sum(axis=1)
        pos = pos + np.clip(move, -12, 12)
    return pos


def layout(net, nodes, local_edges, seed):
    """Deterministic force layout scaled to pixels (~60 px per node), then, for
    small views, overlap removal on node + label boxes at 100% zoom (10 px labels,
    ~5.8 px per character; the page draws labels at a constant screen size)."""
    n_nodes = len(nodes)
    if n_nodes == 0:
        return np.zeros((0, 2))
    iters = 400 if n_nodes > 150 else 350
    pos = fruchterman_reingold(n_nodes, local_edges, iters=iters, seed=seed, width=1.0)
    pos = pos * (60.0 * math.sqrt(n_nodes))
    if n_nodes <= SPREAD_MAX_NODES:
        g = net.graph
        boxes = np.array([[max(11.0, 2.9 * len(label_text(net, v)) + 4),
                           -9.0 if not g.is_reaction[v] else -7.0, 27.0] for v in nodes])
        pos = spread(pos, boxes)
    return pos


# ---- payload --------------------------------------------------------------------
class Tables:
    """Global node and edge tables; views refer to them by index."""

    def __init__(self, net):
        self.net = net
        self.node_g, self.edge_g = {}, {}       # graph index -> global index
        self.nodes, self.edges = [], []

    def node(self, v):
        v = int(v)
        if v not in self.node_g:
            self.node_g[v] = len(self.nodes)
            self.nodes.append(v)
        return self.node_g[v]

    def edge(self, e):
        e = int(e)
        if e not in self.edge_g:
            self.edge_g[e] = len(self.edges)
            self.edges.append(e)
            m, r = self.net.graph.edges[e]
            self.node(m), self.node(r)
        return self.edge_g[e]

    def payload(self):
        g, net = self.net.graph, self.net
        subs = sorted({net.sub_of[v] for v in self.nodes if g.is_reaction[v]})
        sub_idx = {s: i for i, s in enumerate(subs)}
        return {
            "subsystems": subs,
            "nodes": {
                "id": [g.node_ids[v] for v in self.nodes],
                "name": [g.node_names[v] for v in self.nodes],
                "kind": [int(g.is_reaction[v]) for v in self.nodes],
                "genes": [";".join(net.symbols[v]) if g.is_reaction[v] else "" for v in self.nodes],
                "sub": [sub_idx[net.sub_of[v]] if g.is_reaction[v] else -1 for v in self.nodes],
                "rev": [int(bool(g.reversible.get(v, False))) if g.is_reaction[v] else 0 for v in self.nodes],
            },
            "edges": {
                "m": [self.node_g[int(g.edges[e, 0])] for e in self.edges],
                "r": [self.node_g[int(g.edges[e, 1])] for e in self.edges],
                "side": [0 if g.edge_side[e] == "substrate" else 1 for e in self.edges],
                "dir": [int(net.edge_dir[e]) for e in self.edges],
            },
        }


def build_case(net, tables, c, data, tmp_csv):
    """Everything the page shows for one case."""
    g = net.graph
    z, ann, cohorts = data[c.dataset]
    samples = [s for s in cohorts.get(c.cohort, []) if s in z.columns]
    daa = load_case_daa(net, group_daa(z, ann, samples), tmp_csv)
    on = daa.dropna(subset=["node"]).astype({"node": int})
    sources = on[on.role == "source"].node.tolist()
    targets = on[on.role == "target"].node.tolist()
    ground = on[on.role == ""].node.tolist() if bool(sources) != bool(targets) else None

    symbols = c.causal_genes.split(";")
    ens = [net.symbol_to_ens.get(s) for s in symbols]
    causal_ens = [e for e in ens if e and e in net.universe_set]
    causal_set = set(causal_ens)
    causal_rxn = [int(i) for i in net.rxn_nodes if causal_set & set(g.reaction_genes[i])]

    res = {m: run_method(net, m, on, sources, targets, ground, causal_ens) for m in METHODS}
    cond = {m: res[m]["flow"].conductivity for m in METHODS}
    emax = np.maximum(*(res[m]["flow"].edge_max_current for m in METHODS))
    rank_of = {}
    for m in METHODS:
        rxn = res[m]["rxn"]
        rank_of[m] = {} if rxn is None else {int(i): k + 1 for k, i in enumerate(rxn.index)}

    # -- view 1: the current-carrying subnetwork
    endpoints = sorted(set(sources) | set(targets))
    for thr in NET_THRESHOLDS:
        hot = np.flatnonzero(emax >= thr)
        nodes = sorted(set(g.edges[hot].ravel().tolist()) | set(endpoints))
        if len(nodes) <= NET_MAX_NODES:
            break
    views = [{"id": "net", "kind": "net", "title": "Current-carrying subnetwork", "threshold": thr,
              "nodes": nodes, "note": f"edges with max current ≥ {thr:g} under either method, plus every "
                                      f"source and target"}]

    # -- pathway views: union of each method's picks, in order of best p
    picks = {m: pick_pathways(res[m]["pathways"]) for m in METHODS}
    pw_rows = {m: {r.subsystem: r for r in res[m]["pathways"].itertuples()} for m in METHODS}
    # ties in p (same hits and size) fall back to the name, so the order does not
    # depend on set iteration (string hashing differs between runs)
    union = sorted(set(picks["flow"]) | set(picks["walk"]),
                   key=lambda s: (min(pw_rows[m][s].p for m in METHODS if s in pw_rows[m]), s))
    for sub in union:
        members = [int(i) for i in net.rxn_nodes if net.sub_of[int(i)] == sub]
        total, rule = len(members), ""
        if total > PW_MAX_REACTIONS:
            members = [i for i in members if any(cond[m][i] > 0 for m in METHODS)]
            rule = "reactions carrying current under either method"
        mets = {int(g.edges[e, 0]) for i in members for e in net.edges_of.get(i, [])}
        if rule and len(members) == total:
            rule = ""                       # every reaction of the subsystem carries current
        note = f"showing {len(members)} of {total} reactions ({rule})" if rule else f"all {total} reactions"
        views.append({"id": "pw-" + slug(sub), "kind": "pathway", "title": sub, "subsystem": sub,
                      "nodes": sorted(set(members) | mets), "shown": len(members), "total": total, "note": note,
                      "selected_by": [m for m in METHODS if sub in picks[m]]})

    # -- the causal gene's neighbourhood: its reactions and their metabolites, plus
    #    the reactions next to those metabolites that carry current (max >= thr)
    if causal_rxn:
        core_mets = {int(g.edges[e, 0]) for i in causal_rxn for e in net.edges_of.get(i, [])}
        core = set(causal_rxn) | core_mets
        for thr_c in NET_THRESHOLDS + (math.inf,):
            near = {int(g.edges[e, 1]) for m in core_mets for e in net.edges_of.get(m, [])
                    if emax[e] >= thr_c} - core
            nodes_c = core | near
            if len(nodes_c) <= CAUSAL_MAX_NODES:
                break
        note = (f"the causal gene's {len(causal_rxn)} reaction{'s' if len(causal_rxn) > 1 else ''} and "
                f"{len(core_mets)} metabolites" +
                (f", plus {len(near)} neighbouring reactions with max current ≥ {thr_c:g} under either method"
                 if near else "; no neighbouring reaction carries max current ≥ 0.1"))
        views.append({"id": "causal", "kind": "causal", "title": "Causal-gene neighbourhood",
                      "nodes": sorted(nodes_c), "threshold": None if thr_c == math.inf else thr_c, "note": note})

    # -- layouts, global indices, per-view edges
    case_edges, case_rxn = set(), set(causal_rxn)
    for k, v in enumerate(views):
        local = {n: j for j, n in enumerate(v["nodes"])}
        edges = induced_edges(net, v["nodes"])
        seed = int(hashlib.sha256(f"{c.case_id}/{v['id']}".encode()).hexdigest()[:8], 16)
        pos = layout(net, v["nodes"], [(local[int(g.edges[e, 0])], local[int(g.edges[e, 1])]) for e in edges], seed)
        case_edges.update(edges)
        case_rxn.update(n for n in v["nodes"] if g.is_reaction[n])
        v["pos"] = [int(round(x)) for x in pos.ravel()]
        v["edges"] = [tables.edge(e) for e in edges]
        v["nodes"] = [tables.node(n) for n in v["nodes"]]
    case_edges, case_rxn = sorted(case_edges), sorted(case_rxn)

    # -- metabolites: every mapped metabolite on the graph (others cannot be drawn)
    mets = {"node": [tables.node(n) for n in on.node], "z": [sig(x) for x in on.log2fc],
            "q": [sig(x) for x in on.p], "role": on.role.tolist(), "input": on.input_id.astype(str).tolist()}

    # -- per method
    methods = {}
    for m in METHODS:
        r, flow, rxn = res[m], res[m]["flow"], res[m]["rxn"]
        rows = []
        for sub in union:
            row = pw_rows[m].get(sub)
            base = {"subsystem": sub, "selected": sub in picks[m]}
            if row is None:            # no top-100 reaction of this method falls in the subsystem
                k = next(v["total"] for v in views if v.get("subsystem") == sub)
                rows.append({**base, "reactions_in_network": k, "in_top": 0, "expected": None, "fold": None,
                             "p": None, "q": None, "significant": False})
            else:
                rows.append({**base, "reactions_in_network": int(row.reactions_in_network), "in_top": int(row.in_top),
                             "expected": sig(row.expected), "fold": sig(row.fold), "p": sig(row.p), "q": sig(row.q),
                             "significant": bool(row.q < 0.05)})
        in_tbl = set() if rxn is None else set(rxn.index)
        get = lambda col, i: sig(rxn.at[i, col]) if i in in_tbl else None
        methods[m] = {
            "causal_rank": r["causal_rank"],
            "causal_gene_ranks": {net.model.genes[e]: v for e, v in r["gene_ranks"].items()},
            "pairs": len(flow.pairs),
            "reactions_scored": len(in_tbl),
            "walk_success_fraction": sig(flow.walk_info["walk_success_fraction"]) if flow.walk_info else None,
            "sources_unreached": flow.walk_info["walk_sources_unreached"] if flow.walk_info else None,
            "pathways": rows,
            "rxn": {"score": [get("final_score", i) for i in case_rxn],
                    "rank": [rank_of[m].get(i) for i in case_rxn],
                    "cond": [get("conductivity", i) for i in case_rxn],
                    "bott": [get("bottleneck", i) for i in case_rxn],
                    "nf": ([sig(flow.net_forward[i]) for i in case_rxn] if flow.net_forward is not None else None)},
            "edge_mean": [cur(flow.edge_current[e]) for e in case_edges],
            "edge_max": [cur(flow.edge_max_current[e]) for e in case_edges],
        }

    return {
        "id": c.case_id, "disorder": c.disorder, "dataset": c.dataset, "status": c.status, "category": c.category,
        "patients": len(samples), "causal_genes": symbols,
        "causal_in_network": [net.model.genes[e] for e in causal_ens],
        "causal_rxn": [tables.node(i) for i in causal_rxn],
        "measured": int(len(daa)), "mapped_on_graph": int(len(on)),
        "sources": len(sources), "targets": len(targets), "one_sided": bool(ground),
        "ground": len(ground or []),
        "rxn": [tables.node(i) for i in case_rxn],
        "edges": [tables.edge(e) for e in case_edges],
        "mets": mets, "methods": methods,
        "views": [{k: v[k] for k in v} for v in views],
    }


def check_ranks(payload):
    """The causal-gene ranks must equal the benchmark's (raw scoring)."""
    rows, bad = [], 0
    for m, d in BENCH_DIRS.items():
        tsv = BENCH / d / "group_results.tsv"
        if not tsv.exists():
            print(f"  (no {tsv.relative_to(ROOT)}; rank check skipped for {m})")
            continue
        ref = pd.read_csv(tsv, sep="\t").set_index("case").rank_metaboenrich_raw
        for c in payload["cases"]:
            ours, theirs = c["methods"][m]["causal_rank"], ref.get(c["id"])
            ok = (ours is None and pd.isna(theirs)) or (ours is not None and abs(ours - theirs) < 1e-9)
            bad += not ok
            rows.append((c["id"], m, ours, theirs, ok))
    return rows, bad


def build_payload(case_ids=None, model="data/Human-GEM.xml", yml="data/Human-GEM.yml"):
    t0 = time.time()
    net = Network(ROOT / model, ROOT / yml)
    print(f"Network: {net.graph.n} nodes, {len(net.graph.edges)} edges ({time.time() - t0:.0f}s)")
    data = {"t2020": load_t2020(), "miller": load_miller()}
    cases = pd.read_csv(ROOT / CASES, sep="\t").fillna("")
    cases = cases[cases.status.isin(["primary", "secondary"])]
    if case_ids:
        missing = set(case_ids) - set(cases.case_id)
        if missing:
            raise SystemExit(f"unknown or excluded case(s): {', '.join(sorted(missing))}")
        cases = cases[cases.case_id.isin(case_ids)]
    tables = Tables(net)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for _, c in cases.iterrows():
            t = time.time()
            out.append(build_case(net, tables, c, data, Path(tmp) / "daa.csv"))
            cs = out[-1]
            print(f"  {c.case_id:6s} src={cs['sources']:3d} tgt={cs['targets']:3d} views={len(cs['views']):2d} "
                  f"rank flow={cs['methods']['flow']['causal_rank']} walk={cs['methods']['walk']['causal_rank']} "
                  f"({time.time() - t:.0f}s)")
    payload = {
        "meta": {
            "generated": time.strftime("%Y-%m-%d"), "git": git_label(),
            "network": {"nodes": net.graph.n, "edges": int(len(net.graph.edges)), "genes": len(net.universe)},
            "significance": "BH q < 0.05 and |mean z| >= 1", "z_clip": 4,
            "net_threshold_default": NET_THRESHOLDS[0], "net_max_nodes": NET_MAX_NODES,
            "pathway_rule": f"per method: top {PW_TOP} subsystems with q < 0.05 (by p), or the top {PW_MIN} by p "
                            f"if fewer than {PW_MIN} qualify; over-representation among the top {PW_TOP_N} reactions",
            "pathway_max_reactions": PW_MAX_REACTIONS, "zero_current": ZERO,
            "methods": {"flow": "Current flow", "walk": "Absorbing walk"},
            "cytoscape": CYTOSCAPE_VERSION,
        },
        **tables.payload(),
        "cases": out,
    }
    print(f"Built {len(out)} cases in {time.time() - t0:.0f}s")
    return payload


# ---- page -----------------------------------------------------------------------
def cytoscape_js():
    """The pinned cytoscape.js build: downloaded once, hash-checked every time."""
    if not CYTOSCAPE_FILE.exists():
        CYTOSCAPE_FILE.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading cytoscape.js {CYTOSCAPE_VERSION} to {CYTOSCAPE_FILE.relative_to(ROOT)} ...")
        urllib.request.urlretrieve(CYTOSCAPE_URL, CYTOSCAPE_FILE)
    blob = CYTOSCAPE_FILE.read_bytes()
    if hashlib.sha256(blob).hexdigest() != CYTOSCAPE_SHA256:
        raise SystemExit(f"{CYTOSCAPE_FILE} has an unexpected SHA-256; delete it to download again")
    return blob.decode("utf-8")


def render_page(payload, template=TEMPLATE):
    """Fill the template's two placeholders: the data JSON and the library."""
    page = Path(template).read_text(encoding="utf-8")
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).replace("</", "<\\/")
    for ph in ("/*__DATA__*/", "/*__CYTOSCAPE__*/"):
        if page.count(ph) != 1:
            raise SystemExit(f"template must contain {ph} exactly once")
    # Library first: the data JSON could contain the placeholder text, the library cannot.
    return page.replace("/*__CYTOSCAPE__*/", cytoscape_js()).replace("/*__DATA__*/", data)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="docs/network_report.html")
    p.add_argument("--cases", nargs="+", default=None, help="only these case ids (for quick tests)")
    p.add_argument("--json-only", action="store_true", help="write the data to --json and stop")
    p.add_argument("--json", default="results/network_report/data.json")
    p.add_argument("--from-json", default=None, metavar="PATH",
                   help="render the page from a data file written by --json-only instead of recomputing")
    args = p.parse_args(argv)

    if args.from_json:
        payload = json.loads((ROOT / args.from_json).read_text(encoding="utf-8"))
        if args.cases:
            payload["cases"] = [c for c in payload["cases"] if c["id"] in set(args.cases)]
    else:
        payload = build_payload(args.cases)
    rows, bad = check_ranks(payload)
    print(f"Causal-gene ranks vs benchmark: {len(rows) - bad}/{len(rows)} match")
    if bad:
        for r in rows:
            if not r[4]:
                print(f"  MISMATCH {r[0]} {r[1]}: ours {r[2]} vs benchmark {r[3]}")
        raise SystemExit("causal-gene ranks differ from the benchmark results")

    if args.json_only:
        out = ROOT / args.json
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, separators=(",", ":"), ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {rel(out)} ({out.stat().st_size / 1e6:.2f} MB)")
        return
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_page(payload), encoding="utf-8")
    print(f"Wrote {rel(out)} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
