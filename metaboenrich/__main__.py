"""Metaboenrich prototype: current-flow analysis of a DAA on Human-GEM.

    python -m metaboenrich --daa examples/example_daa.csv --out results/example
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .cytoscape import write_graphml
from . import __version__
from .daa import load_daa, load_name_bridge
from .flow import GROUND
from .flow import current_flow
from .gem import download_human_gem, load_sbml, load_subsystems
from .scoring import gene_table, node_table, pathway_table, rank_score, reaction_table, transporter_table
from .network import (DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits, metabolite_degrees,
                      resolve_currency, transport_genes)

DEFAULT_MODEL = "data/Human-GEM.xml"


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="metaboenrich", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="SBML model; the default path is downloaded (Human-GEM v2.0.1) if missing")
    p.add_argument("--daa", required=True, help="DAA table (.csv or .tsv)")
    p.add_argument("--out", default="results", help="output directory (default: %(default)s)")
    p.add_argument("--id-col", nargs="+", default=["metabolite"],
                   help="metabolite identifier column(s), tried in order: Human-GEM id, HMDB, KEGG, ChEBI "
                        "or name, e.g. --id-col HMDB KEGG Name (default: %(default)s)")
    p.add_argument("--fc-col", default="log2FC", help="log2 fold-change column (default: %(default)s)")
    p.add_argument("--fc-linear", action="store_true", help="--fc-col holds linear fold change, not log2")
    p.add_argument("--p-col", default="padj", help="p-value column (default: %(default)s)")
    p.add_argument("--alpha", type=float, default=0.05, help="significance cutoff (default: %(default)s)")
    p.add_argument("--min-log2fc", type=float, default=0.0,
                   help="sources/targets also need |log2FC| >= this, e.g. 0.585 for 1.5-fold (default: %(default)s)")
    p.add_argument("--mode", choices=["undirected", "directed"], default="undirected",
                   help="directed keeps only pairs joined by a directed s->t path (protocol §2.2)")
    p.add_argument("--pair-weight", choices=["unit", "fc"], default="unit",
                   help="unit: every pair injects 1 A (protocol as written); "
                        "fc: weight pair by (|log2FC_s| + |log2FC_t|) / 2")
    p.add_argument("--currency-mode", choices=["role", "remove"], default="role",
                   help="role: remove inorganic currency everywhere, and cofactors (ATP, NAD+, CoA, ...) only "
                        "where they act as exchangers, keeping their synthesis/catabolism reactions; "
                        "remove: remove all of them everywhere (default: %(default)s)")
    p.add_argument("--no-default-currency", action="store_true", help="don't apply the curated currency lists")
    p.add_argument("--currency", nargs="*", default=[], metavar="MET_ID",
                   help="extra metabolites to remove everywhere (Human-GEM ids, e.g. MAM01261)")
    p.add_argument("--currency-degree", type=int, default=None,
                   help="also treat any metabolite in more than N reactions as currency")
    p.add_argument("--reaction-edits", default=str(DEFAULT_REACTION_EDITS),
                   help="TSV of (reaction_id, met_id) pairs to remove from the model, correcting "
                        "annotation problems; 'none' to disable (default: the bundled reaction_edits.tsv)")
    p.add_argument("--max-reaction-size", type=int, default=20,
                   help="drop pseudo-reactions with more than N metabolites (default: %(default)s)")
    p.add_argument("--scoring", choices=["raw", "loo"], default="raw",
                   help="rank on raw conductivity/bottleneck (default; best on the inborn-error benchmark) "
                        "or on leave-one-out versions, which discount reactions whose score rests on a "
                        "single source or target. Both are always written to reactions.tsv")
    p.add_argument("--no-leave-one-out", action="store_true", help=argparse.SUPPRESS)  # old spelling of --scoring raw
    p.add_argument("--one-sided", choices=["auto", "off"], default="auto",
                   help="auto: if nothing significant goes down (or up), withdraw (or supply) current evenly "
                        "through the measured, unchanged metabolites (default: %(default)s)")
    p.add_argument("--id-translation", default=None, metavar="CSV",
                   help="synonym table (KEGG, HMDB, NAME1..NAMEn) used to map metabolites given by name")
    p.add_argument("--subsystems", default="data/Human-GEM.yml",
                   help="Human-GEM YAML with reaction subsystems, for pathways.tsv (downloaded if missing)")
    p.add_argument("--pathway-top", type=int, default=100,
                   help="pathways.tsv tests subsystems for over-representation among the top N reactions "
                        "(default: %(default)s)")
    p.add_argument("--cytoscape-min-current", type=float, default=0.1,
                   help="current_subnetwork.graphml keeps edges carrying at least this share of the "
                        "current for at least one source-target pair (default: %(default)s)")
    p.add_argument("--bottleneck-tau", type=float, default=0.5,
                   help="threshold for the bottleneck_frac column: fraction of pairs in which a node "
                        "carries >= tau of the current (default: %(default)s)")
    return p.parse_args(argv)


def write_cytoscape(out, graph, model, flow, rxn, inter, ep, on, sources, targets, min_current):
    """Write cytoscape/full_network.graphml (every node, with scores) and
    cytoscape/current_subnetwork.graphml (edges carrying >= min_current of
    the unit current for at least one source-target pair, plus every
    source/target).
    Returns (nodes, edges) in the subnetwork."""
    cyto = out / "cytoscape"
    cyto.mkdir(exist_ok=True)
    n = graph.n
    col = lambda: [None] * n

    role = ["reaction" if r else "metabolite" for r in graph.is_reaction]
    for v in sources:
        role[v] = "source"
    for v in targets:
        role[v] = "target"
    attrs = {"label": list(graph.node_names),
             "type": ["reaction" if r else "metabolite" for r in graph.is_reaction],
             "role": role}
    for k in ("log2fc", "p"):
        attrs[k] = col()
        for node, v in zip(on.node, on[k]):
            attrs[k][node] = float(v)
    for k in ("final_score", "rank", "conductivity_loo", "bottleneck_loo", "conductivity", "bottleneck",
              "accumulated_current", "genes", "only_reaction_of"):
        attrs[k] = col()
    for table in (rxn, inter):
        ranks = table.final_score.rank(ascending=False, method="min")
        for i, r in table.iterrows():
            for k in ("final_score", "conductivity_loo", "bottleneck_loo", "conductivity", "bottleneck",
                      "accumulated_current"):
                attrs[k][i] = float(r[k])
            attrs["rank"][i] = int(ranks[i])
            if graph.is_reaction[i]:
                attrs["genes"][i] = r.genes
                attrs["only_reaction_of"][i] = r.only_reaction_of
    for _, r in ep.iterrows():
        i = graph.index[r.met_id]
        attrs["final_score"][i] = float(r.final_score)
        attrs["conductivity"][i] = float(r.conductivity)   # mean effective conductance for endpoints
        attrs["bottleneck"][i] = float(r.bottleneck)
    for i in range(n):
        if role[i] == "metabolite" and attrs["final_score"][i] is not None:
            role[i] = "intermediate"

    edge_attrs = {"mean_current": [float(x) for x in flow.edge_current],
                  "max_current": [float(x) for x in flow.edge_max_current],
                  "side": graph.edge_side,
                  "reversible": [bool(graph.reversible[r]) for r in graph.edges[:, 1]]}

    write_graphml(cyto / "full_network.graphml", graph, attrs, edge_attrs)
    e_mask = flow.edge_max_current >= min_current
    n_mask = np.zeros(n, bool)
    n_mask[graph.edges[e_mask].ravel()] = True
    n_mask[list(sources) + list(targets)] = True
    return write_graphml(cyto / "current_subnetwork.graphml", graph, attrs, edge_attrs, n_mask, e_mask)


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if not Path(args.model).exists():
        if args.model != DEFAULT_MODEL:
            raise SystemExit(f"Model file not found: {args.model}")
        download_human_gem(args.model)
    model = load_sbml(args.model)
    print(f"Loaded {model.id} {model.version}: {len(model.reactions)} reactions, "
          f"{len(model.species)} species, {len(model.genes)} genes")

    # --- currency metabolites -------------------------------------------------
    rules, missing = resolve_currency(model, args.currency_mode, not args.no_default_currency,
                                      args.currency, args.currency_degree, args.max_reaction_size)
    if missing:
        print(f"  note: curated currency names not in model: {', '.join(missing)}")
    if args.reaction_edits.lower() != "none":
        rules.edits, unknown = load_reaction_edits(args.reaction_edits, model)
        n_edits = sum(len(v) for v in rules.edits.values())
        print(f"Reaction edits: {n_edits} metabolite(s) removed from {len(rules.edits)} reaction(s) "
              f"({args.reaction_edits})")
        if unknown:
            print(f"  warning: reaction edits not matching the model: {unknown}")

    bridge = load_name_bridge(args.id_translation) if args.id_translation else None
    daa, unmapped, dupes = load_daa(args.daa, model, args.id_col, args.fc_col, args.p_col,
                                    args.alpha, args.fc_linear, args.min_log2fc, name_bridge=bridge)
    via = daa.matched_on.str.contains("id_translation|loose").sum()
    print(f"DAA: {len(daa) + len(unmapped)} rows, {len(daa)} mapped ({via} by name lookup), {len(unmapped)} unmapped, "
          f"{(daa.role == 'source').sum()} sources, {(daa.role == 'target').sum()} targets")

    lost = daa[daa.met_id.isin(rules.removed) & (daa.role != "")]
    if len(lost):
        print(f"  warning: {len(lost)} significant DAA metabolites are currency and were removed: "
              + ", ".join(lost.input_id.astype(str)))
        daa.loc[lost.index, "role"] = "currency (removed)"

    # --- graph and current flow --------------------------------------------
    graph = build_graph(model, rules, args.max_reaction_size)
    n_rxn = int(graph.is_reaction.sum())
    audit = pd.DataFrame(graph.currency_audit, columns=["met_id", "reaction_id", "decision", "partner"])
    print(f"Graph: {graph.n - n_rxn} metabolite nodes, {n_rxn} reaction nodes, {len(graph.edges)} edges "
          f"({len(rules.removed)} currency metabolites removed everywhere"
          + (f"; {len(rules.family_of)} cofactors kept in {(audit.decision == 'kept').sum()} "
             f"synthesis/catabolism reactions, removed from {(audit.decision == 'exchange').sum()} "
             f"exchange reactions" if rules.family_of else "") + ")")

    deg, met_names = metabolite_degrees(model, args.max_reaction_size)
    rxn_names = {r.id: r.name for r in model.reactions.values()}
    daa_by_met = daa.set_index("met_id")
    counts = audit.groupby(["met_id", "decision"]).size().unstack(fill_value=0)
    cur = pd.DataFrame({"met_id": list(deg), "name": [met_names[m] for m in deg],
                        "n_reactions": list(deg.values())})
    cur["handling"] = cur.met_id.map(
        lambda m: f"removed ({rules.removed[m]})" if m in rules.removed
        else f"role-filtered ({rules.family_of[m]})" if m in rules.family_of else "")
    for col in ("kept", "exchange"):
        cur[f"{col}_reactions"] = cur.met_id.map(counts[col] if col in counts else {}).fillna(0).astype(int)
    cur["in_daa"] = cur.met_id.isin(daa_by_met.index)
    cur["daa_log2fc"] = cur.met_id.map(daa_by_met["log2fc"])
    cur = cur.sort_values("n_reactions", ascending=False)
    cur.to_csv(out / "currency_metabolites.tsv", sep="\t", index=False)
    if len(audit):
        audit.insert(1, "met_name", audit.met_id.map(met_names))
        audit.insert(3, "reaction_name", audit.reaction_id.map(rxn_names))
        audit["partner_name"] = audit.partner.map(met_names).fillna("")
        audit.sort_values(["met_name", "decision", "reaction_id"]).to_csv(
            out / "currency_edges.tsv", sep="\t", index=False)

    daa["node"] = daa.met_id.map(graph.index)
    off_graph = daa[daa.node.isna() & daa.role.isin(["source", "target"])]
    if len(off_graph):
        print(f"  warning: {len(off_graph)} significant DAA metabolites have no usable reactions: "
              + ", ".join(off_graph.input_id.astype(str)))
        daa.loc[off_graph.index, "role"] = "not in graph"
    on = daa.dropna(subset=["node"]).astype({"node": int})
    sources = on[on.role == "source"].node.tolist()
    targets = on[on.role == "target"].node.tolist()
    phi = dict(zip(on.node, on.phi)) if args.pair_weight == "fc" else None

    # One-sided data: nothing significant in one direction. The ground is the
    # measured metabolites that did not change.
    ground = None
    if args.one_sided == "auto" and bool(sources) != bool(targets):
        if args.mode == "directed":
            print("  note: one-sided mode needs undirected flow; running without it")
        else:
            ground = on[on.role == ""].node.tolist()
            side = "sources" if sources else "targets"
            print(f"One-sided: only {side} are significant; current is exchanged with {len(ground)} "
                  f"measured, unchanged metabolites (the ground)")
    flow = current_flow(graph, sources, targets, weights=phi,
                        directed=args.mode == "directed", tau=args.bottleneck_tau, ground=ground)
    print(f"Current flow: {len(flow.pairs)} source-target pairs retained, {len(flow.excluded)} excluded "
          f"({args.mode}{', one-sided' if ground else ''})")

    # --- scoring (§1.5) -------------------------------------------------------
    nodes = node_table(graph, flow)
    use_loo = args.scoring == "loo" and not args.no_leave_one_out
    score_cols = ["conductivity_loo", "bottleneck_loo"] if use_loo else ["conductivity", "bottleneck"]
    endpoints = set(sources) | set(targets)

    rxn = reaction_table(nodes, graph, model, endpoints, score_cols)
    rxn.drop(columns="is_reaction").rename(columns={"node_id": "reaction_id"}).to_csv(
        out / "reactions.tsv", sep="\t", index=False)

    inter = nodes[~nodes.is_reaction & (nodes.conductivity > 0) & ~nodes.index.isin(endpoints)].copy()
    inter["final_score"] = rank_score(inter, score_cols)
    inter = inter.sort_values(["final_score", "conductivity"], ascending=False)
    inter.drop(columns="is_reaction").rename(columns={"node_id": "met_id"}).to_csv(
        out / "intermediate_metabolites.tsv", sep="\t", index=False)

    ep_rows = []
    for _, r in on[on.node.isin(flow.endpoint.keys())].iterrows():
        e = flow.endpoint[r.node]
        ep_rows.append({"input_id": r.input_id, "met_id": r.met_id, "name": graph.node_names[r.node],
                        "role": r.role, "log2fc": r.log2fc, "p": r.p, "pairs": e["pairs"],
                        # conductivity: mean effective conductance 1/R_eff to its partners
                        "conductivity": e["eff_conductance"],
                        # bottleneck: max share of current it relays between *other* pairs
                        "bottleneck": (flow.bottleneck_loo if use_loo else flow.bottleneck)[r.node],
                        "conduit_current": flow.conductivity[r.node],
                        # mean share of its own current leaving through a single reaction
                        "exit_concentration": e["max_edge_share"]})
    ep = pd.DataFrame(ep_rows)
    if len(ep):
        ep["final_score"] = pd.concat([rank_score(g, ["conductivity", "bottleneck"])
                                       for _, g in ep.groupby("role")])
        ep = ep.sort_values(["role", "final_score"], ascending=[True, False])
    ep.to_csv(out / "source_target_metabolites.tsv", sep="\t", index=False)

    genes = gene_table(rxn, graph, model)
    genes.to_csv(out / "genes.tsv", sep="\t", index=False)

    # Pathways (Human-GEM subsystems) over-represented among the top reactions.
    subsystems = load_subsystems(args.subsystems)
    pathways = pathway_table(rxn, graph, subsystems, args.pathway_top)
    pathways.to_csv(out / "pathways.tsv", sep="\t", index=False)

    # Transporters, scored by what they carry (kept apart from the enzyme ranking).
    changed = on[on.role.isin(["source", "target"])]
    met_scores = dict(zip(changed.met_id, zip(changed.log2fc.abs().rank(pct=True), ["measured change"] * len(changed))))
    for i, r in inter.iterrows():
        met_scores.setdefault(r.node_id, (float(r.final_score), "carries current"))
    transporters = transporter_table(transport_genes(model, rules), met_scores, model)
    if len(transporters):
        transporters["best_metabolite"] = transporters.best_metabolite.map(lambda m: f"{m} {met_names.get(m, '')}")
    transporters.to_csv(out / "transporters.tsv", sep="\t", index=False)

    daa.drop(columns="node").to_csv(out / "daa_mapping.tsv", sep="\t", index=False)
    if len(unmapped):
        unmapped.to_csv(out / "daa_unmapped.tsv", sep="\t", index=False)
    if flow.excluded:
        label = lambda v: "ground" if v == GROUND else graph.node_ids[v]
        pd.DataFrame([(label(s), label(t), why) for s, t, why in flow.excluded],
                     columns=["source", "target", "reason"]).to_csv(out / "excluded_pairs.tsv", sep="\t", index=False)

    # --- Cytoscape ----------------------------------------------------------
    n_cyto = write_cytoscape(out, graph, model, flow, rxn, inter, ep, on, sources, targets,
                             args.cytoscape_min_current)

    edge_list = "\n".join(sorted(f"{graph.node_ids[m]}\t{graph.node_ids[r]}\t{s}"
                                 for (m, r), s in zip(graph.edges, graph.edge_side)))
    summary = {
        "metaboenrich_version": __version__,
        "network_edges": len(graph.edges),
        # sha256 of the sorted edge list; equals network/network_info.json's
        # edges_sha256 when the run used the locked network
        "network_edges_sha256": hashlib.sha256(edge_list.encode()).hexdigest(),
        "scoring": "leave-one-out" if use_loo else "raw", "one_sided": bool(ground),
        "id_translation": str(args.id_translation or ""),
        "model": f"{model.id} {model.version}", "daa": str(args.daa), "mode": args.mode,
        "pair_weight": args.pair_weight, "alpha": args.alpha, "bottleneck_tau": args.bottleneck_tau,
        "currency_mode": args.currency_mode,
        "currency_removed_everywhere": len(rules.removed),
        "cofactors_role_filtered": len(rules.family_of),
        "reaction_edits": sum(len(v) for v in rules.edits.values()),
        "graph_metabolites": graph.n - n_rxn, "graph_reactions": n_rxn,
        "sources": len(sources), "targets": len(targets),
        "pairs_retained": len(flow.pairs), "pairs_excluded": len(flow.excluded),
        "reactions_carrying_current": len(rxn), "genes_scored": len(genes),
        "cytoscape_min_current": args.cytoscape_min_current,
        "cytoscape_subnetwork_nodes": n_cyto[0], "cytoscape_subnetwork_edges": n_cyto[1],
        "runtime_s": round(time.time() - t0, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nTop reactions:")
    for _, r in rxn.head(10).iterrows():
        note = f"  <- only reaction of {r.only_reaction_of}" if r.only_reaction_of else ""
        c, b = r[score_cols[0]], r[score_cols[1]]
        print(f"  {r.final_score:.3f}  cond={c:.3f} bott={b:.3f}  "
              f"{r.node_id} {r['name'][:50]}  [{r.genes[:40]}]{note}")
    if len(genes):
        print(f"Top genes: " + ", ".join(f"{g.symbol or g.ensembl_id}" for _, g in genes.head(15).iterrows()))
    print(f"Cytoscape: {out / 'cytoscape'}/current_subnetwork.graphml ({n_cyto[0]} nodes, {n_cyto[1]} edges "
          f"carrying >= {args.cytoscape_min_current} of some pair's current) and full_network.graphml")
    print(f"\nWrote results to {out}/ in {summary['runtime_s']}s")


if __name__ == "__main__":
    main()
