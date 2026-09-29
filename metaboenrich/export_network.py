"""Export the analysis network (after currency handling and reaction edits)
as plain tables and GraphML, with a record of how it was built.

    python -m metaboenrich.export_network --out network

This depends only on the model and the network settings, not on a DAA, so
the exported copy is the fixed network every analysis runs on.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from .cytoscape import write_graphml
from .gem import HUMAN_GEM_SHA256, HUMAN_GEM_URL, download_human_gem, load_sbml
from .network import (COFACTOR_FAMILIES, DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits,
                      resolve_currency)

DEFAULT_MODEL = "data/Human-GEM.xml"


def main(argv=None):
    p = argparse.ArgumentParser(prog="metaboenrich.export_network", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="network")
    p.add_argument("--currency-mode", choices=["role", "remove"], default="role")
    p.add_argument("--reaction-edits", default=str(DEFAULT_REACTION_EDITS))
    p.add_argument("--max-reaction-size", type=int, default=20)
    args = p.parse_args(argv)

    if not Path(args.model).exists():
        if args.model != DEFAULT_MODEL:
            raise SystemExit(f"Model file not found: {args.model}")
        download_human_gem(args.model)
    sha = hashlib.sha256(Path(args.model).read_bytes()).hexdigest()
    model = load_sbml(args.model)
    rules, _ = resolve_currency(model, args.currency_mode, max_reaction_size=args.max_reaction_size)
    if args.reaction_edits.lower() != "none":
        rules.edits, _ = load_reaction_edits(args.reaction_edits, model)
    graph = build_graph(model, rules, args.max_reaction_size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    species = {}
    for s in model.species.values():
        species.setdefault(s.met_id, s)

    rows = []
    for i, nid in enumerate(graph.node_ids):
        if graph.is_reaction[i]:
            genes = graph.reaction_genes[i]
            rows.append({"node_id": nid, "name": graph.node_names[i], "type": "reaction",
                         "reversible": graph.reversible[i],
                         "genes": ";".join(model.genes.get(g) or g for g in genes),
                         "ensembl_ids": ";".join(genes)})
        else:
            s = species[nid]
            rows.append({"node_id": nid, "name": graph.node_names[i], "type": "metabolite",
                         "formula": s.formula, "hmdb": ";".join(s.xrefs.get("hmdb", [])),
                         "kegg": ";".join(s.xrefs.get("kegg", [])),
                         "chebi": ";".join(s.xrefs.get("chebi", [])),
                         "cofactor_family": rules.family_of.get(nid, "")})
    nodes = pd.DataFrame(rows)
    nodes.to_csv(out / "nodes.tsv", sep="\t", index=False)

    edges = pd.DataFrame({
        "metabolite": [graph.node_ids[m] for m in graph.edges[:, 0]],
        "reaction": [graph.node_ids[r] for r in graph.edges[:, 1]],
        "side": graph.edge_side,
        "reversible": [graph.reversible[r] for r in graph.edges[:, 1]],
    })
    edges.to_csv(out / "edges.tsv", sep="\t", index=False)

    node_attrs = {c: nodes[c].where(nodes[c].notna(), None).tolist() for c in nodes.columns if c != "node_id"}
    node_attrs["label"] = node_attrs.pop("name")
    write_graphml(out / "network.graphml", graph, node_attrs,
                  {"side": graph.edge_side, "reversible": edges.reversible.tolist()})

    names = {s.met_id: s.name for s in model.species.values()}
    audit = pd.DataFrame(graph.currency_audit, columns=["met_id", "reaction_id", "decision", "partner"])
    audit.insert(1, "met_name", audit.met_id.map(names))
    audit["partner_name"] = audit.partner.map(names).fillna("")
    audit.sort_values(["met_name", "decision", "reaction_id"]).to_csv(
        out / "currency_edges.tsv", sep="\t", index=False)

    n_rxn = int(graph.is_reaction.sum())
    info = {
        "model": f"{model.id} {model.version}",
        "model_url": HUMAN_GEM_URL if sha == HUMAN_GEM_SHA256 else str(args.model),
        "model_sha256": sha,
        "currency_mode": args.currency_mode,
        "removed_everywhere": {names[m]: why for m, why in sorted(rules.removed.items(), key=lambda x: names[x[0]])},
        "cofactor_families": {f: spec["members"] for f, spec in COFACTOR_FAMILIES.items()}
        if args.currency_mode == "role" else {},
        "pool_metabolites_dropped_with_their_reactions": sorted(names[m] for m in rules.pools),
        "reaction_edits": {r: sorted(names[m] for m in ms) for r, ms in rules.edits.items()},
        "max_reaction_size": args.max_reaction_size,
        "metabolite_nodes": graph.n - n_rxn,
        "reaction_nodes": n_rxn,
        "edges": len(graph.edges),
        # same fingerprint as each run's summary.json network_edges_sha256
        "edges_sha256": hashlib.sha256("\n".join(sorted(
            f"{graph.node_ids[m]}\t{graph.node_ids[r]}\t{s}" for (m, r), s in zip(graph.edges, graph.edge_side)
        )).encode()).hexdigest(),
        "cofactor_edges_kept": int((audit.decision == "kept").sum()),
        "cofactor_edges_removed_as_exchange": int((audit.decision == "exchange").sum()),
    }
    (out / "network_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Exported {info['metabolite_nodes']} metabolites, {n_rxn} reactions, {len(graph.edges)} edges to {out}/")


if __name__ == "__main__":
    main()
