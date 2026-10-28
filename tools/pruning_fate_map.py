"""The whole unpruned network, each node and link coloured by the step that prunes it.

    python tools/pruning_fate_map.py --out results/three_rule_diagrams

Raw Human-GEM (compartments merged; every metabolite, reaction and
metabolite-reaction link) as one Cytoscape.js file. The network is rebuilt
after each step of the method, cumulatively and in order, and every element
gets the step after which it never comes back:

    Step 1   pools, biomass, transporters      Rule 3   carriers
    Rule 1   carbon                            Rule 4   carbon budget
    Rule 2   couples                           size     over 20 metabolites

Elements in the final network are 'kept', 'kept by safeguard' (gone after
Rules 1-3, restored by the keep-rarest safeguard) or, for reactions, 'merged
into a compartment copy'. A link is a metabolite-reaction pair: present in a
network if the metabolite is attached to the reaction or any of its carbon
links. Final-network islands have a dotted border. raw_degree / final_degree
count links (metabolite-reaction pairs, compartment copies counted apart), so
a hub that is kept but lost most links shows a low links_kept_pct.

Writes its own style, fate_map_style.xml ("Metaboenrich fate map": small
nodes, hairline edges, large hub labels). Positions are stored (see layout);
do not run a layout in Cytoscape. Also writes pruning_fate_nodes.tsv and
pruning_fate_summary.tsv.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metaboenrich import structure                                   # noqa: E402
from metaboenrich.atoms import (COUPLE_MIN_REACTIONS, carbon_budget_pairs, carbon_method_pairs,  # noqa: E402
                                carbon_method_rules, carbon_rule1_pairs, carbon_unbalanced, find_carriers,
                                kegg_pairs_for_model, load_atom_pairs, pool_metabolites, transport_reactions)
from metaboenrich.gem import load_sbml                               # noqa: E402
from metaboenrich.network import build_graph                         # noqa: E402

# (key, label, colour): Okabe-Ito, distinguishable with colour-vision deficiency
FATES = [
    ("step1", "Step 1 (pools, biomass, transporters)", "#E69F00"),
    ("rule1", "Rule 1 (carbon)", "#56B4E9"),
    ("rule2", "Rule 2 (couples)", "#D55E00"),
    ("rule3", "Rule 3 (carriers)", "#009E73"),
    ("rule4", "Rule 4 (carbon budget)", "#CC79A7"),
    ("size", "Size limit (over 20 metabolites)", "#F0E442"),
    ("safeguard", "Kept by safeguard", "#0072B2"),
    ("merged", "Kept, merged into a compartment copy", "#9AA3AE"),
    ("kept", "Kept", "#12161B"),
]
LABEL = {k: lab for k, lab, _ in FATES}
COLOUR = {k: c for k, _, c in FATES}
HUB_LABEL_DEGREE = 150      # metabolites with at least this many raw links are labelled


def _passthrough(prop, column, col_type, default):
    return (f'      <visualProperty name="{prop}" default="{default}">\n'
            f'        <passthroughMapping attributeName="{column}" attributeType="{col_type}"/>\n'
            f'      </visualProperty>\n')


# The pruning style (metaboenrich.stage_diagrams) plus per-node label size and
# per-edge width: 11,000 nodes need small marks and hairline edges to be read.
STYLE_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<vizmap documentVersion="3.1" id="metaboenrich-fate-map">\n'
    '  <visualStyle name="Metaboenrich fate map">\n'
    '    <network>\n'
    '      <visualProperty name="NETWORK_BACKGROUND_PAINT" default="#FBFCFD"/>\n'
    '    </network>\n'
    '    <node>\n'
    '      <dependency name="nodeSizeLocked" value="true"/>\n'
    + _passthrough("NODE_FILL_COLOR", "fill", "string", "#AAB2BD")
    + _passthrough("NODE_SIZE", "size", "double", "4.0")
    + _passthrough("NODE_SHAPE", "shape", "string", "ELLIPSE")
    + _passthrough("NODE_BORDER_WIDTH", "border_width", "double", "0.0")
    + _passthrough("NODE_BORDER_PAINT", "border_color", "string", "#12161B")
    + _passthrough("NODE_BORDER_STROKE", "border_style", "string", "SOLID")
    + _passthrough("NODE_TRANSPARENCY", "opacity", "integer", "255")
    + _passthrough("NODE_LABEL", "label_text", "string", "")
    + _passthrough("NODE_LABEL_FONT_SIZE", "label_size", "integer", "12")
    + '      <visualProperty name="NODE_LABEL_COLOR" default="#12161B"/>\n'
    '      <visualProperty name="NODE_LABEL_FONT_FACE" default="SansSerif.bold,bold,12"/>\n'
    '      <visualProperty name="NODE_LABEL_POSITION" default="N,S,c,0.00,4.00"/>\n'
    '    </node>\n'
    '    <edge>\n'
    + _passthrough("EDGE_STROKE_UNSELECTED_PAINT", "color", "string", "#B9C1CB")
    + _passthrough("EDGE_TRANSPARENCY", "opacity", "integer", "100")
    + _passthrough("EDGE_WIDTH", "width", "double", "0.3")
    + '    </edge>\n'
    '  </visualStyle>\n'
    '</vizmap>\n'
)


def incidences(g):
    """{(met_id, Human-GEM reaction id)} of a graph, carbon links folded back into their reaction."""
    out = set()
    for m, r in g.edges.tolist():
        for rid in g.reactions_of(r):
            out.add((g.node_ids[m], rid))
    return out


def networks(model, data):
    """[(fate key, graph)] after each step, cumulative, in method order; then the final network."""
    union = load_atom_pairs(data / "atom_pairs_union.json")
    pr = structure._pruning_stages(model)
    raw = pr[0]
    pools = set(pr[4][3].pools) | pool_metabolites(model)
    cargo = transport_reactions(model, carbon_cargo=True)
    shared = transport_reactions(model, carbon_cargo=False)
    pool_rxns = {rid for rid, r in model.reactions.items()
                 if {model.species[s].met_id for s in (*r.substrates, *r.products)} & pools}
    drop = {"drop_reactions": cargo}

    rule1 = carbon_rule1_pairs(model, union)
    method, couples = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS)
    carriers, carrier_pairs = find_carriers(model, union, data / "Human-GEM_metabolites.tsv")
    c_pairs, _ = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS, carriers=carrier_pairs)
    s_pairs, _ = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS, keep_rarest=True, no_rescue=shared,
                                     carriers=carrier_pairs)
    s_rules = carbon_method_rules(model, s_pairs, couples, pools, keep_rarest=True, no_rescue=shared,
                                  carriers=carriers)
    kegg = kegg_pairs_for_model(model, data / "Human-GEM.xml", data / "kegg_rclass_pairs.json")
    r4_pairs = carbon_budget_pairs(model, s_pairs, {rid: {tuple(p) for p in ps} for rid, ps in kegg.items()},
                                   carbon_unbalanced(model))
    final = next(s for s in structure.stages(model, "three_rule", atom_dir=str(data)) if s[0] == "three_rule")

    # the size limit comes last, so the intermediate networks have none
    return [
        ("raw", build_graph(model, raw[3], None, raw[5])),
        ("step1", build_graph(model, raw[3], None, raw[5], drop_reactions=cargo | pool_rxns)),
        ("rule1", build_graph(model, carbon_method_rules(model, rule1, {}, pools), None, True,
                              atom_pairs=rule1, **drop)),
        ("rule2", build_graph(model, carbon_method_rules(model, method, couples, pools), None, True,
                              atom_pairs=method, **drop)),
        ("rule3", build_graph(model, carbon_method_rules(model, c_pairs, couples, pools, carriers=carriers), None,
                              True, atom_pairs=c_pairs, **drop)),
        ("safeguard", build_graph(model, s_rules, None, True, atom_pairs=s_pairs, **drop)),
        ("rule4", build_graph(model, s_rules, None, True, atom_pairs=r4_pairs, **drop)),
        ("size", build_graph(model, final[3], final[4], final[5], **final[6])),
    ]


def layout(ids, links, degree, hub_degree=100, width=6000.0):
    """{node id: (x, y)}. Metabolites in more than `hub_degree` links (H+, H2O,
    ATP, CoA ...) would pull everything into one knot, so the rest is laid out
    without them: the largest piece force-directed in the centre, the pieces
    that fall away from it packed in rows to its right. Each hub then sits at
    the centre of its neighbours in the largest piece. Takes a few minutes for
    raw Human-GEM."""
    import networkx as nx
    hubs = {n for n in ids if degree[n] > hub_degree and not n.startswith("R_")}
    g = nx.Graph()
    g.add_nodes_from(n for n in ids if n not in hubs)
    g.add_edges_from((a, b) for a, b in links if a not in hubs and b not in hubs)
    pieces = sorted(nx.connected_components(g), key=len, reverse=True)
    pos = {n: (x * width / 2, y * width / 2)
           for n, (x, y) in nx.spring_layout(g.subgraph(pieces[0]), iterations=40, seed=0).items()}
    x0, y, row_h, x = width / 2 + 400, -width / 2, 0.0, width / 2 + 400
    for piece in pieces[1:]:
        size = 40.0 * math.sqrt(len(piece))
        sub = nx.spring_layout(g.subgraph(piece), seed=0) if len(piece) > 1 else {next(iter(piece)): (0.0, 0.0)}
        if x + size > x0 + width:
            x, y, row_h = x0, y + row_h + 60, 0.0
        for n, (px, py) in sub.items():
            pos[n] = (x + size / 2 + px * size / 2, y + size / 2 + py * size / 2)
        x, row_h = x + size + 60, max(row_h, size)
    main_piece, nbrs = pieces[0], {}
    for a, b in links:
        if a in hubs and b in main_piece:        # the packed pieces would pull hubs out of the centre
            nbrs.setdefault(a, []).append(pos[b])
    centre = np.mean([np.mean(v, axis=0) for v in nbrs.values()], axis=0) if nbrs else np.zeros(2)
    for k, h in enumerate(sorted(hubs, key=lambda n: -degree[n])):
        # hubs all sit near the middle; spread them on a sunflower spiral so their labels can be read
        r, a = 120 * math.sqrt(k + 1), k * 2.39996
        pos[h] = (float(centre[0]) + r * math.cos(a), float(centre[1]) + r * math.sin(a))
    return pos


def fate(present):
    """present: [bool] per network in networks() order, raw first. The step after which it never returns."""
    order = ["step1", "rule1", "rule2", "rule3", "safeguard", "rule4", "size"]
    if present[-1]:
        return "kept" if present[4] else "safeguard"       # present[4]: after Rule 3
    k = len(present) - 1
    while k > 1 and not present[k - 1]:
        k -= 1
    return order[k - 1]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "results/three_rule_diagrams"))
    args = ap.parse_args(argv)
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_sbml(str(data / "Human-GEM.xml"))
    nets = networks(model, data)
    raw, final = nets[0][1], nets[-1][1]
    inc = [incidences(g) for _, g in nets]
    mets = [{g.node_ids[i] for i in range(g.n) if not g.is_reaction[i]} for _, g in nets]
    rxns = [{r for i in range(g.n) if g.is_reaction[i] for r in g.reactions_of(i)} for _, g in nets]
    merged = {r: final.node_ids[k] for k, rs in final.merged_copies.items() for r in rs}
    labels = final.components()
    main_lab = np.bincount(labels).argmax()
    in_main = {final.node_ids[i] for i in range(final.n) if labels[i] == main_lab}
    raw_deg = Counter(m for m, _ in inc[0]) + Counter(r for _, r in inc[0])
    final_deg = Counter(m for m, _ in inc[-1]) + Counter(r for _, r in inc[-1])
    sym = model.genes

    nodes, rows = [], []
    for i in range(raw.n):
        nid = raw.node_ids[i]
        is_rxn = bool(raw.is_reaction[i])
        f = fate([nid in (s if is_rxn else m) for s, m in zip(rxns, mets)])
        if is_rxn and f == "kept" and nid in merged:
            f = "merged"
        island = f in ("kept", "safeguard") and not is_rxn and nid not in in_main
        deg = raw_deg[nid]
        genes = ";".join(sym.get(x, x) for x in raw.reaction_genes.get(i, [])) if is_rxn else ""
        d = {"id": nid, "name": raw.node_names[i], "type": "reaction" if is_rxn else "metabolite",
             "pruned_at": LABEL[f], "fate": f, "genes": genes, "raw_degree": deg,
             "final_degree": final_deg[nid], "links_kept_pct": round(100 * final_deg[nid] / deg, 1) if deg else 0.0,
             "merged_into": merged.get(nid, ""),
             "in_main_network": (nid in in_main) if not is_rxn else None,
             "fill": COLOUR[f], "shape": "ROUND_RECTANGLE" if is_rxn else "ELLIPSE",
             "size": 3.0 if is_rxn else round(4 + 2 * math.log2(1 + deg), 1),
             "label_size": 60 if deg >= HUB_LABEL_DEGREE else 12,
             "border_width": 1.0 if island else 0.0, "border_color": "#12161B",
             "border_style": "DOT" if island else "SOLID", "opacity": 255,
             "label_text": raw.node_names[i] if not is_rxn and deg >= HUB_LABEL_DEGREE else ""}
        nodes.append({"data": d})
        rows.append({k: d[k] for k in ("id", "name", "type", "pruned_at", "raw_degree", "final_degree",
                                       "links_kept_pct", "merged_into", "in_main_network", "genes")})

    edges, edge_fate = [], Counter()
    for (m, r), side in zip(raw.edges.tolist(), raw.edge_side):
        key = (raw.node_ids[m], raw.node_ids[r])
        f = fate([key in s for s in inc])
        if f == "kept" and key[1] in merged:
            f = "merged"
        edge_fate[f] += 1
        edges.append({"data": {"id": f"{key[0]}__{key[1]}", "source": key[0], "target": key[1], "side": side,
                               "pruned_at": LABEL[f], "fate": f, "color": COLOUR[f],
                               "opacity": 140 if f in ("kept", "safeguard", "merged") else 60,
                               "width": 0.4 if f in ("kept", "safeguard", "merged") else 0.25}})

    pos = layout(raw.node_ids, [(e["data"]["source"], e["data"]["target"]) for e in edges], raw_deg)
    for n in nodes:
        x, y = pos[n["data"]["id"]]
        n["position"] = {"x": round(x, 1), "y": round(y, 1)}

    doc = {"format_version": "1.0", "generated_by": "metaboenrich tools/pruning_fate_map.py",
           "data": {"name": "Raw Human-GEM, coloured by pruning step",
                    "legend": "; ".join(f"{lab} = {c}" for _, lab, c in FATES)},
           "elements": {"nodes": nodes, "edges": edges}}
    (out / "raw_pruning_fate.cyjs").write_text(json.dumps(doc), encoding="utf-8")
    (out / "fate_map_style.xml").write_text(STYLE_XML, encoding="utf-8")
    table = pd.DataFrame(rows)
    table.to_csv(out / "pruning_fate_nodes.tsv", sep="\t", index=False)

    summary = []
    for k, lab, _ in FATES:
        summary.append({"pruned_at": lab,
                        "metabolites": int(((table.type == "metabolite") & (table.pruned_at == lab)).sum()),
                        "reactions": int(((table.type == "reaction") & (table.pruned_at == lab)).sum()),
                        "links": edge_fate[k]})
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "pruning_fate_summary.tsv", sep="\t", index=False)
    print(summary.to_string(index=False))
    print(f"raw network: {len(nodes)} nodes, {len(edges)} links -> {out / 'raw_pruning_fate.cyjs'}")


if __name__ == "__main__":
    main()
