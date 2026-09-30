"""Pruning-stage network diagrams on one shared layout, for Cytoscape.

Each stage of pruning (see structure.py) gets a Cytoscape.js file with the
same node positions, so flipping between stages shows what each step removes,
cuts off, or turns into a bottleneck. Only the nodes the structural check is
about are drawn, taken across all stages:

* pocket nodes: nodes that alone cut off >= POCKET_MIN metabolites, their
  pocket members, and their strongest neighbours outside the pocket
* hubs: the nodes carrying the most structural current
* islands: pieces of >= ISLAND_MIN metabolites cut off from the main network

Layout regions: nodes removed by pruning on the left (grouped by the stage
that removed them), the main network in the centre, islands on the right.

    python -m metaboenrich.stage_diagrams --structure results/structure/structure.json \
        --out results/stage_diagrams

Needs results/structure/structure.json from `python -m metaboenrich.structure`.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .gem import load_sbml
from .network import build_graph
from .structure import ALTERNATIVE_STAGES, LAYOUT_STAGE, stages

POCKET_MIN = 5
ISLAND_MIN = 4
HUBS_PER_STAGE = 12
NEIGHBOURS_PER_POCKET = 4
LABEL_POCKET_MIN = 10      # smaller pockets are drawn but only named on hover / in the table

# Same encodings as the Pruning Atlas (light theme values).
BIN_COLORS = ["#aab2bd", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
BIN_LABELS = ["0", "1", "2–4", "5–19", "20+"]
INK, MUTED, GHOST, EDGE, EDGE_HUB, ISLAND_BORDER = "#12161b", "#737c88", "#dfe4ea", "#b9c1cb", "#dfe4ea", "#737c88"


def pocket_bin(p):
    return 0 if p == 0 else 1 if p == 1 else 2 if p <= 4 else 3 if p <= 19 else 4


def adjacency(graph):
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    return sp.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(graph.n, graph.n))


def pocket_members(adj, labels, v):
    """Nodes cut off from the largest remaining piece of v's component when v is removed."""
    comp = np.flatnonzero(labels == labels[v])
    rest = comp[comp != v]
    if not len(rest):
        return []
    sub = adj[rest][:, rest]
    _, lab = connected_components(sub, directed=False)
    sizes = np.bincount(lab)
    return rest[lab != sizes.argmax()].tolist()


def fruchterman_reingold(n, edges, iters=350, seed=0, width=1.0):
    """Plain force-directed layout on n nodes (edges: list of (i, j))."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(-width / 2, width / 2, (n, 2))
    if n == 1:
        return np.zeros((1, 2))
    k = 1.6 * width / math.sqrt(n)
    e = np.array(edges, dtype=int).reshape(-1, 2)
    t = width / 10
    for _ in range(iters):
        delta = pos[:, None, :] - pos[None, :, :]
        dist = np.linalg.norm(delta, axis=2) + 1e-9
        disp = (delta / dist[..., None] * (k * k / dist)[..., None]).sum(axis=1)
        if len(e):
            d = pos[e[:, 0]] - pos[e[:, 1]]
            dl = np.linalg.norm(d, axis=1) + 1e-9
            f = (d / dl[:, None]) * (dl * dl / k)[:, None]
            np.add.at(disp, e[:, 0], -f)
            np.add.at(disp, e[:, 1], f)
        disp -= pos * 0.02 * n / width          # gravity keeps pieces together
        length = np.linalg.norm(disp, axis=1) + 1e-9
        pos += disp / length[:, None] * np.minimum(length, t)[:, None]
        t *= 0.985
    pos -= np.median(pos, axis=0)
    # Fit to `width`: scale so 95% of nodes fall inside the circle, then pull
    # the few stragglers (nodes with one weak tie) back to its rim.
    r = np.linalg.norm(pos, axis=1)
    scale = (width / 2) / (np.percentile(r, 95) or 1.0)
    pos *= scale
    r = np.linalg.norm(pos, axis=1)
    far = r > width / 2
    pos[far] *= ((width / 2) * (1 + 0.08 * np.log1p((r[far] - width / 2) / width)) / r[far])[:, None]
    return pos


def main(argv=None):
    p = argparse.ArgumentParser(prog="metaboenrich.stage_diagrams", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="data/Human-GEM.xml")
    p.add_argument("--structure", default="results/structure/structure.json")
    p.add_argument("--out", default="results/stage_diagrams")
    args = p.parse_args(argv)

    data = json.loads(Path(args.structure).read_text(encoding="utf-8"))
    meta = data["nodes"]                       # [id, name, type, genes]
    model = load_sbml(args.model)
    stage_set = data.get("stage_set", "pruning")
    stage_defs = stages(model, stage_set)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ---- per stage: graphs, pockets, islands, hubs -------------------------
    per = []
    for st, (key, title, desc, rules, max_size, drop_obj, atom_pairs) in zip(data["stages"], stage_defs):
        g = build_graph(model, rules, max_size, drop_obj, atom_pairs=atom_pairs)
        assert [meta[k][0] for k in st["node"]] == g.node_ids, f"stage {key} does not match structure.json"
        adj = adjacency(g)
        labels = g.components()
        main_lab = np.bincount(labels).argmax()
        pocket = np.array(st["pocket"])
        flow = np.array(st["flow"])
        focus, role, pocket_of = set(), {}, {}

        cut_nodes = [v for v in np.flatnonzero(pocket >= POCKET_MIN)]
        for v in sorted(cut_nodes, key=lambda v: -pocket[v]):
            members = pocket_members(adj, labels, v)
            focus.add(v)
            role[v] = "cuts off a pocket"
            for m in members:
                focus.add(m)
                role.setdefault(m, "in a pocket")
                pocket_of.setdefault(m, g.node_names[v])
            outside = [w for w in adj[v].indices if w not in set(members)]
            for w in sorted(outside, key=lambda w: -flow[w])[:NEIGHBOURS_PER_POCKET]:
                focus.add(w)
                role.setdefault(w, "attachment")

        for comp in np.unique(labels):
            if comp == main_lab:
                continue
            nodes = np.flatnonzero(labels == comp)
            if (~g.is_reaction[nodes]).sum() >= ISLAND_MIN:
                for v in nodes:
                    focus.add(v)
                    role.setdefault(v, "island")

        for v in np.argsort(-flow)[:HUBS_PER_STAGE]:
            focus.add(v)
            role[v] = role.get(v) if role.get(v) == "cuts off a pocket" else "hub"

        channel_bases = {b for i, b in g.base_reaction.items() if g.node_ids[i] != b}
        per.append({"key": key, "title": title, "description": desc, "graph": g, "labels": labels,
                    "main": main_lab, "pocket": pocket, "flow": flow, "focus": focus, "role": role,
                    "pocket_of": pocket_of, "degree": np.array(st["degree"]), "channel_bases": channel_bases})

    # ---- union of focus nodes, by Human-GEM id ------------------------------
    union = []
    for s in per:
        for v in s["focus"]:
            nid = s["graph"].node_ids[v]
            if nid not in union:
                union.append(nid)
    uidx = {nid: i for i, nid in enumerate(union)}
    final = next(s for s in per if s["key"] == LAYOUT_STAGE[stage_set])
    stage_order = [s["key"] for s in per if s["key"] not in ALTERNATIVE_STAGES]

    # A reaction split into carbon channels ("R_MAR03827#1", "#2") keeps its
    # place on the map: layout is by reaction identity, channels are offset.
    key_of = lambda nid: nid.split("#", 1)[0]

    def status(s, nid):
        """present | hidden (shown instead as its whole reaction / its channels) | removed"""
        if nid in s["graph"].index:
            return "present"
        base = key_of(nid)
        if nid != base and base in s["graph"].index:
            return "hidden"
        if nid == base and base in s["channel_bases"]:
            return "hidden"
        return "removed"

    def present(s, nid):
        return status(s, nid) != "removed"

    def removed_at(nid):
        """First main stage in which nid is absent (None if present at the end)."""
        for key in stage_order:
            s = next(x for x in per if x["key"] == key)
            if not present(s, nid):
                return key
        return None

    keys = list(dict.fromkeys(key_of(nid) for nid in union))
    members = {}
    for nid in union:
        members.setdefault(key_of(nid), []).append(nid)
    g5 = final["graph"]
    # the final-stage node(s) standing for each key
    fin_nodes = {}
    for i, nid in enumerate(g5.node_ids):
        if key_of(nid) in members:
            fin_nodes.setdefault(key_of(nid), []).append(i)

    # ---- layout ---------------------------------------------------------------
    kpos = {}
    in5 = [k for k in keys if k in fin_nodes]
    main5 = [k for k in in5 if any(final["labels"][i] == final["main"] for i in fin_nodes[k])]
    island_keys = [k for k in in5 if k not in set(main5)]
    shelf = [k for k in keys if k not in fin_nodes]

    # Main network: force-directed on the final network's edges among drawn nodes.
    local = {k: i for i, k in enumerate(main5)}
    kid = lambda i: key_of(g5.node_ids[i])
    e5 = sorted({(local[kid(a)], local[kid(b)]) for a, b in g5.edges
                 if kid(a) in local and kid(b) in local and kid(a) != kid(b)})
    # Nodes drawn but not linked to anything else drawn would drift; tie them to
    # a neighbour in the final network.
    linked = {i for ab in e5 for i in ab}
    adj5 = adjacency(g5)
    for k, i in local.items():
        if i not in linked:
            nb = [kid(w) for v in fin_nodes[k] for w in adj5[v].indices if kid(w) in local and kid(w) != k]
            if nb:
                e5.append((i, local[nb[0]]))
    W_MAIN = 2400.0
    if main5:
        for k, xy in zip(main5, fruchterman_reingold(len(main5), e5, width=W_MAIN)):
            kpos[k] = xy

    # Islands: each laid out on its own, packed in rows to the right.
    islands = {}
    for k in island_keys:
        islands.setdefault(final["labels"][fin_nodes[k][0]], []).append(k)
    groups = sorted(islands.values(), key=len, reverse=True)
    x0, row_w = W_MAIN / 2 + 300, 1300.0
    cx, cy, row_h = x0, -W_MAIN / 2 + 60, 0.0
    for ks in groups:
        loc = {k: i for i, k in enumerate(ks)}
        ee = sorted({(loc[kid(a)], loc[kid(b)]) for a, b in g5.edges
                     if kid(a) in loc and kid(b) in loc and kid(a) != kid(b)})
        size = 60.0 + 34.0 * math.sqrt(len(ks))          # box edge for this island
        sub = fruchterman_reingold(len(ks), ee, iters=200, width=size * 0.7)
        if cx + size > x0 + row_w:                            # wrap to the next row
            cx, cy, row_h = x0, cy + row_h + 40, 0.0
        for k, xy in zip(ks, sub + [cx + size / 2, cy + size / 2]):
            kpos[k] = xy
        cx += size + 40
        row_h = max(row_h, size)
    island_center_x = x0 + row_w / 2

    # Shelf: removed nodes in columns by the stage that removed them.
    by_stage = {}
    for k in shelf:
        rep = k if k in uidx else members[k][0]
        by_stage.setdefault(removed_at(rep) or next(iter(ALTERNATIVE_STAGES)), []).append(k)
    shelf_x = -W_MAIN / 2 - 300
    col = 0
    n_groups = 0
    captions = [("cap_main", "Main network", 0.0, -W_MAIN / 2 - 170),
                ("cap_islands", "Islands (cut off from the main network)", island_center_x, -W_MAIN / 2 - 170)]
    for key in stage_order:
        nodes = by_stage.get(key, [])
        if not nodes:
            continue
        s = next(x for x in per if x["key"] == key)
        prev = per[[x["key"] for x in per].index(key) - 1]
        pflow = lambda k: -prev["flow"][prev["graph"].index[k]] if k in prev["graph"].index else 0
        nodes.sort(key=pflow)
        per_col = 26
        ncols = math.ceil(len(nodes) / per_col)
        for i, k in enumerate(nodes):
            c, r = divmod(i, per_col)
            kpos[k] = np.array([shelf_x - (col + c) * 90, -W_MAIN / 2 + 40 + r * 80])
        # captions alternate between two heights so narrow neighbouring groups never overlap
        captions.append((f"cap_removed_{key}", f"Removed at {s['title'].split('.')[0]}",
                         shelf_x - (col + (ncols - 1) / 2) * 90, -W_MAIN / 2 - 60 - 55 * (n_groups % 2)))
        n_groups += 1
        col += ncols + 2                      # room for the next group's caption
    captions.append(("cap_removed", "Removed by pruning", shelf_x - (col - 3) * 45, -W_MAIN / 2 - 170))

    # Node positions: the reaction's place, channels spread slightly around it.
    pos = np.zeros((len(union), 2))
    for k, nids in members.items():
        xy = np.asarray(kpos.get(k, np.zeros(2)), dtype=float)
        chans = sorted(n for n in nids if n != k)
        for n in nids:
            dx = (chans.index(n) - (len(chans) - 1) / 2) * 16 if n != k and len(chans) > 1 else 0.0
            pos[uidx[n]] = xy + [dx, 0.0 if n == k else 10.0]

    # ---- write one Cytoscape.js file per stage -------------------------------
    names = {nid: None for nid in union}
    type_of, genes_of = {}, {}
    for s in per:
        g = s["graph"]
        for nid in union:
            if nid in g.index and names[nid] is None:
                i = g.index[nid]
                names[nid] = g.node_names[i]
                type_of[nid] = "reaction" if g.is_reaction[i] else "metabolite"
                genes_of[nid] = ";".join(model.genes.get(x) or x for x in g.reaction_genes.get(i, [])) \
                    if g.is_reaction[i] else ""
    max_flow = max(float(s["flow"].max()) for s in per) or 1.0

    viewer = {"nodes": [[nid, names[nid], "r" if type_of[nid] == "reaction" else "m", genes_of[nid],
                         round(float(pos[uidx[nid], 0]), 1), round(float(pos[uidx[nid], 1]), 1)] for nid in union],
              "captions": [[c[1], round(c[2], 1), round(c[3], 1)] for c in captions],
              "stages": []}
    for si, s in enumerate(per):
        g = s["graph"]
        nodes_json, edges_json, vstage = [], [], {"key": s["key"], "title": s["title"], "nodes": [], "edges": []}
        for nid in union:
            i = g.index.get(nid)
            is_r = type_of[nid] == "reaction"
            x, y = pos[uidx[nid]]
            if i is None and status(s, nid) == "hidden":
                vstage["nodes"].append([uidx[nid], 0, 0, 0, 0, "hidden"])
                continue
            if i is None:
                d = {"id": nid, "name": names[nid], "type": type_of[nid], "status": "removed",
                     "removed_at": (next((x["title"] for x in per if x["key"] == removed_at(nid)), "") if removed_at(nid) else ""),
                     "fill": GHOST, "size": 8.0 if is_r else 14.0, "shape": "ROUND_RECTANGLE" if is_r else "ELLIPSE",
                     "border_width": 0.0, "border_color": GHOST, "border_style": "SOLID", "opacity": 110,
                     "label_text": "", "genes": genes_of[nid]}
                vstage["nodes"].append([uidx[nid], 0, 0, 0, 0, "removed"])
            else:
                pk, fl = int(s["pocket"][i]), float(s["flow"][i])
                role = s["role"].get(i, "context")
                in_main = s["labels"][i] == s["main"]
                if not in_main and role not in ("cuts off a pocket", "hub"):
                    role = "island"
                size = (7.0 if is_r else 12.0) + (26.0 if is_r else 46.0) * math.sqrt(fl / max_flow)
                border_w, border_c, border_s = 0.0, INK, "SOLID"
                if role == "cuts off a pocket":
                    border_w, border_c = 3.5, INK
                elif role == "in a pocket":
                    border_w, border_c, border_s = 1.6, BIN_COLORS[3], "EQUAL_DASH"
                elif role == "island":
                    border_w, border_c, border_s = 1.6, ISLAND_BORDER, "DOT"
                label = ""
                short = genes_of[nid].split(";")[0] if is_r and genes_of[nid] else names[nid]
                if role == "cuts off a pocket" and pk >= LABEL_POCKET_MIN:
                    label = f"{short[:28]} · cuts off {pk}"
                elif role == "hub":
                    label = short[:28]
                d = {"id": nid, "name": names[nid], "type": type_of[nid], "status": "present",
                     "role": role, "pocket": pk, "pocket_bin": BIN_LABELS[pocket_bin(pk)],
                     "pocket_of": s["pocket_of"].get(i, ""), "structural_current": fl,
                     "degree": int(s["degree"][i]), "in_main_network": bool(in_main),
                     "fill": BIN_COLORS[pocket_bin(pk)], "size": round(size, 2),
                     "shape": "ROUND_RECTANGLE" if is_r else "ELLIPSE",
                     "border_width": border_w, "border_color": border_c, "border_style": border_s,
                     "opacity": 255, "label_text": label, "genes": genes_of[nid]}
                vstage["nodes"].append([uidx[nid], pk, round(fl, 6), int(s["degree"][i]),
                                        int(bool(in_main)), role])
            nodes_json.append({"data": d, "position": {"x": round(float(x), 1), "y": round(float(y), 1)}})
        hubs = {g.index[nid] for nid in union if nid in g.index and s["role"].get(g.index[nid]) == "hub"}
        for a, b in g.edges:
            na, nb = g.node_ids[a], g.node_ids[b]
            if na in uidx and nb in uidx:
                hub_edge = a in hubs or b in hubs
                edges_json.append({"data": {"id": f"{na}__{nb}", "source": na, "target": nb,
                                            "color": EDGE_HUB if hub_edge else EDGE, "opacity": 150 if hub_edge else 230}})
                vstage["edges"] += [uidx[na], uidx[nb]]
        for cid, text, cx, cy in captions:
            nodes_json.append({"data": {"id": cid, "name": text, "type": "caption", "status": "caption",
                                        "fill": "#ffffff", "size": 1.0, "shape": "RECTANGLE", "border_width": 0.0,
                                        "border_color": "#ffffff", "border_style": "SOLID", "opacity": 0,
                                        "label_text": text, "caption": True},
                               "position": {"x": round(cx, 1), "y": round(cy, 1)}})
        doc = {"format_version": "1.0", "generated_by": "metaboenrich.stage_diagrams",
               "data": {"name": s["title"], "description": s["description"]},
               "elements": {"nodes": nodes_json, "edges": edges_json}}
        if s["key"] == "alt_remove":
            fname = "alt_cofactors_removed_everywhere.cyjs"
        elif s["key"] in ALTERNATIVE_STAGES:
            fname = f"alt_{s['key']}.cyjs"
        else:
            fname = f"{stage_order.index(s['key']) + 1}_{s['key']}.cyjs"
        (out / fname).write_text(json.dumps(doc), encoding="utf-8")
        viewer["stages"].append(vstage)
        print(f"{fname:42s} {sum(1 for n in nodes_json if n['data']['status'] == 'present'):4d} present, "
              f"{sum(1 for n in nodes_json if n['data']['status'] == 'removed'):4d} removed, {len(edges_json):5d} edges")

    (out / "diagrams.json").write_text(json.dumps(viewer, separators=(",", ":")), encoding="utf-8")
    (out / "metaboenrich_pruning_style.xml").write_text(STYLE_XML, encoding="utf-8")
    print(f"Wrote {len(per)} stage diagrams ({len(union)} nodes on one layout) and the style to {out}/")


def _passthrough(prop, column, col_type, default):
    return (f'      <visualProperty name="{prop}" default="{default}">\n'
            f'        <passthroughMapping attributeName="{column}" attributeType="{col_type}"/>\n'
            f'      </visualProperty>\n')


STYLE_XML = (
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
    '<vizmap documentVersion="3.1" id="metaboenrich-pruning">\n'
    '  <visualStyle name="Metaboenrich pruning">\n'
    '    <network>\n'
    '      <visualProperty name="NETWORK_BACKGROUND_PAINT" default="#FBFCFD"/>\n'
    '    </network>\n'
    '    <node>\n'
    '      <dependency name="nodeSizeLocked" value="true"/>\n'
    '      <dependency name="nodeCustomGraphicsSizeSync" value="true"/>\n'
    + _passthrough("NODE_FILL_COLOR", "fill", "string", "#AAB2BD")
    + _passthrough("NODE_SIZE", "size", "double", "12.0")
    + _passthrough("NODE_SHAPE", "shape", "string", "ELLIPSE")
    + _passthrough("NODE_BORDER_WIDTH", "border_width", "double", "0.0")
    + _passthrough("NODE_BORDER_PAINT", "border_color", "string", "#12161B")
    + _passthrough("NODE_BORDER_STROKE", "border_style", "string", "SOLID")
    + _passthrough("NODE_TRANSPARENCY", "opacity", "integer", "255")
    + _passthrough("NODE_LABEL", "label_text", "string", "")
    + '      <visualProperty name="NODE_LABEL_COLOR" default="#12161B"/>\n'
    '      <visualProperty name="NODE_LABEL_FONT_SIZE" default="11"/>\n'
    '      <visualProperty name="NODE_LABEL_POSITION" default="N,S,c,0.00,4.00"/>\n'
    '      <visualProperty name="NODE_BORDER_TRANSPARENCY" default="255"/>\n'
    '    </node>\n'
    '    <edge>\n'
    '      <dependency name="arrowColorMatchesEdge" value="false"/>\n'
    + _passthrough("EDGE_STROKE_UNSELECTED_PAINT", "color", "string", "#B9C1CB")
    + _passthrough("EDGE_TRANSPARENCY", "opacity", "integer", "230")
    + '      <visualProperty name="EDGE_WIDTH" default="1.0"/>\n'
    '    </edge>\n'
    '  </visualStyle>\n'
    '</vizmap>\n'
)


if __name__ == "__main__":
    main()
