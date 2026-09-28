"""Write the metabolite-reaction network as GraphML for Cytoscape
(File > Import > Network from File). Node and edge attributes become columns
in Cytoscape's tables and can be mapped to colour, size and width in the
Style panel."""
import math
from xml.sax.saxutils import escape

import numpy as np

_TYPES = {bool: "boolean", int: "int", float: "double", str: "string"}


def _attr_type(values):
    for v in values:
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if isinstance(v, (bool, np.bool_)):
            return "boolean"
        if isinstance(v, (int, np.integer)):
            return "int"
        if isinstance(v, (float, np.floating)):
            return "double"
        return "string"
    return "string"


def _fmt(v, typ):
    if typ == "boolean":
        return "true" if v else "false"
    if typ == "double":
        return repr(float(v))
    return escape(str(v))


def _missing(v):
    return v is None or (isinstance(v, (float, np.floating)) and math.isnan(v)) or v == ""


def write_graphml(path, graph, node_attrs, edge_attrs, node_mask=None, edge_mask=None):
    """node_attrs / edge_attrs: {column name: sequence aligned with graph nodes / edges}.

    Edges point substrate -> reaction -> product. Masks select a subnetwork;
    edges are kept only if both ends are kept.
    """
    n_mask = np.ones(graph.n, bool) if node_mask is None else np.asarray(node_mask, bool)
    e_mask = np.ones(len(graph.edges), bool) if edge_mask is None else np.asarray(edge_mask, bool)
    e_mask &= n_mask[graph.edges[:, 0]] & n_mask[graph.edges[:, 1]]

    n_keys = {k: _attr_type([v[i] for i in np.flatnonzero(n_mask)]) for k, v in node_attrs.items()}
    e_keys = {k: _attr_type([v[i] for i in np.flatnonzero(e_mask)]) for k, v in edge_attrs.items()}

    with open(path, "w", encoding="utf-8") as fh:
        fh.write('<?xml version="1.0" encoding="UTF-8"?>\n'
                 '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')
        for i, (k, t) in enumerate(n_keys.items()):
            fh.write(f'  <key id="n{i}" for="node" attr.name="{escape(k)}" attr.type="{t}"/>\n')
        for i, (k, t) in enumerate(e_keys.items()):
            fh.write(f'  <key id="e{i}" for="edge" attr.name="{escape(k)}" attr.type="{t}"/>\n')
        fh.write('  <graph id="metaboenrich" edgedefault="directed">\n')
        for node in np.flatnonzero(n_mask):
            fh.write(f'    <node id="{escape(graph.node_ids[node])}">')
            for i, (k, t) in enumerate(n_keys.items()):
                v = node_attrs[k][node]
                if not _missing(v):
                    fh.write(f'<data key="n{i}">{_fmt(v, t)}</data>')
            fh.write("</node>\n")
        for e in np.flatnonzero(e_mask):
            met, rxn = graph.edges[e]
            src, dst = (met, rxn) if graph.edge_side[e] == "substrate" else (rxn, met)
            fh.write(f'    <edge source="{escape(graph.node_ids[src])}" target="{escape(graph.node_ids[dst])}">')
            for i, (k, t) in enumerate(e_keys.items()):
                v = edge_attrs[k][e]
                if not _missing(v):
                    fh.write(f'<data key="e{i}">{_fmt(v, t)}</data>')
            fh.write("</edge>\n")
        fh.write("  </graph>\n</graphml>\n")
    return int(n_mask.sum()), int(e_mask.sum())
