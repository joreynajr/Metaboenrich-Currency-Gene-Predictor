"""Scoring shared by the command line and the benchmark (protocol §1.5):
node table from a current-flow result, reaction scores, and gene ranking."""
import numpy as np
import pandas as pd


def rank_score(df, cols):
    """Final score = mean percentile rank of the given columns (placeholder for
    the protocol's unspecified combination of conductivity and bottleneck)."""
    if df.empty:
        return pd.Series(dtype=float)
    return sum(df[c].rank(pct=True) for c in cols) / len(cols)


def node_table(graph, flow):
    return pd.DataFrame({
        "node_id": graph.node_ids,
        "name": graph.node_names,
        "is_reaction": graph.is_reaction,
        "accumulated_current": flow.accumulated,
        "conductivity": flow.conductivity,
        "bottleneck": flow.bottleneck,
        "bottleneck_frac": flow.bottleneck_frac,
        "conductivity_loo": flow.conductivity_loo,
        "bottleneck_loo": flow.bottleneck_loo,
    })


def reaction_table(nodes, graph, model, endpoints, score_cols):
    """Reactions carrying current, scored and sorted best first (index = node index)."""
    rxn = nodes[nodes.is_reaction & (nodes.conductivity > 0)].copy()
    rxn["genes"] = [";".join(model.genes.get(g) or g for g in graph.reaction_genes[i]) for i in rxn.index]
    # A source/target in a single reaction sends all its current through it,
    # so that reaction's bottleneck = 1 by construction; name such endpoints.
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n)
    forced = {}
    for v in endpoints:
        if degree[v] == 1:
            r = graph.edges[graph.edges[:, 0] == v, 1][0]
            forced.setdefault(r, []).append(graph.node_names[v])
    rxn["only_reaction_of"] = [";".join(forced.get(i, [])) for i in rxn.index]
    rxn["final_score"] = rank_score(rxn, score_cols)
    return rxn.sort_values(["final_score", "conductivity"], ascending=False)


def gene_table(rxn, graph, model):
    """Genes inherit the best score among the reactions they catalyse (GPR
    and/or logic is ignored in this prototype). Sorted best first."""
    gene_rows = {}
    for i, r in rxn.iterrows():
        for g in graph.reaction_genes[i]:
            row = gene_rows.setdefault(g, {"ensembl_id": g, "symbol": model.genes.get(g, ""),
                                           "best_score": -1.0, "best_reaction": "", "n_reactions": 0,
                                           "total_accumulated_current": 0.0})
            row["n_reactions"] += 1
            row["total_accumulated_current"] += r.accumulated_current
            if r.final_score > row["best_score"]:
                row["best_score"], row["best_reaction"] = r.final_score, f"{r.node_id} {r['name']}"
    genes = pd.DataFrame(gene_rows.values())
    if len(genes):
        genes = genes.sort_values(["best_score", "total_accumulated_current"], ascending=False)
    return genes
