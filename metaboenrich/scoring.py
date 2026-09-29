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


def pathway_table(rxn, graph, subsystems, top_n=100):
    """Human-GEM subsystems over-represented among the top_n reactions.

    Background: every reaction in the network. Hypergeometric test,
    Benjamini-Hochberg across subsystems. `rxn` is the scored reaction table
    (best first); reactions without current count as not in the top.
    """
    from scipy.stats import hypergeom
    net = [graph.node_ids[i] for i in np.flatnonzero(graph.is_reaction)]
    sub_of = {r: (subsystems.get(r) or ["(none)"])[0] for r in net}
    top = [r for r in rxn.node_id.head(top_n)]
    N, n = len(net), len(top)
    size = pd.Series(sub_of).value_counts()
    hits = pd.Series([sub_of[r] for r in top]).value_counts() if top else pd.Series(dtype=int)
    rows = []
    for sub, k in hits.items():
        K = int(size[sub])
        p = hypergeom.sf(k - 1, N, K, n)
        genes = []
        for r in top:
            if sub_of[r] == sub:
                i = graph.index[r]
                genes += [g for g in rxn.loc[i, "genes"].split(";") if g and g not in genes]
        rows.append({"subsystem": sub, "reactions_in_network": K, "in_top": int(k),
                     "expected": round(K * n / N, 2), "fold": round(k / (K * n / N), 2), "p": p,
                     "top_genes": ";".join(genes[:12])})
    df = pd.DataFrame(rows)
    if len(df):
        p = df.p.values
        order = np.argsort(p)
        q = np.empty(len(p))
        q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
        df["q"] = np.minimum(q, 1.0)
        df = df.sort_values(["p", "fold"], ascending=[True, False])
    return df


def transporter_table(transport, met_scores, model):
    """Transporter genes scored by the metabolites they carry.

    met_scores: {met_id: (score in [0, 1], basis)} where a measured, changed
    metabolite scores by the percentile of |log2FC| among changed metabolites,
    and an unmeasured intermediate by its current-flow final score. A
    transporter takes its best carried metabolite. Kept separate from the
    enzyme ranking: every changed metabolite has transporters, so merging
    them would crowd enzymes out of the top ranks.
    """
    rows = {}
    for met, genes in transport.items():
        if met not in met_scores:
            continue
        score, basis = met_scores[met]
        for g in genes:
            r = rows.setdefault(g, {"ensembl_id": g, "symbol": model.genes.get(g, ""), "best_score": -1.0,
                                    "best_metabolite": "", "basis": "", "n_metabolites": 0})
            r["n_metabolites"] += 1
            if score > r["best_score"]:
                r["best_score"], r["best_metabolite"], r["basis"] = score, met, basis
    df = pd.DataFrame(rows.values())
    if len(df):
        df = df.sort_values(["best_score", "n_metabolites"], ascending=[False, True])
    return df


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
