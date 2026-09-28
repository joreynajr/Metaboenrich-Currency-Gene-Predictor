"""Kirchhoff-Ohm current flow over source-target pairs (protocol §1.4, §2.2).

For each pair (s, t) the protocol solves L x = b with b_s = +1, b_t = -1.
Rather than forming the pseudoinverse of L (dense, n^3), each connected
component's Laplacian is grounded at one node and LU-factorised once. By
linearity x_st = y_s - y_t, where y_v solves the grounded system for a unit
injection at v, so only |S| + |T| solves are needed for all |S| x |T| pairs.
Grounding shifts every potential by a constant, which leaves edge currents
and effective resistances unchanged.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse.linalg import splu


@dataclass
class FlowResult:
    pairs: list                # retained (s, t) node-index pairs
    excluded: list             # (s, t, reason)
    # Per-node arrays. Throughput is the fraction of a pair's unit current
    # passing through a node; a node's own pairs (as s or t) are excluded.
    accumulated: np.ndarray    # sum over pairs of w * throughput  (reaction "voltage" update, §1.4)
    conductivity: np.ndarray   # weighted mean throughput
    bottleneck: np.ndarray     # max throughput over pairs (1 = the only route for some pair)
    bottleneck_frac: np.ndarray  # weighted fraction of pairs with throughput >= tau
    # Leave-one-out versions: the lowest value of the metric after dropping
    # all pairs of any single source or target. A reaction that scores only
    # because one metabolite has a single route out drops to ~0.
    conductivity_loo: np.ndarray
    bottleneck_loo: np.ndarray
    endpoint: dict             # node -> {"pairs", "eff_conductance", "max_edge_share"}
    edge_current: np.ndarray = None  # per edge: weighted mean |current| over pairs
    edge_max_current: np.ndarray = None  # per edge: max |current| over pairs


def _potentials(graph, laplacian, labels, nodes):
    """y_v for every v in `nodes`: potentials for a unit injection at v with
    the first node of v's component grounded. Returns (n x len(nodes))."""
    Y = np.zeros((graph.n, len(nodes)))
    col = {v: j for j, v in enumerate(nodes)}
    for comp in np.unique(labels[nodes]):
        members = np.flatnonzero(labels == comp)
        keep = members[1:]  # members[0] is grounded (potential 0)
        needed = [v for v in nodes if labels[v] == comp]
        if len(keep) == 0:
            continue
        lu = splu(laplacian[keep][:, keep].tocsc())
        pos = {v: i for i, v in enumerate(keep)}
        rhs = np.zeros((len(keep), len(needed)))
        for j, v in enumerate(needed):
            if v in pos:
                rhs[pos[v], j] = 1.0
        sol = lu.solve(rhs)
        for j, v in enumerate(needed):
            Y[keep, col[v]] = sol[:, j]
    return Y, col


def select_pairs(graph, sources, targets, directed):
    """All S x T pairs, minus those that cannot carry current: different
    connected components, or (directed mode, §2.2) no directed s -> t path."""
    labels = graph.components()
    pairs, excluded = [], []
    reach = {s: graph.reachable_from(s) for s in sources} if directed else {}
    for s in sources:
        for t in targets:
            if labels[s] != labels[t]:
                excluded.append((s, t, "disconnected"))
            elif directed and not reach[s][t]:
                excluded.append((s, t, "no directed path"))
            else:
                pairs.append((s, t))
    return pairs, excluded, labels


def _leave_one_out(acc, total_w, sums, w_by, maxes):
    """Leave-one-out conductivity and bottleneck over one side (sources or
    targets). sums/maxes are (n x k): per node, the weighted throughput sum and
    max over the pairs of each of the k metabolites; w_by their pair weights.

    Metabolites whose removal would leave no pairs (e.g. the only target) are
    not dropped: the question is whether a score rests on one metabolite
    *among several*.
    """
    n = len(acc)
    droppable = (w_by > 0) & (w_by < total_w)
    if not droppable.any():
        return np.full(n, np.inf), np.full(n, np.inf)
    rest_w = total_w - w_by[droppable]
    cond = ((acc[:, None] - sums[:, droppable]) / rest_w).min(axis=1)
    # Max over the remaining pairs = the largest per-metabolite max once the
    # dropped metabolite's own max is gone; worst case is dropping the top one.
    m = maxes[:, droppable]
    bott = np.sort(m, axis=1)[:, -2] if m.shape[1] > 1 else np.zeros(n)
    return cond, bott


def current_flow(graph, sources, targets, weights=None, directed=False, tau=0.5, chunk=128):
    """Run current flow for all retained source-target pairs.

    weights: optional {node: phi}; a pair's weight is (phi_s + phi_t) / 2.
             None gives every pair weight 1 (unit injection, as in the protocol).
    """
    pairs, excluded, labels = select_pairs(graph, sources, targets, directed)
    n = graph.n
    acc = np.zeros(n)
    bott = np.zeros(n)
    frac = np.zeros(n)
    edge_acc = np.zeros(len(graph.edges))
    edge_max = np.zeros(len(graph.edges))
    endpoint = {}
    total_w = 0.0
    s_pos = {s: i for i, s in enumerate(sources)}
    t_pos = {t: i for i, t in enumerate(targets)}
    sum_s, max_s = np.zeros((n, len(sources))), np.zeros((n, len(sources)))
    sum_t, max_t = np.zeros((n, len(targets))), np.zeros((n, len(targets)))
    w_s, w_t = np.zeros(len(sources)), np.zeros(len(targets))

    if pairs:
        B = graph.incidence()                 # m x n, unit conductance per edge
        L = (B.T @ B).tocsr()
        absB_T = abs(B).T.tocsr()
        daa_nodes = sorted({v for p in pairs for v in p})
        Y, col = _potentials(graph, L, labels, daa_nodes)

        # Edges incident to each endpoint, to measure how concentrated the
        # injected current is on a single reaction.
        incident = {v: np.flatnonzero((graph.edges == v).any(axis=1)) for v in daa_nodes}

        for start in range(0, len(pairs), chunk):
            block = pairs[start:start + chunk]
            s_idx = np.array([s for s, _ in block])
            t_idx = np.array([t for _, t in block])
            X = Y[:, [col[s] for s in s_idx]] - Y[:, [col[t] for t in t_idx]]
            I = np.asarray(B @ X)                         # edge currents (Ohm's law)
            T = 0.5 * np.asarray(absB_T @ np.abs(I))      # node throughput
            k = np.arange(len(block))
            T[s_idx, k] = 0.0                             # endpoints trivially carry all current
            T[t_idx, k] = 0.0
            w = (np.array([(weights[s] + weights[t]) / 2 for s, t in block])
                 if weights else np.ones(len(block)))

            acc += T @ w
            edge_acc += np.abs(I) @ w
            edge_max = np.maximum(edge_max, np.abs(I).max(axis=1))
            bott = np.maximum(bott, T.max(axis=1))
            frac += (T >= tau) @ w
            total_w += w.sum()

            si = np.array([s_pos[s] for s in s_idx])
            ti = np.array([t_pos[t] for t in t_idx])
            Tw = T * w
            np.add.at(sum_s.T, si, Tw.T)
            np.add.at(sum_t.T, ti, Tw.T)
            np.maximum.at(max_s.T, si, T.T)
            np.maximum.at(max_t.T, ti, T.T)
            np.add.at(w_s, si, w)
            np.add.at(w_t, ti, w)

            r_eff = X[s_idx, k] - X[t_idx, k]
            for j, (s, t) in enumerate(block):
                for v in (s, t):
                    e = endpoint.setdefault(v, {"pairs": 0, "eff_conductance": 0.0, "max_edge_share": 0.0})
                    e["pairs"] += 1
                    e["eff_conductance"] += 1.0 / r_eff[j] if r_eff[j] > 0 else 0.0
                    e["max_edge_share"] += np.abs(I[incident[v], j]).max()

        for e in endpoint.values():
            e["eff_conductance"] /= e["pairs"]
            e["max_edge_share"] /= e["pairs"]

    denom = total_w or 1.0
    cond = acc / denom
    if pairs:
        cs, bs = _leave_one_out(acc, total_w, sum_s, w_s, max_s)
        ct, bt = _leave_one_out(acc, total_w, sum_t, w_t, max_t)
        cond_loo, bott_loo = np.minimum(cs, ct), np.minimum(bs, bt)
        # Only one source and one target: nothing can be dropped.
        cond_loo = np.where(np.isinf(cond_loo), cond, cond_loo)
        bott_loo = np.where(np.isinf(bott_loo), bott, bott_loo)
    else:
        cond_loo, bott_loo = cond, bott
    return FlowResult(pairs, excluded, acc, cond, bott, frac / denom,
                      np.maximum(cond_loo, 0.0), bott_loo, endpoint, edge_acc / denom, edge_max)
