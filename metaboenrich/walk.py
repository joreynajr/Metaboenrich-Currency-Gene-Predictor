"""Absorbing random walk from sources to targets on the directed
metabolite-reaction graph (experimental; an alternative to protocol §2.3).

Walkers start at the sources and step along reaction directions only:
substrate -> reaction -> product for a forward reaction, the reverse for a
backward one. A reversible reaction is split into a forward and a backward
state, so a walker that enters through a substrate can only leave through a
product. A walker may not undo the step it just took (forward through a
reaction, then straight back through its backward state), which would
otherwise let it hop between co-substrates in two steps. To enforce that, a
walker on a metabolite remembers which reaction state it arrived from.

The targets absorb walkers. The DAA enters in two places:

* hook 1, start weights: walkers are released at source s in proportion to
  phi_s = |log2FC_s|
* hook 2, partial absorption (optional, kappa): a walker arriving at target t
  is absorbed with probability a_t = 1 - exp(-kappa * phi_t) and otherwise
  walks on, so a deep drop catches nearly everything and a mild one lets
  walkers through to targets further along. kappa=None absorbs every walker.

A walker with no way out (a dead end) leaks: it is lost. Scores use only
walks that reach a target. This is the Doob h-transform: with h(u) the
probability that a walk from state u is absorbed at a target, the expected
traffic of successful walks over the arc u -> v, for walks from s, is

    n_s(u) Q_uv h(v) / h(s),   where n_s = (I - Q)^-T e_s

(Q: transitions between transient states, with the rows of target arrival
states scaled by 1 - a_t). One sparse LU of I - Q gives h and every n_s. Arc
traffic is folded back onto the original metabolite-reaction edges as net
flow, and a node's throughput is half the summed |net flow| on its edges, as
in flow.py.

Per source s, the conditioned walk is one unit of flow from s to the targets.
Scores average these per-source flows, weighting source s by phi_s * h(s):
its share of all released walkers that reach a target. Bottleneck is the
largest per-source throughput. Directed cycles (all-irreversible loops) can
carry circulating flow, so throughput can exceed 1, unlike in current flow.
"""
from collections import deque
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

from .flow import FlowResult

ANY = -2   # stands in for "any target" / "any source" in excluded rows
_EPS = 1e-12


@dataclass
class WalkStates:
    """State space of the walk. States 0..graph.n-1 are the graph's nodes: a
    metabolite's state is where walkers start, a reaction's state is one of
    its directions. Then come the second directions of reversible reactions,
    then one arrival state per (reaction direction -> metabolite) step."""
    n_states: int
    node: np.ndarray        # graph node of each state
    arrival: np.ndarray     # bool: state = "on metabolite node[i], just arrived from a reaction"
    arc_u: np.ndarray       # walk arcs u -> v ...
    arc_v: np.ndarray
    arc_edge: np.ndarray    # ... the graph edge each arc runs along
    arc_sign: np.ndarray    # +1 metabolite -> reaction, -1 reaction -> metabolite


def walk_states(graph):
    """Build the walk's state space. Reaction directions come from graph.arcs
    read against graph.edge_side."""
    arcset = set(map(tuple, np.asarray(graph.arcs).reshape(-1, 2).tolist()))
    edges = graph.edges.tolist()
    fwd, bwd = {}, {}
    for (m, r), side in zip(edges, graph.edge_side):
        into, out = (m, r) in arcset, (r, m) in arcset
        if side == "product":
            into, out = out, into          # leaving the reaction towards a product is the forward step
        fwd[r] = fwd.get(r, False) or into
        bwd[r] = bwd.get(r, False) or out

    node = list(range(graph.n))
    fstate, bstate, opposite = {}, {}, {}
    for r in sorted(set(fwd) | set(bwd)):
        if fwd[r]:
            fstate[r] = r
        if bwd[r]:
            bstate[r] = len(node) if fwd[r] else r
            if fwd[r]:
                node.append(r)
        if fwd[r] and bwd[r]:
            opposite[fstate[r]], opposite[bstate[r]] = bstate[r], fstate[r]

    inputs = {}    # metabolite -> [(direction state it feeds, edge)]
    outputs = {}   # direction state -> [(metabolite it makes, edge)]
    for e, ((m, r), side) in enumerate(zip(edges, graph.edge_side)):
        # forward: substrates in, products out; backward: the other way round
        for d, is_input in ((fstate.get(r), side != "product"), (bstate.get(r), side == "product")):
            if d is None:
                continue
            if is_input:
                inputs.setdefault(m, []).append((d, e))
            else:
                outputs.setdefault(d, []).append((m, e))

    u, v, ae, sg = [], [], [], []
    for m, ins in inputs.items():       # start state: any reaction the metabolite feeds
        for d, e in ins:
            u.append(m); v.append(d); ae.append(e); sg.append(1.0)
    arrival = [False] * len(node)
    for d, outs in outputs.items():
        for m, e in outs:
            a = len(node)
            node.append(m)
            arrival.append(True)
            u.append(d); v.append(a); ae.append(e); sg.append(-1.0)
            for d2, e2 in inputs.get(m, []):
                if d2 != opposite.get(d):   # no immediate reversal
                    u.append(a); v.append(d2); ae.append(e2); sg.append(1.0)
    return WalkStates(len(node), np.array(node), np.array(arrival), np.array(u, dtype=np.int64),
                      np.array(v, dtype=np.int64), np.array(ae, dtype=np.int64), np.array(sg))


def _can_reach(n_states, arc_u, arc_v, goals):
    """Boolean mask of states from which some state in `goals` is reachable."""
    pred = sp.csr_matrix((np.ones(len(arc_u)), (arc_v, arc_u)), shape=(n_states, n_states))
    seen = np.zeros(n_states, dtype=bool)
    seen[goals] = True
    queue = deque(int(g) for g in goals)
    while queue:
        x = queue.popleft()
        for p in pred.indices[pred.indptr[x]:pred.indptr[x + 1]]:
            if not seen[p]:
                seen[p] = True
                queue.append(p)
    return seen


def absorbing_walk(graph, sources, targets, phi, kappa=None, tau=0.5, ground=None, chunk=128):
    """Run the absorbing walk. Returns a FlowResult, so scoring is shared with
    current_flow.

    phi:    {node: |log2FC|} for sources and targets (hooks 1 and 2).
    kappa:  partial-absorption strength; None absorbs every walker at a target.
    ground: optional node list for one-sided data. With no targets, the ground
            nodes absorb every walker; with no sources, walkers start evenly
            from the ground nodes.

    FlowResult fields as in current_flow, except: the leave-one-out columns are
    NaN (not defined here); endpoint["eff_conductance"] is, for a source, the
    probability that its walkers reach a target and, for a target, the share
    of successful walkers it absorbs; net_forward is a reaction's net traffic
    in its forward direction (negative = runs backward); walk_info holds
    run-level numbers.
    """
    sources, targets = list(sources), list(targets)
    ground = [g for g in (ground or []) if g not in set(sources) | set(targets)]
    weight = {s: float(phi.get(s, 0.0)) for s in sources}
    absorb_p = {t: 1.0 if kappa is None else 1.0 - float(np.exp(-kappa * float(phi.get(t, 0.0))))
                for t in targets}
    one_sided = ""
    if ground and sources and not targets:
        targets, absorb_p, one_sided = list(ground), {g: 1.0 for g in ground}, "ground absorbs"
    elif ground and targets and not sources:
        sources, weight, one_sided = list(ground), {g: 1.0 for g in ground}, "ground releases"
    if sources and sum(weight.values()) <= 0:
        weight = {s: 1.0 for s in sources}
    real_targets = targets if one_sided != "ground absorbs" else []
    real_sources = sources if one_sided != "ground releases" else []

    n, m_edges = graph.n, len(graph.edges)
    ws = walk_states(graph)
    k_of = {t: k for k, t in enumerate(targets)}
    absorb = np.zeros(ws.n_states)
    tgt_states = np.flatnonzero(ws.arrival & np.isin(ws.node, targets))
    absorb[tgt_states] = [absorb_p[t] for t in ws.node[tgt_states]]
    goals = tgt_states[absorb[tgt_states] > 0]
    live = _can_reach(ws.n_states, ws.arc_u, ws.arc_v, goals) if len(goals) else np.zeros(ws.n_states, bool)

    outdeg = np.bincount(ws.arc_u, minlength=ws.n_states)
    keep = live[ws.arc_u] & live[ws.arc_v]
    au, av, ae, asg = ws.arc_u[keep], ws.arc_v[keep], ws.arc_edge[keep], ws.arc_sign[keep]
    q = (1.0 - absorb[au]) / outdeg[au]                       # Q_uv on live arcs

    live_idx = np.flatnonzero(live)
    pos = np.full(ws.n_states, -1)
    pos[live_idx] = np.arange(len(live_idx))
    h = np.zeros(ws.n_states)
    lu = None
    if len(live_idx):
        Q = sp.csc_matrix((q, (pos[au], pos[av])), shape=(len(live_idx),) * 2)
        lu = splu((sp.identity(len(live_idx), format="csc") - Q).tocsc())
        h[live_idx] = lu.solve(absorb[live_idx])

    reached = [s for s in sources if h[s] > _EPS]
    w_raw = np.array([weight[s] * h[s] for s in reached])
    released = sum(weight.values())
    w = w_raw / w_raw.sum() if len(reached) else w_raw

    acc, bott, frac, fwd_acc = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    edge_acc, edge_max = np.zeros(m_edges), np.zeros(m_edges)
    endpoint, pairs, excluded = {}, [], []
    for s in real_sources:
        if h[s] <= _EPS:
            excluded += [(s, t, "no directed path to a target") for t in (real_targets or [ANY])]

    absB_T = abs(graph.incidence()).T.tocsr()                 # node x edge
    M = sp.csr_matrix((asg, (ae, np.arange(len(ae)))), shape=(m_edges, len(ae)))   # arc -> net edge flow
    is_sub = np.array([side != "product" for side in graph.edge_side], dtype=bool)
    S_fwd = sp.csr_matrix((np.ones(int(is_sub.sum())), (graph.edges[is_sub, 1], np.flatnonzero(is_sub))),
                          shape=(n, m_edges))                 # substrate-edge net flow summed per reaction
    G = sp.csr_matrix((absorb[tgt_states], ([k_of[t] for t in ws.node[tgt_states]], tgt_states)),
                      shape=(len(targets), ws.n_states))      # absorption at each target
    tgt = np.array(targets, dtype=np.int64)
    tgt_share, tgt_edge_share = np.zeros(len(tgt)), np.zeros(len(tgt))
    tgt_pairs = np.zeros(len(tgt), dtype=int)
    incident = {v: np.flatnonzero((graph.edges == v).any(axis=1)) for v in set(sources) | set(targets)}
    ep_real = set(real_sources)

    for start in range(0, len(reached), chunk):
        block = reached[start:start + chunk]
        wb = w[start:start + chunk]
        cols = np.arange(len(block))
        E = np.zeros((len(live_idx), len(block)))
        E[pos[block], cols] = 1.0
        N = np.zeros((ws.n_states, len(block)))
        N[live_idx] = lu.solve(E, trans="T")                  # expected visits, walks from each source
        hs = h[block]
        C = N[au] * (q * h[av])[:, None] / hs[None, :]        # conditioned arc traffic
        net = np.asarray(M @ C)                               # net flow per graph edge (met -> rxn positive)
        T = 0.5 * np.asarray(absB_T @ np.abs(net))
        absorbed = np.asarray(G @ N) / hs[None, :]            # share of s's successful walks ending at t
        T[tgt] -= 0.5 * absorbed                              # walkers absorbed at t stop: count relay only
        T[block, cols] = 0.0                                  # a source trivially carries its own walkers
        T = np.maximum(T, 0.0)

        acc += T @ wb
        bott = np.maximum(bott, T.max(axis=1))
        frac += (T >= tau) @ wb
        edge_acc += np.abs(net) @ wb
        edge_max = np.maximum(edge_max, np.abs(net).max(axis=1))
        fwd_acc += np.asarray(S_fwd @ net) @ wb
        tgt_share += absorbed @ wb
        hit = absorbed > _EPS
        tgt_pairs += hit.sum(axis=1)
        for k, t in enumerate(targets):
            if t in incident and len(incident[t]):
                tgt_edge_share[k] += np.abs(net[incident[t]]).max(axis=0) @ wb
        for j, s in enumerate(block):
            src = s if s in ep_real else ANY
            for k, t in enumerate(targets):
                if hit[k, j]:
                    pairs.append((src, t if real_targets else ANY))
                elif real_targets and src != ANY:
                    excluded.append((s, t, "no walk reaches this target"))
            if src != ANY:
                endpoint[s] = {"pairs": int(hit[:, j].sum()), "eff_conductance": float(hs[j]),
                               "max_edge_share": float(np.abs(net[incident[s], j]).max())}

    for k, t in enumerate(targets):
        if not real_targets:
            break
        if tgt_pairs[k]:
            endpoint[t] = {"pairs": int(tgt_pairs[k]), "eff_conductance": float(tgt_share[k]),
                           "max_edge_share": float(tgt_edge_share[k])}
        elif not real_sources:                                # one-sided, ground releases
            excluded.append((ANY, t, "no walk reaches this target"))
    if not real_targets or not real_sources:                  # one-sided: one row per real node
        pairs = sorted(set(pairs))

    nan = np.full(n, np.nan)
    info = {"walk_kappa": kappa, "walk_one_sided": one_sided,
            "walk_success_fraction": float(w_raw.sum() / released) if released else 0.0,
            "walk_sources_unreached": int(sum(h[s] <= _EPS for s in real_sources)),
            "walk_states": int(ws.n_states), "walk_live_states": int(len(live_idx)),
            "walk_arcs": int(len(ws.arc_u))}
    return FlowResult(pairs, excluded, acc, acc, bott, frac, nan, nan.copy(), endpoint, edge_acc, edge_max,
                      net_forward=np.where(graph.is_reaction, fwd_acc, np.nan), walk_info=info)
