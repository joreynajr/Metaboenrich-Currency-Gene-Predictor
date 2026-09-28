"""Structural (data-free) bottleneck analysis across network pruning stages.

For each stage of pruning Human-GEM, every node gets:

* degree        number of edges (reactions for a metabolite, metabolites for a reaction)
* flow          structural current load: mean throughput when randomly sampled
                metabolites act as sources and targets, i.e. how much current
                the node carries from network wiring alone, with no data
* pocket        metabolites cut off from the rest of the network if the node
                is removed (0 for most nodes; the sole link to a dead end or
                an island-in-waiting otherwise)

    python -m metaboenrich.structure --out results/structure
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from .flow import current_flow
from .gem import download_human_gem, load_sbml
from .network import (CurrencyRules, DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits,
                      resolve_currency)

DEFAULT_MODEL = "data/Human-GEM.xml"


def stages(model):
    """(key, title, description, rules, max_reaction_size, drop_objective) per pruning stage."""
    role, _ = resolve_currency(model, "role")
    remove, _ = resolve_currency(model, "remove")
    edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, model)

    def rules(base, removed=None, families=True, pools=False, edit=False):
        return CurrencyRules(
            removed=dict(base.removed if removed is None else removed),
            family_of=dict(base.family_of) if families else {},
            members=base.members, loaded_core=base.loaded_core, formulas=base.formulas,
            pools=set(base.pools) if pools else set(), edits=edits if edit else {},
            names=base.names, loaded_tokens=base.loaded_tokens)

    empty = rules(role, removed={}, families=False)
    return [
        ("raw", "1. Raw Human-GEM",
         "Compartments merged; transport/exchange reactions (one metabolite after merging) dropped. Nothing else removed.",
         empty, None, False),
        ("inorganic", "2. Inorganics and carriers removed",
         "H+, H2O, Pi, PPi, O2, CO2, NH3, ions, [protein], ferredoxins, cytochrome b5 and thioredoxins removed from every reaction.",
         rules(role, families=False), None, False),
        ("cofactors", "3. Cofactors role-filtered",
         "ATP, NAD(P)(H), CoA, SAM, ... removed where they exchange with a partner form; kept in their synthesis and breakdown.",
         rules(role), None, False),
        ("pools", "4. Pools and pseudo-reactions dropped",
         "Biomass reactions, lumped pools (cofactors and vitamins, xenobiotics, ...) and reactions with > 20 metabolites dropped.",
         rules(role, pools=True), 20, True),
        ("edits", "5. Reaction edits applied",
         "SCLY (MAR07133) B6 vitamers removed. This is the network the analysis uses.",
         rules(role, pools=True, edit=True), 20, True),
        ("alt_remove", "Alternative: cofactors removed everywhere",
         "The original approach: every cofactor removed from every reaction, plus the stage 4-5 cleanup. For comparison with stage 5.",
         rules(remove, families=False, pools=True, edit=True), 20, True),
    ]


def pocket_sizes(graph):
    """Metabolites separated from the largest remaining piece of their
    component when each node is removed (iterative Tarjan articulation search)."""
    n = graph.n
    u, v = graph.edges[:, 0], graph.edges[:, 1]
    adj = sp.csr_matrix((np.ones(2 * len(u)), (np.r_[u, v], np.r_[v, u])), shape=(n, n))
    ptr, nbr = adj.indptr, adj.indices
    is_met = (~graph.is_reaction).astype(np.int64)

    disc = np.full(n, -1)
    low = np.zeros(n, np.int64)
    parent = np.full(n, -1)
    sub = is_met.copy()           # metabolites in DFS subtree
    sep_sum = np.zeros(n, np.int64)
    sep_max = np.zeros(n, np.int64)
    comp = np.full(n, -1)
    t = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = t
        t += 1
        comp[root] = root
        stack = [(root, ptr[root])]
        while stack:
            x, i = stack[-1]
            if i < ptr[x + 1]:
                stack[-1] = (x, i + 1)
                w = nbr[i]
                if disc[w] == -1:
                    parent[w], comp[w] = x, root
                    disc[w] = low[w] = t
                    t += 1
                    stack.append((w, ptr[w]))
                elif w != parent[x]:
                    low[x] = min(low[x], disc[w])
            else:
                stack.pop()
                p = parent[x]
                if p != -1:
                    low[p] = min(low[p], low[x])
                    sub[p] += sub[x]
                    if low[x] >= disc[p]:
                        sep_sum[p] += sub[x]
                        sep_max[p] = max(sep_max[p], sub[x])
    comp_mets = np.bincount(comp, weights=is_met, minlength=n).astype(np.int64)
    rest = comp_mets[comp] - is_met - sep_sum
    return (sep_sum + rest - np.maximum(sep_max, rest)).astype(np.int64)


def analyse_stage(graph, samples, rng):
    labels = graph.components()
    main = np.bincount(labels).argmax()
    in_main = labels == main
    degree = np.bincount(graph.edges.ravel(), minlength=graph.n)

    mets = np.flatnonzero(in_main & ~graph.is_reaction)
    pick = rng.choice(mets, size=min(2 * samples, len(mets)), replace=False)
    flow = current_flow(graph, list(pick[:samples]), list(pick[samples:]))
    pocket = pocket_sizes(graph)
    return degree, flow.conductivity, pocket, in_main


def concentration(values, top=0.01):
    """Share of total structural current carried by the top `top` fraction of nodes."""
    v = np.sort(values)[::-1]
    k = max(1, int(round(top * len(v))))
    return float(v[:k].sum() / v.sum()) if v.sum() else 0.0


def main(argv=None):
    p = argparse.ArgumentParser(prog="metaboenrich.structure", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--out", default="results/structure")
    p.add_argument("--samples", type=int, default=150,
                   help="random sources and targets per stage (pairs = samples^2; default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if not Path(args.model).exists():
        if args.model != DEFAULT_MODEL:
            raise SystemExit(f"Model file not found: {args.model}")
        download_human_gem(args.model)
    model = load_sbml(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    node_index, node_meta = {}, []   # shared across stages: id -> position; [id, name, type, genes]
    result = {"model": f"{model.id} {model.version}", "samples": args.samples, "seed": args.seed,
              "nodes": node_meta, "stages": []}
    for key, title, desc, rules, max_size, drop_obj in stages(model):
        g = build_graph(model, rules, max_size, drop_obj)
        degree, flow, pocket, in_main = analyse_stage(g, args.samples, np.random.default_rng(args.seed))
        ids = []
        for i, nid in enumerate(g.node_ids):
            if nid not in node_index:
                node_index[nid] = len(node_meta)
                genes = ";".join(model.genes.get(x) or x for x in g.reaction_genes.get(i, [])) \
                    if g.is_reaction[i] else ""
                node_meta.append([nid, g.node_names[i], "r" if g.is_reaction[i] else "m", genes])
            ids.append(node_index[nid])

        is_rxn = g.is_reaction
        n_rxn = int(is_rxn.sum())
        top_m = np.argsort(-np.where(is_rxn, -1, degree))[:1][0]
        summary = {
            "metabolites": g.n - n_rxn, "reactions": n_rxn, "edges": len(g.edges),
            "components": int(len(np.unique(g.components()))),
            "metabolites_outside_main": int((~in_main & ~is_rxn).sum()),
            "reactions_pocket_ge1": int(((pocket >= 1) & is_rxn).sum()),
            "reactions_pocket_ge5": int(((pocket >= 5) & is_rxn).sum()),
            "metabolites_pocket_ge1": int(((pocket >= 1) & ~is_rxn).sum()),
            "top1pct_flow_share": round(concentration(flow), 4),
            "max_metabolite_degree": int(degree[top_m]),
            "max_degree_metabolite": g.node_names[top_m],
        }
        result["stages"].append({
            "key": key, "title": title, "description": desc, "summary": summary,
            "node": ids, "degree": degree.tolist(),
            "flow": [round(float(x), 6) for x in flow],
            "pocket": pocket.tolist(), "main": in_main.astype(int).tolist(),
        })
        print(f"{title:45s} {summary}")

    (out / "structure.json").write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {out / 'structure.json'}")


if __name__ == "__main__":
    main()
