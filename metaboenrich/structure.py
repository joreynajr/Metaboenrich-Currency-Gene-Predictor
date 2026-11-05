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


STAGE_SETS = ("pruning", "carbon", "three_rule")
# Stages that are side-by-side alternatives rather than steps in the main
# sequence; the diagrams place removed nodes by the main sequence only.
ALTERNATIVE_STAGES = {"alt_remove", "carbon_mapped_only", "first_method", "combined", "acyl_merged",
                      "acyl_merged_keep_rarest", "rules12_keep_rarest"}
# The stage whose network sets the shared layout of the diagrams.
LAYOUT_STAGE = {"pruning": "edits", "carbon": "carbon_method", "three_rule": "three_rule"}


def stages(model, stage_set="pruning", atom_dir="data", couple_min_reactions=None):
    """(key, title, description, rules, max_reaction_size, drop_objective,
    extra build_graph options) per stage. 'pruning': the currency-pruning steps. 'carbon': the carbon-
    skeleton channel networks (needs data/atom_pairs*.json from
    python -m metaboenrich.atoms)."""
    if stage_set == "carbon":
        return _carbon_stages(model, atom_dir, couple_min_reactions)
    if stage_set == "three_rule":
        return _three_rule_stages(model, atom_dir, couple_min_reactions)
    if stage_set == "lipid_merge":
        return _lipid_merge_stages(model, atom_dir, couple_min_reactions)
    return [s + ({},) for s in _pruning_stages(model)]


def _carbon_stages(model, atom_dir, couple_min_reactions=None):
    """The carbon method on its own (no hand-made currency lists), with the
    first method alongside for reference. Pairs: KEGG's curated pairs plus
    cleaned atom maps (data/atom_pairs_union.json)."""
    from .atoms import (COUPLE_MIN_REACTIONS, carbon_method_pairs, carbon_method_rules, carbon_rule1_pairs,
                        load_atom_pairs)
    n_min = couple_min_reactions or COUPLE_MIN_REACTIONS
    pr = _pruning_stages(model)
    raw, edits = pr[0], pr[4]
    pools = edits[3].pools
    union = load_atom_pairs(Path(atom_dir) / "atom_pairs_union.json")
    rule1 = carbon_rule1_pairs(model, union)
    method, couples = carbon_method_pairs(model, union, n_min)
    r1_rules = carbon_method_rules(model, rule1, {}, pools)
    m_rules = carbon_method_rules(model, method, couples, pools)
    return [
        ("raw", "1. Raw Human-GEM", raw[2], raw[3], raw[4], raw[5], {}),
        ("carbon_rule1", "2. Rule 1: follow carbon",
         "Each reaction is split into its substrate-product pairs that share carbon atoms; molecules with no carbon "
         "(water, H+, phosphate, O2, NH3, ions) never link. Biomass and pool pseudo-reactions dropped. Reactions "
         "that cannot be atom-mapped are kept whole, minus their carbon-free molecules.",
         r1_rules, 20, True, {"atom_pairs": rule1}),
        ("carbon_method", "3. Rules 1 + 2: the carbon method",
         f"Rule 1, then any carbon pair found in {n_min} or more reactions is treated as a cofactor couple "
         f"(ATP/ADP, NAD+/NADH, CoA/acetyl-CoA ...) and removed everywhere; {len(couples)} couples detected from "
         "the network itself. No hand-made lists.",
         m_rules, 20, True, {"atom_pairs": method}),
        ("carbon_mapped_only", "4. Carbon method, mapped reactions only",
         "Network 3 without the reactions that could not be atom-mapped: shows what the method gives when every "
         "link is backed by an atom map or KEGG pair.",
         m_rules, 20, True, {"atom_pairs": method, "drop_unmapped": True}),
        ("first_method", "Reference: first method",
         "The current analysis network (stage 5 of the Pruning Atlas): hand-made lists of inorganics and cofactors, "
         "cofactors kept only where made or broken down. Reactions are not split.",
         edits[3], edits[4], edits[5], {}),
        ("combined", "Reference: first method + carbon split",
         "The first method's network with each reaction then split into its carbon pairs (the version shown in the "
         "earlier draft of this report).", edits[3], edits[4], edits[5], {"atom_pairs": union}),
    ]


# Filled by the most recent _three_rule_stages call: the rescues made by the
# keep-rarest safeguard in 3-rule pruning (list of dicts, see
# metaboenrich.atoms.carbon_method_pairs), the same without Rule 3, and the
# carriers found by Rule 3 ({carrier: {loaded forms}}), and the pairs Rule 4
# removed (list of dicts, see metaboenrich.atoms.carbon_budget_pairs).
KEPT_RAREST_LOG = []
KEPT_RAREST_LOG_RULES12 = []
CARRIERS_FOUND = {}
CARBON_BUDGET_LOG = []
RESCUE_PICK = "rarest"   # the safeguard's choice in the final stage (atoms.RESCUE_PICKS)


def _three_rule_stages(model, atom_dir, couple_min_reactions=None):
    """The carbon method revised: pool pseudo-metabolites and transporters
    dropped, carriers removed (Rule 3: CoA, carnitine, glutathione ... found
    from structure), and the keep-rarest safeguard (a reaction whose carbon
    pairs were all removed keeps its rarest one instead of dropping out).
    The carbon method itself, earlier variants and the first method are shown
    for comparison; the carbon stage set is not changed."""
    from .atoms import (COUPLE_MIN_REACTIONS, acyl_coa_merge, carbon_budget_pairs, carbon_method_pairs,
                        carbon_method_rules, carbon_unbalanced, find_carriers, kegg_pairs_for_model,
                        load_atom_pairs, pool_metabolites, transport_reactions)
    n_min = couple_min_reactions or COUPLE_MIN_REACTIONS
    carbon = {s[0]: s for s in _carbon_stages(model, atom_dir, couple_min_reactions)}
    pr = _pruning_stages(model)
    union = load_atom_pairs(Path(atom_dir) / "atom_pairs_union.json")
    pools = set(pr[4][3].pools) | pool_metabolites(model)
    cargo = transport_reactions(model, carbon_cargo=True)
    shared = transport_reactions(model, carbon_cargo=False)
    drop = {"drop_reactions": cargo, "merge_duplicates": True}

    method, couples = carbon_method_pairs(model, union, n_min)
    clean_rules = carbon_method_rules(model, method, couples, pools)

    carriers, carrier_pairs = find_carriers(model, union, Path(atom_dir) / "Human-GEM_metabolites.tsv")
    CARRIERS_FOUND.clear()
    CARRIERS_FOUND.update(carriers)
    cr = {"carriers": carrier_pairs}
    c_pairs, _ = carbon_method_pairs(model, union, n_min, **cr)
    c_rules = carbon_method_rules(model, c_pairs, couples, pools, carriers=carriers)
    log = []
    cl_pairs, _ = carbon_method_pairs(model, union, n_min, keep_rarest=True, no_rescue=shared, log=log,
                                      rescue_pick=RESCUE_PICK, **cr)
    kegg = kegg_pairs_for_model(model, Path(atom_dir) / "Human-GEM.xml", Path(atom_dir) / "kegg_rclass_pairs.json")
    CARBON_BUDGET_LOG.clear()
    cl_pairs = carbon_budget_pairs(model, cl_pairs, {rid: {tuple(p) for p in ps} for rid, ps in kegg.items()},
                                   carbon_unbalanced(model), CARBON_BUDGET_LOG)
    cl_rules = carbon_method_rules(model, cl_pairs, couples, pools, keep_rarest=True, no_rescue=shared, log=log,
                                   carriers=carriers, rescue_pick=RESCUE_PICK)
    KEPT_RAREST_LOG[:] = log

    log0 = []
    ll_pairs, _ = carbon_method_pairs(model, union, n_min, keep_rarest=True, no_rescue=shared, log=log0)
    ll_rules = carbon_method_rules(model, ll_pairs, couples, pools, keep_rarest=True, no_rescue=shared, log=log0)
    KEPT_RAREST_LOG_RULES12[:] = log0

    merge, merge_names, _ = acyl_coa_merge(model, Path(atom_dir) / "Human-GEM_metabolites.tsv")
    mg = {"merge": merge, "merge_names": merge_names}
    m_pairs, m_couples = carbon_method_pairs(model, union, n_min, merge=merge)
    m_rules = carbon_method_rules(model, m_pairs, m_couples, pools, merge=merge)
    ml_pairs, _ = carbon_method_pairs(model, union, n_min, keep_rarest=True, no_rescue=shared, merge=merge)
    ml_rules = carbon_method_rules(model, ml_pairs, m_couples, pools, keep_rarest=True, no_rescue=shared,
                                   merge=merge)
    n_lc = sum(v == "ACYLCOA_LC" for v in merge.values())
    n_vlc = len(merge) - n_lc
    names = {s.met_id: s.name for s in model.species.values()}
    carrier_text = ", ".join(f"{names[c]} ({len(x)})" for c, x in sorted(carriers.items(), key=lambda kv: -len(kv[1])))
    raw, cm, first = carbon["raw"], carbon["carbon_method"], carbon["first_method"]
    return [
        raw,
        ("carbon_method", "2. Carbon method (Rules 1 + 2)", cm[2], cm[3], cm[4], cm[5], cm[6]),
        ("carbon_cleaned", "3. Pools and transporters dropped",
         f"Network 2 without reactions that involve any of Human-GEM's {len(pools)} pool pseudo-metabolites "
         f"(acyl-CoA pool, PI pool, ...) and without {len(cargo):,} transporters, i.e. reactions carrying a "
         "carbon-containing metabolite across a membrane (ATP-driven pumps, vesicle secretion). Their genes stay "
         "in the transporter table.",
         clean_rules, 20, True, {"atom_pairs": method, **drop}),
        ("carbon_carriers", "4. Carriers removed (Rule 3)",
         "Network 3 without the links between a carrier and its loaded forms. A carrier is found from structure: "
         f"a molecule found whole inside at least 10 of its partners, always attached at the same atom through a "
         f"heteroatom (thioester, ester, amide). Carriers found: {carrier_text}. Links between two loaded forms "
         "(acetyl-CoA to acetoacetyl-CoA) stay.",
         c_rules, 20, True, {"atom_pairs": c_pairs, **drop}),
        ("three_rule", "5. 3-rule pruning (final)",
         "Network 4, but a reaction whose carbon pairs were all removed keeps the rarest one instead of dropping "
         "out (glutamate dehydrogenase keeps glutamate/2-oxoglutarate, 41 reactions, over NAD+/NADH, 427). "
         "Reactions that carry anything across a membrane are never rescued. Rule 4 then removes "
         f"{len(CARBON_BUDGET_LOG)} carbon pairs the reaction's own carbon counts rule out (transaminase "
         "cross-pairs, ATP to the aminoacyl-tRNA), never a pair KEGG lists.",
         cl_rules, 20, True, {"atom_pairs": cl_pairs, **drop}),
        ("rules12_keep_rarest", "Alternative: Rules 1-2 with the keep-rarest safeguard",
         "Network 3 with the keep-rarest safeguard but without removing carriers.",
         ll_rules, 20, True, {"atom_pairs": ll_pairs, **drop}),
        ("acyl_merged", "Alternative: long acyl-CoAs merged",
         f"Network 3 with the {n_lc} long-chain (C13-C21) and {n_vlc} very-long-chain (C22+) acyl-CoAs merged into "
         "one node per class, found from structure. Short and medium acyl-CoAs, including acetyl- and malonyl-CoA, "
         "stay individual. Couples are counted on the merged network, so CoA / long-chain acyl-CoA becomes one "
         "frequent pair.",
         m_rules, 20, True, {"atom_pairs": m_pairs, **drop, **mg}),
        ("acyl_merged_keep_rarest", "Alternative: long acyl-CoAs merged, keep-rarest safeguard",
         "The merged network with the keep-rarest safeguard.",
         ml_rules, 20, True, {"atom_pairs": ml_pairs, **drop, **mg}),
        first,
    ]


LIPID_MERGE_LOG = {}   # stage key -> [{reaction, genes, kept_pair, kept_pair_count}] for reactions the merge empties
LIPID_MERGE_MEMBERS = {}


def _lipid_merge_stages(model, atom_dir, couple_min_reactions=None):
    """The final 3-rule network with lipid chain-length ladders merged, one
    change at a time (experiment; the three_rule stages are not changed).

    The final network's pairs are renamed through the merge and pairs inside
    one merged node dropped; nothing else is recomputed (couples, carriers and
    Rule 4 stay as in the final network). A reaction the merge leaves with no
    pair gets the safeguard: its rarest other carbon pair, never one Rule 4
    removed, never in a membrane-crossing reaction; with none it drops out."""
    from .atoms import (ACYL_COA_CLASSES, LIPID_MIN_CARBONS, carbon_rule1_pairs, lipid_merge, load_atom_pairs,
                        transport_reactions)
    base = {s[0]: s for s in _three_rule_stages(model, atom_dir, couple_min_reactions)}["three_rule"]
    rules, extra = base[3], base[6]
    final = extra["atom_pairs"]
    rule4 = {(x["reaction"], tuple(x["pair"])) for x in CARBON_BUDGET_LOG}
    rule1 = carbon_rule1_pairs(model, load_atom_pairs(Path(atom_dir) / "atom_pairs_union.json"))
    freq = {}
    for ps in rule1.values():
        for p in {tuple(sorted(p)) for p in ps}:
            freq[p] = freq.get(p, 0) + 1
    shared = transport_reactions(model, carbon_cargo=False)
    smiles = Path(atom_dir) / "Human-GEM_metabolites.tsv"
    c12 = ((LIPID_MIN_CARBONS, 999, "ACYLCOA_LONG", f"long-chain acyl-CoA (C{LIPID_MIN_CARBONS}+)"),)
    variants = (
        ("lipid_coa_classes", "acyl-CoA C13-C21 and C22+ (fatty only)", {"coa_classes": ACYL_COA_CLASSES}),
        ("lipid_coa12", "acyl-CoA C12+ (fatty only)", {"coa_classes": c12}),
        ("lipid_coa12_acp", "acyl-CoA C12+ and acyl-[ACP] C12+", {"coa_classes": c12, "acp": True}),
        ("lipid_coa12_acp_ag3p", "acyl-CoA, acyl-[ACP] and 1-acylglycerol-3P C12+",
         {"coa_classes": c12, "acp": True, "acylglycerol3p": True}),
    )
    out = [base]
    for key, title, opts in variants:
        merge, names, members = lipid_merge(model, smiles, **opts)
        LIPID_MERGE_MEMBERS[key] = members

        def ren(p):
            return merge.get(p[0], p[0]), merge.get(p[1], p[1])
        pairs, log = {}, []
        for rid, ps in final.items():
            kept = list(dict.fromkeys(ren(p) for p in ps if ren(p)[0] != ren(p)[1]))
            if ps and not kept:
                cands = sorted({tuple(sorted(p)) for p in rule1.get(rid, []) if ren(p)[0] != ren(p)[1]
                                and (rid, p) not in rule4} if rid not in shared else set())
                best = min(cands, key=lambda p: (freq[p], p)) if cands else None
                kept = [ren(p) for p in rule1[rid] if tuple(sorted(p)) == best][:1] if best else []
                log.append({"reaction": rid, "genes": sorted(model.reactions[rid].genes), "kept_pair": best,
                            "kept_pair_count": freq.get(best) if best else None, "lost_pairs": ps})
            pairs[rid] = kept
        LIPID_MERGE_LOG[key] = log
        r = CurrencyRules(**{**rules.__dict__, "edits": {rid: {merge.get(m, m) for m in ms}
                                                         for rid, ms in rules.edits.items()}})
        desc = (f"The final network with {title} merged ({len(merge)} metabolites into {len(names)} node(s)); "
                f"{len(log)} reactions lose every pair, "
                f"{sum(x['kept_pair'] is None for x in log)} of them have no other pair and drop out.")
        out.append((key, f"Lipid merge: {title}", desc, r, base[4], base[5],
                    {**extra, "atom_pairs": pairs, "merge": merge, "merge_names": names}))
    return out


def _pruning_stages(model):
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
    p.add_argument("--stage-set", choices=STAGE_SETS, default="pruning",
                   help="pruning: currency-pruning steps; carbon: carbon-skeleton channel networks "
                        "(needs python -m metaboenrich.atoms first) (default: %(default)s)")
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
              "stage_set": args.stage_set, "nodes": node_meta, "stages": []}
    for key, title, desc, rules, max_size, drop_obj, extra in stages(model, args.stage_set):
        g = build_graph(model, rules, max_size, drop_obj, **extra)
        atom_pairs = extra.get("atom_pairs")
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
            "reactions_original": len({r for i in np.flatnonzero(is_rxn) for r in g.reactions_of(i)}),
            "reactions_kept_whole": len(g.unmapped_reactions) if atom_pairs is not None else None,
            "reactions_dropped_no_carbon_pair": len(g.no_pair_reactions) if atom_pairs is not None else None,
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
