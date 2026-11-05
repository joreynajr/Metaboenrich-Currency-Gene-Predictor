"""Spreadsheet of what each step of the revised carbon method removes.

Follows every chemical reaction of the raw network (Human-GEM, compartments
merged) through the steps of 3-rule pruning, in order:

    Step 1     pools, biomass and transporters dropped (whole reactions)
    Rule 1     links only between carbon-sharing molecules
    Rule 2     cofactor couples (pairs in 10+ reactions) removed
    Rule 3     carrier / loaded-form links removed (CoA, carnitine, ...)
    safeguard  a reaction left with no link keeps its rarest one
    size       reactions still listing more than 20 metabolites dropped

    python tools/rules_audit.py --out results/three_rule/rules_audit.xlsx

Sheets: About, Summary, one tab per step listing what it acts on (Step 1
pools, Rule 1 no carbon, Rule 2 couples, Rule 3 carriers), Reactions (one row
per reaction: what is left after each step, where it drops out and why), Links
(every carbon pair of the atom-mapped reactions and the step that removes it)
and Genes lost. The links come partly from KEGG's curated pairs, so keep the
file out of the public repository.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from openpyxl.styles import Alignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metaboenrich import structure                                   # noqa: E402
from metaboenrich.atoms import (COUPLE_MIN_REACTIONS, carbon_balance, carbon_budget, carbon_method_pairs,  # noqa: E402
                                carbon_unbalanced, find_carriers, kegg_pairs_for_model, load_atom_pairs,
                                pool_metabolites, transport_reactions)
from metaboenrich.gem import load_sbml, load_subsystems              # noqa: E402
from metaboenrich.network import build_graph, parse_formula          # noqa: E402

STEPS = ["Step 1 (pools, biomass, transporters)", "Rule 1 (carbon)", "Rule 2 (couples)", "Rule 3 (carriers)"]
SIZE = "Too large (over 20 metabolites)"
RULE4 = "Rule 4 (carbon budget)"     # removes links only; a reaction never drops out at it
MERGE = "Compartment copies merged"  # removes nothing; identical reactions share a node


def safeguard_tab(model, names, sym):
    """One row per reaction the keep-rarest safeguard kept: every candidate pair with its count."""
    rows = []
    for x in structure.KEPT_RAREST_LOG:
        r = model.reactions[x["reaction"]]
        cands = sorted(x["candidates"], key=lambda c: (c[1], c[0]))
        tied = [p for p, n in cands if n == x["kept_pair_count"]]
        rows.append({"reaction": x["reaction"].replace("R_", ""), "name": r.name,
                     "genes": ";".join(sym.get(g, g) for g in r.genes), "atom-mapped": x["mapped"],
                     "link kept": f"{names[x['kept_pair'][0]]} / {names[x['kept_pair'][1]]}",
                     "reactions that pair is found in": x["kept_pair_count"],
                     "tie broken by metabolite id": len(tied) > 1,
                     "candidates (pair: reactions found in), rarest first":
                         "; ".join(f"{names[a]} / {names[b]}: {n}" for (a, b), n in cands)})
    return pd.DataFrame(rows).sort_values(["reactions that pair is found in", "reaction"])


def _equation(model, rxn, names):
    """'2 ATP (10 C) + alanine (3 C) -> ...' with compartments merged and carbon per molecule."""
    def side(refs):
        out = {}
        for s, n in refs.items():
            m = model.species[s].met_id
            out[m] = out.get(m, 0) + n
        return " + ".join(f"{'' if n == 1 else f'{n:g} '}{names[m]} ({parse_formula(model.species[s].formula)['C']} C)"
                          for (m, n), s in zip(out.items(), [next(s for s in refs if model.species[s].met_id == m)
                                                            for m in out]))
    return f"{side(rxn.substrates)} -> {side(rxn.products)}"


def _assignment(model, rid, names, limit=16):
    """One way of sending each carbon substrate molecule whole into a product molecule with
    matching carbon counts (what carbon_budget searches for), or '' if there is none."""
    rxn, carbons, net = model.reactions[rid], {}, {}
    for sign, refs in ((-1, rxn.substrates), (1, rxn.products)):
        for s, n in refs.items():
            m = model.species[s].met_id
            carbons[m] = parse_formula(model.species[s].formula)["C"]
            net[m] = net.get(m, 0) + sign * n
    subs = [m for m, n in net.items() if n < 0 and carbons[m] > 0 for _ in range(round(-n))]
    prods = [m for m, n in net.items() if n > 0 and carbons[m] > 0 for _ in range(round(n))]
    if len(subs) + len(prods) > limit:
        return ""
    subs.sort(key=lambda m: -carbons[m])
    cap, into = [carbons[p] for p in prods], [None] * len(subs)

    def go(i):
        if i == len(subs):
            return not any(cap)
        for k in range(len(prods)):
            if cap[k] >= carbons[subs[i]]:
                cap[k] -= carbons[subs[i]]
                into[i] = k
                if go(i + 1):
                    return True
                cap[k] += carbons[subs[i]]
        return False
    if not go(0):
        return ""
    return "; ".join(f"{names[s]} ({carbons[s]} C) -> {names[prods[k]]}" for s, k in zip(subs, into))


def rule4_tab(model, names, union, carrier_pairs, unbalanced, R, final):
    """Every pair Rule 4 judged in a reaction of the final network, with the arithmetic and the decision."""
    shared = transport_reactions(model, carbon_cargo=False)
    pairs, _ = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS, keep_rarest=True, no_rescue=shared,
                                   carriers=carrier_pairs)
    kegg = {rid: {tuple(p) for p in ps} for rid, ps in kegg_pairs_for_model(model).items()}
    removed = {(x["reaction"], tuple(x["pair"])) for x in structure.CARBON_BUDGET_LOG}
    keep_rxn = set(R[R["in 3-rule pruning"] | (R["drops out at"] == SIZE)].reaction)
    kept_node = {r: final.base_reaction[k] for k, rs in final.merged_copies.items() for r in rs}
    review = {}
    path = ROOT / "handoff/rule4_reviewed.csv"
    if path.exists():
        rv = pd.read_csv(path)
        review = {("R_" + b, s, p): w for b, s, p, w in zip(rv.base, rv.substrate, rv["product"], rv.review)}
    rows = []
    for rid in sorted(pairs):
        ps = list(dict.fromkeys(pairs[rid]))
        if rid.replace("R_", "") not in keep_rxn or not ps:
            continue
        v = carbon_budget(model, rid, ps, unbalanced)
        rxn = model.reactions[rid]
        assign = _assignment(model, rid, names) if any(x == "impossible" for x in v.values()) else ""
        for s, p in ps:
            verdict, listed = v[(s, p)], (s, p) in kegg.get(rid, ())
            action = ("removed by Rule 4" if (rid, (s, p)) in removed else
                      "kept: KEGG lists the pair" if verdict == "impossible" and listed else
                      "kept: every pair of the reaction is impossible (a reaction is never left with none)"
                      if verdict == "impossible" else f"kept: {verdict}")
            rows.append({"reaction": rid.replace("R_", ""), "name": rxn.name,
                         "equation (compartments merged)": _equation(model, rxn, names),
                         "substrate": names[s], "product": names[p], "verdict": verdict, "KEGG lists pair": listed,
                         "decision": action,
                         "a whole-molecule assignment that fits (shown for reactions with an impossible pair)": assign,
                         "chemistry review": review.get((kept_node.get(rid, rid), s, p), "")})
    return pd.DataFrame(rows)


def copies_tab(model, names, sym, final):
    """One row per merged compartment copy: the node it joined."""
    rows = []
    for k, rs in sorted(final.merged_copies.items(), key=lambda kv: final.node_ids[kv[0]]):
        keep = final.base_reaction[k]
        for r in rs:
            rows.append({"kept node": final.node_ids[k].replace("R_", ""), "kept reaction": keep.replace("R_", ""),
                         "name": model.reactions[keep].name, "merged copy": r.replace("R_", ""),
                         "copy's genes": ";".join(sym.get(g, g) for g in model.reactions[r].genes),
                         "kept node's genes (union)": ";".join(sym.get(g, g) for g in final.reaction_genes[k])})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "data/Human-GEM.xml"))
    ap.add_argument("--out", default=str(ROOT / "results/three_rule/rules_audit.xlsx"))
    args = ap.parse_args()

    model = load_sbml(args.model)
    names = {s.met_id: s.name for s in model.species.values()}
    formula = {s.met_id: s.formula for s in model.species.values()}
    carbon = {s.met_id: parse_formula(s.formula)["C"] > 0 for s in model.species.values()}
    sym = model.genes
    subsystems = load_subsystems(ROOT / "data/Human-GEM.yml")
    stages = {s[0]: s for s in structure.stages(model, "three_rule", atom_dir=str(ROOT / "data"))}
    rescued = {x["reaction"]: x for x in structure.KEPT_RAREST_LOG}

    def graph(key):
        s = stages[key]
        return build_graph(model, s[3], s[4], s[5], **s[6])

    raw, final = graph("raw"), graph("three_rule")
    in_raw = set(raw.base_reaction.values())
    links_final = Counter(r for i in range(final.n) if final.is_reaction[i] for r in final.reactions_of(i))

    union = load_atom_pairs(ROOT / "data/atom_pairs_union.json")
    _, couples = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS)
    carriers, carrier_pairs = find_carriers(model, union, ROOT / "data/Human-GEM_metabolites.tsv")
    pools = set(stages["carbon_cleaned"][3].pools) | pool_metabolites(model)
    cargo = transport_reactions(model, carbon_cargo=True)
    balance, unbalanced = carbon_balance(model), carbon_unbalanced(model)
    rule4 = {(x["reaction"], tuple(x["pair"])) for x in structure.CARBON_BUDGET_LOG}
    partner = {}
    for a, b in couples:
        partner.setdefault(a, set()).add(b)
        partner.setdefault(b, set()).add(a)

    def label(p):
        return f"{names[p[0]]} -> {names[p[1]]}"

    reactions, links = [], []
    for rid in sorted(in_raw):
        rxn = model.reactions[rid]
        subs = {model.species[s].met_id for s in rxn.substrates}
        prods = {model.species[s].met_id for s in rxn.products}
        both = subs & prods
        subs, prods = subs - both, prods - both
        row = {"reaction": rid.replace("R_", ""), "name": rxn.name,
               "genes": ";".join(sym.get(g, g) for g in rxn.genes),
               "subsystem": "; ".join(subsystems.get(rid, [])),
               "metabolites": len(subs | prods), "atom_mapped": rid in union}
        why1 = ("biomass reaction" if rxn.is_objective else
                "pool pseudo-metabolite" if (subs | prods | both) & pools else
                "transporter (carbon cargo)" if rid in cargo else "")
        if rid in union:
            pairs = list(dict.fromkeys((a, b) for a, b, *_ in union[rid]))
            r1 = [p for p in pairs if carbon.get(p[0]) and carbon.get(p[1])]
            r2 = [p for p in r1 if tuple(sorted(p)) not in couples]
            r3 = [p for p in r2 if tuple(sorted(p)) not in carrier_pairs]
            counts = {STEPS[1]: len(r1), STEPS[2]: len(r2), STEPS[3]: len(r3)}
            removed_by = {STEPS[1]: [label(p) for p in pairs if p not in r1],
                          STEPS[2]: [f"{label(p)} ({couples[tuple(sorted(p))]})" for p in r1 if p not in r2],
                          STEPS[3]: [f"{label(p)} (carrier: {names[carrier_pairs[tuple(sorted(p))]]})"
                                     for p in r2 if p not in r3]}
            budget = carbon_budget(model, rid, pairs, unbalanced)
            for p in pairs:
                sp = tuple(sorted(p))
                fate = (STEPS[1] if p not in r1 else STEPS[2] if p not in r2 else STEPS[3] if p not in r3 else
                        RULE4 if (rid, p) in rule4 else "kept")
                links.append({"reaction": row["reaction"], "reaction_name": rxn.name, "substrate": names[p[0]],
                              "product": names[p[1]], "removed_by": fate,
                              "detail": (f"couple in {couples[sp]} reactions" if fate == STEPS[2] else
                                         f"carrier: {names[carrier_pairs[sp]]}" if fate == STEPS[3] else
                                         "a partner has no carbon" if fate == STEPS[1] else
                                         "carbon counts rule it out" if fate == RULE4 else ""),
                              "restored_by_safeguard": rid in rescued and rid in links_final
                              and sp == tuple(rescued[rid]["kept_pair"]),
                              "carbon budget": budget[p]})
        else:
            m1 = {m for m in subs | prods if carbon.get(m)}
            hit2 = {m for m in subs if partner.get(m, set()) & prods} | {m for m in prods if partner.get(m, set()) & subs}
            hit3 = {c for c in subs if carriers.get(c, set()) & prods} | {c for c in prods if carriers.get(c, set()) & subs}
            counts = {STEPS[1]: len(m1), STEPS[2]: len(m1 - hit2), STEPS[3]: len(m1 - hit2 - hit3)}
            removed_by = {STEPS[1]: sorted(names[m] for m in (subs | prods) - m1),
                          STEPS[2]: sorted(names[m] for m in m1 & hit2),
                          STEPS[3]: sorted(names[m] for m in (m1 - hit2) & hit3)}
        need = 1 if rid in union else 2          # links for a mapped reaction, metabolites for a whole one
        if why1:
            out_at, why = STEPS[0], why1
        else:
            out_at = next((s for s in STEPS[1:] if counts[s] < need), "")
            why = {STEPS[1]: "no carbon pair" if rid in union else "fewer than 2 carbon molecules",
                   STEPS[2]: "all its links were cofactor couples",
                   STEPS[3]: "all its remaining links were carrier links"}.get(out_at, "")
        saved = rid in rescued and rid in links_final and out_at != STEPS[0]
        row.update({"drops out at": out_at, "why": why, "kept by safeguard": saved,
                    "rarest link kept": label(rescued[rid]["kept_pair"]) if saved else "",
                    "in 3-rule pruning": rid in links_final, "links in 3-rule pruning": links_final.get(rid, 0),
                    "carbon in": round(balance[rid][0], 4), "carbon out": round(balance[rid][1], 4),
                    "carbon unbalanced": rid in unbalanced})
        row.update({f"left after {s}": counts[s] for s in STEPS[1:]})
        row.update({f"removed by {s}": "; ".join(removed_by[s]) for s in STEPS[1:]})
        if not row["in 3-rule pruning"] and (not out_at or rid in rescued):
            row["drops out at"], row["why"] = SIZE, "more than 20 metabolites after the rules"
            row["kept by safeguard"], row["rarest link kept"] = False, ""
        reactions.append(row)
    R = pd.DataFrame(reactions)
    L = pd.DataFrame(links)

    # one tab per step: what it acts on
    raw_deg = Counter()
    for m_i, _ in raw.edges:
        raw_deg[raw.node_ids[m_i]] += 1
    removed_in = Counter()
    for x in links:
        if x["removed_by"] != "kept":
            for nm in (x["substrate"], x["product"]):
                removed_in[(x["removed_by"], nm)] += 1
    P = pd.DataFrame([{"pool pseudo-metabolite": names[m], "reactions in raw network": raw_deg.get(m, 0)}
                      for m in sorted(pools, key=lambda m: (-raw_deg.get(m, 0), names[m]))])
    nocarbon = [n for n in raw.node_ids if not raw.is_reaction[raw.index[n]] and not carbon.get(n)]
    N = pd.DataFrame([{"molecule with no carbon": names[m], "formula": formula.get(m, ""),
                       "reactions in raw network": raw_deg.get(m, 0)}
                      for m in sorted(nocarbon, key=lambda m: (-raw_deg.get(m, 0), names[m]))])
    # what happens to each couple: per reaction it appears in (counted once per reaction)
    fate_of = dict(zip(R.reaction, R["drops out at"]))
    in_final = dict(zip(R.reaction, R["in 3-rule pruning"]))
    by_pair = {}
    for x in links:
        key = (x["reaction"], frozenset((x["substrate"], x["product"])))
        if key in by_pair:
            by_pair[key] = by_pair[key] or x["restored_by_safeguard"]
        else:
            by_pair[key] = x["restored_by_safeguard"]
    k_rows = []
    for (a, b), n in sorted(couples.items(), key=lambda kv: -kv[1]):
        pair = frozenset((names[a], names[b]))
        rxns = [(r, rest) for (r, p), rest in by_pair.items() if p == pair]
        step1 = sum(fate_of[r] == STEPS[0] for r, _ in rxns)
        restored = sum(rest for _, rest in rxns)
        k_rows.append({"partner A": names[a], "partner B": names[b],
                       "reactions the pair appears in (count that makes it a couple)": n,
                       "already gone: not in the raw network": n - len(rxns),
                       "already gone: reaction dropped at Step 1": step1,
                       "link removed by Rule 2": len(rxns) - step1 - restored,
                       "link kept by safeguard": restored})
    K = pd.DataFrame(k_rows)
    links_by = {}
    for x in links:
        links_by.setdefault((x["reaction"], frozenset((x["substrate"], x["product"]))), []).append(x)
    C = pd.DataFrame([{"carrier": names[c], "loaded forms": len(x),
                       "links removed": sum(1 for (r, pr), rest in by_pair.items()
                                            if names[c] in pr and not rest and fate_of[r] != STEPS[0]
                                            and any(x["removed_by"] == STEPS[3] for x in links_by[(r, pr)])),
                       "examples of loaded forms": "; ".join(sorted(names[y] for y in x)[:8])}
                      for c, x in sorted(carriers.items(), key=lambda kv: -len(kv[1])) if c not in pools])
    # summary: one row per step, start and end
    what = {STEPS[0]: ("Whole reactions: biomass, reactions with a pool pseudo-metabolite, transporters carrying "
                       "a carbon-containing molecule", "Step 1 pools; Reactions (filter 'why')"),
            STEPS[1]: ("Links to molecules with no carbon (water, H+, phosphate, O2, ions)", "Rule 1 no carbon"),
            STEPS[2]: ("Links between cofactor couples: pairs found in 10 or more reactions (ATP/ADP, NAD+/NADH)",
                       "Rule 2 couples"),
            STEPS[3]: ("Links between a carrier and its loaded forms (CoA and acyl-CoAs, carnitine and "
                       "acylcarnitines)", "Rule 3 carriers"),
            RULE4: ("Links the reaction's own carbon counts rule out (transaminase cross-pairs, ATP to the "
                    "aminoacyl-tRNA), unless KEGG lists the pair or it is the reaction's last pair",
                    "Rule 4 carbon budget"),
            SIZE: ("Whole reactions still listing more than 20 metabolites after the rules (protein-synthesis "
                   "pseudo-reactions)", "Reactions (filter 'drops out at')"),
            MERGE: ("Nothing: reactions identical once compartments are merged share one node, with the union of "
                    "their genes", "Compartment copies")}
    def links_removed(step):
        """Links (one per reaction and pair) the step removes from reactions that survived Step 1,
        not counting links the keep-rarest safeguard puts back."""
        seen = {(x["reaction"], frozenset((x["substrate"], x["product"])))
                for x in links if x["removed_by"] == step and fate_of[x["reaction"]] != STEPS[0]}
        return sum(not by_pair[k] for k in seen)

    alive = set().union(*[set(g.split(";")) for g in R.genes if g])
    still = set().union(*[set(g.split(";")) for g in R[R["in 3-rule pruning"]].genes if g])
    genes_at, left = {}, len(R)
    summary = [{"step": "Start: raw Human-GEM", "what it removes": "", "reactions dropped": None,
                "reactions saved by safeguard": None, "links removed": None, "genes that leave the network": None,
                "reactions left": left, "details on tab": "Reactions"}]
    for s in STEPS + [RULE4, SIZE]:
        out_s = R[(R["drops out at"] == s) & ~R["kept by safeguard"]]
        saved = R[(R["drops out at"] == s) & R["kept by safeguard"]]
        gone = set().union(*[set(g.split(";")) for g in out_s.genes if g]) if len(out_s) else set()
        genes_at[s] = sorted((gone - still) & alive)
        left -= len(out_s)
        summary.append({"step": s, "what it removes": what[s][0], "reactions dropped": len(out_s),
                        "reactions saved by safeguard": len(saved) or None,
                        "links removed": links_removed(s) if s in STEPS[1:] + [RULE4] else None,
                        "genes that leave the network": len(genes_at[s]), "reactions left": left,
                        "details on tab": what[s][1]})
    summary.append({"step": "End: 3-rule pruning", "what it removes": "", "reactions dropped": None,
                    "reactions saved by safeguard": int(R["kept by safeguard"].sum()), "links removed": None,
                    "genes that leave the network": None, "reactions left": int(R["in 3-rule pruning"].sum()),
                    "details on tab": "Reactions"})
    n_merged = sum(len(v) for v in final.merged_copies.values())
    summary.insert(-1, {"step": MERGE, "what it removes": what[MERGE][0], "reactions dropped": None,
                        "reactions saved by safeguard": None, "links removed": None,
                        "genes that leave the network": 0, "reactions left": int(R["in 3-rule pruning"].sum()),
                        "details on tab": what[MERGE][1],
                        "note": f"{n_merged} reaction nodes merged into {len(final.merged_copies)}; reactions "
                                "still count as present"})
    S = pd.DataFrame(summary)
    G = pd.DataFrame([{"step": s, "gene": g} for s in genes_at for g in genes_at[s]])
    SG, B4, CP = safeguard_tab(model, names, sym), rule4_tab(model, names, union, carrier_pairs, unbalanced,
                                                            R, final), copies_tab(model, names, sym, final)

    # ---- before vs after: raw network against 3-rule pruning -------------
    def describe(g):
        deg = Counter()
        for a_i, b_i in g.edges:
            deg[a_i] += 1
            deg[b_i] += 1
        lab = g.components()
        main_lab = Counter(lab[i] for i in range(g.n) if not g.is_reaction[i]).most_common(1)[0][0]
        mets = [i for i in range(g.n) if not g.is_reaction[i]]
        rxn_nodes = [i for i in range(g.n) if g.is_reaction[i]]
        per_met = {g.node_ids[i]: (deg[i], lab[i] == main_lab) for i in mets}
        sides = {}
        for (a_i, b_i), side in zip(g.edges, g.edge_side):
            m_i, r_i = (a_i, b_i) if g.is_reaction[b_i] else (b_i, a_i)
            sides.setdefault(r_i, {"substrate": [], "product": []})[side].append(g.node_names[m_i])
        per_rxn = {}
        for r_i in rxn_nodes:
            s = sides.get(r_i, {"substrate": [], "product": []})
            for rid in g.reactions_of(r_i):
                per_rxn.setdefault(rid, []).append(
                    " + ".join(sorted(s["substrate"])) + " -> " + " + ".join(sorted(s["product"])))
        met_rxns = {}
        for a_i, b_i in g.edges:
            m_i, r_i = (a_i, b_i) if g.is_reaction[b_i] else (b_i, a_i)
            met_rxns.setdefault(g.node_ids[m_i], set()).update(g.reactions_of(r_i))
        gene_rxns = {}
        for r_i in rxn_nodes:
            for gg in g.reaction_genes[r_i]:
                gene_rxns.setdefault(sym.get(gg, gg), set()).update(g.reactions_of(r_i))
        return {"deg": {k: v[0] for k, v in per_met.items()}, "main": {k: v[1] for k, v in per_met.items()},
                "met_rxns": met_rxns, "rxn": per_rxn, "genes": gene_rxns, "n_edges": len(g.edges),
                "n_rxn_nodes": len(rxn_nodes), "mets": len(mets)}

    before, after = describe(raw), describe(final)
    xref = {}
    for sp in model.species.values():
        xref.setdefault(sp.met_id, sp.xrefs)

    def met_status(m):
        if m in after["deg"]:
            return "kept" if after["main"][m] else "kept, but cut off from the main network"
        if not carbon.get(m):
            return "removed: no carbon (Rule 1)"
        if m in pools:
            return "removed: pool pseudo-metabolite (Step 1)"
        return "removed: every reaction it was in dropped out or lost this link"

    MB = pd.DataFrame([{
        "metabolite": names[m],
        "KEGG": "; ".join(xref.get(m, {}).get("kegg", []) or []),
        "HMDB": "; ".join(xref.get(m, {}).get("hmdb", []) or []),
        "reactions, raw": len(before["met_rxns"].get(m, ())),
        "reactions, revised": len(after["met_rxns"].get(m, ())),
        "links, revised": after["deg"].get(m, 0),
        "change in reactions": len(after["met_rxns"].get(m, ())) - len(before["met_rxns"].get(m, ())),
        "in 3-rule pruning": m in after["deg"],
        "status": met_status(m)} for m in before["deg"]]).sort_values(
        ["reactions, raw", "metabolite"], ascending=[False, True])
    RB = pd.DataFrame([{
        "reaction": rid.replace("R_", ""), "name": model.reactions[rid].name,
        "genes": ";".join(sym.get(g_, g_) for g_ in model.reactions[rid].genes),
        "raw (whole reaction, all metabolites)": "; ".join(before["rxn"].get(rid, [])),
        "revised (carbon links)": "; ".join(after["rxn"].get(rid, [])),
        "links, revised": len(after["rxn"].get(rid, [])),
        "status": ("kept by safeguard" if r_kept else "kept" if rid in after["rxn"] else f"dropped: {r_out}")}
        for rid, r_kept, r_out in zip(sorted(in_raw), R["kept by safeguard"], R["drops out at"])])
    GB = pd.DataFrame([{
        "gene": gname, "reactions, raw": len(before["genes"].get(gname, ())),
        "reactions, revised": len(after["genes"].get(gname, ())),
        "status": ("kept" if len(after["genes"].get(gname, ())) == len(before["genes"][gname]) else
                   "kept, fewer reactions" if gname in after["genes"] else
                   "leaves the network: " + next((s for s in genes_at if gname in genes_at[s]), "reactions dropped"))}
        for gname in sorted(before["genes"])])

    def top_hubs(d, n=25):
        return [(names[m], k) for m, k in sorted(((m, len(r)) for m, r in d["met_rxns"].items()),
                                                 key=lambda x: -x[1])[:n]]
    totals = pd.DataFrame([
        {"measure": "Metabolites", "raw Human-GEM": before["mets"], "3-rule pruning": after["mets"]},
        {"measure": "Human-GEM reactions represented", "raw Human-GEM": len(before["rxn"]),
         "3-rule pruning": len(after["rxn"])},
        {"measure": "Reaction nodes (a split reaction counts once per carbon link)",
         "raw Human-GEM": before["n_rxn_nodes"], "3-rule pruning": after["n_rxn_nodes"]},
        {"measure": "Edges (metabolite to reaction node)", "raw Human-GEM": before["n_edges"],
         "3-rule pruning": after["n_edges"]},
        {"measure": "Genes", "raw Human-GEM": len(before["genes"]), "3-rule pruning": len(after["genes"])},
        {"measure": "Metabolites cut off from the main network",
         "raw Human-GEM": sum(not v for v in before["main"].values()),
         "3-rule pruning": sum(not v for v in after["main"].values())},
        {"measure": "Metabolites in 100+ reactions",
         "raw Human-GEM": sum(len(v) >= 100 for v in before["met_rxns"].values()),
         "3-rule pruning": sum(len(v) >= 100 for v in after["met_rxns"].values())},
    ])
    hb, ha = top_hubs(before), top_hubs(after)
    hubs = pd.DataFrame({"rank": range(1, len(hb) + 1),
                         "raw Human-GEM: metabolite": [x[0] for x in hb], "raw: reactions": [x[1] for x in hb],
                         "3-rule pruning: metabolite": [x[0] for x in ha], "revised: reactions": [x[1] for x in ha]})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        notes = pd.DataFrame({"About this workbook": [
            "Every chemical reaction of raw Human-GEM (compartments merged, 7,080 reactions) followed through the "
            "revised carbon method. The steps run in order: Step 1 drops whole reactions (pools, biomass, "
            "transporters); Rules 1, 2 and 3 remove links inside reactions; a reaction left with no link drops out, "
            "unless the keep-rarest safeguard keeps its rarest link; Rule 4 removes links the carbon counts rule "
            "out; reactions still listing more than 20 metabolites drop out; finally reactions identical once "
            "compartments are merged share one node.",
            "Before vs after: the raw network and 3-rule pruning side by side (totals and the 25 biggest "
            "hubs). Metabolites / Reactions / Genes before-after: every member of the raw network, how many "
            "reactions it is in before and after, and what happened to it. Reactions before-after writes each "
            "reaction out as in raw Human-GEM and as the carbon links that remain.",
            "The tabs after these give the detail for each step of the method.",
            "Summary: one row per step, with the tab that lists what that step removes.",
            "Step 1 pools / Rule 1 no carbon / Rule 2 couples / Rule 3 carriers: the molecules or pairs each step "
            "acts on.",
            "Reactions: one row per reaction. 'drops out at' is the first step that leaves it with nothing; 'kept by "
            "safeguard' marks those saved anyway. 'left after' counts links for atom-mapped reactions and carbon "
            "molecules for reactions with no atom map (kept whole). 'carbon in' / 'carbon out' count carbons from "
            "the formulas times the stoichiometry; 'carbon unbalanced' marks reactions where they differ, in which "
            "carbon transfer between two molecules is not well defined.",
            "Links: every carbon pair of the atom-mapped reactions and the step that removes it. 'carbon budget' "
            "is Rule 4's verdict (impossible / possible / inconclusive / unbalanced); Rule 4 removes an impossible "
            "pair unless KEGG lists it or it is the reaction's last pair.",
            "Safeguard: every reaction Rules 2-3 left with no link and that the safeguard kept. It puts back one "
            "of the reaction's removed pairs: the one found in the fewest reactions network-wide (the rarest, so "
            "the least likely to be cofactor bookkeeping); ties go to the lower metabolite id. All candidates are "
            "listed with their counts. Reactions that carry anything across a membrane are never rescued.",
            "Rule 4 carbon budget: every pair Rule 4 judged in the final network. It assumes each substrate "
            "molecule's carbon goes whole into one product molecule, and looks for ways to assign the substrates "
            "(stoichiometry counted) to products so that carbon counts match exactly. A pair no such assignment "
            "uses is 'impossible'; 'inconclusive' if no assignment exists at all (a substrate must split); "
            "'unbalanced' if carbon in and out differ. For reactions with an impossible pair, one fitting "
            "assignment is shown. 'chemistry review' is a by-hand judgement of the 48 flagged pairs: 9 are real "
            "chemistry (group transfers: SCOT, GALT, BHMT, AGPS) and are kept by the two guards.",
            "Compartment copies: reactions identical once compartments are merged; each copy joins the kept node, "
            "whose genes are the union.",
            "Genes lost: genes whose every reaction drops out, by step.",
            "Contains KEGG-derived pairs: do not post publicly."]})
        notes.to_excel(xw, sheet_name="About", index=False)
        totals.to_excel(xw, sheet_name="Before vs after", index=False)
        hubs.to_excel(xw, sheet_name="Before vs after", index=False, startrow=len(totals) + 3)
        MB.to_excel(xw, sheet_name="Metabolites before-after", index=False)
        RB.to_excel(xw, sheet_name="Reactions before-after", index=False)
        GB.to_excel(xw, sheet_name="Genes before-after", index=False)
        S.to_excel(xw, sheet_name="Summary", index=False)
        P.to_excel(xw, sheet_name="Step 1 pools", index=False)
        N.to_excel(xw, sheet_name="Rule 1 no carbon", index=False)
        K.to_excel(xw, sheet_name="Rule 2 couples", index=False)
        C.to_excel(xw, sheet_name="Rule 3 carriers", index=False)
        R.to_excel(xw, sheet_name="Reactions", index=False)
        L.to_excel(xw, sheet_name="Links", index=False)
        SG.to_excel(xw, sheet_name="Safeguard", index=False)
        B4.to_excel(xw, sheet_name="Rule 4 carbon budget", index=False)
        CP.to_excel(xw, sheet_name="Compartment copies", index=False)
        G.to_excel(xw, sheet_name="Genes lost", index=False)
        for ws in xw.book.worksheets:
            ws.freeze_panes = "A2"
            if ws.title != "Before vs after":
                ws.auto_filter.ref = ws.dimensions
            for col in ws.columns:
                width = max(len(str(c.value or "")) for c in list(col)[:200])
                ws.column_dimensions[col[0].column_letter].width = min(max(10, width + 2), 60)
        xw.book["About"].column_dimensions["A"].width = 140
        xw.book["Summary"].column_dimensions["B"].width = 70
        for sheet in ("About", "Summary"):
            for row in xw.book[sheet].iter_rows():
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
    print(S.drop(columns="what it removes").to_string(index=False))
    mismatch = R[(R["drops out at"].eq("") | R["kept by safeguard"]) != R["in 3-rule pruning"]]
    print(f"\nreactions: {len(R)}; links: {len(L)}; check mismatches: {len(mismatch)}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
