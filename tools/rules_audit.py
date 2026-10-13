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
from metaboenrich.atoms import (COUPLE_MIN_REACTIONS, carbon_method_pairs, find_carriers,  # noqa: E402
                                load_atom_pairs, pool_metabolites, transport_reactions)
from metaboenrich.gem import load_sbml, load_subsystems              # noqa: E402
from metaboenrich.network import build_graph, parse_formula          # noqa: E402

STEPS = ["Step 1 (pools, biomass, transporters)", "Rule 1 (carbon)", "Rule 2 (couples)", "Rule 3 (carriers)"]
SIZE = "Too large (over 20 metabolites)"


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
    links_final = Counter(final.base_reaction[i] for i in range(final.n) if final.is_reaction[i])

    union = load_atom_pairs(ROOT / "data/atom_pairs_union.json")
    _, couples = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS)
    carriers, carrier_pairs = find_carriers(model, union, ROOT / "data/Human-GEM_metabolites.tsv")
    pools = set(stages["carbon_cleaned"][3].pools) | pool_metabolites(model)
    cargo = transport_reactions(model, carbon_cargo=True)
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
            for p in pairs:
                sp = tuple(sorted(p))
                fate = (STEPS[1] if p not in r1 else STEPS[2] if p not in r2 else STEPS[3] if p not in r3 else "kept")
                links.append({"reaction": row["reaction"], "reaction_name": rxn.name, "substrate": names[p[0]],
                              "product": names[p[1]], "removed_by": fate,
                              "detail": (f"couple in {couples[sp]} reactions" if fate == STEPS[2] else
                                         f"carrier: {names[carrier_pairs[sp]]}" if fate == STEPS[3] else
                                         "a partner has no carbon" if fate == STEPS[1] else ""),
                              "restored_by_safeguard": rid in rescued and rid in links_final
                              and sp == tuple(rescued[rid]["kept_pair"])})
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
                    "in 3-rule pruning": rid in links_final, "links in 3-rule pruning": links_final.get(rid, 0)})
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
            SIZE: ("Whole reactions still listing more than 20 metabolites after the rules (protein-synthesis "
                   "pseudo-reactions)", "Reactions (filter 'drops out at')")}
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
    for s in STEPS + [SIZE]:
        out_s = R[(R["drops out at"] == s) & ~R["kept by safeguard"]]
        saved = R[(R["drops out at"] == s) & R["kept by safeguard"]]
        gone = set().union(*[set(g.split(";")) for g in out_s.genes if g]) if len(out_s) else set()
        genes_at[s] = sorted((gone - still) & alive)
        left -= len(out_s)
        summary.append({"step": s, "what it removes": what[s][0], "reactions dropped": len(out_s),
                        "reactions saved by safeguard": len(saved) or None,
                        "links removed": links_removed(s) if s in STEPS[1:] else None,
                        "genes that leave the network": len(genes_at[s]), "reactions left": left,
                        "details on tab": what[s][1]})
    summary.append({"step": "End: 3-rule pruning", "what it removes": "", "reactions dropped": None,
                    "reactions saved by safeguard": int(R["kept by safeguard"].sum()), "links removed": None,
                    "genes that leave the network": None, "reactions left": int(R["in 3-rule pruning"].sum()),
                    "details on tab": "Reactions"})
    S = pd.DataFrame(summary)
    G = pd.DataFrame([{"step": s, "gene": g} for s in genes_at for g in genes_at[s]])

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
            per_rxn.setdefault(g.base_reaction[r_i], []).append(
                " + ".join(sorted(s["substrate"])) + " -> " + " + ".join(sorted(s["product"])))
        met_rxns = {}
        for a_i, b_i in g.edges:
            m_i, r_i = (a_i, b_i) if g.is_reaction[b_i] else (b_i, a_i)
            met_rxns.setdefault(g.node_ids[m_i], set()).add(g.base_reaction[r_i])
        gene_rxns = {}
        for r_i in rxn_nodes:
            for gg in g.reaction_genes[r_i]:
                gene_rxns.setdefault(sym.get(gg, gg), set()).add(g.base_reaction[r_i])
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
            "unless the keep-rarest safeguard keeps its rarest link; finally reactions still listing more than 20 "
            "metabolites drop out.",
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
            "molecules for reactions with no atom map (kept whole).",
            "Links: every carbon pair of the atom-mapped reactions and the step that removes it.",
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
