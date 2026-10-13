"""Build the Carbon Channel Atlas page: the carbon method (Rules 1 + 2), with
the first method (the Pruning Atlas) for reference.

Needs, from the repository root:
    python -m metaboenrich.atoms                          # carbon pairs (data/atom_pairs*.json)
    python -m metaboenrich.structure --stage-set carbon --out results/structure_carbon
    python -m metaboenrich.stage_diagrams --structure results/structure_carbon/structure.json \
        --out results/carbon_diagrams
    benchmark runs (see BENCHMARKS below)

Then:
    python tools/atlas/build_carbon_atlas.py --id-translation PATH/TO/id_translation.csv \
        --out docs/carbon_atlas.html
"""
import argparse
import html
import json
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from build_atlas import build_html                        # noqa: E402
from metaboenrich.atoms import (COUPLE_MIN_REACTIONS, carbon_method_pairs, carbon_rule1_pairs, evaluate_rules,  # noqa: E402
                                kegg_pairs_for_model, load_atom_pairs, load_smiles, reaction_sides)
from metaboenrich.gem import load_sbml                    # noqa: E402
from metaboenrich.network import parse_formula            # noqa: E402

e = html.escape
# network label -> group-level benchmark folder (results/...)
BENCHMARKS = [
    ("1. Raw Human-GEM", "benchmark_stage/pruning_raw"),
    ("2. Rule 1: follow carbon", "benchmark_stage/carbon_carbon_rule1"),
    ("<b>3. Rules 1 + 2: the carbon method</b>", "benchmark_stage/carbon_carbon_method"),
    ("&nbsp;&nbsp;&nbsp;same, couple = pair in 5+ reactions", "benchmark_stage/carbon_carbon_method_n5"),
    ("&nbsp;&nbsp;&nbsp;same, couple = pair in 20+ reactions", "benchmark_stage/carbon_carbon_method_n20"),
    ("4. Carbon method, mapped reactions only", "benchmark_stage/carbon_carbon_mapped_only"),
    ("Reference: first method", "benchmark_v02"),
    ("Reference: first method + carbon split", "benchmark_atoms_union"),
]
REFERENCE = "benchmark_v02"


def table(headers, rows, num=()):
    th = "".join(f'<th class="{"num" if i in num else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    tr = "".join("<tr>" + "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(r)) + "</tr>"
                 for r in rows)
    return f'<div class="table-scroll"><table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table></div>'


def example_svg():
    """Aspartate aminotransferase: one reaction node vs two carbon links."""
    def node(x, y, label, anchor="end"):
        dx = -12 if anchor == "end" else 12
        return (f'<circle cx="{x}" cy="{y}" r="7" fill="var(--p2)"></circle>'
                f'<text x="{x + dx}" y="{y + 4}" text-anchor="{anchor}" class="ex-l">{e(label)}</text>')

    def rxn(x, y):
        return f'<rect x="{x - 7}" y="{y - 7}" width="14" height="14" rx="2" fill="var(--ink-2)"></rect>'

    def ln(x1, y1, x2, y2):
        return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="var(--axis)" stroke-width="2"></line>'

    left = (ln(120, 60, 200, 95) + ln(120, 130, 200, 95) + ln(200, 95, 280, 60) + ln(200, 95, 280, 130) + rxn(200, 95)
            + node(120, 60, "2-oxoglutarate") + node(120, 130, "aspartate")
            + node(280, 60, "glutamate", "start") + node(280, 130, "oxaloacetate", "start")
            + '<text x="200" y="24" text-anchor="middle" class="ex-h">Human-GEM: one reaction node</text>'
            + '<text x="200" y="172" text-anchor="middle" class="ex-n">current can flow aspartate → glutamate</text>')
    right = (ln(540, 60, 620, 60) + ln(620, 60, 700, 60) + ln(540, 130, 620, 130) + ln(620, 130, 700, 130)
             + rxn(620, 60) + rxn(620, 130)
             + node(540, 60, "2-oxoglutarate") + node(540, 130, "aspartate")
             + node(700, 60, "glutamate", "start") + node(700, 130, "oxaloacetate", "start")
             + '<text x="620" y="24" text-anchor="middle" class="ex-h">Rule 1: one link per carbon skeleton</text>'
             + '<text x="620" y="172" text-anchor="middle" class="ex-n">only the amino group moves between them</text>')
    return (f'<figure class="example"><svg viewBox="0 0 820 185" role="img" aria-label="Aspartate aminotransferase '
            f'as one reaction node and as two carbon links">{left}{right}</svg>'
            '<figcaption>Aspartate aminotransferase (GOT1/GOT2). Squares are reactions, circles metabolites. Both '
            'links carry the enzyme\'s genes.</figcaption></figure>')


SHOW_NAMES = {"AKG": "2-oxoglutarate", "OAA": "oxaloacetate", "L-lactate": "lactate",
              "fructose-6-phosphate": "fructose-6-P", "glucose-6-phosphate": "glucose-6-P"}
COUNT_EXAMPLES = [  # pairs shown in the Rule 2 counting chart
    ("NADP+", "NADPH"), ("NAD+", "NADH"), ("ADP", "ATP"), ("acetyl-CoA", "CoA"), ("AKG", "glutamate"),
    ("CoA", "palmitoyl-CoA"), ("CoA", "lauroyl-CoA"),
    ("alanine", "pyruvate"), ("UDP", "UDP-glucose"), ("ubiquinol", "ubiquinone"), ("L-lactate", "pyruvate"),
    ("aspartate", "OAA"), ("fructose-6-phosphate", "glucose-6-phosphate"),
]


def rule2_count_svg(freq, names, n_min):
    """Rule 2 itself: carbon pairs ranked by the number of reactions they appear in, with the cutoff."""
    by_name = {frozenset((names[a], names[b])): n for (a, b), n in freq.items()}
    rows = [(a, b, by_name[frozenset((a, b))]) for a, b in COUNT_EXAMPLES if frozenset((a, b)) in by_name]
    rows.sort(key=lambda r: -r[2])
    x0, x1, top, rh = 262, 700, 44, 21
    X = lambda v: x0 + (x1 - x0) * math.log10(v) / 3       # 1 .. 1000 reactions
    out = []
    bottom = top + rh * len(rows)
    for t in (1, 10, 100, 1000):
        out.append(f'<line x1="{X(t):.1f}" y1="{top - 6}" x2="{X(t):.1f}" y2="{bottom}" stroke="var(--grid)" stroke-width="1"></line>'
                   f'<text x="{X(t):.1f}" y="{bottom + 16}" text-anchor="middle" class="ex-t">{t:,}</text>')
    for k, (a, b, n) in enumerate(rows):
        y = top + k * rh
        couple = n >= n_min
        label = f"{SHOW_NAMES.get(a, a)} / {SHOW_NAMES.get(b, b)}"
        w = max(X(n) - x0, 3)
        out.append(f'<g><title>{e(label)}: {n:,} reaction{"s" if n != 1 else ""} ({"cofactor couple, removed" if couple else "kept"})</title>'
                   f'<text x="{x0 - 10}" y="{y + 14}" text-anchor="end" class="ex-l">{e(label)}</text>'
                   f'<rect x="{x0}" y="{y + 4}" width="{w:.1f}" height="13" rx="3" '
                   f'fill="{"var(--ink-2)" if couple else "var(--p2)"}"></rect>'
                   f'<text x="{x0 + w + 6:.1f}" y="{y + 14}" class="ex-v">{n:,}</text></g>')
    xc = X(n_min)
    out.append(f'<line x1="{xc:.1f}" y1="{top - 14}" x2="{xc:.1f}" y2="{bottom + 2}" stroke="var(--ink)" '
               f'stroke-width="1.5" stroke-dasharray="5 4"></line>'
               f'<text x="{xc + 6:.1f}" y="{top - 18}" class="ex-h">cutoff: {n_min} reactions</text>')
    n_above = sum(1 for _, _, n in rows if n >= n_min)
    ymid_a, ymid_b = top + rh * n_above / 2, top + rh * (n_above + len(rows)) / 2
    out.append(f'<text x="{x1 + 30}" y="{ymid_a:.0f}" class="ex-h">cofactor couple</text>'
               f'<text x="{x1 + 30}" y="{ymid_a + 16:.0f}" class="ex-n">removed everywhere</text>'
               f'<text x="{x1 + 30}" y="{ymid_b:.0f}" class="ex-h">kept</text>'
               f'<text x="{x1 + 30}" y="{ymid_b + 16:.0f}" class="ex-n">carbon link stays</text>'
               f'<line x1="{x0 - 250}" y1="{top + rh * n_above:.1f}" x2="{x1 + 150}" y2="{top + rh * n_above:.1f}" '
               f'stroke="var(--border)" stroke-width="1"></line>')
    out.append(f'<text x="{(x0 + x1) / 2}" y="{bottom + 34}" text-anchor="middle" class="ex-n">'
               'number of reactions the pair appears in (log scale)</text>')
    h = bottom + 44
    total, above = len(freq), sum(1 for n in freq.values() if n >= n_min)
    once = sum(1 for n in freq.values() if n == 1)
    alt = {c: sum(1 for n in freq.values() if n >= c) for c in (5, 20)}
    return (f'<figure class="example"><svg viewBox="0 0 860 {h}" role="img" aria-label="Carbon pairs ranked by how many '
            f'reactions they appear in, with the cutoff at {n_min}">{"".join(out)}</svg>'
            f'<figcaption><b>Rule 2 is a count.</b> After Rule 1 there are {total:,} distinct carbon pairs. Each is counted by '
            f'the number of reactions it appears in. The {above} pairs at {n_min} or more are cofactor couples and are removed '
            f'everywhere; the other {total - above:,} are kept ({once:,} of them appear in only one reaction). The chart shows '
            'examples. Just below the line, UDP / UDP-glucose and ubiquinol / ubiquinone behave like cofactors but are kept, '
            f'and alanine / pyruvate is real chemistry, kept correctly. Above it, 2-oxoglutarate / glutamate, the amino-group carrier of transaminases, is removed as a couple. The cutoff matters: at 5, {alt[5]} pairs would count as '
            f'couples; at 20, {alt[20]}.</figcaption></figure>')


def rule2_svg(couples, names):
    """Lactate dehydrogenase: after Rule 1 (two carbon links, one a cofactor couple) and after Rule 2."""
    nad = next((n for (a, b), n in couples.items() if {names[a], names[b]} == {"NAD+", "NADH"}), 0)

    def node(x, y, label, anchor="end", faint=False):
        dx = -12 if anchor == "end" else 12
        fill, cls = ("var(--grid)", "ex-l ex-faint") if faint else ("var(--p2)", "ex-l")
        return (f'<circle cx="{x}" cy="{y}" r="7" fill="{fill}"></circle>'
                f'<text x="{x + dx}" y="{y + 4}" text-anchor="{anchor}" class="{cls}">{e(label)}</text>')

    def rxn(x, y, size=14, faint=False):
        h = size / 2
        return (f'<rect x="{x - h}" y="{y - h}" width="{size}" height="{size}" rx="2" '
                f'fill="{"var(--grid)" if faint else "var(--ink-2)"}"></rect>')

    def ln(x1, y1, x2, y2, faint=False):
        dash = ' stroke-dasharray="4 4"' if faint else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{"var(--grid)" if faint else "var(--axis)"}" '
                f'stroke-width="2"{dash}></line>')

    # left: after Rule 1, NADH also reaches the other reactions that use the couple
    fan = "".join(ln(300, 130, 352, 100 + k * 15) + rxn(352, 100 + k * 15, 8) for k in range(5))
    left = (ln(80, 60, 160, 60) + ln(160, 60, 240, 60) + ln(80, 130, 160, 130) + ln(160, 130, 240, 130)
            + fan + rxn(160, 60) + rxn(160, 130)
            + node(80, 60, "lactate") + node(80, 130, "NAD+")
            + node(240, 60, "pyruvate", "start") + node(240, 130, "NADH", "start")
            + f'<text x="364" y="96" class="ex-n" text-anchor="start">{nad - 1:,} other</text>'
            + '<text x="364" y="110" class="ex-n" text-anchor="start">reactions</text>'
            + '<text x="200" y="24" text-anchor="middle" class="ex-h">After Rule 1: two carbon links</text>'
            + f'<text x="200" y="172" text-anchor="middle" class="ex-n">NAD+ / NADH pair appears in {nad:,} reactions</text>')
    right = (ln(540, 60, 620, 60) + ln(620, 60, 700, 60) + ln(540, 130, 620, 130, True) + ln(620, 130, 700, 130, True)
             + rxn(620, 60) + rxn(620, 130, faint=True)
             + node(540, 60, "lactate") + node(540, 130, "NAD+", faint=True)
             + node(700, 60, "pyruvate", "start") + node(700, 130, "NADH", "start", faint=True)
             + '<text x="620" y="24" text-anchor="middle" class="ex-h">Rule 2: the cofactor couple is removed</text>'
             + '<text x="620" y="172" text-anchor="middle" class="ex-n">only the lactate skeleton is followed</text>')
    return (f'<figure class="example"><svg viewBox="0 0 820 185" role="img" aria-label="Lactate dehydrogenase '
            f'before and after removing the NAD+/NADH cofactor couple">{left}{right}</svg>'
            '<figcaption><b>What removal does.</b> Lactate dehydrogenase (LDHA/LDHB). After Rule 1, NAD+ → NADH is a genuine carbon link, but the '
            f'same pair turns up in {nad:,} reactions. Left in, it would let current run from lactate to any of them. '
            'Rule 2 removes it everywhere; faint dashed items are removed.</figcaption></figure>')


def recipe_table(structure):
    """How each network is made, one row per network, with its size."""
    S = {st["key"]: st["summary"] for st in json.loads(Path(structure).read_text())["stages"]}
    Y, N = "yes", "–"
    rows = [
        ("raw", "1. Raw Human-GEM", N, N, N, N, "–"),
        ("carbon_rule1", "2. Rule 1: follow carbon", Y, N, Y, "yes (no carbon)", "kept whole"),
        ("carbon_method", "<b>3. Rules 1 + 2: the carbon method</b>", Y, Y, Y, "yes (no carbon)", "kept whole"),
        ("carbon_mapped_only", "4. Carbon method, mapped reactions only", Y, Y, Y, "yes (no carbon)", "dropped"),
        ("first_method", "Reference: first method", N, N, Y, "hand-made lists", "–"),
        ("combined", "Reference: first method + carbon split", Y, N, Y, "hand-made lists", "kept whole"),
    ]
    out = [[label, split, rule2, pseudo, small, unm, f"{S[key]['metabolites']:,}", f"{S[key]['reactions']:,}"]
           for key, label, split, rule2, pseudo, small, unm in rows]
    return table(["Network", "Split into carbon links (Rule 1)", "Cofactor couples removed (Rule 2)",
                  "Biomass and pool pseudo-reactions dropped", "Water, H+, ions, cofactors removed",
                  "Reactions with no atom map", "Metabolites", "Reaction links"], out, num=(6, 7))


def method_section(model, couples, n_min, structure, union):
    names = {s.met_id: s.name for s in model.species.values()}
    freq = {}
    for ps in carbon_rule1_pairs(model, union).values():
        for p in {tuple(sorted(p)) for p in ps}:
            freq[p] = freq.get(p, 0) + 1
    chips = "".join(f'<li>{e(names[a])} / {e(names[b])} <span class="n">{n}</span></li>'
                    for (a, b), n in sorted(couples.items(), key=lambda kv: -kv[1]))
    return f"""<section class="method" aria-label="How the carbon method works">
    <h2>How the carbon method works</h2>
    <ol class="steps">
      <li><b>Start from raw Human-GEM</b> (compartments merged) and drop the biomass and pool pseudo-reactions, which are
        bookkeeping, not chemistry.</li>
      <li><b>Find where each substrate's carbon atoms go.</b> KEGG's curated substrate–product pairs where KEGG has the
        reaction, plus atom maps computed with RXNMapper from Human-GEM's own structures elsewhere. A computed link is kept
        only if the substrate is the product's main carbon source, or the product is the substrate's main carbon
        destination (accuracy below).</li>
      <li><b>Rule 1: split each reaction into one link per carbon-sharing pair.</b> Pairs where either molecule has no
        carbon are dropped, so water, H+, phosphate, O2, NH3 and ions link nothing. The enzyme's genes attach to all of
        its links.</li>
      <li><b>Rule 2: remove cofactor couples.</b> Count how many reactions each carbon pair appears in. A pair found in
        {n_min} or more reactions is a cofactor couple and is removed everywhere; {len(couples)} couples are found this
        way (list below). A reaction whose only pairs were couples is dropped.</li>
      <li><b>Reactions with no atom map</b> (a structure is missing, or the reaction is too large for the mapper) are kept
        whole, with the two rules applied from chemical formulas: carbon-free molecules removed, and a couple removed
        when one partner is a substrate and the other a product.</li>
    </ol>
    {example_svg()}
    {rule2_count_svg(freq, names, n_min)}
    {rule2_svg(couples, names)}
    <h2>How each network on this page is made</h2>
    <p class="desc-wide">Networks 1–3 are the steps of the carbon method; network 3 is the method itself. Network 4
    checks what the reactions kept whole contribute. The two reference networks come from the first method (the Pruning
    Atlas) and use its hand-made lists; they are here only for comparison. Charts and diagrams below cover all six.</p>
    {recipe_table(structure)}
    <details class="couples"><summary>The {len(couples)} cofactor couples Rule 2 detected (number of reactions each
      pair appears in)</summary><ul>{chips}</ul></details>
  </section>"""


def evidence_section(model, raw, kegg, method_pairs, bench_note=""):
    carbon_free = {s.met_id for s in model.species.values() if parse_formula(s.formula)["C"] == 0}
    rows = evaluate_rules(model, raw, kegg, carbon_free)
    acc = table(["Cleaning rule", "Reactions", "Links kept that KEGG confirms", "KEGG links recovered", "Reactions exactly right"],
                [[e(r["rule"]), f'{e(r["reactions"])} ({r["n"]:,})', f'{r["precision"]:.0%}', f'{r["recall"]:.0%}',
                  f'{r["exactly_right"]:.0%}'] for r in rows], num=(2, 3, 4))

    # chemical reactions with no atom map
    smiles = load_smiles(ROOT / "data/Human-GEM_metabolites.tsv")
    names = {s.met_id: s.name for s in model.species.values()}
    reasons, blockers = Counter(), Counter()
    for rid, rxn in model.reactions.items():
        subs, prods = reaction_sides(model, rxn)
        if rid in method_pairs or rxn.is_objective or not (set(subs) - set(prods) and set(prods) - set(subs)):
            continue  # mapped, biomass, or transport/exchange (not a chemical reaction)
        v = raw.get(rid, {})
        if v.get("error", "").startswith("no SMILES"):
            reasons["A metabolite in the reaction has no chemical structure in Human-GEM"] += 1
            for x in subs + prods:
                if x not in smiles:
                    blockers[x] += 1
        elif "error" in v:
            reasons["Reaction too large for the atom-mapping tool"] += 1
        else:
            reasons["Other"] += 1
    unm = table(["Why a reaction has no atom map", "Reactions"],
                [[e(k), f"{v:,}"] for k, v in reasons.most_common()] +
                [["<b>Total</b>", f"<b>{sum(reasons.values()):,}</b>"]], num=(1,))
    top = ", ".join(f"{e(names[x])} ({n})" for x, n in blockers.most_common(10))

    # benchmark by network
    base = pd.read_csv(ROOT / "results" / REFERENCE / "group_results.tsv", sep="\t")
    base = base[base.status == "primary"].set_index("case").rank_metaboenrich_raw
    brows = []
    for label, folder in BENCHMARKS:
        path = ROOT / "results" / folder
        if not (path / "group_results.tsv").exists():
            brows.append([label, "not run", "", "", "", ""])
            continue
        d = pd.read_csv(path / "group_results.tsv", sep="\t")
        d = d[d.status == "primary"].set_index("case")
        r = d.rank_metaboenrich_raw
        U = json.loads((path / "run_info.json").read_text())["gene_universe"]
        med = f"{r.median():,.1f}".rstrip("0").rstrip(".")
        vs = "" if folder == REFERENCE else f"{int((r < base).sum())} / {int((r > base).sum())}"
        brows.append([label, f"{med} of {U:,} ({100 * r.median() / U:.1f}%)", str(int((r <= 10).sum())),
                      str(int((r <= 50).sum())), str(int((r <= 100).sum())), vs])
    bench = table(["Network", "Median rank of the faulty enzyme", "Top 10", "Top 50", "Top 100",
                   "Better / worse than first method"], brows, num=(2, 3, 4, 5))
    return f"""<section class="evidence" aria-label="Evidence">
    <h2>Does it find the faulty enzyme?</h2>
    <p class="desc-wide">The same test as the benchmark report: plasma metabolomics from patients with a known enzyme
    defect (20 disorders; Miller 2015, Thistlethwaite 2022), and the rank of the faulty enzyme's gene among all
    network genes. Lower is better; a random guess lands around the middle. The last column counts disorders where the
    network ranks the faulty gene better / worse than the first method.</p>
    {bench}
    {bench_note}

    <h2>How accurate are the carbon links?</h2>
    <p class="desc-wide">On the reactions where KEGG has curated pairs, how often each rule for cleaning the computed
    atom maps agrees with KEGG. Only links between carbon-containing molecules are counted.</p>
    {acc}

    <h2>Reactions with no atom map</h2>
    {unm}
    <p class="desc-wide">Most missing structures are not naming problems: Human-GEM records no structure or external ID
    for them. They are mostly drugs from the model's drug-metabolism reactions, glycan and protein-anchor chains, and
    specific lipid intermediates, which are largely outside a polar metabolomics panel. Most frequent: {top}.</p>
  </section>"""


EXTRA_CSS = """<style>
  .method, .evidence { display: grid; gap: 14px; }
  .method h2, .evidence h2 { margin-top: 8px; }
  .steps { margin: 0; padding-left: 1.3em; display: grid; gap: 8px; max-width: 78ch; color: var(--ink-2); }
  .steps b { color: var(--ink); }
  .example { margin: 0; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 12px 14px; display: grid; gap: 6px; }
  .example svg { width: 100%; height: auto; display: block; }
  .ex-l { font: 13px var(--sans); fill: var(--ink); }
  .ex-h { font: 600 13.5px var(--sans); fill: var(--ink); }
  .ex-n { font: 12.5px var(--sans); fill: var(--muted); font-style: italic; }
  .ex-faint { fill: var(--muted); }
  .ex-t { font: 11px var(--mono); fill: var(--muted); }
  .ex-v { font: 12px var(--mono); fill: var(--ink-2); }
  .example figcaption { font-size: 13px; color: var(--ink-2); }
  .couples summary { cursor: pointer; font-weight: 600; }
  .couples ul { list-style: none; padding: 0; margin: 10px 0 0; display: flex; flex-wrap: wrap; gap: 6px; }
  .couples li { border: 1px solid var(--border); border-radius: 4px; padding: 2px 8px; font-size: 13px; background: var(--surface); }
  .couples .n { color: var(--muted); font-family: var(--mono); font-size: 12px; margin-left: 4px; }
</style>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default=str(ROOT / "results/structure_carbon/structure.json"))
    ap.add_argument("--diagrams", default=str(ROOT / "results/carbon_diagrams/diagrams.json"))
    ap.add_argument("--model", default=str(ROOT / "data/Human-GEM.xml"))
    ap.add_argument("--id-translation", required=True)
    ap.add_argument("--bench-note", default="", help="HTML shown under the benchmark table")
    ap.add_argument("--out", default=str(ROOT / "docs/carbon_atlas.html"))
    args = ap.parse_args()

    model = load_sbml(args.model)
    raw = json.loads((ROOT / "data/atom_pairs_raw.json").read_text())
    kegg = kegg_pairs_for_model(model, args.model, ROOT / "data/kegg_rclass_pairs.json")
    union = load_atom_pairs(ROOT / "data/atom_pairs_union.json")
    method, couples = carbon_method_pairs(model, union, COUPLE_MIN_REACTIONS)
    page = build_html(args.structure, args.diagrams, args.model, args.id_translation,
                      Path(__file__).with_name("carbon_atlas_template.html"),
                      {"__METHOD__": EXTRA_CSS + method_section(model, couples, COUPLE_MIN_REACTIONS, args.structure, union),
                       "__EVIDENCE__": evidence_section(model, raw, kegg, method, args.bench_note)})
    page = page.replace("python -m metaboenrich.structure --out results/structure",
                        "python -m metaboenrich.structure --stage-set carbon --out results/structure_carbon")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
