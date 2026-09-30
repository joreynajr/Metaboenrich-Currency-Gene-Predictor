"""Build the Carbon Channel Atlas page (companion to the Pruning Atlas).

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
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from build_atlas import build_html                        # noqa: E402
from metaboenrich.atoms import evaluate_rules, kegg_pairs_for_model, load_atom_pairs, load_smiles, reaction_sides  # noqa: E402
from metaboenrich.gem import load_sbml                    # noqa: E402
from metaboenrich.network import DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits, resolve_currency  # noqa: E402

e = html.escape
# network label -> group-level benchmark folder (results/...)
BENCHMARKS = [
    ("1. Raw Human-GEM", "benchmark_stage/pruning_raw"),
    ("2. Current analysis network", "benchmark_v02"),
    ("3. Carbon links (main version)", "benchmark_atoms_union"),
    ("Alternative: KEGG pairs first", "benchmark_atoms"),
    ("Alternative: atom maps first", "benchmark_atoms_mapperfirst"),
    ("Alternative: carbon links on the raw network", "benchmark_stage/carbon_carbon_only"),
]


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
            + '<text x="200" y="24" text-anchor="middle" class="ex-h">Current network: one reaction node</text>'
            + '<text x="200" y="172" text-anchor="middle" class="ex-n">current can flow aspartate → glutamate</text>')
    right = (ln(540, 60, 620, 60) + ln(620, 60, 700, 60) + ln(540, 130, 620, 130) + ln(620, 130, 700, 130)
             + rxn(620, 60) + rxn(620, 130)
             + node(540, 60, "2-oxoglutarate") + node(540, 130, "aspartate")
             + node(700, 60, "glutamate", "start") + node(700, 130, "oxaloacetate", "start")
             + '<text x="620" y="24" text-anchor="middle" class="ex-h">Carbon links: one link per carbon skeleton</text>'
             + '<text x="620" y="172" text-anchor="middle" class="ex-n">only the amino group moves between them</text>')
    return (f'<figure class="example"><svg viewBox="0 0 820 185" role="img" aria-label="Aspartate aminotransferase '
            f'as one reaction node and as two carbon links">{left}{right}</svg>'
            '<figcaption>Aspartate aminotransferase (GOT1/GOT2). Squares are reactions, circles metabolites. Both enzymes '
            'links carry the same genes.</figcaption></figure>')


def method_section(model, pairs_kegg_first, raw):
    src = Counter(v["source"] for v in json.loads((ROOT / "data/atom_pairs.json").read_text()).values())
    mapped = sum(1 for v in raw.values() if v.get("pairs"))
    return f"""<section class="method" aria-label="How the carbon links are made">
    <h2>How the carbon links are made</h2>
    <ol class="steps">
      <li><b>Start from the current network</b>, after the currency rules (inorganics removed, cofactors kept only
        where they are made or broken down, pools dropped, the SCLY edit).</li>
      <li><b>Find where each substrate's carbon atoms go.</b> Two sources: KEGG's curated substrate–product pairs for
        {src.get('kegg', 0):,} Human-GEM reactions, and atom maps computed with RXNMapper from Human-GEM's own
        structures for {mapped:,} reactions.</li>
      <li><b>Clean the computed maps.</b> Atom maps sometimes invent links in reactions with similar molecules on
        both sides (transaminases). A link is kept only if the substrate is the product's main carbon source, or the
        product is the substrate's main carbon destination (accuracy below).</li>
      <li><b>Split each reaction into one link per carbon pair.</b> The enzyme's genes attach to all of its links.
        Reactions that cannot be mapped are kept whole, not dropped.</li>
    </ol>
    {example_svg()}
    <p class="desc-wide">Three ways of combining the two sources are shown. The main version uses both KEGG's pairs and
    the cleaned atom maps, because KEGG lists only the main pair of each reaction and misses smaller carbon donors
    (carbamoyl-phosphate → citrulline in ornithine transcarbamylase).</p>
  </section>"""


def evidence_section(model, raw, kegg, rules):
    excluded = set(rules.removed) | set(rules.family_of)
    rows = evaluate_rules(model, raw, kegg, excluded)
    acc = table(["Cleaning rule", "Reactions", "Links kept that KEGG confirms", "KEGG links recovered", "Reactions exactly right"],
                [[e(r["rule"]), f'{e(r["reactions"])} ({r["n"]:,})', f'{r["precision"]:.0%}', f'{r["recall"]:.0%}',
                  f'{r["exactly_right"]:.0%}'] for r in rows], num=(2, 3, 4))

    # what could not be split, in the main carbon network
    g = build_graph(model, rules, atom_pairs=load_atom_pairs(ROOT / "data/atom_pairs_union.json"))
    smiles = load_smiles(ROOT / "data/Human-GEM_metabolites.tsv")
    names = {s.met_id: s.name for s in model.species.values()}
    reasons, blockers = Counter(), Counter()
    for rid in g.unmapped_reactions:
        v = raw.get(rid, {})
        if v.get("error", "").startswith("no SMILES"):
            reasons["A metabolite in the reaction has no chemical structure in Human-GEM"] += 1
            for x in sum(reaction_sides(model, model.reactions[rid]), []):
                if x not in smiles:
                    blockers[x] += 1
        elif "error" in v:
            reasons["Reaction too large for the atom-mapping tool"] += 1
        else:
            reasons["Other"] += 1
    n_rxn = len({g.base_reaction.get(i, g.node_ids[i]) for i in range(g.n) if g.is_reaction[i]}) + len(g.no_pair_reactions)
    unm = table(["Why a reaction was kept whole", "Reactions"],
                [[e(k), f"{v:,}"] for k, v in reasons.most_common()] +
                [["<b>Total kept whole</b>", f"<b>{len(g.unmapped_reactions):,}</b> of {n_rxn:,}"]], num=(1,))
    top = ", ".join(f"{e(names[x])} ({n})" for x, n in blockers.most_common(10))

    # benchmark by network
    base = pd.read_csv(ROOT / "results/benchmark_v02/group_results.tsv", sep="\t")
    base = base[base.status == "primary"].set_index("case").rank_metaboenrich_raw
    brows = []
    for label, folder in BENCHMARKS:
        path = ROOT / "results" / folder
        if not (path / "group_results.tsv").exists():
            brows.append([e(label), "not run", "", "", "", ""])
            continue
        d = pd.read_csv(path / "group_results.tsv", sep="\t")
        d = d[d.status == "primary"].set_index("case")
        r = d.rank_metaboenrich_raw
        U = json.loads((path / "run_info.json").read_text())["gene_universe"]
        med = f"{r.median():,.1f}".rstrip("0").rstrip(".")
        vs = "" if folder == "benchmark_v02" else f"{int((r < base).sum())} / {int((r > base).sum())}"
        brows.append([e(label), f"{med} of {U:,} ({100 * r.median() / U:.1f}%)", str(int((r <= 10).sum())),
                      str(int((r <= 50).sum())), str(int((r <= 100).sum())), vs])
    bench = table(["Network", "Median rank of the faulty enzyme", "Top 10", "Top 50", "Top 100", "Better / worse than current"],
                  brows, num=(2, 3, 4, 5))
    return f"""<section class="evidence" aria-label="Evidence">
    <h2>Does it find the faulty enzyme better?</h2>
    <p class="desc-wide">The same test as the benchmark report: plasma metabolomics from patients with a known enzyme
    defect (20 disorders; Miller 2015, Thistlethwaite 2022), and the rank of the faulty enzyme's gene among all
    network genes. Lower is better; a random guess lands around the middle.</p>
    {bench}
    <p class="desc-wide">Overall the carbon links perform about the same as the current network. Individual disorders
    move a lot in both directions, and with 20 disorders differences of this size are within the noise.</p>

    <h2>How accurate are the carbon links?</h2>
    <p class="desc-wide">On the reactions where KEGG has curated pairs, how often each rule for cleaning the computed
    atom maps agrees with KEGG. Only links between non-currency metabolites are counted, as those are the ones the
    network uses.</p>
    {acc}

    <h2>What could not be split</h2>
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
  .example figcaption { font-size: 13px; color: var(--ink-2); }
</style>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default=str(ROOT / "results/structure_carbon/structure.json"))
    ap.add_argument("--diagrams", default=str(ROOT / "results/carbon_diagrams/diagrams.json"))
    ap.add_argument("--model", default=str(ROOT / "data/Human-GEM.xml"))
    ap.add_argument("--id-translation", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs/carbon_atlas.html"))
    args = ap.parse_args()

    model = load_sbml(args.model)
    rules, _ = resolve_currency(model, "role")
    rules.edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, model)
    raw = json.loads((ROOT / "data/atom_pairs_raw.json").read_text())
    kegg = kegg_pairs_for_model(model, args.model, ROOT / "data/kegg_rclass_pairs.json")
    page = build_html(args.structure, args.diagrams, args.model, args.id_translation,
                      Path(__file__).with_name("carbon_atlas_template.html"),
                      {"__METHOD__": EXTRA_CSS + method_section(model, kegg, raw),
                       "__EVIDENCE__": evidence_section(model, raw, kegg, rules)})
    page = page.replace("python -m metaboenrich.structure --out results/structure",
                        "python -m metaboenrich.structure --stage-set carbon --out results/structure_carbon")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
