"""Build the 3-Rule Pruning Atlas: the carbon method revised (pools and transporters
dropped, carriers removed, rarest link kept), with earlier variants and the first
method for comparison.

Needs, from the repository root:
    python -m metaboenrich.structure --stage-set three_rule --out results/structure_three_rule
    python -m metaboenrich.stage_diagrams --structure results/structure_three_rule/structure.json \
        --out results/three_rule_diagrams
    benchmark runs (see BENCHMARKS below)

Then:
    python tools/atlas/build_three_rule_atlas.py --id-translation PATH/TO/id_translation.csv \
        --out docs/three_rule_atlas.html

Also writes results/three_rule/rescued_reactions.csv and results/three_rule/metrics.tsv.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from build_atlas import build_html                         # noqa: E402
from build_carbon_atlas import EXTRA_CSS, e, table         # noqa: E402
from metaboenrich import structure                         # noqa: E402
from metaboenrich.atoms import pool_metabolites, transport_reactions  # noqa: E402
from metaboenrich.gem import load_sbml                     # noqa: E402
from metaboenrich.network import build_graph               # noqa: E402

# network key -> group-level benchmark folder (results/...); all run with the id_translation name lookup
KEYS = ["raw", "carbon_method", "carbon_cleaned", "carbon_carriers", "three_rule", "rules12_keep_rarest",
        "acyl_merged", "acyl_merged_keep_rarest", "first_method"]
BENCHMARKS = {k: f"benchmark_lookup/three_rule_{k}" for k in KEYS}
LABELS = {
    "raw": "1. Raw Human-GEM",
    "carbon_method": "2. Carbon method (Rules 1 + 2)",
    "carbon_cleaned": "3. Pools and transporters dropped",
    "carbon_carriers": "4. Carriers removed (Rule 3)",
    "three_rule": "<b>5. 3-rule pruning (final)</b>",
    "rules12_keep_rarest": "Alternative: Rules 1-2 with the keep-rarest safeguard",
    "acyl_merged": "Alternative: long acyl-CoAs merged (on 3)",
    "acyl_merged_keep_rarest": "Alternative: long acyl-CoAs merged, keep-rarest safeguard",
    "first_method": "Reference: first method",
}
SHORT = {"raw": "1", "carbon_method": "2", "carbon_cleaned": "3", "carbon_carriers": "4", "three_rule": "5",
         "rules12_keep_rarest": "R1-2 + safeguard", "acyl_merged": "Acyl", "acyl_merged_keep_rarest": "Acyl + safeguard", "first_method": "First"}


# Interpretation shown under the metrics table (written from the benchmark of 2026-10-02; update it if the
# networks or the benchmark change).
READING = (
    '<p class="desc-wide"><b>Reading this table.</b> Rule 3 is the change that matters. Removing carriers ends the '
    'CoA short circuit: current spreads more evenly (the busiest 1% of nodes carry the least of any network) and '
    'the disorders of CoA-ester metabolism recover, HMG-CoA lyase deficiency most of all (rank 445 to 83; first '
    'method 55). The keep-rarest safeguard on top of Rule 3 adds the missing enzymes at little cost, and 3-rule '
    'pruning (5) has the best median of the carbon variants. It still trails the first method overall, mostly on '
    'methylmalonic acidemia and citrullinemia, and beats it on propionic acidemia and glutaric aciduria. The cost '
    'of Rule 3 is connectivity: 84 more metabolites are cut off from the main network, mostly xenobiotics '
    '(naphthalene derivatives, halogenated solvents) and their glutathione conjugates, whose only route into the '
    'network ran through glutathione. The safeguard without Rule 3, and both acyl-CoA merges, do not help. With 20 '
    'disorders, differences of a few dozen ranks are within the noise.</p>')


def glud_svg():
    """Glutamate dehydrogenase under the carbon method and under 3-rule pruning."""
    def node(x, y, label, anchor="end", faint=False):
        dx = -12 if anchor == "end" else 12
        return (f'<circle cx="{x}" cy="{y}" r="7" fill="{"var(--grid)" if faint else "var(--p2)"}"></circle>'
                f'<text x="{x + dx}" y="{y + 4}" text-anchor="{anchor}" class="ex-l{" ex-faint" if faint else ""}">'
                f'{e(label)}</text>')

    def rxn(x, y, faint=False):
        return f'<rect x="{x - 7}" y="{y - 7}" width="14" height="14" rx="2" fill="{"var(--grid)" if faint else "var(--ink-2)"}"></rect>'

    def ln(x1, y1, x2, y2, faint=False):
        dash = ' stroke-dasharray="4 4"' if faint else ""
        return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{"var(--grid)" if faint else "var(--axis)"}" '
                f'stroke-width="2"{dash}></line>')

    def pair(cx, y, a, b, count, faint):
        return (ln(cx - 80, y, cx, y, faint) + ln(cx, y, cx + 80, y, faint) + rxn(cx, y, faint)
                + node(cx - 80, y, a, faint=faint) + node(cx + 80, y, b, "start", faint)
                + f'<text x="{cx}" y="{y - 14}" text-anchor="middle" class="ex-t">couple · {count} reactions</text>')

    left = (pair(200, 66, "glutamate", "2-oxoglutarate", 41, True) + pair(200, 136, "NAD+", "NADH", 427, True)
            + '<text x="200" y="24" text-anchor="middle" class="ex-h">Carbon method: both pairs are couples</text>'
            + '<text x="200" y="176" text-anchor="middle" class="ex-n">no link left, so GLUD1 leaves the network</text>')
    right = (pair(620, 66, "glutamate", "2-oxoglutarate", 41, False) + pair(620, 136, "NAD+", "NADH", 427, True)
             + '<text x="620" y="24" text-anchor="middle" class="ex-h">Safeguard: the rarer couple stays</text>'
             + '<text x="620" y="176" text-anchor="middle" class="ex-n">41 &lt; 427, so glutamate → 2-oxoglutarate is kept</text>')
    return (f'<figure class="example"><svg viewBox="0 0 820 190" role="img" aria-label="Glutamate dehydrogenase under '
            f'the carbon method and with the rarest link kept">{left}{right}</svg>'
            '<figcaption>Glutamate dehydrogenase (GLUD1/GLUD2): glutamate + NAD(P)+ ⇌ 2-oxoglutarate + NH3 + NAD(P)H. '
            'Water and ammonia carry no carbon, so after Rule 1 it has two pairs, and Rule 2 marks both as couples. Faint '
            'dashed items are removed.</figcaption></figure>')


def method_section(model, stage_rows, n_pools, n_cargo, carriers):
    S = {k: s for k, s in stage_rows}
    rows = []
    recipe_rows = {  # pools+transporters, Rule 3, safeguard, acyl merge
        "raw": ("–", "–", "–", "–"), "carbon_method": ("–", "–", "–", "–"),
        "carbon_cleaned": ("yes", "–", "–", "–"), "carbon_carriers": ("yes", "yes", "–", "–"),
        "three_rule": ("yes", "yes", "yes", "–"), "rules12_keep_rarest": ("yes", "–", "yes", "–"),
        "acyl_merged": ("yes", "–", "–", "yes"), "acyl_merged_keep_rarest": ("yes", "–", "yes", "yes"),
        "first_method": ("pools only (its own list)", "–", "–", "–")}
    for key in KEYS:
        s = S[key]
        rows.append([LABELS[key], "yes" if key not in ("raw", "first_method") else "–", *recipe_rows[key],
                     f"{s['metabolites']:,}", f"{s['reactions']:,}"])
    recipe = table(["Network", "Carbon method (Rules 1 + 2)", "Pools and transporters dropped", "Carriers removed (Rule 3)",
                    "Rarest link kept (safeguard)", "Long acyl-CoAs merged", "Metabolites", "Reaction links"], rows, num=(6, 7))
    return f"""<section class="method" aria-label="What changed">
    <h2>What changed</h2>
    <ol class="steps">
      <li><b>Start from the carbon method</b> (network 3 of the Carbon Channel Atlas): Rule 1 splits reactions into
        carbon-sharing links, Rule 2 removes the 48 cofactor couples found in 10 or more reactions. Couples are counted
        exactly as before.</li>
      <li><b>Drop pools and transporters.</b> Reactions involving any of Human-GEM's {n_pools} pool pseudo-metabolites
        (acyl-CoA pool, PI pool, fatty acid-VLDL pool ...) leave the network, as do {n_cargo:,} transporters: reactions
        that carry a carbon-containing metabolite across a membrane. Once compartments are merged, an ATP-driven pump
        is left with ATP → ADP as its only carbon pair. Transporter genes stay in the transporter table, linked to their
        cargo.</li>
      <li><b>Rule 3: remove carriers.</b> A carrier is found from structure: a molecule found whole inside at least 10
        of its carbon partners, every time attached at the same atom and through a heteroatom (a thioester, ester or
        amide bond). Found: {carriers}. The links between a carrier and its loaded forms (CoA and methylmalonyl-CoA,
        carnitine and propionylcarnitine) are loading and unloading, and are removed everywhere; links between two
        loaded forms (acetyl-CoA to acetoacetyl-CoA) and the carrier's own synthesis and breakdown stay. A bond between
        two carbons is never counted as loading, so glycine to 5-aminolevulinate (heme synthesis) and thiolase stay.
        Glycine only just misses (36 of its 37 conjugates attach at the nitrogen) and is not treated as a carrier.</li>
      <li><b>Safeguard: keep the rarest link.</b> A reaction whose carbon pairs were all removed would lose every link and drop out
        with its genes. Instead it keeps the couple found in the fewest reactions (ties broken by metabolite id).
        Reactions that carry anything across a membrane, including ion pumps, are never rescued. The same rule
        applies to reactions with no atom map, using the couple counts from mapped reactions.</li>
      <li><b>Alternative: merge long acyl-CoAs.</b> Acyl-CoAs are found from structure (the CoA substructure plus a
        thioester in Human-GEM's SMILES); chain length is the carbon count minus CoA's 21. Long-chain (C13–C21) and
        very-long-chain (C22+) acyl-CoAs each become one node. Short and medium acyl-CoAs, including acetyl- and
        malonyl-CoA, stay individual, as do the acylcarnitines a polar panel measures. Couples are counted on the
        merged network.</li>
    </ol>
    {glud_svg()}
    <h2>How each network on this page is made</h2>
    <p class="desc-wide">Networks 1–5 are the steps; network 5 is 3-rule pruning. The alternatives show the safeguard
    without Rule 3 and two tests of merging long acyl-CoAs; the first method is shown for reference. Charts and
    diagrams below cover all nine.</p>
    {recipe}
  </section>"""


def evidence_section(model, graphs, stage_rows, log, out_dir):
    names = {s.met_id: s.name for s in model.species.values()}
    sym = model.genes
    S = {k: s for k, s in stage_rows}
    genes = {k: {x for v in g.reaction_genes.values() for x in v} for k, g in graphs.items()}

    # rescue log
    recs = []
    for x in log:
        r = model.reactions[x["reaction"]]
        a, b = x["kept_pair"]
        recs.append({"reaction": x["reaction"].replace("R_", ""), "name": r.name,
                     "genes": ";".join(sym.get(gg, gg) for gg in r.genes),
                     "kept_pair": f"{names[a]} / {names[b]}", "kept_pair_count": x["kept_pair_count"],
                     "candidates": "; ".join(f"{names[p]} / {names[q]} ({n})" for (p, q), n in x["candidates"]),
                     "atom_mapped": x["mapped"]})
    rdf = pd.DataFrame(recs).sort_values(["kept_pair_count", "reaction"])
    rdf.to_csv(out_dir / "rescued_reactions.csv", index=False)
    back = genes["three_rule"] - genes["carbon_carriers"]
    recv = Counter()
    for x in log:
        for m in x["kept_pair"]:
            recv[names[m]] += 1

    # metrics table
    mrows, metrics = [], []
    for key in KEYS[1:]:
        s = S[key]
        path = ROOT / "results" / BENCHMARKS[key]
        med = top5 = U = None
        if (path / "group_results.tsv").exists():
            d = pd.read_csv(path / "group_results.tsv", sep="\t")
            r = d[d.status == "primary"].rank_metaboenrich_raw
            U = json.loads((path / "run_info.json").read_text())["gene_universe"]
            med, top5 = float(r.median()), int((r <= 0.05 * U).sum())
        metrics.append({"network": key, "metabolites": s["metabolites"], "reaction_links": s["reactions"],
                        "largest_hub": f"{s['max_degree_metabolite']} ({s['max_metabolite_degree']})",
                        "genes": len(genes[key]), "top1pct_current": round(s["top1pct_flow_share"], 4),
                        "cut_off": s["metabolites_outside_main"], "benchmark_median_rank": med,
                        "benchmark_top5pct": top5, "n_genes_ranked": U})
        mrows.append([LABELS[key], f"{s['metabolites']:,}", f"{s['reactions']:,}",
                      f"{e(s['max_degree_metabolite'])} ({s['max_metabolite_degree']:,})", f"{len(genes[key]):,}",
                      f"{100 * s['top1pct_flow_share']:.1f}%", f"{s['metabolites_outside_main']:,}",
                      "not run" if med is None else f"{med:g}", "" if top5 is None else f"{top5} / 20",
                      "" if U is None else f"{U:,}"])
    pd.DataFrame(metrics).to_csv(out_dir / "metrics.tsv", sep="\t", index=False)
    mtab = table(["Network", "Metabolites", "Reaction links", "Largest hub (links)", "Genes",
                  "Current through busiest 1%", "Cut off", "Benchmark median rank", "Faulty gene in top 5%",
                  "Genes ranked"], mrows, num=(1, 2, 4, 5, 6, 7, 8, 9))

    # benchmark by network, per disorder
    per = {}
    for key, folder in BENCHMARKS.items():
        f = ROOT / "results" / folder / "group_results.tsv"
        if f.exists():
            d = pd.read_csv(f, sep="\t")
            per[key] = d[d.status == "primary"].set_index("case").rank_metaboenrich_raw
    cols = [k for k in KEYS if k in per]
    cases = per["three_rule"].sort_values().index if "three_rule" in per else per[cols[0]].index
    moved = table(["Disorder"] + [SHORT[k] for k in cols],
                  [[e(c)] + [f"{int(per[k][c]):,}" for k in cols] for c in cases], num=tuple(range(1, len(cols) + 1)))

    chips = "".join(f"<li>{e(n)} <span class=\"n\">{c}</span></li>" for n, c in recv.most_common(30))
    rtab = table(["Reaction", "Name", "Genes", "Kept pair (reactions)", "Other couples"],
                 [[e(r.reaction), e(r["name"]), e(r.genes), f"{e(r.kept_pair)} ({r.kept_pair_count})",
                   e("; ".join(c for c in r.candidates.split("; ") if not c.startswith(r.kept_pair + " (")))]
                  for _, r in rdf.iterrows()])
    back_syms = sorted(sym.get(x, x) for x in back)
    return f"""<section class="evidence" aria-label="Evidence">
    <h2>What it changes</h2>
    <p class="desc-wide">Structure and benchmark for every network. The benchmark matches metabolite names through
    the lab's id_translation synonyms as well as Human-GEM's names and ids (propionylcarnitine is listed as
    "O-propanoylcarnitine" in Human-GEM), so its numbers differ slightly from the Carbon Channel Atlas. "Faulty gene in
    top 5%" counts the 20 disorders whose faulty enzyme ranks in the top 5% of network genes.</p>
    {mtab}
    __READING__
    <h2>Rank of the faulty enzyme, by disorder</h2>
    <p class="desc-wide">Lower is better; about 2,100 to 2,500 genes are ranked in each network. Column numbers are the
    networks above. Sorted by 3-rule pruning (5).</p>
    {moved}
    <h2>The {len(log)} rescued reactions (network 5)</h2>
    <p class="desc-wide">{len(back)} genes return to the network: {e(", ".join(back_syms))}.</p>
    <details class="couples"><summary>Metabolites that received rescued links (number of rescued links)</summary>
      <ul>{chips}</ul></details>
    <details class="couples"><summary>All {len(log)} rescued reactions, rarest kept pair first</summary>{rtab}</details>
  </section>"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default=str(ROOT / "results/structure_three_rule/structure.json"))
    ap.add_argument("--diagrams", default=str(ROOT / "results/three_rule_diagrams/diagrams.json"))
    ap.add_argument("--model", default=str(ROOT / "data/Human-GEM.xml"))
    ap.add_argument("--id-translation", required=True)
    ap.add_argument("--out", default=str(ROOT / "docs/three_rule_atlas.html"))
    ap.add_argument("--reading", default=READING, help="HTML paragraph under the metrics table")
    args = ap.parse_args()

    model = load_sbml(args.model)
    defs = structure.stages(model, "three_rule", atom_dir=str(ROOT / "data"))
    log = list(structure.KEPT_RAREST_LOG)
    graphs = {k: build_graph(model, r, ms, do, **ex) for k, _, _, r, ms, do, ex in defs}
    stage_rows = [(st["key"], st["summary"]) for st in json.loads(Path(args.structure).read_text())["stages"]]
    out_dir = ROOT / "results/three_rule"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = {s.met_id: s.name for s in model.species.values()}
    carriers = ", ".join(f"{names[c]} ({len(x)} loaded forms)" for c, x in
                         sorted(structure.CARRIERS_FOUND.items(), key=lambda kv: -len(kv[1])) if "pool" not in names[c])
    page = build_html(args.structure, args.diagrams, args.model, args.id_translation,
                      Path(__file__).with_name("three_rule_atlas_template.html"),
                      {"__METHOD__": EXTRA_CSS + method_section(model, stage_rows, len(pool_metabolites(model) | set(defs[2][3].pools)),
                                                                len(transport_reactions(model)), carriers),
                       "__EVIDENCE__": evidence_section(model, graphs, stage_rows, log, out_dir).replace(
                           "__READING__", args.reading)})
    atp = next(m for m, s in ((s.met_id, s) for s in model.species.values()) if s.name == "ATP")
    deg = {k: int(((g.edges == g.index[atp]).any(axis=1)).sum()) for k, g in graphs.items()}
    back = {x for v in graphs["three_rule"].reaction_genes.values() for x in v} - \
           {x for v in graphs["carbon_carriers"].reaction_genes.values() for x in v}
    coa = next(s.met_id for s in model.species.values() if s.name == "CoA")
    coa_deg = {k: int(((g.edges == g.index[coa]).any(axis=1)).sum()) if coa in g.index else 0 for k, g in graphs.items()}
    for k, v in {"__N_RESCUED__": len(log), "__N_GENES_BACK__": len(back),
                 "__ATP_METHOD__": deg["carbon_method"], "__ATP_LAST__": deg["three_rule"],
                 "__COA_CLEAN__": coa_deg["carbon_cleaned"], "__COA_CARR__": coa_deg["carbon_carriers"],
                 "__CARRIERS__": carriers}.items():
        page = page.replace(k, str(v))
    page = page.replace("python -m metaboenrich.structure --out results/structure",
                        "python -m metaboenrich.structure --stage-set three_rule --out results/structure_three_rule")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
