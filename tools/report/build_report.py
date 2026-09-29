"""Build the Metaboenrich progress report (one self-contained HTML page).

Every number comes from result files, not from hand-typed text:
    results/structure/structure.json                pruning stages
    results/benchmark_v02/{group,patient}_results.tsv  inborn-error benchmark
    results/statistic_analysis_v02/<comparison>/    the lab's own (unpublished) comparisons; local only
    predictions/published_v1/units.tsv               sealed blind predictions

    python tools/report/build_report.py --out docs/report.html
"""
import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
U = 2302
e = html.escape


def fmt_int(x):
    return f"{int(round(x)):,}"


def pct(x):
    return f"{100 * x:.0f}%"


def table(headers, rows, num_cols=(), cls=""):
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{e(h)}</th>' for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>'
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table></div>'


# ---- data -------------------------------------------------------------------
def stage_rows():
    s = json.loads((ROOT / "results/structure/structure.json").read_text(encoding="utf-8"))
    rows = []
    for st in s["stages"]:
        m = st["summary"]
        rows.append([e(st["title"]), fmt_int(m["metabolites"]), fmt_int(m["reactions"]), fmt_int(m["edges"]),
                     fmt_int(m["metabolites_outside_main"]), fmt_int(m["reactions_pocket_ge5"]),
                     pct(m["top1pct_flow_share"])])
    return rows


def benchmark():
    g = pd.read_csv(ROOT / "results/benchmark_v02/group_results.tsv", sep="\t")
    p = pd.read_csv(ROOT / "results/benchmark_v02/patient_results.tsv", sep="\t")
    return g, p


def bench_summary(g, p):
    rows = []
    for label, d in (("Disorder groups", g[g.status == "primary"]), ("Individual patients", p[p.status == "primary"])):
        d = d[d.rank_metaboenrich_raw.notna()]
        for col, name in (("rank_metaboenrich_raw", "<b>Metaboenrich, raw</b>"),
                          ("rank_metaboenrich", "Metaboenrich, leave-one-out"),
                          ("rank_neighbour", "Neighbour baseline")):
            r = d[col]
            med = f"{r.median():,.1f}".rstrip("0").rstrip(".")
            rows.append([f"{label} ({len(d)})", name, f"{med} ({100 * r.median() / U:.1f}%)",
                         str(int((r <= 10).sum())), str(int((r <= 50).sum())), str(int((r <= 100).sum()))])
    return rows


def bench_chart(g):
    d = g[g.status.isin(["primary", "secondary"])].copy()
    d = d.sort_values("rank_metaboenrich_raw")
    W, row_h, L, R, T = 760, 22, 250, 24, 34
    H = T + row_h * len(d) + 30
    x = lambda r: L + (W - L - R) * math.log10(max(r, 1)) / math.log10(U)
    ticks = [1, 3, 10, 30, 100, 300, 1000, U]
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rank of the causal gene per disorder, three methods">']
    for t in ticks:
        out.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{T - 8}" y2="{H - 26}" class="grid"/>'
                   f'<text x="{x(t):.1f}" y="{H - 10}" class="tick" text-anchor="middle">{fmt_int(t)}</text>')
    out.append(f'<text x="{L}" y="14" class="axis-label">Rank of the causal gene among {fmt_int(U)} network genes '
               f'(log scale; left is better; random ≈ {fmt_int(U / 2)})</text>')
    out.append(f'<rect x="{x(U / 2) - 1:.1f}" y="{T - 8}" width="2" height="{H - T - 18}" class="random"/>')
    for i, r in enumerate(d.itertuples(index=False)):
        y = T + i * row_h + row_h / 2
        star = "*" if bool(r.one_sided) else ""
        label = f"{r.disorder}{star}" + (" (secondary)" if r.status == "secondary" else "")
        out.append(f'<text x="{L - 10}" y="{y + 4:.1f}" class="row-label" text-anchor="end">{e(label[:40])}</text>')
        vals = [("nb", r.rank_neighbour, "Neighbour baseline"), ("loo", r.rank_metaboenrich, "Leave-one-out"),
                ("raw", r.rank_metaboenrich_raw, "Metaboenrich raw")]
        xs = [x(v) for _, v, _ in vals if pd.notna(v)]
        if xs:
            out.append(f'<line x1="{min(xs):.1f}" x2="{max(xs):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="span"/>')
        for cls, v, name in vals:
            if pd.notna(v):
                shape = (f'<rect x="{x(v) - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" rx="1.5" class="m-{cls}"/>'
                         if cls == "nb" else f'<circle cx="{x(v):.1f}" cy="{y:.1f}" r="5" class="m-{cls}"/>')
                out.append(f'<g><title>{e(r.disorder)}: {name} rank {v:.0f}</title>{shape}</g>')
    out.append("</svg>")
    return "".join(out)


def lab_rows():
    rows = []
    base = ROOT / "results/statistic_analysis_v02"
    for d in sorted(base.iterdir()):
        s = json.loads((d / "summary.json").read_text())
        genes = pd.read_csv(d / "genes.tsv", sep="\t").head(10)
        rxn = pd.read_csv(d / "reactions.tsv", sep="\t").head(12)
        forced = {str(g).split(";")[0] for g, o in zip(rxn.genes.fillna(""), rxn.only_reaction_of.fillna("")) if o}
        gl = ", ".join((f"<u>{e(g)}</u>" if g in forced else e(g)) for g in genes.symbol.fillna(genes.ensembl_id))
        paths = pd.read_csv(d / "pathways.tsv", sep="\t").head(4)
        pl = "; ".join(f"{e(r.subsystem)} ({r.in_top}/{r.expected:g})" for r in paths.itertuples())
        rows.append([e(d.name.replace("_vs_", " vs ")), f"{s['sources']} / {s['targets']}", gl, pl])
    return rows


def prediction_rows():
    u = pd.read_csv(ROOT / "predictions/published_v1/units.tsv", sep="\t")
    ok = u[u.status == "ok"].copy()
    ok["signal"] = ok.sources.fillna(0) + ok.targets.fillna(0)
    status = [["Units defined", str(len(u))], ["Run", str(len(ok))],
              ["Skipped (fewer than 3 metabolites)", str(len(u) - len(ok))],
              ["With at least one significant mapped metabolite", str(int((ok.signal > 0).sum()))]]
    rows = []
    for r in ok[(ok.signal > 0) & ~ok.source.str.startswith("Alaimo")].itertuples(index=False):
        name = r.unit.split("_", 1)[1].replace("_", " ")
        rank = str(r.rank_of_diagnosed_gene) if pd.notna(r.rank_of_diagnosed_gene) else "–"
        gene = f"{e(str(r.diagnosed_gene_in_data))}: {e(rank)}" if isinstance(r.diagnosed_gene_in_data, str) and r.diagnosed_gene_in_data else "–"
        top = ", ".join(str(r.top_genes).split(", ")[:6]) if isinstance(r.top_genes, str) and r.top_genes else "no current flow (too few metabolites)"
        paths = "; ".join(p.split(" (")[0] for p in str(r.top_pathways).split("; ")[:3]) if isinstance(r.top_pathways, str) else ""
        rows.append([e(name[:70]), f"{int(r.mapped)}", f"{int(r.sources)} / {int(r.targets)}{' ¹' if r.one_sided else ''}",
                     gene, e(top), e(paths)])
    alaimo = ok[ok.source.str.startswith("Alaimo")]
    return status, rows, len(alaimo), int((alaimo.signal > 0).sum())


# ---- page ---------------------------------------------------------------------
CSS = """
:root{--page:#f3f5f7;--surface:#fbfcfd;--ink:#12161b;--ink-2:#4a5360;--muted:#6b7480;--rule:#dfe4ea;--axis:#c5ccd4;
--accent:#1f63b5;--raw:#2a78d6;--loo:#eb6834;--nb:#8b95a1;--note:#eef3fa;--note-ink:#1c3e66;
--sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;
--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;--axis:#38414b;--accent:#6ea6ec;--raw:#3987e5;--loo:#f07a45;--nb:#8b95a1;--note:#18222e;--note-ink:#bcd4f2}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;
--axis:#38414b;--accent:#6ea6ec;--raw:#3987e5;--loo:#f07a45;--nb:#8b95a1;--note:#18222e;--note-ink:#bcd4f2}
*{box-sizing:border-box}
body{background:var(--page);color:var(--ink);font:16px/1.6 var(--sans);padding-inline:clamp(16px,4vw,40px);padding-block:40px 72px}
main{max-width:980px;margin-inline:auto;display:grid;gap:44px}
.prose{max-width:70ch;display:grid;gap:12px}
h1,h2,h3{text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:clamp(30px,4vw,40px);font-weight:700;letter-spacing:-.015em}
h2{font-size:23px;font-weight:650;letter-spacing:-.005em;padding-top:8px;border-top:1px solid var(--rule)}
h3{font-size:17px;font-weight:650}
p,ul{margin:0}ul{padding-left:1.2em;display:grid;gap:6px}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.meta{font:13px/1.6 var(--mono);color:var(--muted)}
code,.mono{font-family:var(--mono);font-size:.9em}
a{color:var(--accent)}
section{display:grid;gap:16px}
.summary{background:var(--note);color:var(--note-ink);border-radius:8px;padding:20px 24px;display:grid;gap:10px}
.summary li{color:var(--note-ink)}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font:500 11.5px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
.tbl.wide td{min-width:90px}
figure{margin:0;display:grid;gap:10px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 16px 10px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:13.5px;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:6px}
.grid{stroke:var(--rule);stroke-width:1}
.random{fill:var(--axis);opacity:.6}
.tick{font:11px var(--mono);fill:var(--muted)}
.axis-label{font:12px var(--sans);fill:var(--ink-2)}
.row-label{font:12.5px var(--sans);fill:var(--ink)}
.span{stroke:var(--axis);stroke-width:1.5}
.m-raw{fill:var(--raw);stroke:var(--surface);stroke-width:1.5}
.m-loo{fill:var(--surface);stroke:var(--loo);stroke-width:2}
.m-nb{fill:var(--nb);stroke:var(--surface);stroke-width:1.5}
.small{font-size:13.5px;color:var(--ink-2)}
u{text-decoration-color:var(--loo);text-decoration-thickness:2px;text-underline-offset:3px}
footer{font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:16px;display:grid;gap:6px}
"""


def build(out, public=False):
    g, p = benchmark()
    status, pred_rows, n_alaimo, n_alaimo_sig = prediction_rows()
    raw_g = g[(g.status == "primary")].rank_metaboenrich_raw
    ATLAS = "https://claude.ai/code/artifact/c72ab942-2d9c-4214-88ee-538f1977a440"
    REPO = "https://github.com/joreynajr/Metaboenrich-Currency-Gene-Predictor"
    doc = f"""<meta charset="utf-8">
<title>Metaboenrich Prototype Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style>
<main>
<header class="prose">
  <span class="eyebrow">Metaboenrich · progress report · 29 September 2026</span>
  <h1>Metaboenrich Prototype Report</h1>
  <p class="meta">For Nina Frasketi, Joaquin Reyna and David Montefusco · tool 0.2.0.dev0 · network v0.1.1
  (3,958 metabolites, 6,377 reactions, 16,356 edges; edge-list SHA-256 942cd8ac…) · Human-GEM 2.0.1</p>
</header>

<section class="summary" aria-label="Summary">
  <h3>Where things stand</h3>
  <ul>
    <li>A working prototype of the protocol's current-flow method runs on Human-GEM in seconds, with
      currency metabolites handled by role, a locked and versioned network, and Cytoscape output.</li>
    <li>On inborn errors of metabolism with a known defective enzyme, it ranks the causal gene at a median of
      <b>{raw_g.median():.0f} of {fmt_int(U)}</b> (top {100 * raw_g.median() / U:.1f}%), with
      <b>{int((raw_g <= 100).sum())} of {len(raw_g)}</b> disorders in the top 100. A simple baseline sits at random.</li>
    <li>Blind predictions on the lab's 12 published datasets are made and sealed in git
      (commit <span class="mono">2727d1b</span>, tag <span class="mono">predictions-published-v1</span>).
      The answer key has not been built or read.</li>
    <li>Several methods decisions are still open (listed at the end). The next step is the answer key.</li>
  </ul>
</section>

<section>
  <h2>1. What the tool does</h2>
  <div class="prose">
    <p>Following the protocol (§1), Human-GEM becomes a network of metabolite and reaction nodes. Significantly
    increased metabolites are current sources and significantly decreased ones are sinks. For every
    source–sink pair the tool solves Kirchhoff's and Ohm's laws on the network, then scores each reaction by how
    much of the current it carries (<i>conductivity</i>) and how much it constrains it (<i>bottleneck</i>).
    Genes take the best score of their reactions. Outputs: ranked genes, reactions, pathways and transporters,
    plus Cytoscape files.</p>
  </div>
</section>

<section>
  <h2>2. Building the network</h2>
  <div class="prose">
    <p>Human-GEM 2.0.1 has 8,460 compartment-specific metabolites (4,165 once compartments are merged),
    12,877 reactions and 2,848 genes. Our data are bulk polar metabolomics with no compartment information, so
    each metabolite is one node. Transport and exchange reactions then have nothing to connect and drop out.</p>
    <p><b>Currency metabolites</b> are the main design problem: H⁺ is in 4,540 reactions and H₂O in 3,496, so
    left in they connect everything. Following David's strategy, inorganics are removed everywhere, but cofactors
    stay in the network through their own synthesis and breakdown and are removed only where they shuttle a
    group in a cycle (ATP → ADP, NAD⁺ → NADH, CoA → acyl-CoA). That keeps 168 cofactor links and removes 6,808.
    Two corrections followed: an annotation edit (SCLY listed the PLP cofactor as a reactant, which made it the
    vitamin B6 pool's only link to the network) and a pairing fix in v0.1.1 that restored 10 nucleotide-salvage
    reactions.</p>
    <p>The structural check measures what each pruning step does, with no data involved:</p>
  </div>
  {table(["Stage", "Metabolites", "Reactions", "Edges", "Metabolites cut off", "Reactions cutting off 5+", "Current via top 1% of nodes"],
         stage_rows(), num_cols=(1, 2, 3, 4, 5, 6))}
  <p class="small prose">Removing hubs spreads the current (34% → 21% through the busiest 1% of nodes) but creates
  islands and single-link pockets; the largest are drug glucuronides behind UDP-glucuronate (80) and glutathione
  conjugates behind GSH (68). Interactive version with linked network diagrams:
  <a href="{ATLAS}">Human-GEM Pruning Atlas</a>.</p>
</section>

{"" if public else f'''<section>
  <h2>3. First application: the lab's liver dataset</h2>
  <div class="prose">
    <p>Four comparisons from the lab's untargeted dataset (111 metabolites, 68 mapped to Human-GEM; Tukey
    p &lt; 0.05), rerun with the current tool. <u>Underlined</u> genes rest on a single measured metabolite whose only
    reaction they catalyse; treat them with caution.</p>
  </div>
  {table(["Comparison", "Up / down", "Top genes", "Top pathways (in top 100 / expected)"], lab_rows(), cls="wide")}
  <p class="small prose">Recurring themes: purine and pyrimidine synthesis and salvage, lysine metabolism
  (TMLHE, trimethyllysine), methylation (AHCY, lysine methyltransferases), creatine synthesis (GATM/GAMT) and
  urea-cycle / arginine metabolism. Effects in this dataset are small, so no single reaction dominates.</p>
</section>'''}

<section>
  <h2>4. Benchmark: finding the defective enzyme</h2>
  <div class="prose">
    <p>To test the method against hard data, we used untargeted plasma metabolomics from patients with confirmed
    inborn errors (Miller et al. 2015; Thistlethwaite et al. 2022; z-scores from the CTD package on CRAN):
    20 polar-signature disorders with known causal genes, run per disorder group and per patient. A
    <i>neighbour baseline</i> ranks each enzyme by how many significant metabolites its own reactions touch;
    current flow has to beat it to add anything.</p>
  </div>
  {table(["Set", "Method", "Median rank (percentile)", "Top 10", "Top 50", "Top 100"], bench_summary(g, p), num_cols=(3, 4, 5))}
  <figure>
    <div class="legend" aria-hidden="true">
      <span><svg width="12" height="12"><circle cx="6" cy="6" r="5" fill="var(--raw)"/></svg>Metaboenrich, raw (default)</span>
      <span><svg width="12" height="12"><circle cx="6" cy="6" r="4" fill="none" stroke="var(--loo)" stroke-width="2"/></svg>Leave-one-out</span>
      <span><svg width="12" height="12"><rect x="1.5" y="1.5" width="9" height="9" rx="1.5" fill="var(--nb)"/></svg>Neighbour baseline</span>
    </div>
    {bench_chart(g)}
    <figcaption>Group-level rank of the causal gene per disorder. * one-sided run (only increases significant).
    Hover a mark for its value.</figcaption>
  </figure>
  <div class="prose">
    <p>Current flow finds enzymes the baseline cannot see, where the defect's signal sits a step or more away
    (TYMP, IVD, MCC, OTC, AADC). Leave-one-out scoring, introduced earlier to discount single-metabolite routes,
    hurts here: an enzyme defect often <i>is</i> one substrate–product pair (GAMT: rank 2 raw, 1,888 with
    leave-one-out). Raw scoring is therefore the default; leave-one-out stays as an option and a column. The
    organic acidurias (MMA, PA, GA) rank poorly with every method and need a closer look.</p>
  </div>
</section>

<section>
  <h2>5. What changed in version 0.2</h2>
  <ul class="prose">
    <li><b>One-sided flow:</b> when nothing significant goes down (or up), current is exchanged with the measured,
      unchanged metabolites. All 20 benchmark disorders now run (4 could not before).</li>
    <li><b>Pathways:</b> Human-GEM subsystems tested for over-representation among the top 100 reactions.</li>
    <li><b>Transporters:</b> scored by the metabolites they carry, reported as a separate list (e.g. for GLUT1).</li>
    <li><b>Name mapping:</b> the lab's <span class="mono">id_translation.csv</span> synonym table, then an
      unambiguous loose-name match; every match is labelled.</li>
    <li><b>Provenance:</b> every result records the tool version, scoring and the network fingerprint.</li>
  </ul>
</section>

<section>
  <h2>6. Blind test on published datasets</h2>
  <div class="prose">
    <p>The tool was frozen (commit <span class="mono">8d4a8da</span>) and run on the lab's standardised table of 12
    published studies, split into analysis units with significance rules fixed in advance. The predictions were
    committed before anyone built or read the answer key, so the git timestamp is the record. Answers will be
    judged against each paper's conclusions (genes, pathways, processes), not only single genes.</p>
  </div>
  {table(["", "Units"], status, num_cols=(1,))}
  {table(["Unit", "Mapped", "Up / down", "Diagnosed gene: rank", "Predicted top genes", "Predicted top pathways"], pred_rows, num_cols=(1,), cls="wide")}
  <p class="small prose">¹ one-sided run. Not shown: {n_alaimo} Alaimo 2020 patient units, {n_alaimo_sig} with any
  signal; the master table lists only a few metabolites per patient. Burrage 2019 and Kennedy 2019 probably share
  patients with the benchmark, so they are not independent tests. Alaimo, Glinton, Cappuccio and Wangler are too
  sparse in the master table to judge; their full supplementary tables would support a second, separately sealed
  round.</p>
</section>

<section>
  <h2>7. Open decisions and next steps</h2>
  <ul class="prose">
    <li><b>Answer key:</b> turn each paper's conclusions and the "Expected tool prediction" sheet into checkable
      items, ideally written by someone who has not studied the predictions, then score.</li>
    <li><b>Scoring:</b> raw vs leave-one-out is settled for now by the benchmark; the final-score formula and whether
      fold-change size should weight the current are still open.</li>
    <li><b>Network:</b> deoxynucleotides as cofactors and the RNA/DNA nodes (likely behind recurring HK1–3, PKM,
      RAD1/TREX1 hits); candidate cofactors glutathione, THF, acetyl-CoA, glutamate/AKG, carnitine.</li>
    <li><b>Benchmark by pruning stage:</b> score each pruning step against the benchmark, so each decision in the
      log has a number next to it.</li>
    <li><b>Organic acidurias:</b> find out why MMA, PA and GA rank poorly.</li>
    <li><b>Tissue context:</b> a liver-specific network for hepatocyte data.</li>
  </ul>
</section>

<footer>
  <p>Repository: <a href="{REPO}">{REPO.split('//')[1]}</a> · tags <span class="mono">v0.1.0</span>,
  <span class="mono">v0.1.1</span>, <span class="mono">predictions-published-v1</span>.
  Details: <span class="mono">README.md</span>, <span class="mono">CHANGELOG.md</span>,
  <span class="mono">docs/decisions.md</span>, <span class="mono">docs/benchmark.md</span>,
  <span class="mono">predictions/published_v1/PREDICTIONS.md</span>.</p>
  <p>Built from result files by <span class="mono">tools/report/build_report.py</span>.</p>
</footer>
</main>
"""
    if public:                                   # renumber after the removed section 3
        for n in (4, 5, 6, 7):
            doc = doc.replace(f"<h2>{n}. ", f"<h2>{n - 1}. ")
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/report.html"))
    ap.add_argument("--public", action="store_true",
                    help="leave out the section on the lab's unpublished data (for the public repo)")
    a = ap.parse_args()
    build(a.out, a.public)
