"""Build the testing-data report (one self-contained HTML page): what data the
method is tested on, broken down by paper, disease and analysis unit.

Two collections:
  1. Inborn-error benchmark (answer key known): plasma z-scores from the CTD R
     package (Thistlethwaite 2020 dataset, Miller 2015), cases in benchmarks/cases.tsv.
  2. Blind published datasets (answer key withheld, decision D14): the analysis
     units in predictions/published_v2/units.tsv. Only the data description is
     used; prediction ranks are left out, and the answer key is never read.

Inputs (data/ and results/ are not committed):
    data/benchmark/{thistlethwaite2020.csv, miller2015.csv, cohorts_coded.tsv}   python benchmarks/get_benchmark_data.py
    benchmarks/cases.tsv
    results/bench_rw/flow/group_results.tsv        python -m metaboenrich.benchmark --levels group --out results/bench_rw/flow
    results/bench_rw_p/flow/patient_results.tsv    python -m metaboenrich.benchmark --levels patient --out results/bench_rw_p/flow
    predictions/published_v2/units.tsv

    python tools/report/build_testing_data_report.py --out docs/testing_data.html
"""
import argparse
import html
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from metaboenrich.benchmark import load_miller, load_t2020   # noqa: E402

GROUP = ROOT / "results" / "bench_rw" / "flow" / "group_results.tsv"
PATIENT = ROOT / "results" / "bench_rw_p" / "flow" / "patient_results.tsv"
UNITS = ROOT / "predictions" / "published_v2" / "units.tsv"


def sources_are(run_dir):
    """Which significant metabolites a run used as sources (D20). Runs made
    before the option existed used the increased ones."""
    try:
        return json.loads((Path(run_dir) / "run_info.json").read_text())["orientation"]["sources"]
    except (FileNotFoundError, KeyError):
        return "increased"


def up_down(df, orient):
    """(increased, decreased) count columns of a results table with sources/targets columns."""
    return (df.targets, df.sources) if orient == "decreased" else (df.sources, df.targets)


ORIENT = sources_are(GROUP.parent)
ROLE_OF = {"decreased": "sources" if ORIENT == "decreased" else "targets",
           "increased": "targets" if ORIENT == "decreased" else "sources"}
CASES = ROOT / "benchmarks" / "cases.tsv"

# Thistlethwaite 2020 cohorts that are not disorders.
T2020_NON_DISEASE = {"edta_refs": "EDTA-plasma reference samples", "hep_refs": "Heparin-plasma reference samples",
                     "alaimo": "Alaimo et al. samples (not used by the benchmark)"}
# Blind-set comparisons named only by disorder: the standard causal gene, added
# here for the cross-reference (marked as added in the page). A comparison that
# names two disorders (Sidorina 2024, e.g. "MMA vs CblC") gets both genes.
BLIND_ALIASES = [
    ("Arginase deficiency", "ARG1"), ("Argininosuccinate lyase deficiency", "ASL"),
    ("Citrullinemia type 1", "ASS1"), ("Ornithine transcarbamylase deficiency", "OTC"),
    ("MNGIE", "TYMP"), ("Lesch-Nyhan", "HPRT1"), ("VLCADD", "ACADVL"), ("Fumarase", "FH"),
    ("MMA", "MMUT"), ("PA vs", "PCCA;PCCB"), ("CblC", "MMACHC"),
]


def esc(x):
    return html.escape("" if x is None else str(x))


def fint(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "–"
    return f"{int(round(float(x))):,}"


def git_label():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return f"branch {run('rev-parse', '--abbrev-ref', 'HEAD')} · commit {run('rev-parse', '--short', 'HEAD')}"
    except OSError:
        return "unknown commit"


def table(headers, rows, num=(), cls=""):
    head = "".join(f'<th class="num">{h}</th>' if i in num else f"<th>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def genes_html(s):
    return " ".join(f"<code>{esc(g)}</code>" for g in str(s).split(";") if g)


# ---- benchmark --------------------------------------------------------------------
def anticoagulant(samples):
    kinds = {"HEP": "heparin", "EDTA": "EDTA"}
    found = sorted({kinds[s.split("-")[0]] for s in samples if s.split("-")[0] in kinds})
    return " + ".join(found) or "–"


def benchmark_data():
    zt, annt, ct = load_t2020()
    zm, annm, cm = load_miller()
    cases = pd.read_csv(CASES, sep="\t").fillna("")
    grp = pd.read_csv(GROUP, sep="\t").set_index("case")
    pat = pd.read_csv(PATIENT, sep="\t")
    t_raw = pd.read_csv(ROOT / "data" / "benchmark" / "thistlethwaite2020.csv", index_col=0, low_memory=False)

    datasets = {
        "t2020": {"z": zt, "ann": annt, "cohorts": ct},
        "miller": {"z": zm, "ann": annm, "cohorts": cm},
    }
    rows = []
    for _, c in cases.iterrows():
        d = datasets[c.dataset]
        samples = [s for s in d["cohorts"].get(c.cohort, []) if s in d["z"].columns]
        sub = d["z"][samples] if samples else None
        n = len(samples)
        need = max(2, -(-n // 2)) if n > 1 else 1
        measured = int((sub.notna().sum(axis=1) >= need).sum()) if n else 0
        row = {"case": c.case_id, "disorder": c.disorder, "dataset": c.dataset, "cohort": c.cohort,
               "genes": c.causal_genes, "category": c.category, "status": c.status, "note": c.note,
               "patients": n, "measured": measured,
               "plasma": anticoagulant(samples) if c.dataset == "t2020" else "heparin (Miller 2015)"}
        if c.case_id in grp.index:
            g = grp.loc[c.case_id]
            g_up, g_down = up_down(g, ORIENT)
            row.update(mapped=g.mapped, sig=g.sig_mapped, up=g_up, down=g_down, one_sided=bool(g.one_sided))
        p = pat[pat.case == c.case_id]
        if len(p):
            p_up, p_down = up_down(p, sources_are(PATIENT.parent))
            row.update(p_n=len(p), p_up=float(p_up.median()), p_down=float(p_down.median()),
                       p_none=int(((p.sources == 0) & (p.targets == 0)).sum()))
        rows.append(row)
    bench = pd.DataFrame(rows)

    t_info = {
        "samples": zt.shape[1], "metabolites": zt.shape[0],
        "hmdb": int((annt.HMDB != "").sum()), "kegg": int((annt.KEGG != "").sum()),
        "super": t_raw.SUPER_PATHWAY.dropna().value_counts().to_dict(),
        "platform": t_raw.PLATFORM.dropna().value_counts().to_dict(),
        "cohorts": {k: len(v) for k, v in ct.items() if k != "cohort"},
        "anticoag": {k: anticoagulant(v) for k, v in ct.items()},
    }
    m_info = {
        "samples": zm.shape[1], "metabolites": zm.shape[0], "hmdb": int((annm.HMDB != "").sum()),
        "unnamed": int(annm.name.astype(str).str.startswith("x -").sum()),
        "diagnoses": {k: len(v) for k, v in cm.items()},
    }
    return bench, t_info, m_info


# ---- blind set ----------------------------------------------------------------------
def subjects_in_rule(rule):
    m = re.search(r"over (\d+) subjects", str(rule))
    if m:
        return int(m.group(1))
    return 1 if "one subject" in str(rule) else None


def blind_data():
    u = pd.read_csv(UNITS, sep="\t")
    u["subjects"] = u.rule.map(subjects_in_rule)
    u["ok"] = u.status.eq("ok")
    for c in ("Comparison", "Matrix", "Disease_Group", "diagnosed_gene_in_data", "species", "Subject_ID"):
        u[c] = u[c].fillna("").astype(str)
    u["Subject_ID"] = u.Subject_ID.str.replace(r"\.0$", "", regex=True)
    u["year"] = u.source.str.extract(r"(\d{4})").astype(int)
    u["paper"] = u.source.str.replace(r" et al ", " et al. ", regex=False)

    def gene_for(r):
        if r.diagnosed_gene_in_data:
            return r.diagnosed_gene_in_data, False
        text = f"{r.Comparison} {r.Disease_Group}"
        found = [g for key, g in BLIND_ALIASES if key in text]
        return ";".join(found), bool(found)
    gg = u.apply(gene_for, axis=1)
    u["gene"] = [g for g, _ in gg]
    u["gene_added"] = [a for _, a in gg]
    return u


# ---- charts -------------------------------------------------------------------------
def patients_chart(bench):
    """Horizontal bars: patients per disorder, primary and secondary only, largest first."""
    b = bench[bench.status != "excluded"].sort_values(["patients", "disorder"], ascending=[False, True])
    W, left, right, bar, gap = 760, 300, 60, 16, 6
    xmax = max(40, int(b.patients.max()))
    ticks = [0, 10, 20, 30, 40] if xmax <= 40 else list(range(0, xmax + 10, 10))
    xmax = ticks[-1]
    sx = lambda v: left + (W - left - right) * v / xmax
    H = 30 + len(b) * (bar + gap) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="pc-t"><title id="pc-t">Patients per benchmark disorder</title>']
    for t in ticks:
        out.append(f'<line class="grid" x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="20" y2="{H - 26}"/>'
                   f'<text class="tick" x="{sx(t):.1f}" y="{H - 10}" text-anchor="middle">{t}</text>')
    y = 24
    for _, r in b.iterrows():
        cls = "bar-t" if r.dataset == "t2020" else "bar-m"
        label = f"{r.disorder}" + (" (secondary)" if r.status == "secondary" else "")
        out.append(f'<text class="lab" x="{left - 8}" y="{y + bar - 4}" text-anchor="end">{esc(label)}</text>'
                   f'<rect class="{cls}" x="{left}" y="{y}" width="{sx(r.patients) - left:.1f}" height="{bar}" rx="2"/>'
                   f'<text class="val" x="{sx(r.patients) + 5:.1f}" y="{y + bar - 4}">{r.patients}</text>')
        y += bar + gap
    out[0] = out[0].replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H + 22}"')
    out.append(f'<text class="tick" x="{(left + W - right) / 2:.0f}" y="{H + 14}" text-anchor="middle">patients</text></svg>')
    return "".join(out)


def signal_chart(bench):
    """Diverging bars: significant metabolites on the network at group level, increased vs decreased."""
    b = bench[bench.status != "excluded"].dropna(subset=["up"]).sort_values("up", ascending=False)
    W, mid, bar, gap = 800, 560, 16, 6
    half = 200
    vmax = max(50, int(max(b.up.max(), b.down.max())))
    vmax = int(np.ceil(vmax / 10) * 10)
    s = lambda v: half * v / vmax
    H = 30 + len(b) * (bar + gap) + 30
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-labelledby="sg-t"><title id="sg-t">Significant metabolites on the network per disorder, decreased and increased</title>']
    for t in range(0, vmax + 1, 10):
        for xx in {mid - s(t), mid + s(t)}:
            out.append(f'<line class="grid" x1="{xx:.1f}" x2="{xx:.1f}" y1="20" y2="{H - 26}"/>'
                       f'<text class="tick" x="{xx:.1f}" y="{H - 10}" text-anchor="middle">{t}</text>')
    out.append(f'<text class="tick" x="{mid - half / 2:.0f}" y="14" text-anchor="middle">decreased ({ROLE_OF["decreased"]})</text>'
               f'<text class="tick" x="{mid + half / 2:.0f}" y="14" text-anchor="middle">increased ({ROLE_OF["increased"]})</text>')
    y = 24
    for _, r in b.iterrows():
        out.append(f'<text class="lab" x="{mid - half - 10}" y="{y + bar - 4}" text-anchor="end">{esc(r.disorder)}</text>'
                   f'<rect class="bar-down" x="{mid - s(r.down):.1f}" y="{y}" width="{s(r.down):.1f}" height="{bar}" rx="2"/>'
                   f'<rect class="bar-up" x="{mid}" y="{y}" width="{s(r.up):.1f}" height="{bar}" rx="2"/>')
        if r.down:
            out.append(f'<text class="val" x="{mid - s(r.down) - 4:.1f}" y="{y + bar - 4}" text-anchor="end">{int(r.down)}</text>')
        out.append(f'<text class="val" x="{mid + s(r.up) + 4:.1f}" y="{y + bar - 4}">{int(r.up)}</text>')
        y += bar + gap
    out.append(f'<line class="axis" x1="{mid}" x2="{mid}" y1="20" y2="{H - 26}"/></svg>')
    return "".join(out)


# ---- page ---------------------------------------------------------------------------
CSS = """
:root{
  /* layout: one reading column (max 1040px) of sections; tables and charts scroll inside their own boxes */
  --page:#f3f5f7;--surface:#fbfcfd;--ink:#12161b;--ink-2:#4a5360;--muted:#6b7480;--rule:#dfe4ea;--axis:#c5ccd4;
  --accent:#1f63b5;--note:#eef3fa;--note-ink:#1c3e66;
  --t2020:#2a78d6;--miller:#d27a12;--up:#c2410c;--down:#1d6fb8;--warn-bg:#fdf3e7;--warn-ink:#7a4210;
  --ok:#1d7a4f;--skip:#8a6a3b;
  --sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
  color-scheme:light;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;--axis:#38414b;
  --accent:#6ea6ec;--note:#18222e;--note-ink:#bcd4f2;
  --t2020:#5c9be6;--miller:#eda04a;--up:#f08a5d;--down:#5ea3e6;--warn-bg:#2a2016;--warn-ink:#f1c48e;--ok:#5cc596;--skip:#c09a63}}
:root[data-theme="dark"]{
  color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;--axis:#38414b;
  --accent:#6ea6ec;--note:#18222e;--note-ink:#bcd4f2;
  --t2020:#5c9be6;--miller:#eda04a;--up:#f08a5d;--down:#5ea3e6;--warn-bg:#2a2016;--warn-ink:#f1c48e;--ok:#5cc596;--skip:#c09a63}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 var(--sans);padding-inline:clamp(16px,4vw,40px);padding-block:40px 72px}
main{max-width:1040px;margin-inline:auto;display:grid;gap:48px}
.prose{max-width:72ch;display:grid;gap:12px}
h1,h2,h3{text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:clamp(30px,4vw,40px);font-weight:700;letter-spacing:-.015em}
h2{font-size:23px;font-weight:650;padding-top:10px;border-top:1px solid var(--rule)}
h3{font-size:18px;font-weight:650}
p,ul{margin:0}
ul{padding-left:1.2em;display:grid;gap:6px}
a{color:var(--accent)}
a:focus-visible,summary:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
code{font-family:var(--mono);font-size:.86em}
section{display:grid;gap:18px}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:18px;color:var(--ink-2);max-width:68ch}
.small{font-size:14px;color:var(--ink-2);max-width:72ch}
.toc{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:14px}
.summary{background:var(--note);color:var(--note-ink);border-radius:8px;padding:18px 22px;display:grid;gap:10px}
.summary p,.summary li{color:var(--note-ink)}
.warn{background:var(--warn-bg);color:var(--warn-ink);border-radius:8px;padding:14px 18px;display:grid;gap:8px;font-size:14.5px}
.warn li,.warn p{color:var(--warn-ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 14px;display:grid;gap:2px;min-width:0}
.tile b{font:600 24px/1.2 var(--sans);font-variant-numeric:tabular-nums}
.tile span{font-size:13px;color:var(--ink-2)}
.papers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.card{background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:18px 20px;display:grid;gap:10px;align-content:start;min-width:0}
.card h3 small{display:block;font:400 13px var(--sans);color:var(--muted);margin-top:4px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:14px;margin:0}
.kv dt{color:var(--muted)}.kv dd{margin:0;min-width:0}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{text-align:left;padding:8px 11px;border-bottom:1px solid var(--rule);vertical-align:top}
thead th{font:500 11.5px/1.3 var(--mono);color:var(--muted);white-space:nowrap;background:var(--surface)}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
tr.group td{background:var(--page);font:500 11px var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);padding:7px 11px}
td code{font-size:12px;white-space:nowrap}
.pill{display:inline-block;font:500 11.5px/1.5 var(--mono);border-radius:999px;padding:0 8px;border:1px solid currentColor;white-space:nowrap}
.pill.primary{color:var(--ok)}.pill.secondary{color:var(--accent)}.pill.excluded{color:var(--muted)}
.pill.ok{color:var(--ok)}.pill.skip{color:var(--skip)}
.pill.t2020{color:var(--t2020)}.pill.miller{color:var(--miller)}
.added{color:var(--muted);font-size:12px}
figure{margin:0;display:grid;gap:10px;background:var(--surface);border:1px solid var(--rule);border-radius:10px;padding:16px}
.scroll{overflow-x:auto}
.scroll svg{display:block;min-width:640px;width:100%;height:auto}
figcaption{font-size:14px;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;font-size:13px;color:var(--ink-2)}
.sw{display:inline-block;width:11px;height:11px;border-radius:2px;margin-right:6px;vertical-align:-1px}
svg text{font-family:var(--sans);fill:var(--ink)}
svg .lab{font-size:12px}svg .val{font:11px var(--mono);fill:var(--ink-2)}svg .tick{font:10.5px var(--mono);fill:var(--muted)}
svg .grid{stroke:var(--rule);stroke-width:1}svg .axis{stroke:var(--axis);stroke-width:1.2}
svg .bar-t{fill:var(--t2020)}svg .bar-m{fill:var(--miller)}svg .bar-up{fill:var(--up)}svg .bar-down{fill:var(--down)}
details{background:var(--surface);border:1px solid var(--rule);border-radius:10px}
details[open]{padding-bottom:12px}
summary{cursor:pointer;padding:14px 18px;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 14px;list-style-position:outside}
summary h3{font-size:16.5px;display:inline}
summary .meta{font-size:13.5px;color:var(--ink-2)}
details > :not(summary){margin-inline:18px}
details .tbl{margin-top:4px}
footer{font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:16px;display:grid;gap:6px}
@media (prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
"""


def build(out):
    bench, t, m = benchmark_data()
    u = blind_data()
    used = bench[bench.status != "excluded"]
    prim = bench[bench.status == "primary"]

    # ---- headline tiles
    tiles = [
        (f"{len(prim)}", "primary benchmark disorders"),
        (f"{int(prim.patients.sum())}", "primary benchmark patients"),
        (f"{u.paper.nunique()}", "published papers in the blind set"),
        (f"{len(u)} / {int(u.ok.sum())}", "blind analysis units / run"),
    ]
    tiles_html = "".join(f'<div class="tile"><b>{a}</b><span>{esc(b)}</span></div>' for a, b in tiles)

    # ---- benchmark paper cards
    super_txt = ", ".join(f"{k.lower()} {v}" for k, v in list(t["super"].items())[:6])
    plat_txt = ", ".join(f"{k} {v}" for k, v in t["platform"].items())
    t_dis = [k for k in t["cohorts"] if k not in T2020_NON_DISEASE]
    m_used = sorted(set(bench[bench.dataset == "miller"].cohort))
    cards = f"""
<div class="papers">
  <article class="card">
    <h3><span class="pill t2020">t2020</span> Thistlethwaite et al.<small>Sci Rep 12:6556 (2022), doi:10.1038/s41598-022-10415-5.
    Dataset <code>Thistlethwaite2020</code> in the CTD R package.</small></h3>
    <dl class="kv">
      <dt>Samples</dt><dd>{t['samples']:,} plasma samples in {len(t['cohorts'])} cohorts ({len(t_dis)} disorders, 2 reference groups, the Alaimo set)</dd>
      <dt>Metabolites</dt><dd>{t['metabolites']:,} (HMDB id for {t['hmdb']:,}, KEGG id for {t['kegg']:,})</dd>
      <dt>Values</dt><dd>z-scores per metabolite against reference plasma</dd>
      <dt>Plasma</dt><dd>heparin (<code>HEP-</code> samples) and EDTA (<code>EDTA-</code>) cohorts</dd>
      <dt>Annotated classes</dt><dd>{esc(super_txt)}</dd>
      <dt>Platforms</dt><dd>{esc(plat_txt)}</dd>
      <dt>Used for</dt><dd>{len(bench[(bench.dataset == 't2020') & (bench.status != 'excluded')])} benchmark disorders (main set)</dd>
    </dl>
  </article>
  <article class="card">
    <h3><span class="pill miller">miller</span> Miller et al.<small>J Inherit Metab Dis 38:1029 (2015), doi:10.1007/s10545-015-9843-7.
    Dataset <code>Miller2015</code> in the CTD R package.</small></h3>
    <dl class="kv">
      <dt>Samples</dt><dd>{m['samples']:,} plasma samples, {len(m['diagnoses'])} diagnosis groups (incl. {m['diagnoses'].get('No biochemical genetic diagnosis', 0)} with no biochemical genetic diagnosis)</dd>
      <dt>Metabolites</dt><dd>{m['metabolites']:,} ({m['unnamed']:,} unnamed <code>x - …</code> features; HMDB id for {m['hmdb']:,})</dd>
      <dt>Values</dt><dd>z-scores against the study's reference samples</dd>
      <dt>Plasma</dt><dd>heparin</dd>
      <dt>Used for</dt><dd>{len(m_used)} disorders not in Thistlethwaite (the rest of its heparin cohorts are also in Thistlethwaite, so those come from there)</dd>
    </dl>
  </article>
</div>"""

    # ---- benchmark per-disease table
    rows = []
    for status in ("primary", "secondary", "excluded"):
        sub = bench[bench.status == status].sort_values(["category", "disorder"])
        if not len(sub):
            continue
        rows.append([f'<tr class="group"><td colspan="11">{status} ({len(sub)})</td></tr>'])
        for _, r in sub.iterrows():
            rows.append([
                f"<b>{esc(r.disorder)}</b>" + (f'<br><span class="added">{esc(r.note)}</span>' if r.note else ""),
                esc(r.category), genes_html(r.genes),
                f'<span class="pill {r.dataset}">{r.dataset}</span>', esc(r.plasma), fint(r.patients), fint(r.measured),
                fint(r.get("sig")), f'{fint(r.get("up"))} / {fint(r.get("down"))}' + (" ¹" if r.get("one_sided") is True else ""),
                f'{fint(r.get("p_up"))} / {fint(r.get("p_down"))}' if pd.notna(r.get("p_n")) else "–",
                fint(r.get("p_none")) if pd.notna(r.get("p_n")) else "–"])
    head = ["Disorder", "Category", "Causal gene(s)", "Data", "Plasma", "Patients", "Metabolites tested",
            "Significant (group)", "Up / down on network", "Patient median up / down", "Patients with no signal"]
    body = []
    for r in rows:
        if len(r) == 1:
            body.append(r[0])
        else:
            body.append("<tr>" + "".join(f'<td class="num">{c}</td>' if i in (5, 6, 7, 8, 9, 10) else f"<td>{c}</td>"
                                         for i, c in enumerate(r)) + "</tr>")
    hd = "".join(f'<th class="num">{h}</th>' if i in (5, 6, 7, 8, 9, 10) else f"<th>{h}</th>" for i, h in enumerate(head))
    disease_table = f'<div class="tbl"><table><thead><tr>{hd}</tr></thead><tbody>{"".join(body)}</tbody></table></div>'

    # ---- by category
    cat = used.groupby("category").agg(disorders=("case", "count"), patients=("patients", "sum"),
                                       names=("disorder", lambda s: "; ".join(sorted(s)))).reset_index()
    cat = cat.sort_values("patients", ascending=False)
    cat_table = table(["Category", "Disorders", "Patients", "Which"],
                      [[esc(r.category), r.disorders, r.patients, esc(r.names)] for _, r in cat.iterrows()], num=(1, 2))

    # ---- cohorts of Thistlethwaite not used + Miller diagnoses
    case_by_cohort = {(r.dataset, r.cohort): r for _, r in bench.iterrows()}
    t_rows = []
    for k, n in sorted(t["cohorts"].items(), key=lambda x: -x[1]):
        r = case_by_cohort.get(("t2020", k))
        use = (f'<span class="pill {r.status}">{r.status}</span> {esc(r.disorder)}' if r is not None
               else esc(T2020_NON_DISEASE.get(k, "not used")))
        t_rows.append([f"<code>{esc(k)}</code>", n, esc(t["anticoag"].get(k, "")), use])
    m_rows = []
    for k, n in sorted(m["diagnoses"].items(), key=lambda x: -x[1]):
        r = case_by_cohort.get(("miller", k))
        if r is not None:
            use = f'<span class="pill {r.status}">{r.status}</span> {esc(r.disorder)}'
        elif k == "No biochemical genetic diagnosis":
            use = "not used (no diagnosis)"
        else:
            use = "taken from Thistlethwaite instead (same heparin cohort)"
        m_rows.append([esc(k), n, use])

    # ---- blind set, per paper
    paper_blocks = []
    for paper, g in sorted(u.groupby("paper"), key=lambda x: (x[1].year.iloc[0], x[0])):
        species = "; ".join(sorted({s for s in g.species if s}))
        matrices = "; ".join(sorted({s for s in g.Matrix if s}))
        genes = sorted({x for x in g.gene if x})
        added = g.gene_added.any()
        rule_txt = "; ".join(sorted({re.sub(r"over \d+ subjects", "over N subjects", str(x)) for x in g.rule}))
        is_alaimo = paper.startswith("Alaimo")
        if is_alaimo:
            ok = g[g.ok]
            unit_rows = [[esc(r.Subject_ID), genes_html(r.gene), fint(r.metabolites), fint(r.mapped),
                          f"{fint(r.increased)} / {fint(r.decreased)}", '<span class="pill ok">run</span>']
                         for _, r in ok.sort_values("gene").iterrows()]
            skipped = g[~g.ok]
            skip_genes = ", ".join(sorted({x for x in skipped.gene if x}))
            unit_html = table(["Patient", "Diagnosed gene", "Metabolites", "Mapped", "Up / down", "Status"],
                              unit_rows, num=(2, 3, 4))
            unit_html += (f'<p class="small">{len(skipped)} more patients were skipped because fewer than 3 metabolites '
                          f'passed the rule. Their genes: {esc(skip_genes) or "–"}.</p>')
        else:
            unit_rows = []
            for _, r in g.iterrows():
                status = '<span class="pill ok">run</span>' if r.ok else f'<span class="pill skip">{esc(r.status)}</span>'
                gene = genes_html(r.gene) + (' <span class="added">(added)</span>' if r.gene_added and r.gene else "")
                unit_rows.append([esc(r.Comparison), esc(r.Disease_Group) or "–", esc(r.Matrix), gene,
                                  fint(r.subjects), fint(r.metabolites), fint(r.mapped),
                                  f"{fint(r.increased)} / {fint(r.decreased)}" if r.ok else "–", status])
            unit_html = table(["Comparison", "Disease group", "Matrix", "Gene", "Subjects", "Metabolites", "Mapped",
                               "Up / down", "Status"], unit_rows, num=(4, 5, 6, 7))
        gene_txt = ", ".join(genes[:14]) + (f" and {len(genes) - 14} more" if len(genes) > 14 else "")
        paper_blocks.append(f"""
<details{' open' if paper.startswith('Adam') else ''}>
  <summary><h3>{esc(paper)}</h3><span class="meta">{len(g)} unit{'s' if len(g) != 1 else ''}, {int(g.ok.sum())} run ·
  {esc(species)} · {esc(matrices)}</span></summary>
  <dl class="kv">
    <dt>Comparisons</dt><dd>{esc('; '.join(sorted(set(g.Comparison))))}</dd>
    <dt>Genes</dt><dd>{esc(gene_txt) or '–'}{' <span class="added">(some added from the disorder name)</span>' if added else ''}</dd>
    <dt>Rule</dt><dd>{esc(rule_txt)}</dd>
  </dl>
  {unit_html}
</details>""")

    papers_summary = table(
        ["Paper", "Species", "Sample type", "Units", "Run", "Metabolites per unit (median)"],
        [[esc(p), esc("; ".join(sorted({s for s in g.species if s}))), esc("; ".join(sorted({s for s in g.Matrix if s}))),
          len(g), int(g.ok.sum()), fint(g.metabolites.median())]
         for p, g in sorted(u.groupby("paper"), key=lambda x: (x[1].year.iloc[0], x[0]))], num=(3, 4, 5))

    # ---- disease crosswalk: benchmark genes that also appear in the blind set
    cross = []
    for _, r in bench.iterrows():
        gs = set(r.genes.split(";"))
        hit = u[u.gene.map(lambda s: bool(gs & set(re.findall(r"[A-Z][A-Z0-9]+", str(s)))))]
        if len(hit):
            papers = "; ".join(f"{p} ({len(h)} unit{'s' if len(h) != 1 else ''})" for p, h in hit.groupby("paper"))
            cross.append([esc(r.disorder), genes_html(r.genes), f'<span class="pill {r.status}">{r.status}</span>',
                          fint(r.patients), esc(papers)])
    cross_table = table(["Disorder", "Gene(s)", "Benchmark status", "Benchmark patients", "Also in the blind set"],
                        cross, num=(3,))

    # ---- overlaps
    burrage = u[u.paper.str.startswith("Burrage")][["Disease_Group", "subjects"]]
    burrage_txt = "; ".join(f"{esc(r.Disease_Group)}: {fint(r.subjects)}" for _, r in burrage.iterrows())
    bench_uc = "; ".join(f"{esc(r.disorder)}: {r.patients}" for _, r in bench[bench.case.isin(["otc", "asld", "arg", "cit"])].iterrows())

    page = f"""<title>Metaboenrich Testing Data</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style>
<main>
<header class="prose">
  <div class="eyebrow">Metaboenrich · testing data</div>
  <h1>Testing data</h1>
  <p class="lede">What the method is tested on: which papers the data come from, which diseases they cover, how many
  patients each has, and how a metabolite counts as changed. Two collections, kept apart on purpose.</p>
  <nav class="toc"><a href="#overview">Overview</a><a href="#bench-papers">Benchmark: papers</a>
  <a href="#bench-diseases">Benchmark: diseases</a><a href="#bench-cohorts">Benchmark: cohorts</a>
  <a href="#blind">Blind set: papers</a><a href="#crosswalk">Disease crosswalk</a><a href="#overlap">Overlaps</a></nav>
</header>

<section id="overview">
  <div class="tiles">{tiles_html}</div>
  <div class="summary">
    <p><strong>1. Inborn-error benchmark (answer known).</strong> Untargeted plasma metabolomics from patients with a
    confirmed single-enzyme defect, as per-sample z-scores against reference plasma. The question is where the method ranks
    the known causal gene among the 2,302 genes on the network. These data were used to choose settings (raw scoring,
    decision D9) and to compare methods (current flow vs the absorbing walk, D17).</p>
    <p><strong>2. Blind published datasets (answer withheld).</strong> Metabolite tables the lab collected from published
    papers, split into analysis units. Predictions were made and committed before anyone opened the answer key (each
    paper's conclusions and the "Expected tool prediction" sheet; decision D14). This page uses only the data description
    in <code>predictions/published_v2/units.tsv</code>. It shows no prediction results and reads no answer key.</p>
  </div>
</section>

<section id="bench-papers">
  <h2>1 · Benchmark: source papers</h2>
  <p class="small">Both datasets ship with the CTD R package on CRAN (v1.3, MIT licence) and are fetched by
  <code>python benchmarks/get_benchmark_data.py</code>. Cases and causal genes are in <code>benchmarks/cases.tsv</code>.</p>
  {cards}
</section>

<section id="bench-diseases">
  <h2>1 · Benchmark: diseases</h2>
  <p class="small">{len(prim)} primary disorders with a polar-metabolite signature and one known causal gene (or enzyme complex),
  one secondary (cobalamin defects, many possible genes), and {len(bench[bench.status == 'excluded'])} excluded with the reason given.
  <b>Metabolites tested</b>: measured in at least half of the disorder's patients (at least 2), the group-level rule's
  requirement. <b>Up / down on network</b>: significant metabolites that map to a network node, increased ({ROLE_OF["increased"]}) and
  decreased ({ROLE_OF["decreased"]}); these drive the analysis.</p>
  {disease_table}
  <p class="small">¹ One-sided: only increases are significant, so the analysis exchanges current with the measured,
  unchanged metabolites. Significance at group level: z-test on the mean z, BH q &lt; 0.05 and |mean z| ≥ 1. At patient
  level: |z| ≥ 2. Counts come from the current-flow benchmark run (<code>results/bench_rw</code>); the walk uses the same
  significant metabolites.</p>
  <figure>
    <div class="legend"><span><i class="sw" style="background:var(--t2020)"></i>Thistlethwaite</span>
    <span><i class="sw" style="background:var(--miller)"></i>Miller 2015</span></div>
    <div class="scroll">{patients_chart(bench)}</div>
    <figcaption>Patients per disorder, largest first. {int((used.patients <= 4).sum())} of {len(used)} disorders
    have 4 patients or fewer, which makes their group-level signal fragile.</figcaption>
  </figure>
  <figure>
    <div class="legend"><span><i class="sw" style="background:var(--down)"></i>decreased ({ROLE_OF["decreased"]})</span>
    <span><i class="sw" style="background:var(--up)"></i>increased ({ROLE_OF["increased"]})</span></div>
    <div class="scroll">{signal_chart(bench)}</div>
    <figcaption>Significant metabolites on the network at group level. Disorders with no decreases run one-sided.
    Fewest: {esc(', '.join(f"{r.disorder} ({int(r.up + r.down)})" for _, r in used.dropna(subset=['up']).assign(n=lambda d: d.up + d.down).nsmallest(3, 'n').iterrows()))}.</figcaption>
  </figure>
  <h3>By category</h3>
  {cat_table}
</section>

<section id="bench-cohorts">
  <h2>1 · Benchmark: cohorts in each dataset</h2>
  <p class="small">Every cohort in the two datasets, and what the benchmark does with it.</p>
  <h3>Thistlethwaite</h3>
  {table(["Cohort", "Samples", "Plasma", "Use"], t_rows, num=(1,))}
  <h3>Miller 2015</h3>
  {table(["Diagnosis", "Samples", "Use"], m_rows, num=(1,))}
</section>

<section id="blind">
  <h2>2 · Blind set: papers</h2>
  <p class="small">Built by <code>benchmarks/published_predictions.py</code> from the lab's standardised master table.
  One unit per (paper, comparison, sample type), split further by disease group for z-score data and by patient for
  Alaimo 2020. Rules were fixed before any result was seen: log2FC data, p &lt; 0.05; several subjects with z-scores,
  BH q &lt; 0.05 and |mean z| ≥ 1; one subject, |z| ≥ 2. Units with fewer than 3 usable metabolites are skipped.</p>
  {papers_summary}
  {''.join(paper_blocks)}
</section>

<section id="crosswalk">
  <h2>Disease crosswalk</h2>
  <p class="small">Benchmark disorders whose causal gene also appears in the blind set. Where a blind unit names only the
  disorder (Burrage 2019, Du 2023, Sidorina 2024, Reisz 2023, Sebaa 2023, Hou 2017), its standard gene was added here
  for matching; those are marked "added" in the paper tables.</p>
  {cross_table}
</section>

<section id="overlap">
  <h2>Possible patient overlap</h2>
  <div class="warn">
    <p>Several papers in both collections come from the same clinical metabolomics programme, so some patients may appear
    in both. Nothing here proves an overlap; it should be checked before blind results are scored.</p>
    <ul>
      <li>The Thistlethwaite data include an <code>alaimo</code> cohort of {t['cohorts'].get('alaimo', 0)} samples
      (<code>ALAIMO-…</code>). The benchmark does not use it, but the blind set's Alaimo 2020 patients may be the same people.</li>
      <li>Urea-cycle disorders. Burrage 2019 groups (subjects): {burrage_txt}. Benchmark cohorts (patients): {bench_uc}.</li>
      <li>GABA-transaminase deficiency appears in the benchmark ({fint(bench.set_index('case').patients.get('abat'))} patients)
      and in Kennedy 2019.</li>
      <li>Miller 2015's heparin cohorts are also in Thistlethwaite; the benchmark takes each disorder from one dataset only,
      so no patient is counted twice within the benchmark.</li>
    </ul>
  </div>
</section>

<footer>
  <span>Built by <code>tools/report/build_testing_data_report.py</code> from the CTD data, <code>benchmarks/cases.tsv</code>,
  the current-flow benchmark results and <code>predictions/published_v2/units.tsv</code> ({esc(git_label())}).</span>
  <span>Related: <a href="benchmark.md">benchmark.md</a> · <a href="rw_report.html">rw_report.html</a> ·
  <a href="decisions.md">decisions.md</a> (D9, D14, D17) · <a href="../predictions/published_v2/PREDICTIONS.md">PREDICTIONS.md</a></span>
</footer>
</main>
"""
    doc = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
           '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
           + page.replace("<main>", "</head>\n<body>\n<main>", 1) + "</body>\n</html>\n")
    Path(out).write_text(doc, encoding="utf-8")
    print(f"wrote {out} ({Path(out).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "testing_data.html"))
    build(ap.parse_args().out)
