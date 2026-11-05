"""Build the absorbing-walk report (one self-contained HTML page): the
inborn-error benchmark and the lab's published datasets.

Group level only (decision D21): each disorder's patients are pooled. The walk
is reported under both orientations (D20):
    sources = up    significantly increased metabolites are the sources,
                    decreased ones the targets (--sources increased; protocol §1.3)
    sources = down  significantly decreased metabolites are the sources,
                    increased ones the targets (--sources decreased; the default)

Every number comes from benchmark result files (results/ is not committed):
    results/bench_rw/<cfg>/group_results.tsv      sources = down
    results/bench_rw_up/<cfg>/group_results.tsv   sources = up
    cfg = flow, walk_full, walk_k0.5, walk_k1, walk_k2
Each run's run_info.json must record the orientation its folder claims.

Produce them with
    python -m metaboenrich.benchmark --levels group --out results/bench_rw/flow
    python -m metaboenrich.benchmark --levels group --method walk --out results/bench_rw/walk_full
    python -m metaboenrich.benchmark --levels group --method walk --walk-kappa K --out results/bench_rw/walk_kK
        (K = 0.5, 1, 2)
and the same five commands with --sources increased and results/bench_rw_up/ in
place of results/bench_rw/, for example
    python -m metaboenrich.benchmark --levels group --sources increased --out results/bench_rw_up/flow
    python -m metaboenrich.benchmark --levels group --sources increased --method walk --out results/bench_rw_up/walk_full

The published datasets: the analysis units in predictions/published_v2/units.tsv
(read as on the testing-data page) and the causal gene's rank per unit and method in
    results/published_eval/units.tsv     python benchmarks/published_eval.py --out results/published_eval

Then:
    python tools/report/build_rw_report.py --out docs/rw_report.html
"""
import argparse
import html
import json
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from build_testing_data_report import published_data     # the published units, causal genes completed by the answer key

ROOT = Path(__file__).resolve().parents[2]
# tag -> (folder, the orientation its runs must record)
RUNS = {"up": (ROOT / "results" / "bench_rw_up", "increased"),
        "down": (ROOT / "results" / "bench_rw", "decreased")}
KAPPAS = ["0.5", "1", "2"]
CFGS = ["flow", "walk_full"] + [f"walk_k{k}" for k in KAPPAS]
RANK = "rank_metaboenrich_raw"
LABEL = {"up": "sources = up", "down": "sources = down"}
EVAL = ROOT / "results" / "published_eval" / "units.tsv"
EVAL_STATUS = {"ok": "evaluated", "skipped: fewer than 3 metabolites": "skipped: &lt; 3 metabolites",
               "left out: single subject (D21)": "left out: one subject (D21)"}


# ---- formatting -----------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def isnan(x):
    return x is None or (isinstance(x, float) and math.isnan(x))


def fr(x):
    """A rank: integer when whole, else up to two decimals (ties are averaged,
    and a median of averaged ranks can end in .25 or .75)."""
    if isnan(x):
        return "–"
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.2f}".rstrip("0")


def fw(x):
    """A walk rank; no rank means no walker reached a target."""
    return "not scorable" if isnan(x) else fr(x)


def share(x):
    """A share of walkers: one decimal, two when one would read as 0.0%."""
    if x == 0:
        return "0%"
    s = f"{100 * x:.1f}"
    return f"{100 * x:.2f}%" if float(s) == 0 else f"{s}%"


def table(headers, rows, num=(), cls=""):
    head = "".join(f'<th class="num">{h}</th>' if i in num else f"<th>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def change(a, b):
    """Glyph for a rank change from current flow (a) to the walk (b); lower rank is better.
    A glyph, so the direction never rests on colour."""
    if isnan(b):
        return ""
    if b < a:
        return '<span class="up" title="walk ranks it higher than current flow">▲</span>'
    if b > a:
        return '<span class="down" title="walk ranks it lower than current flow">▼</span>'
    return '<span class="same" title="same rank as current flow">=</span>'


def plural(n, word, many=None):
    return f"{n} {word if n == 1 else (many or word + 's')}"


def top(s, k):
    return int((s <= k).sum())                  # a missing rank is never in the top k


def genes_code(s):
    return " ".join(f"<code>{esc(g)}</code>" for g in str(s).split(";") if g)


# ---- data -----------------------------------------------------------------------
def run_info(run_dir):
    return json.loads((Path(run_dir) / "run_info.json").read_text())


def check_cfg(info, cfg, where):
    """The folder name must match the method and κ the run recorded."""
    method = "flow" if cfg == "flow" else "walk"
    kappa = None if cfg in ("flow", "walk_full") else float(cfg[len("walk_k"):])
    assert info.get("method") == method, f"{where}: run_info says method = {info.get('method')}, expected {method}"
    assert info.get("walk_kappa") == kappa, f"{where}: run_info says walk_kappa = {info.get('walk_kappa')}, expected {kappa}"


def load():
    raw = {}
    for tag, (folder, expect) in RUNS.items():
        for cfg in CFGS:
            info = run_info(folder / cfg)
            got = info["orientation"]["sources"]      # which significant metabolites were the sources (D20)
            assert got == expect, f"{folder.name}/{cfg}: run_info says sources = {got}, expected {expect}"
            check_cfg(info, cfg, f"{folder.name}/{cfg}")
            raw[tag, cfg] = pd.read_csv(folder / cfg / "group_results.tsv", sep="\t")
    ref = raw["down", "flow"]
    for key, df in raw.items():
        assert (df.case.values == ref.case.values).all(), f"row order differs in {key}"

    fu, fd = raw["up", "flow"], raw["down", "flow"]
    # Current flow is symmetric under the swap: identical ranks, swapped role counts.
    assert fu[RANK].equals(fd[RANK]), "current-flow ranks differ between the two orientations"
    assert fu.rank_neighbour.equals(fd.rank_neighbour), "neighbour ranks differ between the two orientations"
    assert (fu.sources.values == fd.targets.values).all() and (fu.targets.values == fd.sources.values).all()

    grp = fd[["case", "disorder", "status", "causal_in_network", "universe"]].copy()
    grp["up"], grp["down"] = fu.sources.values, fu.targets.values      # sources = up: increased are sources
    grp["nb"] = fd.rank_neighbour.values
    grp["flow"] = fd[RANK].values
    for tag in RUNS:
        grp[f"w_{tag}"] = raw[tag, "walk_full"][RANK].values
        grp[f"s_{tag}"] = raw[tag, "walk_full"].walk_success_fraction.values
        for k in KAPPAS:
            grp[f"k{k}_{tag}"] = raw[tag, f"walk_k{k}"][RANK].values
    return grp, int(grp.universe.iloc[0])


def load_published():
    """Every published analysis unit with its ranks from benchmarks/published_eval.py. Up / down are the
    significant metabolites on the network in the evaluation's own run (current flow, sources = down)."""
    u = published_data().drop(columns=["one_sided"])
    e = pd.read_csv(EVAL, sep="\t", dtype={"causal_in_network": str})
    e = e.fillna({"causal_in_network": "", "kind": ""}).rename(columns={"status": "eval_status"})
    cols = ["unit", "eval_status", "kind", "causal_in_network", "universe", "sources_down", "targets_up", "one_sided",
            "rank_flow", "rank_walk_up", "rank_walk_down", "rank_neighbour", "walk_success_walk_up",
            "walk_success_walk_down"]
    p = u.merge(e[cols], on="unit", how="left", validate="1:1")
    assert p.eval_status.notna().all(), f"units missing from {EVAL}: {', '.join(p.unit[p.eval_status.isna()])}"
    p["short"] = p.paper.str.replace(r" et al\.? ", " ", regex=True)
    p["label"] = p.short + " · " + p.Disease_Group.where(p.Disease_Group != "", p.Comparison)
    p["ranked"] = p.eval_status.eq("ok") & p.causal_in_network.ne("") & p.rank_flow.notna()
    return p.sort_values(["year", "paper", "unit"])


def git_label():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = " + uncommitted changes" if run("status", "--porcelain", "--untracked-files=no") else ""
        return f"branch {run('branch', '--show-current')} · commit {run('rev-parse', '--short', 'HEAD')}{dirty}"
    except OSError:
        return ""


# ---- chart ----------------------------------------------------------------------
def log_x(rank, x0, w, U):
    return x0 + math.log10(max(rank, 1.0)) / math.log10(U) * w


def x_ticks(x0, w, U, y0, y1, label_y):
    out = []
    for v in (1, 10, 100, 1000):
        x = log_x(v, x0, w, U)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}"/>'
                   f'<text class="tick" x="{x:.1f}" y="{label_y}" text-anchor="middle">{v:,}</text>')
    for base in (1, 10, 100, 1000):
        for m in range(2, 10):
            v = base * m
            if v < U:
                x = log_x(v, x0, w, U)
                out.append(f'<line class="grid minor" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}"/>')
    x = log_x(U, x0, w, U)
    out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="{y0}" y2="{y1}"/>')
    if x - log_x(1000, x0, w, U) > 64:          # room for "2,302" beside "1,000"
        out.append(f'<text class="tick" x="{x:.1f}" y="{label_y}" text-anchor="end">{U:,}</text>')
    return "".join(out)


def marker(kind, x, y):
    if kind == "flow":
        return f'<circle class="m-flow" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>'
    if kind == "up":
        return f'<rect class="m-walk" x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" rx="2"/>'
    d = 6.5
    return (f'<polygon class="m-down" points="{x:.1f},{y - d:.1f} {x + d:.1f},{y:.1f} '
            f'{x:.1f},{y + d:.1f} {x - d:.1f},{y:.1f}"/>')


def rank_chart(grp, U, what="disorder"):
    """Per disorder (or published unit): current flow, walk (sources = up) and walk (sources = down) on one log axis."""
    rows = grp.sort_values(["flow", "disorder"])
    W, row_h, top, bottom = 900, 30, 26, 42
    lab_w, x0, w = 280, 296, 440
    vx = x0 + w + 16
    n = len(rows)
    H = top + row_h * n + bottom
    parts = [x_ticks(x0, w, U, top, top + row_h * n, top + row_h * n + 18),
             f'<text class="tick" x="{vx}" y="{top - 10}">flow / up / down</text>']
    for i, (_, r) in enumerate(rows.iterrows()):
        yc = top + row_h * i + row_h / 2
        xs = {"flow": log_x(r.flow, x0, w, U)}
        for tag in RUNS:
            if not isnan(r[f"w_{tag}"]):
                xs[tag] = log_x(r[f"w_{tag}"], x0, w, U)
        name = r.disorder if len(r.disorder) <= 40 else r.disorder[:38] + "…"
        walk_tip = "".join(
            f"<br>walk, {LABEL[tag]}: "
            + ("not scorable (no walker reached a target)" if isnan(r[f"w_{tag}"]) else f"rank {fr(r[f'w_{tag}'])}")
            + f", {share(r[f's_{tag}'])} of walkers reach a target" for tag in RUNS)
        tip = (f"<b>{esc(r.disorder)}</b><br>causal: {esc(r.causal_in_network)}<br>"
               f"{int(r.up)} increased, {int(r.down)} decreased<br>current flow: rank {fr(r.flow)}{walk_tip}"
               f"<br>neighbour baseline: rank {fr(r.nb)}")
        parts.append(f'<g class="row" data-tip="{esc(tip)}">'
                     f'<rect class="row-bg" x="0" y="{yc - row_h / 2:.1f}" width="{W}" height="{row_h}"/>'
                     f'<text class="row-label" x="{lab_w}" y="{yc + 4:.1f}" text-anchor="end">{esc(name)}</text>'
                     f'<line class="span" x1="{min(xs.values()):.1f}" x2="{max(xs.values()):.1f}" '
                     f'y1="{yc:.1f}" y2="{yc:.1f}"/>'
                     + "".join(marker(k, xs[k], yc) for k in ("flow", "up", "down") if k in xs) +
                     f'<text class="row-value" x="{vx}" y="{yc + 4:.1f}">'
                     f'{fr(r.flow)} / {fr(r.w_up)} / {fr(r.w_down)}</text></g>')
    parts.append(f'<text class="axis-label" x="{x0 + w / 2}" y="{H - 6}" text-anchor="middle">'
                 f'Rank of the causal gene among {U:,} genes (log scale; left is better)</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rank of the causal gene per {what}: current flow, '
            f'absorbing walk with sources = up, and absorbing walk with sources = down">{"".join(parts)}</svg>')


# ---- page -----------------------------------------------------------------------
CSS = """
:root{--page:#f3f5f7;--surface:#fbfcfd;--ink:#12161b;--ink-2:#4a5360;--muted:#6b7480;--rule:#dfe4ea;--axis:#c5ccd4;
--accent:#1f63b5;--note:#eef3fa;--note-ink:#1c3e66;--walk:#2a78d6;--walk-down:#1baf7a;--flow:#eb6834;--ref:#7c8794;
--sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0f1216;--surface:#161a1f;
--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;--axis:#38414b;--accent:#6ea6ec;--note:#18222e;
--note-ink:#bcd4f2;--walk:#3987e5;--walk-down:#199e70;--flow:#d95926;--ref:#8d96a1}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;
--rule:#262c33;--axis:#38414b;--accent:#6ea6ec;--note:#18222e;--note-ink:#bcd4f2;--walk:#3987e5;--walk-down:#199e70;
--flow:#d95926;--ref:#8d96a1}
*{box-sizing:border-box}
body{margin:0;background:var(--page);color:var(--ink);font:16px/1.6 var(--sans);padding-inline:clamp(16px,4vw,40px);padding-block:40px 72px}
main{max-width:980px;margin-inline:auto;display:grid;gap:44px}
.prose{max-width:70ch;display:grid;gap:12px}
h1,h2,h3{text-wrap:balance;margin:0;line-height:1.2}
h1{font-size:clamp(30px,4vw,40px);font-weight:700;letter-spacing:-.015em}
h2{font-size:23px;font-weight:650;padding-top:10px;border-top:1px solid var(--rule)}
h3{font-size:17px;font-weight:650}
p,ul{margin:0}ul{padding-left:1.2em;display:grid;gap:6px}
.eyebrow{font:500 12px/1.4 var(--mono);letter-spacing:.08em;text-transform:uppercase;color:var(--muted)}
.lede{font-size:18px;color:var(--ink-2)}
code{font-family:var(--mono);font-size:.88em}
a{color:var(--accent)}
section{display:grid;gap:16px}
main>*,section>*,figure>*{min-width:0}
.summary{background:var(--note);color:var(--note-ink);border-radius:8px;padding:20px 24px;display:grid;gap:10px}
.summary li{color:var(--note-ink)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:14px 16px;display:grid;gap:2px}
.tile .k{font:500 11.5px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}
.tile .v{font-size:28px;font-weight:650;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.tile .v small{font-size:16px;font-weight:500;color:var(--ink-2)}
.tile .s{font-size:13px;color:var(--ink-2)}
.tbl{overflow-x:auto;background:var(--surface);border:1px solid var(--rule);border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:14px}
th,td{text-align:left;padding:8px 12px;border-bottom:1px solid var(--rule);vertical-align:top}
th{font:500 11.5px/1.3 var(--mono);letter-spacing:.05em;text-transform:uppercase;color:var(--muted);white-space:nowrap}
.dense table{font-size:13.5px}.dense th,.dense td{padding:8px 8px}.dense th{letter-spacing:.02em}.dense td:first-child{min-width:15ch}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
.up,.down,.same{font-size:12px;color:var(--ink-2)}
figure{margin:0;display:grid;gap:12px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 16px 12px}
figure svg{width:100%;height:auto;display:block}
figcaption{font-size:13.5px;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw-walk{background:var(--walk);border-radius:2px}.sw-flow{background:var(--flow)}
.sw-down{background:var(--walk-down);border-radius:1px;width:9px;height:9px;transform:rotate(45deg);margin-inline:1px}
svg text{font-family:var(--sans)}
.grid{stroke:var(--rule);stroke-width:1}.grid.minor{opacity:.5}
.tick{font:11px var(--mono);fill:var(--muted)}
.axis-label{font-size:12px;fill:var(--ink-2)}
.row-label{font-size:12.5px;fill:var(--ink)}
.row-value{font:12px var(--mono);fill:var(--ink-2)}
.row-bg{fill:transparent}.row:hover .row-bg{fill:var(--page)}
.span{stroke:var(--axis);stroke-width:2}
.m-walk{fill:var(--walk);stroke:var(--surface);stroke-width:2}
.m-down{fill:var(--walk-down);stroke:var(--surface);stroke-width:2}
.m-flow{fill:var(--flow);stroke:var(--surface);stroke-width:2}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--axis);
border-radius:6px;padding:8px 10px;font-size:13px;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.12);max-width:320px;z-index:10}
.tip[hidden]{display:none}
.small{font-size:13.5px;color:var(--ink-2)}
.added{font-size:12.5px;color:var(--muted)}
.pill{font:500 11px/1.4 var(--mono);color:var(--ink-2);border:1px solid var(--axis);border-radius:10px;padding:1px 7px;white-space:nowrap}
.papers{display:grid;gap:10px}
details{background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:12px 16px}
summary{cursor:pointer;display:flex;flex-wrap:wrap;align-items:baseline;gap:4px 12px}
summary::-webkit-details-marker{display:none}
summary::before{content:"▸";color:var(--muted);font-size:12px}
details[open]>summary::before{content:"▾"}
summary .meta{font-size:13px;color:var(--ink-2)}
.det{display:grid;gap:12px;margin-top:12px}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 16px;margin:0;font-size:14px}
.kv dt{color:var(--muted)}.kv dd{margin:0;min-width:0}
footer{font-size:13px;color:var(--muted);border-top:1px solid var(--rule);padding-top:16px;display:grid;gap:6px}
"""

JS = """
const tip = document.getElementById('tip');
function place(e){ const pad=14, r=tip.getBoundingClientRect();
  let x=e.clientX+pad, y=e.clientY+pad;
  if (x+r.width>innerWidth-8) x=e.clientX-r.width-pad;
  if (y+r.height>innerHeight-8) y=e.clientY-r.height-pad;
  tip.style.left=x+'px'; tip.style.top=y+'px'; }
document.querySelectorAll('[data-tip]').forEach(el=>{
  el.addEventListener('mousemove',e=>{ tip.innerHTML=el.dataset.tip; tip.hidden=false; place(e); });
  el.addEventListener('mouseleave',()=>{ tip.hidden=true; });
});
"""


PUB_COLS = [("Neighbour", "rank_neighbour"), ("Current flow", "rank_flow"),
            ("Walk, sources = up", "rank_walk_up"), ("Walk, sources = down", "rank_walk_down")]


def unit_ranks(r):
    """Neighbour, current flow, walk up, walk down for one published unit; '–' where there is no rank."""
    if not r.ranked:
        return ["–"] * 4
    return [fr(r.rank_neighbour), fr(r.rank_flow), f"{fw(r.rank_walk_up)} {change(r.rank_flow, r.rank_walk_up)}",
            f"{fw(r.rank_walk_down)} {change(r.rank_flow, r.rank_walk_down)}"]


def unit_gene(r):
    note = ""
    if r.eval_status == "ok" and r.gene and not r.causal_in_network:
        note = "not a network gene"
    elif r.eval_status == "ok" and r.causal_in_network and not r.ranked:
        note = "no significant metabolite on the network"
    elif r.kind == "contrast":
        note = "contrast: best rank over both disorders"
    elif r.gene_added and r.gene:
        note = "added from the disorder name"
    return (genes_code(r.gene) or "–") + (f'<br><span class="added">{note}</span>' if note else "")


def published_section(p, U):
    """The published-datasets section and its line for the summary."""
    assert p.universe.dropna().eq(U).all(), "the published evaluation ranks among a different gene universe"
    rk = p[p.ranked]
    dis, con = rk[rk.kind == "disorder"], rk[rk.kind == "contrast"]
    papers = sorted(p.groupby("paper"), key=lambda x: (x[1].year.iloc[0], x[0]))

    # summary table: one disorder vs controls, and two-disorder contrasts
    def cells(df, col):
        s = df[col].dropna()
        return [fr(s.median()) if len(s) else "–", top(s, 10), top(s, 100)]
    sum_rows = [[name] + cells(dis, col) + cells(con, col) for name, col in PUB_COLS]

    # per paper
    paper_rows, blocks = [], []
    for paper, g in papers:
        r_ok = g[g.ranked]
        species = "; ".join(sorted({s for s in g.species if s}))
        matrices = "; ".join(sorted({s for s in g.Matrix if s}))
        paper_rows.append([esc(paper), esc(matrices), len(g), int(g.eval_status.eq("ok").sum()), len(r_ok)]
                          + [fr(r_ok[col].median()) if len(r_ok) else "–" for _, col in PUB_COLS])
        listed, single = g[g.Subject_ID == ""], g[g.Subject_ID != ""]
        unit_html = ""
        if len(listed):
            rows = []
            for _, r in listed.iterrows():
                ud = (f"{fr(r.targets_up)} / {fr(r.sources_down)}" + (" ¹" if str(r.one_sided) == "True" else "")
                      if r.eval_status == "ok" else "–")
                comp = esc(r.Comparison) + (f'<br><span class="added">{esc(r.Disease_Group)}</span>' if r.Disease_Group else "")
                if r.eval_status != "ok":
                    comp += f'<br><span class="pill">{EVAL_STATUS.get(r.eval_status, esc(r.eval_status))}</span>'
                rows.append([comp, esc(r.Matrix), unit_gene(r), fr(r.subjects), ud, *unit_ranks(r)])
            unit_html = table(["Comparison", "Matrix", "Causal gene", "Subjects", "Up / down", "Neighbour", "Current<br>flow",
                               "Walk,<br>sources = up", "Walk,<br>sources = down"], rows, num=(3, 4, 5, 6, 7, 8), cls="dense")
        if len(single):
            left = int(single.eval_status.str.startswith("left out").sum())
            unit_html += (f'<p class="small">{plural(len(single), "single-patient unit")} not listed: {left} left out because '
                          f'one person\'s z-scores are not a differential analysis (D21), {len(single) - left} skipped with '
                          f'fewer than 3 usable metabolites.</p>')
        genes = sorted({x for s in g.gene for x in str(s).split(";") if x})
        gene_txt = ", ".join(genes[:14]) + (f" and {len(genes) - 14} more" if len(genes) > 14 else "")
        rule_txt = "; ".join(sorted({re.sub(r"over \d+ subjects", "over N subjects", str(x)) for x in g.rule}))
        blocks.append(f"""
<details>
  <summary><h3>{esc(paper)}</h3><span class="meta">{plural(len(g), 'unit')} · {int(g.eval_status.eq('ok').sum())} evaluated ·
  {len(r_ok)} ranked · {esc(species)} · {esc(matrices)}</span></summary>
  <div class="det">
  <dl class="kv">
    <dt>Comparisons</dt><dd>{esc('; '.join(sorted(set(g.Comparison))))}</dd>
    <dt>Causal genes</dt><dd>{esc(gene_txt) or '–'}</dd>
    <dt>Rule</dt><dd>{esc(rule_txt)}</dd>
  </dl>
  {unit_html}
  </div>
</details>""")

    # chart of every ranked unit, in the benchmark chart's form
    cd = pd.DataFrame({"disorder": rk.label, "causal_in_network": rk.causal_in_network, "up": rk.targets_up,
                       "down": rk.sources_down, "nb": rk.rank_neighbour, "flow": rk.rank_flow, "w_up": rk.rank_walk_up,
                       "w_down": rk.rank_walk_down, "s_up": rk.walk_success_walk_up, "s_down": rk.walk_success_walk_down})

    # numbers for the prose
    n_ev = int(p.eval_status.eq("ok").sum())
    n_single = int(p.eval_status.str.startswith("left out").sum())
    n_skip = int(p.eval_status.str.startswith("skipped").sum())
    n_nonet = int((p.eval_status.eq("ok") & p.causal_in_network.eq("")).sum())
    n_nosig = n_ev - n_nonet - len(rk)
    up_better = {k: int((d.rank_walk_up < d.rank_flow).sum()) for k, d in (("dis", dis), ("all", rk))}
    down_better = int((rk.rank_walk_down < rk.rank_flow).sum())
    by_paper = rk.groupby("paper")[["rank_flow", "rank_walk_up", "rank_walk_down"]].median()
    biggest = dis.paper.value_counts()
    big_txt = (f"{biggest.iloc[0]} of the {len(dis)} come from {esc(biggest.index[0])}" if len(biggest) else "")

    bullet = (f"<li>Published datasets: {len(dis)} one-disorder units from {dis.paper.nunique()} papers have a ranked causal "
              f"gene. Median rank: current flow {fr(dis.rank_flow.median())}, walk with sources = up "
              f"{fr(dis.rank_walk_up.median())}, walk with sources = down {fr(dis.rank_walk_down.median())}. Sources = up "
              f"ranks the gene above current flow in {up_better['dis']} of {len(dis)}; {big_txt}.</li>")

    html_ = f"""
<section id="published">
  <h2>Published datasets</h2>
  <div class="prose">
    <p>Metabolite tables the lab collected from {len(papers)} published papers, split into analysis units (one per paper,
    comparison and sample type; by disease group for z-score data; by patient for Alaimo 2020). Each group-level unit is
    rebuilt from its saved metabolite mapping and run with the current code under the same methods and orientations as the
    benchmark. Score: the causal gene's rank among the {U:,} genes on network reactions, raw scoring.</p>
    <p class="small">{len(p)} units: {n_ev} evaluated, {n_single} single-patient units left out (D21), {n_skip} skipped with
    fewer than 3 usable metabolites. Of the evaluated units, {n_nonet} have a causal gene that is not a network gene (GLUT1, a
    transporter; MELAS, a mitochondrial tRNA) and {n_nosig} have no significant metabolite on the network, so
    {len(rk)} get a rank: {len(dis)} one disorder vs controls and {len(con)} two-disorder contrasts.</p>
  </div>
  {table(["Method", "One disorder:<br>median", "One disorder:<br>top 10", "One disorder:<br>top 100",
          "Contrasts:<br>median", "Contrasts:<br>top 10", "Contrasts:<br>top 100"], sum_rows,
         num=(1, 2, 3, 4, 5, 6), cls="dense")}
  <p class="small">{len(dis)} one-disorder units, {len(con)} contrasts (Sidorina 2024: e.g. MMA vs CblC, where the best rank
  over both disorders' genes counts). Sources = up ranks the causal gene above current flow in {up_better['all']} of
  {len(rk)} ranked units; sources = down in {down_better}.</p>
  <figure>
    <div class="legend"><span><i class="sw sw-flow"></i>Current flow (either orientation)</span>
      <span><i class="sw sw-walk"></i>Walk, sources = up</span><span><i class="sw sw-down"></i>Walk, sources = down</span></div>
    {rank_chart(cd, U, what="published unit")}
    <figcaption>One row per ranked unit (paper · disease group or comparison), sorted by current-flow rank. Hover a row for
    the causal gene and the share of walkers that reach a target.</figcaption>
  </figure>
  <h3>By paper</h3>
  <p class="small">Units from one paper share samples, lab and platform, so they are not independent ({esc(rk.paper.value_counts().index[0])}
  alone gives {rk.paper.value_counts().iloc[0]} of the {len(rk)} ranked units). Medians per paper over its ranked units give
  each study one vote: the walk with sources = up
  has a lower median than current flow in {int((by_paper.rank_walk_up < by_paper.rank_flow).sum())} of {len(by_paper)}
  papers, sources = down in {int((by_paper.rank_walk_down < by_paper.rank_flow).sum())}.</p>
  {table(["Paper", "Sample type", "Units", "Evaluated", "Ranked", "Neighbour:<br>median", "Current flow:<br>median",
          "Walk, up:<br>median", "Walk, down:<br>median"], paper_rows, num=(2, 3, 4, 5, 6, 7, 8), cls="dense")}
  <h3>Units, paper by paper</h3>
  <div class="papers">{''.join(blocks)}</div>
  <p class="small">Units without ranks are marked under the comparison: skipped (fewer than 3 usable metabolites) or left
  out. Up / down: significant metabolites on the network, increased and decreased. ¹ One-sided. ▲ and ▼ mark a
  walk rank above or below current flow. Causal genes come from the master table's Gene column, completed by
  <code>benchmarks/published_answer_key.tsv</code> where a unit names only the disorder. Rules were fixed before any result
  was seen: log2FC data, p &lt; 0.05; several subjects with z-scores, BH q &lt; 0.05 and |mean z| ≥ 1.</p>
</section>"""
    return html_, bullet


def build(out):
    grp_all, U = load()
    grp = grp_all[grp_all.status == "primary"]
    N = len(grp)
    pub_html, pub_bullet = published_section(load_published(), U)

    stats = {"flow": grp.flow}
    stats.update({tag: grp[f"w_{tag}"] for tag in RUNS})
    unscored = {tag: grp[grp[f"w_{tag}"].isna()] for tag in RUNS}
    vs_flow = {}
    for tag in RUNS:
        w = grp[f"w_{tag}"]
        vs_flow[tag] = {"better": grp[w < grp.flow], "worse": grp[w > grp.flow], "same": grp[w == grp.flow]}
    up_higher = int(((grp.w_up < grp.w_down) | (grp.w_down.isna() & grp.w_up.notna())).sum())
    down_higher = grp[(grp.w_down < grp.w_up) | (grp.w_up.isna() & grp.w_down.notna())]
    succ_med = {tag: float(grp[f"s_{tag}"].median()) for tag in RUNS}
    kappa_shift = {tag: max(abs(grp[f"k{k}_{tag}"].median() - grp[f"w_{tag}"].median()) for k in KAPPAS)
                   for tag in RUNS}

    def median_note(tag):
        n_ok = int(grp[f"w_{tag}"].notna().sum())
        return "" if n_ok == N else f" over the {n_ok} it can score"

    def unscored_note(tag):
        u = unscored[tag]
        return "" if not len(u) else f"; {esc(', '.join(u.disorder))} not scorable"

    # summary --------------------------------------------------------------------
    walk_bullets = "".join(
        f"<li>Walk, {LABEL[tag]}: median {fr(stats[tag].median())}{median_note(tag)}{unscored_note(tag)}. "
        f"Top 10 for {top(stats[tag], 10)} of {N}, top 100 for {top(stats[tag], 100)}. It ranks the causal gene above "
        f"current flow in {len(vs_flow[tag]['better'])} of {N} disorders and below it in {len(vs_flow[tag]['worse'])}.</li>"
        for tag in RUNS)

    # tiles ----------------------------------------------------------------------
    tiles = [("Current flow", fr(grp.flow.median()),
              f"top 10: {top(grp.flow, 10)} of {N} · top 100: {top(grp.flow, 100)} · either orientation")]
    for tag in RUNS:
        s = stats[tag]
        tiles.append((f"Walk, {LABEL[tag]}", fr(s.median()),
                      f"top 10: {top(s, 10)} of {N} · top 100: {top(s, 100)}{median_note(tag).replace(' over', ' · median over')}"))
    tiles_html = "".join(f'<div class="tile"><span class="k">{k}</span><span class="v">{v} <small>median</small></span>'
                         f'<span class="s">{s}</span></div>' for k, v, s in tiles)

    # per-disorder table -----------------------------------------------------------
    group_rows = []
    for _, r in grp.sort_values(["flow", "disorder"]).iterrows():
        group_rows.append([esc(r.disorder), f"<code>{esc(r.causal_in_network).replace(';', ';<wbr>')}</code>", int(r.up), int(r.down),
                           fr(r.nb), fr(r.flow),
                           f"{fw(r.w_up)} {change(r.flow, r.w_up)}", f"{fw(r.w_down)} {change(r.flow, r.w_down)}",
                           share(r.s_up), share(r.s_down)])
    sec = grp_all[grp_all.status == "secondary"]
    sec_note = "; ".join(f"{esc(r.disorder)}: current flow {fr(r.flow)}, walk {fw(r.w_up)} with sources = up, "
                         f"{fw(r.w_down)} with sources = down" for _, r in sec.iterrows())

    # kappa table ------------------------------------------------------------------
    def kappa_cells(col):
        return [fr(col.median()), top(col, 10), top(col, 100)]
    kappa_rows = [["full absorption<br>(default)"] + kappa_cells(grp.w_up) + kappa_cells(grp.w_down)]
    for k in KAPPAS:
        kappa_rows.append([f"κ = {k}"] + kappa_cells(grp[f"k{k}_up"]) + kappa_cells(grp[f"k{k}_down"]))
    kappa_rows.append(["<i>current flow,<br>for reference</i>"] + kappa_cells(grp.flow) + kappa_cells(grp.flow))
    k_unscored = {tag: " / ".join(str(int(grp[c].isna().sum())) for c in [f"w_{tag}"] + [f"k{k}_{tag}" for k in KAPPAS])
                  for tag in RUNS}
    plural_rank = lambda x: f"{fr(x)} rank{'' if x == 1 else 's'}"

    # gains and losses -------------------------------------------------------------
    def items(df, tag, key):
        df = df.assign(r=df.flow / df[f"w_{tag}"]).sort_values("r", ascending=(key != "better"))
        return ", ".join(f"{esc(r.disorder)} {fr(r.flow)} → {fr(r[f'w_{tag}'])}" for _, r in df.iterrows()) or "none"

    gl_html = ""
    for tag in RUNS:
        v = vs_flow[tag]
        same = (f"<li><b>Same rank ({len(v['same'])}):</b> {items(v['same'], tag, 'same')}.</li>" if len(v["same"]) else "")
        uns = (f"<li><b>Not scorable ({len(unscored[tag])}):</b> no walker reaches a target in "
               f"{esc(', '.join(f'{r.disorder} (current flow {fr(r.flow)})' for _, r in unscored[tag].iterrows()))}.</li>"
               if len(unscored[tag]) else "")
        gl_html += f"""
  <h3>Walk, {LABEL[tag]}</h3>
  <ul class="prose">
    <li><b>Higher than current flow ({len(v['better'])}):</b> {items(v['better'], tag, 'better')}.</li>
    <li><b>Lower than current flow ({len(v['worse'])}):</b> {items(v['worse'], tag, 'worse')}.</li>
    {same}{uns}
  </ul>"""
    down_list = ", ".join(f"{esc(r.disorder)} ({fr(r.w_up)} → {fr(r.w_down)})"
                          for _, r in down_higher.assign(q=down_higher.w_up / down_higher.w_down)
                          .sort_values("q", ascending=False).iterrows()) or "none"
    msud = grp_all[grp_all.case == "msud"]
    msud_note = ""
    if len(msud):
        m = msud.iloc[0]
        msud_note = (f" In {esc(m.disorder)}, for example, the causal gene is ranked {fw(m.w_up)} with sources = up and "
                     f"{fw(m.w_down)} with sources = down; current flow ranks it {fr(m.flow)} either way.")
    one_sided = grp[(grp.up == 0) | (grp.down == 0)]
    only_up = one_sided[one_sided.down == 0]
    only_down = one_sided[one_sided.up == 0]

    def os_list(df):
        return esc(", ".join(f"{r.disorder} (current flow {fr(r.flow)}, walk {fr(r.w_up)} up / {fr(r.w_down)} down)"
                             for _, r in df.iterrows()))
    os_text = ""
    if len(only_up):
        os_text += (f" {plural(len(only_up), 'disorder')} {'has' if len(only_up) == 1 else 'have'} significant increases "
                    f"only: {os_list(only_up)}. With sources = up there are then only sources and the ground absorbs; "
                    f"with sources = down there are only targets and the ground releases the walkers.")
    if len(only_down):
        os_text += (f" {plural(len(only_down), 'disorder')} {'has' if len(only_down) == 1 else 'have'} significant decreases "
                    f"only: {os_list(only_down)}. With sources = up there are then only targets and the ground releases "
                    f"the walkers; with sources = down there are only sources and the ground absorbs.")

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Absorbing Walk Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style></head><body><main>

<header class="prose">
  <div class="eyebrow">Metaboenrich · {esc(git_label())}</div>
  <h1>Absorbing random walk: benchmark and published datasets</h1>
  <p class="lede">How well the experimental direction-respecting walk (<code>--mode walk</code>) finds the faulty enzyme under
  each orientation, compared with the current-flow method: first on the inborn-error benchmark, then on the lab's published
  datasets. Group level only.</p>
</header>

<section class="summary">
  <ul>
    <li>The walk is run under two orientations. <b>Sources = up</b>: the significantly increased metabolites are the sources
      and the decreased ones the targets. <b>Sources = down</b>: the significantly decreased metabolites are the sources and the increased
      ones the targets (the package default since decision D20).</li>
    <li>Current flow gives identical ranks under both orientations (checked for all {len(grp_all)} disorders). Median rank
      {fr(grp.flow.median())} of {U:,} genes; top 10 for {top(grp.flow, 10)} of {N} primary disorders, top 100 for
      {top(grp.flow, 100)}.</li>
    {walk_bullets}
    <li>Sources = up ranks the causal gene higher than sources = down in {up_higher} of {N} disorders. The median share of
      walkers that reach a target is {share(succ_med['up'])} with sources = up and {share(succ_med['down'])} with sources = down.</li>
    <li>Partial absorption (hook 2, κ) moves the group median by at most {plural_rank(kappa_shift['up'])} with sources = up and
      {plural_rank(kappa_shift['down'])} with sources = down. Full absorption stays the default.</li>
    {pub_bullet}
    <li>Group level only: each disorder's patients are pooled. Patient-level runs were dropped (decision D21).</li>
  </ul>
</section>

<section class="tiles">{tiles_html}</section>

<section>
  <h2>What was compared</h2>
  <div class="prose">
    <p><b>The test.</b> Untargeted plasma metabolomics from patients with a confirmed inborn error (Thistlethwaite et al.
    2020, Miller et al. 2015). Each disorder's patients are pooled (BH q &lt; 0.05 and |mean z| ≥ 1) and the run goes from
    the pooled metabolite changes to a ranked gene list. The score is where the known causal gene lands among the {U:,} genes
    on network reactions (1 = top). {N} primary disorders count; {plural(len(sec), 'secondary case')} {'is' if len(sec) == 1 else 'are'} shown apart.</p>
    <p><b>Current flow</b> is the package default: Kirchhoff current between every source and
    every target on the undirected network, raw scoring. Swapping sources and targets only reverses the current, so its ranks
    are the same under both orientations.</p>
    <p><b>Absorbing walk</b>: walkers start at the sources in proportion to |log2FC|, follow reaction directions, and stop at
    the targets. Reversible reactions are split in two, a walker can't undo its last step, dead ends leak, and only walks that
    reach a target are scored (decision D17). Because it follows directions, the orientation matters:</p>
    <ul>
      <li><b>Sources = up</b>: significantly increased metabolites are the sources, decreased are the targets
        (<code>--sources increased</code>; protocol §1.3; the orientation used before D20). Runs in
        <code>results/bench_rw_up/</code>.</li>
      <li><b>Sources = down</b>: significantly decreased metabolites are the sources, increased are the targets
        (<code>--sources decreased</code>; the package default since D20). Runs in <code>results/bench_rw/</code>.</li>
    </ul>
    <p><b>Neighbour baseline</b> ranks each enzyme by how many significant metabolites its own reactions touch.</p>
    <p class="small">Same network (3,958 metabolites, 6,377 reactions), same mapping and significance rules for every method;
    only the scoring step and the orientation differ.</p>
  </div>
</section>

<section>
  <h2>Benchmark: disorders</h2>
  <figure>
    <div class="legend"><span><i class="sw sw-flow"></i>Current flow (either orientation)</span>
      <span><i class="sw sw-walk"></i>Walk, sources = up</span><span><i class="sw sw-down"></i>Walk, sources = down</span></div>
    {rank_chart(grp, U)}
    <figcaption>One row per primary disorder, sorted by current-flow rank. The numbers on the right are the three ranks in
    legend order; a dash and a missing marker mean the walk could not score that disorder (no walker reached a target).
    Hover a row for the causal gene, the number of increased and decreased metabolites, and the share of walkers that reached
    a target under each orientation.</figcaption>
  </figure>
  {table(["Disorder", "Causal gene(s)", "Increased", "Decreased", "Neighbour", "Current<br>flow", "Walk,<br>sources = up",
          "Walk,<br>sources = down", "Reaching a<br>target, up", "Reaching a<br>target, down"],
         group_rows, num=(2, 3, 4, 5, 6, 7, 8, 9), cls="dense")}
  <p class="small">Ranks of the causal gene among {U:,} genes; ▲ and ▼ mark a walk rank above or below current flow. The last
  two columns give the share of released walkers that reach a target. Medians: neighbour baseline {fr(grp.nb.median())},
  current flow {fr(grp.flow.median())}, walk with sources = up {fr(grp.w_up.median())}, walk with sources = down
  {fr(grp.w_down.median())}{median_note('down')}. Top 100: {top(grp.nb, 100)}, {top(grp.flow, 100)}, {top(grp.w_up, 100)}
  and {top(grp.w_down, 100)} of {N}. Top 10: {top(grp.nb, 10)}, {top(grp.flow, 10)}, {top(grp.w_up, 10)} and
  {top(grp.w_down, 10)}. Secondary case, not in these counts: {sec_note}.</p>
</section>

<section>
  <h2>Partial absorption (hook 2)</h2>
  <p class="prose">With κ set, a target absorbs a walker with probability 1 − exp(−κ·|log2FC|) and lets the rest walk on to
  targets further along. Under either orientation it barely moves anything, so the simpler full absorption stays the default.</p>
  {table(["Setting", "Sources = up:<br>median", "Sources = up:<br>top 10", "Sources = up:<br>top 100",
          "Sources = down:<br>median", "Sources = down:<br>top 10", "Sources = down:<br>top 100"], kappa_rows,
         num=(1, 2, 3, 4, 5, 6), cls="dense")}
  <p class="small">Counts are of {N} primary disorders; medians are over the disorders each setting can score.
  Disorders the walk cannot score (full absorption / κ = {' / '.join(KAPPAS)}): {k_unscored['up']} with sources = up,
  {k_unscored['down']} with sources = down.</p>
</section>

<section id="gains">
  <h2>Where each orientation gains or loses against current flow</h2>
  <p class="prose">Rank of the causal gene, current flow → walk, sorted from the largest change.</p>
  {gl_html}
  <div class="prose">
    <p><b>Where sources = down does better than sources = up:</b> {down_list} (walk rank up → down).</p>
    <p><b>Substrate and product.</b> In an enzyme defect the substrate side usually rises and the product side falls. With
    sources = up, walkers start at the accumulated substrates and the blocked enzyme lies on their way to the depleted products.
    With sources = down they start on the product side and must follow reaction directions to the increased metabolites
    upstream of the block, so the faulty enzyme is seldom on a successful route.{msud_note} Current flow ignores direction,
    which is why its ranks do not depend on the orientation.</p>
    {f'<p><b>One-sided disorders.</b>{os_text} The measured, unchanged metabolites stand in for the missing side.</p>' if os_text else ""}
    <p class="small">These are readings of the runs, not a test.</p>
  </div>
</section>
{pub_html}

<section>
  <h2>Caveats</h2>
  <ul class="prose">
    <li>The benchmark has {N} disorders, all plasma. The settings were not tuned on it, but the walk was designed after seeing
      current flow's results on the same data, and both orientations were compared on it, so treat any difference as provisional.</li>
    <li>The published datasets are small and clustered: few ranked units, several from the same paper, mixed species and
      sample types. Read the per-paper medians alongside the per-unit counts.</li>
    <li>Both collections have now been seen. No held-out data remain, so a choice of default method or orientation made from
      this page rests on data that also informed it.</li>
    <li>Some patients may be in both collections (urea-cycle disorders in Burrage 2019, GABA-transaminase deficiency in
      Kennedy 2019, the Alaimo samples); see <a href="testing_data.html#overlap">the testing-data page</a>. Until that is
      checked, the two are not fully independent evidence.</li>
    <li>Patient-level runs were dropped (decision D21): one patient's z-scores against a reference population are not a
      differential analysis.</li>
    <li>The walk depends on Human-GEM's reversibility labels (from flux bounds). Wrong labels block or open routes.</li>
    <li>Leave-one-out scoring isn't defined for the walk; all comparisons here use raw scoring for every method.</li>
  </ul>
</section>

<footer>
  <span>Built by <code>tools/report/build_rw_report.py</code> from <code>results/bench_rw/</code> (sources = down) and
  <code>results/bench_rw_up/</code> (sources = up); published datasets from <code>predictions/published_v2/units.tsv</code>
  and <code>results/published_eval/units.tsv</code>. Method: <code>metaboenrich/walk.py</code>; decision log entries D17,
  D20 and D21.</span>
</footer>
</main>
<div class="tip" id="tip" hidden></div>
<script>{JS}</script>
</body></html>
"""
    Path(out).write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({len(page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "rw_report.html"))
    build(ap.parse_args().out)
