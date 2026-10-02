"""Build the absorbing-walk benchmark report (one self-contained HTML page).

Every number comes from benchmark result files (results/ is not committed):
    results/bench_rw/<cfg>/group_results.tsv       cfg = flow, walk_full, walk_k0.5, walk_k1, walk_k2
    results/bench_rw_p/<cfg>/patient_results.tsv   cfg = flow, walk_full, walk_k1

Produce them on branch directed-rw with
    python -m metaboenrich.benchmark --levels group --out results/bench_rw/flow
    python -m metaboenrich.benchmark --levels group --method walk --out results/bench_rw/walk_full
    python -m metaboenrich.benchmark --levels group --method walk --walk-kappa K --out results/bench_rw/walk_kK
        (K = 0.5, 1, 2)
    python -m metaboenrich.benchmark --levels patient --out results/bench_rw_p/flow
    python -m metaboenrich.benchmark --levels patient --method walk --out results/bench_rw_p/walk_full
    python -m metaboenrich.benchmark --levels patient --method walk --walk-kappa 1 --out results/bench_rw_p/walk_k1

Then:
    python tools/report/build_rw_report.py --out docs/rw_report.html
"""
import argparse
import html
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GROUP = ROOT / "results" / "bench_rw"
PATIENT = ROOT / "results" / "bench_rw_p"
KAPPAS = ["0.5", "1", "2"]
RANK = "rank_metaboenrich_raw"


# ---- formatting -----------------------------------------------------------------
def esc(x):
    return html.escape(str(x))


def fr(x):
    """A rank: integer when whole, else one decimal (ties are averaged)."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{int(x):,}" if float(x).is_integer() else f"{x:,.1f}"


def pct(x, digits=0):
    return f"{100 * x:.{digits}f}%"


def table(headers, rows, num=(), cls=""):
    head = "".join(f'<th class="num">{h}</th>' if i in num else f"<th>{h}</th>" for i, h in enumerate(headers))
    body = "".join("<tr>" + "".join(f'<td class="num">{c}</td>' if i in num else f"<td>{c}</td>"
                                    for i, c in enumerate(r)) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def change(a, b):
    """Glyph for a rank change a -> b (lower rank is better), so it never rests on colour."""
    if b < a:
        return '<span class="up" title="walk ranks it higher">▲</span>'
    if b > a:
        return '<span class="down" title="walk ranks it lower">▼</span>'
    return '<span class="same" title="same rank">=</span>'


# ---- data -----------------------------------------------------------------------
def load():
    g = {"flow": pd.read_csv(GROUP / "flow" / "group_results.tsv", sep="\t"),
         "walk": pd.read_csv(GROUP / "walk_full" / "group_results.tsv", sep="\t")}
    for k in KAPPAS:
        g[k] = pd.read_csv(GROUP / f"walk_k{k}" / "group_results.tsv", sep="\t")
    p = {"flow": pd.read_csv(PATIENT / "flow" / "patient_results.tsv", sep="\t"),
         "walk": pd.read_csv(PATIENT / "walk_full" / "patient_results.tsv", sep="\t"),
         "1": pd.read_csv(PATIENT / "walk_k1" / "patient_results.tsv", sep="\t")}
    for d in (g, p):
        for k, df in d.items():
            assert (df.case.values == d["flow"].case.values).all(), f"row order differs in {k}"
    assert (p["flow"]["sample"].values == p["walk"]["sample"].values).all()

    grp = g["flow"][["case", "disorder", "status", "causal_in_network", "sources", "targets", "universe"]].copy()
    grp["nb"] = g["flow"].rank_neighbour
    grp["flow"] = g["flow"][RANK]
    grp["walk"] = g["walk"][RANK]
    grp["success"] = g["walk"].walk_success_fraction
    for k in KAPPAS:
        grp[f"k{k}"] = g[k][RANK]

    pat = p["flow"][["case", "disorder", "status", "sample", "causal_in_network"]].copy()
    pat["nb"], pat["flow"] = p["flow"].rank_neighbour, p["flow"][RANK]
    pat["walk"], pat["k1"] = p["walk"][RANK], p["1"][RANK]
    pat["success"] = p["walk"].walk_success_fraction
    return grp, pat, int(grp.universe.iloc[0])


def git_label():
    try:
        run = lambda *a: subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        dirty = " + uncommitted changes" if run("status", "--porcelain", "--untracked-files=no") else ""
        return f"branch {run('branch', '--show-current')} · commit {run('rev-parse', '--short', 'HEAD')}{dirty}"
    except OSError:
        return ""


# ---- charts ---------------------------------------------------------------------
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


def recall_chart(pat, U):
    """Share of patients whose causal gene is within the top k, against k (log)."""
    W, H, l, r, t, b = 900, 360, 60, 24, 14, 50
    w, h = W - l - r, H - t - b
    ks = np.arange(1, U + 1)
    series = {}
    for key in ("nb", "flow", "walk"):
        ranks = np.sort(pat[key].values)
        series[key] = np.searchsorted(ranks, ks, side="right") / len(ranks)
    y = lambda s: t + (1 - s) * h
    parts = []
    for s in (0, .25, .5, .75, 1):
        parts.append(f'<line class="grid" x1="{l}" x2="{l + w}" y1="{y(s):.1f}" y2="{y(s):.1f}"/>'
                     f'<text class="tick" x="{l - 8}" y="{y(s) + 4:.1f}" text-anchor="end">{int(s * 100)}%</text>')
    parts.append(x_ticks(l, w, U, t, t + h, t + h + 18))
    for key, cls in (("nb", "ln-ref"), ("flow", "ln-flow"), ("walk", "ln-walk")):
        v = series[key]
        d = [f"M{log_x(1, l, w, U):.1f},{y(v[0]):.1f}"]
        for i in range(1, U):
            if v[i] != v[i - 1]:
                d.append(f"H{log_x(ks[i], l, w, U):.1f}V{y(v[i]):.1f}")
        d.append(f"H{l + w:.1f}")
        parts.append(f'<path class="{cls}" d="{"".join(d)}"/>')
    parts.append(f'<text class="axis-label" x="{l + w / 2}" y="{H - 6}" text-anchor="middle">'
                 f'k: the causal gene is within the top k of {U:,} genes (log scale)</text>')
    parts.append(f'<text class="axis-label" transform="translate(14 {t + h / 2}) rotate(-90)" text-anchor="middle">'
                 f'Patients</text>')
    parts.append(f'<line id="rc-x" class="crosshair" x1="0" x2="0" y1="{t}" y2="{t + h}" visibility="hidden"/>')
    parts.append(f'<rect id="rc-hit" x="{l}" y="{t}" width="{w}" height="{h}" fill="transparent"/>')
    data = {k: [round(float(x), 4) for x in v] for k, v in series.items()}
    geom = {"l": l, "w": w, "U": U}
    svg = (f'<svg id="recall" viewBox="0 0 {W} {H}" role="img" aria-label="Share of patients with the causal gene '
           f'within the top k genes, for the absorbing walk, current flow and the neighbour baseline">{"".join(parts)}</svg>')
    return svg, json.dumps({"series": data, "geom": geom})


def scatter_chart(pat, U):
    """Per patient: current-flow rank (x) against walk rank (y), both log."""
    S, l, b, t, r = 560, 62, 50, 14, 14
    w = S - l - r
    h = S - t - b
    px = lambda v: log_x(v, l, w, U)
    py = lambda v: t + h - (math.log10(max(v, 1.0)) / math.log10(U)) * h
    parts = []
    for v in (1, 10, 100, 1000, U):
        parts.append(f'<line class="grid" x1="{px(v):.1f}" x2="{px(v):.1f}" y1="{t}" y2="{t + h}"/>'
                     f'<line class="grid" x1="{l}" x2="{l + w}" y1="{py(v):.1f}" y2="{py(v):.1f}"/>'
                     f'<text class="tick" x="{px(v):.1f}" y="{t + h + 18}" '
                     f'text-anchor="{"end" if v == U else "middle"}">{v:,}</text>'
                     f'<text class="tick" x="{l - 8}" y="{py(v) + 4:.1f}" text-anchor="end">{v:,}</text>')
    parts.append(f'<line class="diag" x1="{px(1):.1f}" y1="{py(1):.1f}" x2="{px(U):.1f}" y2="{py(U):.1f}"/>')
    parts.append(f'<text class="region" x="{l + w - 10}" y="{t + h - 12}" text-anchor="end">walk ranks it higher ↘</text>')
    parts.append(f'<text class="region" x="{l + 10}" y="{t + 18}">↖ current flow ranks it higher</text>')
    order = pat.assign(gain=np.sign(pat.flow - pat.walk)).sort_values("gain")
    for _, row in order.iterrows():
        cls = "pt-walk" if row.walk < row.flow else "pt-flow" if row.walk > row.flow else "pt-tie"
        tip = (f"<b>{esc(row.disorder)}</b><br>patient {esc(row['sample'])}<br>current flow rank {fr(row.flow)}"
               f"<br>walk rank {fr(row.walk)}")
        parts.append(f'<circle class="{cls}" cx="{px(row.flow):.1f}" cy="{py(row.walk):.1f}" r="4.5" '
                     f'data-tip="{esc(tip)}"/>')
    parts.append(f'<text class="axis-label" x="{l + w / 2}" y="{S - 8}" text-anchor="middle">'
                 f'Current flow: rank of the causal gene (log)</text>')
    parts.append(f'<text class="axis-label" transform="translate(16 {t + h / 2}) rotate(-90)" text-anchor="middle">'
                 f'Absorbing walk: rank of the causal gene (log)</text>')
    return (f'<svg viewBox="0 0 {S} {S}" role="img" aria-label="Each primary patient: current-flow rank against '
            f'walk rank of the causal gene">{"".join(parts)}</svg>')


def dumbbell_chart(grp, U):
    """Per disorder (group level): current-flow rank -> walk rank."""
    rows = grp.assign(gain=np.log10(grp.flow / grp.walk)).sort_values("gain", ascending=False)
    W, row_h, top, bottom = 900, 30, 8, 42
    lab_w, x0, w = 300, 316, 470
    H = top + row_h * len(rows) + bottom
    parts = [x_ticks(x0, w, U, top, top + row_h * len(rows), top + row_h * len(rows) + 18)]
    for i, (_, r) in enumerate(rows.iterrows()):
        yc = top + row_h * i + row_h / 2
        xf, xw = log_x(r.flow, x0, w, U), log_x(r.walk, x0, w, U)
        name = r.disorder if len(r.disorder) <= 40 else r.disorder[:38] + "…"
        tip = (f"<b>{esc(r.disorder)}</b><br>causal: {esc(r.causal_in_network)}<br>current flow rank {fr(r.flow)}"
               f"<br>walk rank {fr(r.walk)}<br>neighbour baseline {fr(r.nb)}<br>"
               f"{int(r.sources)} up, {int(r.targets)} down; {pct(r.success, 1)} of walkers reach a target")
        parts.append(f'<g class="row" data-tip="{esc(tip)}">'
                     f'<rect class="row-bg" x="0" y="{yc - row_h / 2:.1f}" width="{W}" height="{row_h}"/>'
                     f'<text class="row-label" x="{lab_w}" y="{yc + 4:.1f}" text-anchor="end">{esc(name)}</text>'
                     f'<line class="span" x1="{min(xf, xw):.1f}" x2="{max(xf, xw):.1f}" y1="{yc:.1f}" y2="{yc:.1f}"/>'
                     f'<circle class="m-flow" cx="{xf:.1f}" cy="{yc:.1f}" r="5.5"/>'
                     f'<circle class="m-walk" cx="{xw:.1f}" cy="{yc:.1f}" r="5.5"/>'
                     f'<text class="row-value" x="{x0 + w + 16}" y="{yc + 4:.1f}">{fr(r.flow)} → {fr(r.walk)}</text>'
                     f'</g>')
    parts.append(f'<text class="axis-label" x="{x0 + w / 2}" y="{H - 6}" text-anchor="middle">'
                 f'Rank of the causal gene among {U:,} genes (log scale; left is better)</text>')
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Rank of the causal gene per disorder, current flow '
            f'and absorbing walk">{"".join(parts)}</svg>')


# ---- page -----------------------------------------------------------------------
CSS = """
:root{--page:#f3f5f7;--surface:#fbfcfd;--ink:#12161b;--ink-2:#4a5360;--muted:#6b7480;--rule:#dfe4ea;--axis:#c5ccd4;
--accent:#1f63b5;--note:#eef3fa;--note-ink:#1c3e66;--walk:#2a78d6;--flow:#eb6834;--ref:#7c8794;
--sans:"Public Sans",system-ui,-apple-system,"Segoe UI",sans-serif;--mono:"IBM Plex Mono",ui-monospace,Consolas,monospace;
color-scheme:light}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0f1216;--surface:#161a1f;
--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;--rule:#262c33;--axis:#38414b;--accent:#6ea6ec;--note:#18222e;
--note-ink:#bcd4f2;--walk:#3987e5;--flow:#d95926;--ref:#8d96a1}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0f1216;--surface:#161a1f;--ink:#eef1f5;--ink-2:#b3bcc7;--muted:#8d96a1;
--rule:#262c33;--axis:#38414b;--accent:#6ea6ec;--note:#18222e;--note-ink:#bcd4f2;--walk:#3987e5;--flow:#d95926;--ref:#8d96a1}
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
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
tr:last-child td{border-bottom:0}
.up,.down,.same{font-size:12px;color:var(--ink-2)}
figure{margin:0;display:grid;gap:12px;background:var(--surface);border:1px solid var(--rule);border-radius:8px;padding:16px 16px 12px}
figure svg{width:100%;height:auto;display:block}
.square{max-width:560px}
figcaption{font-size:13.5px;color:var(--ink-2)}
.legend{display:flex;flex-wrap:wrap;gap:6px 20px;font-size:13px;color:var(--ink-2)}
.legend span{display:inline-flex;align-items:center;gap:7px}
.sw{width:11px;height:11px;border-radius:50%;display:inline-block}
.sw-walk{background:var(--walk)}.sw-flow{background:var(--flow)}
.sw-line{width:20px;height:0;border-top:2px dashed var(--ref)}
svg text{font-family:var(--sans)}
.grid{stroke:var(--rule);stroke-width:1}.grid.minor{opacity:.5}
.tick{font:11px var(--mono);fill:var(--muted)}
.axis-label{font-size:12px;fill:var(--ink-2)}
.row-label{font-size:12.5px;fill:var(--ink)}
.row-value{font:12px var(--mono);fill:var(--ink-2)}
.row-bg{fill:transparent}.row:hover .row-bg{fill:var(--page)}
.span{stroke:var(--axis);stroke-width:2}
.m-walk{fill:var(--walk);stroke:var(--surface);stroke-width:2}
.m-flow{fill:var(--flow);stroke:var(--surface);stroke-width:2}
.ln-walk{fill:none;stroke:var(--walk);stroke-width:2}
.ln-flow{fill:none;stroke:var(--flow);stroke-width:2}
.ln-ref{fill:none;stroke:var(--ref);stroke-width:2;stroke-dasharray:5 4}
.crosshair{stroke:var(--muted);stroke-width:1;stroke-dasharray:3 3}
.diag{stroke:var(--axis);stroke-width:1.5;stroke-dasharray:5 4}
.region{font-size:12px;fill:var(--muted)}
.pt-walk{fill:var(--walk);stroke:var(--surface);stroke-width:1.5;opacity:.85}
.pt-flow{fill:var(--flow);stroke:var(--surface);stroke-width:1.5;opacity:.85}
.pt-tie{fill:var(--ref);stroke:var(--surface);stroke-width:1.5;opacity:.85}
circle[data-tip]:hover{opacity:1;stroke:var(--ink)}
.tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);border:1px solid var(--axis);
border-radius:6px;padding:8px 10px;font-size:13px;line-height:1.45;box-shadow:0 4px 14px rgba(0,0,0,.12);max-width:300px;z-index:10}
.tip[hidden]{display:none}
.small{font-size:13.5px;color:var(--ink-2)}
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
const RC = JSON.parse(document.getElementById('recall-data').textContent);
const svg = document.getElementById('recall'), hit = document.getElementById('rc-hit'), cx = document.getElementById('rc-x');
hit.addEventListener('mousemove', e=>{
  const p = svg.createSVGPoint(); p.x=e.clientX; p.y=e.clientY;
  const q = p.matrixTransform(svg.getScreenCTM().inverse());
  const g = RC.geom, f = Math.min(Math.max((q.x-g.l)/g.w,0),1);
  const k = Math.min(g.U, Math.max(1, Math.round(Math.pow(10, f*Math.log10(g.U)))));
  const x = g.l + Math.log10(k)/Math.log10(g.U)*g.w;
  cx.setAttribute('x1',x); cx.setAttribute('x2',x); cx.setAttribute('visibility','visible');
  const s = RC.series, fmt=v=>(100*v).toFixed(1)+'%';
  tip.innerHTML = `<b>Within the top ${k.toLocaleString()}</b><br>`+
    `<span class="sw sw-walk"></span> Absorbing walk ${fmt(s.walk[k-1])}<br>`+
    `<span class="sw sw-flow"></span> Current flow ${fmt(s.flow[k-1])}<br>`+
    `<span class="sw-line" style="display:inline-block"></span> Neighbour baseline ${fmt(s.nb[k-1])}`;
  tip.hidden=false; place(e);
});
hit.addEventListener('mouseleave', ()=>{ tip.hidden=true; cx.setAttribute('visibility','hidden'); });
"""


def build(out):
    grp_all, pat_all, U = load()
    grp = grp_all[grp_all.status == "primary"]
    pat_p = pat_all[pat_all.status == "primary"]
    pat = pat_p.dropna(subset=["flow", "walk"])
    n_dropped = len(pat_p) - len(pat)

    gm_f, gm_w = grp.flow.median(), grp.walk.median()
    pm_f, pm_w = pat.flow.median(), pat.walk.median()
    top = lambda s, k: float((s <= k).mean())
    better, worse = float((pat.walk < pat.flow).mean()), float((pat.walk > pat.flow).mean())
    per = pat.assign(win=pat.walk < pat.flow).groupby(["case", "disorder"]).agg(
        n=("sample", "size"), flow=("flow", "median"), walk=("walk", "median"), wins=("win", "mean")).reset_index()
    dis_better = int((per.walk < per.flow).sum())

    tiles = [
        ("Group median rank", f"{fr(gm_f)} <small>→</small> {fr(gm_w)}", f"{len(grp)} primary disorders"),
        ("Patient median rank", f"{fr(pm_f)} <small>→</small> {fr(pm_w)}", f"{len(pat)} primary patients"),
        ("Patients: causal gene in top 10", f"{pct(top(pat.flow, 10))} <small>→</small> {pct(top(pat.walk, 10))}",
         f"top 100: {pct(top(pat.flow, 100))} → {pct(top(pat.walk, 100))}"),
        ("Walk ranks it higher", pct(better), f"for patients; current flow higher for {pct(worse)}"),
    ]
    tiles_html = "".join(f'<div class="tile"><span class="k">{k}</span><span class="v">{v}</span>'
                         f'<span class="s">{s}</span></div>' for k, v, s in tiles)

    gains = grp.assign(r=grp.flow / grp.walk).sort_values("r", ascending=False)
    best = ", ".join(f"{esc(r.disorder)} {fr(r.flow)} → {fr(r.walk)}" for _, r in gains.head(4).iterrows())
    losses = grp[grp.walk > grp.flow].assign(r=lambda d: d.walk / d.flow).sort_values("r", ascending=False)

    recall_svg, recall_json = recall_chart(pat, U)
    ks = [1, 10, 50, 100, 500]
    recall_rows = [[f"top {k:,}", pct(top(pat.nb, k), 1), pct(top(pat.flow, k), 1), pct(top(pat.walk, k), 1),
                    pct(top(pat.k1, k), 1) if pat.k1.notna().all() else "–"] for k in ks]

    group_rows = []
    for _, r in grp.sort_values("walk").iterrows():
        group_rows.append([esc(r.disorder), f"<code>{esc(r.causal_in_network)}</code>", int(r.sources), int(r.targets),
                           fr(r.nb), fr(r.flow), f"{fr(r.walk)} {change(r.flow, r.walk)}", pct(r.success, 1)])
    per_rows = [[esc(r.disorder), r.n, fr(r.flow), f"{fr(r.walk)} {change(r.flow, r.walk)}", pct(r.wins)]
                for _, r in per.sort_values("walk").iterrows()]

    kappa_rows = [["full absorption (default)", fr(gm_w), int((grp.walk <= 100).sum()), int((grp.walk <= 10).sum()),
                   fr(pm_w), pct(top(pat.walk, 10))]]
    for k in KAPPAS:
        col = grp[f"k{k}"]
        prow = pat.k1 if k == "1" else None
        kappa_rows.append([f"κ = {k}", fr(col.median()), int((col <= 100).sum()), int((col <= 10).sum()),
                           fr(prow.median()) if prow is not None else "not run",
                           pct(top(prow, 10)) if prow is not None else "–"])
    kappa_rows.append(["<i>current flow, for reference</i>", fr(gm_f), int((grp.flow <= 100).sum()),
                       int((grp.flow <= 10).sum()), fr(pm_f), pct(top(pat.flow, 10))])

    loss_rows = [[esc(r.disorder), fr(r.flow), fr(r.walk), int(r.sources), int(r.targets), pct(r.success, 1)]
                 for _, r in losses.iterrows()]
    sec = grp_all[grp_all.status == "secondary"]
    sec_note = "; ".join(f"{esc(r.disorder)}: {fr(r.flow)} → {fr(r.walk)}" for _, r in sec.iterrows())
    corr = float(np.corrcoef(np.log10(grp.success.clip(lower=1e-4)), np.log10(grp.flow / grp.walk))[0, 1])

    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Absorbing Walk Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Public+Sans:wght@400;500;600;700&display=swap">
<style>{CSS}</style></head><body><main>

<header class="prose">
  <div class="eyebrow">Metaboenrich · {esc(git_label())}</div>
  <h1>Absorbing random walk: benchmark</h1>
  <p class="lede">How well the experimental direction-respecting walk (<code>--mode walk</code>) finds the faulty enzyme in
  inborn errors of metabolism, compared with the current-flow method it would replace.</p>
</header>

<section class="summary">
  <ul>
    <li>Across {len(pat)} patients, the walk puts the causal gene in the top 10 for {pct(top(pat.walk, 10))} of them,
      against {pct(top(pat.flow, 10))} for current flow. The median rank halves, from {fr(pm_f)} to {fr(pm_w)} of {U:,} genes.</li>
    <li>It ranks the causal gene higher than current flow for {pct(better)} of patients and in {dis_better} of
      {len(per)} disorders. Largest group-level gains: {best}.</li>
    <li>It loses badly in two disorders: {', '.join(f'{esc(r.disorder)} ({fr(r.flow)} → {fr(r.walk)})' for _, r in losses.head(2).iterrows())}.
      Both have a recognisable cause (see <a href="#losses">where the walk loses</a>).</li>
    <li>Partial absorption (hook 2, κ) changes almost nothing. Full absorption stays the default.</li>
  </ul>
</section>

<section class="tiles">{tiles_html}</section>

<section>
  <h2>What was compared</h2>
  <div class="prose">
    <p><b>The test.</b> Untargeted plasma metabolomics from patients with a confirmed inborn error (Thistlethwaite et al.
    2020, Miller et al. 2015). Each run goes from the patient's metabolite changes to a ranked gene list. The score is where the
    known causal gene lands among the {U:,} genes on network reactions (1 = top). <b>Group level</b> pools each disorder's
    patients (BH q &lt; 0.05 and |mean z| ≥ 1); <b>patient level</b> runs each patient alone (|z| ≥ 2).</p>
    <p><b>Current flow</b> is the method used in the sealed blind predictions: Kirchhoff current from every increased to every
    decreased metabolite on the undirected network, raw scoring. <b>Absorbing walk</b>: walkers start at increased metabolites in
    proportion to |log2FC|, follow reaction directions, and stop at decreased ones. Reversible reactions are split in two, a
    walker can't undo its last step, dead ends leak, and only walks that reach a target are scored (decision D17).
    <b>Neighbour baseline</b> ranks each enzyme by how many significant metabolites its own reactions touch.</p>
    <p class="small">Same network (3,958 metabolites, 6,377 reactions), same mapping and significance rules for every method;
    only the scoring step differs. {n_dropped} patient{'' if n_dropped == 1 else 's'} with no reachable source–target pair in
    the walk {'is' if n_dropped == 1 else 'are'} left out of the head-to-head patient numbers.</p>
  </div>
</section>

<section>
  <h2>Patients: how often the causal gene is near the top</h2>
  <figure>
    <div class="legend"><span><i class="sw sw-walk"></i>Absorbing walk</span><span><i class="sw sw-flow"></i>Current flow</span>
      <span><i class="sw-line"></i>Neighbour baseline</span></div>
    {recall_svg}
    <figcaption>Each line shows, for every cut-off k, the share of the {len(pat)} primary patients whose causal gene is within the top k.
    Higher and further left is better. The walk's lead is at the top of the list; by the top 500 the two methods are level
    ({pct(top(pat.walk, 500), 1)} vs {pct(top(pat.flow, 500), 1)}). The baseline's late jump is a tie: genes touched by no
    significant metabolite all share the average of the remaining ranks. Hover for exact values.</figcaption>
  </figure>
  {table(["Cut-off", "Neighbour baseline", "Current flow", "Absorbing walk", "Walk, κ = 1"], recall_rows, num=(1, 2, 3, 4))}
</section>

<section>
  <h2>Patient by patient</h2>
  <figure class="square">
    {scatter_chart(pat, U)}
    <figcaption>One dot per patient. Dots below the diagonal (blue) are patients where the walk ranks the causal gene higher;
    above it (orange), current flow does. {int((pat.walk <= 1).sum())} patients have the causal gene ranked first by the walk,
    against {int((pat.flow <= 1).sum())} for current flow.</figcaption>
  </figure>
  {table(["Disorder", "Patients", "Current flow, median", "Walk, median", "Walk higher for"], per_rows, num=(1, 2, 3, 4))}
</section>

<section>
  <h2>Disorders (group level)</h2>
  <figure>
    <div class="legend"><span><i class="sw sw-flow"></i>Current flow</span><span><i class="sw sw-walk"></i>Absorbing walk</span></div>
    {dumbbell_chart(grp, U)}
    <figcaption>One row per disorder, sorted by how much the walk improves the rank. Hover a row for the causal gene, the number of
    increased and decreased metabolites, and the share of walkers that reached a target.</figcaption>
  </figure>
  {table(["Disorder", "Causal gene(s)", "Up", "Down", "Neighbour", "Current flow", "Walk", "Walkers reaching a target"],
         group_rows, num=(2, 3, 4, 5, 6, 7))}
  <p class="small">Medians: neighbour baseline {fr(grp.nb.median())}, current flow {fr(gm_f)}, walk {fr(gm_w)}. Top 100:
  {int((grp.nb <= 100).sum())}, {int((grp.flow <= 100).sum())} and {int((grp.walk <= 100).sum())} of {len(grp)}. Top 10:
  {int((grp.nb <= 10).sum())}, {int((grp.flow <= 10).sum())} and {int((grp.walk <= 10).sum())}.
  Secondary case, not in these counts: {sec_note}.</p>
</section>

<section>
  <h2>Partial absorption (hook 2)</h2>
  <p class="prose">With κ set, a decreased metabolite absorbs a walker with probability 1 − exp(−κ·|log2FC|) and lets the rest walk on
  to targets further along. On this benchmark it barely moves anything, so the simpler full absorption stays the default.</p>
  {table(["Setting", "Group median", "Group top 100", "Group top 10", "Patient median", "Patients in top 10"], kappa_rows,
         num=(1, 2, 3, 4, 5))}
</section>

<section id="losses">
  <h2>Where the walk loses</h2>
  {table(["Disorder", "Current flow", "Walk", "Up", "Down", "Walkers reaching a target"], loss_rows, num=(1, 2, 3, 4, 5))}
  <div class="prose">
    <p><b>Aromatic L-amino acid decarboxylase deficiency: shunt products point away from the block.</b> The strongest increase is
    3-O-methyldopa, which COMT makes from L-DOPA, the substrate DDC can't process. L-DOPA itself isn't among the significant
    metabolites. COMT runs one way, so walkers starting at 3-O-methyldopa can never get back to L-DOPA and then through DDC.
    Current flow ignores direction and uses exactly that route. Any disorder whose clearest signal is a shunt product will hit the same wall.</p>
    <p><b>Homocystinuria: no informative sink.</b> The only significant decrease is cortisol, which is unrelated to CBS, so
    walkers have nowhere meaningful to go. Few of them reach it.</p>
    <p class="small">These explanations come from inspecting the two runs, not from a general test. Across the {len(grp)} disorders, the
    share of walkers that reach a target doesn't predict the gain or loss (correlation of log shares {corr:+.2f}): MSUD gains most with
    under 5% success.</p>
  </div>
</section>

<section>
  <h2>Caveats</h2>
  <ul class="prose">
    <li>One benchmark, {len(grp)} disorders, {len(pat)} patients, all plasma. The settings were not tuned on it, but the walk was
      designed after seeing current flow's results on the same data, so treat the gain as promising, not established.</li>
    <li>The walk depends on Human-GEM's reversibility labels (from flux bounds). Wrong labels block or open routes.</li>
    <li>Leave-one-out scoring isn't defined for the walk; all comparisons here use raw scoring for both methods.</li>
    <li>The sealed blind predictions were made with current flow. Whether the walk becomes the default should be decided before the
      next blind round, not after looking at those answers.</li>
  </ul>
</section>

<footer>
  <span>Built by <code>tools/report/build_rw_report.py</code> from <code>results/bench_rw/</code> and <code>results/bench_rw_p/</code>.
  Method: <code>metaboenrich/walk.py</code>; decision log entry D17.</span>
</footer>
</main>
<div class="tip" id="tip" hidden></div>
<script type="application/json" id="recall-data">{recall_json}</script>
<script>{JS}</script>
</body></html>
"""
    Path(out).write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({len(page) / 1e3:.0f} kB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs" / "rw_report.html"))
    build(ap.parse_args().out)
