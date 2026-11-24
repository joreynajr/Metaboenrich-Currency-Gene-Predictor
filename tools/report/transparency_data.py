"""Transparency displays for how the evaluation data was found, screened and registered.

Each figure makes one claim a reader can check by eye; the title states it. Per-unit and
per-candidate detail (receipt cards, evidence cards, cache receipts, audit rows, quality
flags, decision links, contrast group boxes) also goes to JSON for an interactive page.

    python tools/report/transparency_data.py [--out results/transparency/data] [--seed 26]

Figures (png + pdf):
  recall_scoreboard  registered repository studies the sweep re-found, and why it missed the rest
  source_flow        sweep source -> what the hit is -> what happened to it
  overlap            hand search vs repository sweep, at study level
  failure_ledger     every swept study has its raw response cached; failed fetches per source
  drop_funnel        every registered comparison -> testable -> primary, every drop with its reason and rule
  contrast_table     all C/D contrasts: what separates case from control, what is matched, groups pooled
  contrast_diagrams  four worked examples: every deposited group, case and control marked, unused arms grey
  inclusion_margin   each testable unit's distance above the two inclusion floors (D26/D28, D29)
  daa_recheck        every metabolite of every C/D comparison re-tested row by row vs the DAA output
JSON: cards.json (one card per comparison), contrasts.json (contrast_table's rows), candidates.json (top sweep candidates with evidence
and cache receipts), figures.json (file, title, caption).

Reads results/data_discovery/, benchmarks/data_registry.tsv, results/data_figures/units.tsv,
results/{workbench,metabolights}_daa/, the per-sample tables and docs/decisions.md.
"""
import argparse
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_data_figures as B   # noqa: E402  (also puts the repo and tools/ on sys.path)
from build_data_figures import CC, DEC, FC, FL, INC, INK, INK2, PERT, ROOT, RULE   # noqa: E402

DISC = ROOT / "results" / "data_discovery"
CACHE = DISC / "cache"
MW = "https://www.metabolomicsworkbench.org/rest/study/study_id"
ML = "https://www.ebi.ac.uk/metabolights/ws/studies"
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
GITHUB_DECISIONS = ("https://github.com/joreynajr/Metaboenrich-Currency-Gene-Predictor/blob/null-calibration/"
                    "docs/decisions.md")
# status colours (reserved; always shown with an icon and a label)
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"
RAG = {"green": (GOOD, "✓"), "amber": (WARN, "!"), "red": (CRIT, "✕")}
GREY = "#c6cad0"
SRC = {"workbench": "Metabolomics Workbench", "metabolights": "MetaboLights", "literature": "Europe PMC literature"}
# why the sweep missed a known study (read from the cached mwTab / factors, 2026-11-23)
MISS_WHY = {"ST000698": "gene written 'Sirt5' (mouse-style case), not the human symbol",
            "ST003262": "gene named only as 'creatine kinase', never as CKB",
            "ST004985": "disease named only as 'VLCAD deficient', never as ACADVL"}
# fate -> (decision, short rule) for drop and keep reasons
FATE_RULE = {"primary": ("D26", "loss of function of one network gene"),
             "nonprimary": ("D26", "inhibitor or gain of function of one network gene"),
             "nogene": ("D26", "causal gene must be in the network"),
             "norun": ("D22", "needs significant metabolites in the network (raw p < 0.05)"),
             "single": ("D21", "group level only, no single subjects"),
             "mouse": ("D25", "human only"),
             "lowsignal": ("D29", ">= 3 significantly changed metabolites in the network"),
             "panel": ("D29", "targeted panel never processed; recheck its signal"),
             "fewsamples": ("D26", ">= 3 samples per side"),
             "lt3": ("D22", "too few metabolites in the network to score"),
             "raw": ("D26", "needs per-sample values, not raw spectra"),
             "novalues": ("D26", "deposit must contain values"),
             "tracing": ("D26", "no isotope tracing"),
             "lipid": ("D26", "no lipid-only panels"),
             "caselist": ("D12", "excluded in the benchmark case list"),
             "other": ("D26", "weak design / no single specific target")}


def ckey(url, params=None):
    return hashlib.sha1((url + json.dumps(params or {}, sort_keys=True)).encode()).hexdigest()


def cached(url, kind="json"):
    return CACHE / f"{ckey(url)}.{'json' if kind == 'json' else 'txt'}"


def save(fig, out, name, figs, title, caption):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    figs.append({"file": f"{name}.png", "title": title, "caption": caption})


def decisions():
    """D number -> (heading, GitHub anchor url)."""
    out = {}
    for line in (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (D\d+)\. (.*)", line)
        if m:
            slug = re.sub(r"[^\w\- ]", "", f"{m.group(1)}. {m.group(2)}".lower()).replace(" ", "-")
            out[m.group(1)] = (m.group(2), f"{GITHUB_DECISIONS}#{slug}")
    return out


def d26_added():
    """Repository study ids added by D26 (item 1 of that decision)."""
    txt = (ROOT / "docs" / "decisions.md").read_text(encoding="utf-8")
    sec = txt[txt.index("### D26."):txt.index("Clusters (", txt.index("### D26."))]   # the cluster note names old ids
    return set(re.findall(r"\b(ST\d{6}|MTBLS\d+)\b", sec))


def load():
    c = pd.read_csv(DISC / "candidates.tsv", sep="\t", dtype=str).fillna("")
    s = pd.read_csv(DISC / "screen.tsv", sep="\t", dtype=str).fillna("")
    k = pd.read_csv(DISC / "curated.tsv", sep="\t", dtype=str).fillna("")
    r = pd.read_csv(ROOT / "benchmarks" / "data_registry.tsv", sep="\t", dtype=str).fillna("")
    u = pd.read_csv(ROOT / "results" / "data_figures" / "units.tsv", sep="\t", dtype=str).fillna("")
    return c, s, k, r, u


# ---- 1. recall ---------------------------------------------------------------------
def recall_scoreboard(c, r, out, figs):
    repo = r[r.study.str.match(r"^(ST\d{6}|MTBLS\d+)$") & (r.status == "active")]
    known = sorted(set(repo.study) - d26_added())          # found by the earlier hand search
    gene = repo.drop_duplicates("study").set_index("study").causal_genes
    found = set(c.id)
    hit = [s for s in known if s in found]
    miss = [s for s in known if s not in found]
    fig, ax = plt.subplots(figsize=(7.2, 2.5))
    ax.set_xlim(0, 10); ax.set_ylim(-0.2, 3.4); ax.axis("off")
    for i, s in enumerate(hit + miss):
        x, y = i % 10, 2.6 - (i // 10) * 0.95
        ok = s in found
        ax.add_patch(FancyBboxPatch((x + 0.06, y), 0.88, 0.72, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc="#126b63" if ok else "white", ec="#126b63" if ok else CRIT, lw=1.2))
        ax.text(x + 0.5, y + 0.47, s, ha="center", va="center", fontsize=5.6, color="white" if ok else INK)
        ax.text(x + 0.5, y + 0.2, gene.get(s, "").split(";")[0], ha="center", va="center", fontsize=5.4,
                style="italic", color="white" if ok else INK2)
    y0 = 2.6 - ((len(known) - 1) // 10) * 0.95 - 0.35
    for j, s in enumerate(miss):
        ax.text(0.06, y0 - 0.27 * j, f"✕ {s} ({gene.get(s, '')}): {MISS_WHY.get(s, 'not matched')}", fontsize=6.4,
                color=INK, va="center")
    ax.set_ylim(y0 - 0.27 * len(miss) - 0.05, 3.4)
    title = f"The sweep re-finds {len(hit)} of {len(known)} studies we already knew about"
    ax.set_title(title, loc="left", fontsize=9.5, weight="bold")
    save(fig, out, "recall_scoreboard", figs, title,
         "Every repository study found earlier by hand is a known positive; filled tiles were found by the "
         "automatic sweep, outlined ones were missed, with the reason read from the cached deposit text.")
    return {"known": known, "found": hit, "missed": {s: MISS_WHY.get(s, "") for s in miss}}


# ---- 2. source flow (alluvial) ------------------------------------------------------
def what(row):
    if row.in_network != "True":
        return "gene not in network"
    if row.design_flags:
        return "tracing or lipid-only"
    if row.perturbation == "loss":
        return "network gene, lost"
    if row.perturbation in ("chemical", "gain"):
        return "network gene, inhibitor or gain"
    return "network gene, no perturbation word"


def fate_of(row, scr, cur):
    if row.in_registry == "active":
        return "in the test set"
    if row.in_registry:
        return "registered, dropped with reason"
    if row.source == "literature":
        return "paper lead (needs a deposit)"
    v = cur.get(row.id, "")
    if v in ("register",):
        return "in the test set"
    if v in ("reject", "maybe") or scr.get((row.id, row.gene), "") in ("not usable",):
        return "screened out with reason"
    if (row.id, row.gene) in scr:
        return "screened out with reason"
    return "not screened (filtered earlier)"


def alluvial(ax, cols, flows, colors, gap=0.012, width=0.035):
    """cols: list of ordered node lists; flows: list of dict {(a, b): n} between adjacent columns."""
    tot = sum(flows[0].values())
    xs = np.linspace(0.02, 0.74, len(cols))
    pos = []
    for j, nodes in enumerate(cols):
        sizes = {}
        for n in nodes:
            fl = flows[j - 1] if j else flows[0]
            sizes[n] = sum(v for (a, b), v in fl.items() if (b if j else a) == n)
        span = 1 - gap * (len(nodes) - 1)
        y, p = 1.0, {}
        for n in nodes:
            h = sizes[n] / tot * span
            p[n] = [y - h, y, y, y]          # bottom, top, out-cursor, in-cursor
            y -= h + gap
        pos.append(p)
    for j, fl in enumerate(flows):
        for a in cols[j]:
            for b in cols[j + 1]:
                v = fl.get((a, b), 0)
                if not v:
                    continue
                h = v / tot * (1 - gap * (len(cols[j]) - 1))
                y0a = pos[j][a][2]; pos[j][a][2] -= h
                y0b = pos[j + 1][b][3]; pos[j + 1][b][3] -= h
                x0, x1 = xs[j] + width, xs[j + 1]
                xm = (x0 + x1) / 2
                verts = [(x0, y0a), (xm, y0a), (xm, y0b), (x1, y0b), (x1, y0b - h), (xm, y0b - h), (xm, y0a - h),
                         (x0, y0a - h), (x0, y0a)]
                codes = [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4, MPath.LINETO, MPath.CURVE4,
                         MPath.CURVE4, MPath.CURVE4, MPath.CLOSEPOLY]
                mid = b if j == 0 else a                      # bands wear the middle column's colour
                ax.add_patch(PathPatch(MPath(verts, codes), fc=colors.get(mid, GREY), ec="none", alpha=0.42))
    for j, nodes in enumerate(cols):
        for n in nodes:
            lo, hi = pos[j][n][:2]
            ax.add_patch(Rectangle((xs[j], lo), width, hi - lo, color=colors.get(n, INK2), lw=0))
            nn = round((hi - lo) / (1 - gap * (len(nodes) - 1)) * tot)
            ax.text(xs[j] + width + 0.006, (lo + hi) / 2, f"{n}  {nn}", ha="left", va="center", fontsize=6.4,
                    color=INK, bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85) if 0 < j < len(cols) - 1 else None)
    ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1.01); ax.axis("off")


def source_flow(c, s, k, out, figs):
    scr = {(r.id, r.gene): r.verdict for r in s.itertuples()}
    cur = {r.id: r.verdict for r in k.itertuples()}
    d = c.assign(src=c.source.map({"workbench": "Workbench", "metabolights": "MetaboLights",
                                   "literature": "Literature"}),
                 what=[what(r) for r in c.itertuples()], fate=[fate_of(r, scr, cur) for r in c.itertuples()])
    A = ["Workbench", "MetaboLights", "Literature"]
    Bn = ["network gene, lost", "network gene, inhibitor or gain", "network gene, no perturbation word",
          "tracing or lipid-only", "gene not in network"]
    Cn = ["in the test set", "registered, dropped with reason", "screened out with reason",
          "not screened (filtered earlier)", "paper lead (needs a deposit)"]
    f1 = d.groupby(["src", "what"]).size().to_dict()
    f2 = d.groupby(["what", "fate"]).size().to_dict()
    col = {"Workbench": CC["C"], "MetaboLights": CC["D"], "Literature": "#8c7b6b",
           "network gene, lost": PERT["gene removed or lost"], "network gene, inhibitor or gain": PERT["chemically blocked"],
           "network gene, no perturbation word": GREY, "tracing or lipid-only": FC["tracing"],
           "gene not in network": FC["nogene"], "in the test set": "#126b63", "registered, dropped with reason": INK2,
           "screened out with reason": "#9a9da1", "not screened (filtered earlier)": GREY,
           "paper lead (needs a deposit)": "#8c7b6b"}
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    alluvial(ax, [A, Bn, Cn], [f1, f2], col)
    for x, h in ((0.02, "Source (study or paper × gene)"), (0.38, "What the hit is"), (0.74, "What happened")):
        ax.text(x, 1.045, h, ha="left", fontsize=7, weight="bold", color=INK)
    nt = (d.fate == "in the test set").sum()
    lost = (d.what == "network gene, lost").sum()
    title = (f"{len(d)} hits: {lost} name a lost network gene; {nt} became test-set studies, "
             "the rest left with a reason")
    ax.set_title(title, loc="left", fontsize=9, weight="bold", pad=18)
    fig.tight_layout()
    save(fig, out, "source_flow", figs, title,
         "Follow any band from its source to its end: repositories supply the test-set studies, literature supplies "
         "leads that still need a deposited table, and nothing disappears without a labelled endpoint.")


# ---- 3. overlap ---------------------------------------------------------------------
def overlap(c, r, out, figs):
    repo_ids = r.study[r.study.str.match(r"^(ST\d{6}|MTBLS\d+)$")]
    added = d26_added()
    hand = set(repo_ids) - added                       # registered before the sweep existed
    swept = set(c.id[c.source != "literature"])
    sweep = set(repo_ids) & swept                      # registered studies the sweep finds
    reg_now = set(r.study[(r.status == "active") & r.study.isin(sweep | hand)])
    only_h, both, only_s = hand - sweep, hand & sweep, sweep - hand
    other = len(swept - set(repo_ids))
    lit = c[c.source == "literature"].id.nunique()
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    ax.set_xlim(0.8, 10.2); ax.set_ylim(-0.05, 5.0); ax.set_aspect("equal"); ax.axis("off")
    for x, colr in ((3.3, "#8c7b6b"), (5.6, CC["C"])):
        ax.add_patch(plt.Circle((x, 2.4), 1.9 if x < 4 else 2.25, fc=colr, alpha=0.22, ec=colr, lw=1.2))
    ax.text(2.0, 2.4, f"{len(only_h)}", ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(2.0, 1.85, "hand only", ha="center", fontsize=6.5, color=INK2)
    ax.text(4.35, 2.4, f"{len(both)}", ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(4.35, 1.85, "both", ha="center", fontsize=6.5, color=INK2)
    ax.text(6.6, 2.4, f"{len(only_s)}", ha="center", va="center", fontsize=14, weight="bold", color=INK)
    ax.text(6.6, 1.85, "sweep only", ha="center", fontsize=6.5, color=INK2)
    ax.text(6.6, 1.5, f"({len(added & only_s)} registered in D26)", ha="center", fontsize=6.2, color=INK2)
    ax.text(3.0, 4.55, "Earlier hand search\n(PubMed, Claude Desktop)", ha="center", fontsize=7, color=INK)
    ax.text(6.4, 4.85, "Automatic repository sweep", ha="center", fontsize=7, color=INK)
    ax.text(8.15, 3.6, f"also flagged:\n{other} repository\nstudies screened out\n+ {lit} literature\npaper leads",
            fontsize=6.3, color=INK2, va="top")
    title = (f"The sweep finds {len(sweep)} of {len(hand | sweep)} registered repository studies, "
             f"{len(only_s)} of them new")
    fig.suptitle(title, x=0.012, ha="left", fontsize=9, weight="bold")
    fig.subplots_adjust(left=0, right=1, top=0.9, bottom=0)
    save(fig, out, "overlap", figs, title,
         "Registered repository studies (Workbench ST, MetaboLights MTBLS, any status) by how they were found: the "
         "overlap shows the sweep agrees with the hand search, the right side is what the tool added.")
    return {"hand_only": sorted(only_h), "both": sorted(both), "sweep_only_registered": sorted(added & only_s),
            "registered_active": len(reg_now)}


# ---- 4. failure ledger --------------------------------------------------------------
def failure_ledger(out, figs):
    summ = json.loads(cached(f"{MW}/ST/summary").read_text(encoding="utf-8"))
    wb = [v["study_id"] for v in summ.values() if "sapiens" in v.get("species", "")]
    mlj = json.loads(cached(ML).read_text(encoding="utf-8")).get("content", [])
    rows = []
    have = sum(cached(f"{MW}/{s}/factors").exists() for s in wb)
    have2 = sum(cached(f"{MW}/{s}/mwtab/txt", "txt").exists() for s in wb)
    rows.append(("Workbench", "human studies cached", len(wb), min(have, have2)))
    have = sum(cached(f"{ML}/{s}/organisms").exists() for s in mlj)
    rows.append(("MetaboLights", "studies cached (species check)", len(mlj), have))
    failed = [l for l in (DISC / "failed_fetches.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    nfail = {"Workbench": sum(MW in l for l in failed), "MetaboLights": sum(ML in l for l in failed),
             "Literature": sum(EPMC in l for l in failed)}
    pages = [f for f in CACHE.glob("*.json") if f.stat().st_size > 200_000
             and '"resultList"' in f.open(encoding="utf-8").read(4000)]
    m = re.search(r"literature (\d+)", (DISC / "report.md").read_text(encoding="utf-8"))
    n_pap = int(m.group(1)) if m else 0
    rows.append(("Literature", f"result pages cached ({n_pap:,} papers)", len(pages), len(pages)))
    fig, axs = plt.subplots(1, 3, figsize=(7.6, 1.9))
    ledger = {}
    for ax, (name, what_, n, ok) in zip(axs, rows):
        f = nfail[name]
        rate = f / max(n, 1)
        st = "green" if f == 0 and ok == n else ("amber" if rate < 0.01 else "red")
        colr, icon = RAG[st]
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0,rounding_size=0.05", fc="white",
                                    ec=RULE, lw=1))
        ax.add_patch(Rectangle((0.02, 0.05), 0.035, 0.9, color=colr, lw=0))
        ax.text(0.1, 0.8, name, fontsize=8, weight="bold", color=INK)
        ax.text(0.1, 0.47, f"{ok:,} / {n:,}", fontsize=15, weight="bold", color=INK)
        ax.text(0.1, 0.3, what_, fontsize=6.2, color=INK2)
        ax.text(0.1, 0.13, f"{icon} {f} failed fetch{'es' if f != 1 else ''}"
                + (" (retried later)" if f and ok == n else ""), fontsize=6.6, color=INK)
        ledger[name] = {"items": n, "cached": ok, "failed": f, "status": st}
    title = (f"{sum(nfail.values())} request failed in the whole sweep; every other study's raw response "
             "is on disk")
    fig.suptitle(title, x=0.012, ha="left", fontsize=9, weight="bold")
    fig.tight_layout()
    save(fig, out, "failure_ledger", figs, title,
         "An incomplete sweep cannot pass silently: each source shows how many items have their raw response on disk "
         "(the provenance record) and how many requests failed after three tries.")
    return ledger, failed


# ---- 5. funnel ----------------------------------------------------------------------
def drop_funnel(u, D, out, figs):
    n = len(u)
    keep = ["primary", "nonprimary"]
    drops = u[~u.fate.isin(keep)].fate.value_counts()
    nt = u.fate.isin(keep).sum()
    npri = (u.fate == "primary").sum()
    rows = [(f, drops[f]) for f in drops.index]
    fig, ax = plt.subplots(figsize=(7.6, 0.24 * (len(rows) + 3) + 0.7))
    H = len(rows) + 3
    ax.set_xlim(0, n + 112); ax.set_ylim(H - 1.3, -0.8); ax.axis("off")
    # main trunk
    left = 0
    ax.barh(0, n, color="#5a5c61", height=0.62, lw=0)
    ax.text(n + 2, 0, f"{n} registered comparisons", va="center", fontsize=7, color=INK, weight="bold")
    rem = n
    for i, (f, k) in enumerate(rows, start=1):
        rem -= k
        ax.barh(i, k, left=rem, color=FC.get(f, GREY), height=0.62, lw=0)
        ax.plot([rem, rem], [i - 1 + 0.31, i - 0.31], color=RULE, lw=0.8)
        d, rule = FATE_RULE.get(f, ("", ""))
        ax.text(rem + k + 2, i, f"−{k}  {FL[f]}", va="center", fontsize=6.6, color=INK)
        ax.text(n + 110, i, d, va="center", ha="right", fontsize=6.6, color=INK2, weight="bold")
    i = len(rows) + 1
    ax.barh(i, nt, color=FC["nonprimary"], height=0.62, lw=0)
    ax.barh(i, npri, color=FC["primary"], height=0.62, lw=0)
    ax.text(nt + 2, i, f"{nt} testable ({npri} gene lost = primary, {nt - npri} inhibitor or gain)", va="center",
            fontsize=7, weight="bold", color=INK)
    ax.text(n + 110, -0.75, "rule", ha="right", fontsize=6.4, color=INK2)
    title = f"{n - nt} of {n} comparisons dropped, each with a stated reason and the rule behind it"
    ax.set_title(title, loc="left", fontsize=9, weight="bold")
    fig.tight_layout()
    save(fig, out, "drop_funnel", figs, title,
         "Read down: each grey-to-colour step peels off the comparisons failing one rule, with the decision (D number "
         "in docs/decisions.md) that set it, until only the testable set remains.")
    return {f: int(k) for f, k in drops.items()}


# ---- contrast reconstruction (shared by contrast diagrams, cards and audit) --------
def wb_groups(study, sub):
    import workbench_daa as W
    spec = W.C.get((study, sub))
    an = sub.split("_")[0]
    f = W.PER_SAMPLE / f"{study}_{an}.csv"
    if spec is None or spec.get("not_computable") or not f.exists():
        return None
    raw = pd.read_csv(f, dtype=str)
    labels = raw.loc[raw.iloc[:, 0] == "#GROUP"].iloc[0, 1:]
    case, ctrl = W.select(labels, spec["case"]), W.select(labels, spec["control"])

    def cond(lab):
        return " | ".join(f"{k}:{v}" for k, v in W.parse_label(lab).items()
                          if k not in ("Sample Group", "Injection order"))
    g = {}
    for col, lab in labels.items():
        role = "case" if col in case else "control" if col in ctrl else (
            "QC / blank" if W.QC_PAT.search(str(lab)) else "not used")
        key = (cond(lab), role)
        g[key] = g.get(key, 0) + 1
    return dict(kind="workbench", file=f, raw=raw, labels=labels, case=case, control=ctrl, reason=spec["reason"],
                groups=[{"group": k[0], "role": k[1], "n": n} for k, n in g.items()])


def ml_groups(study, sub):
    import metabolights_daa as M
    if (study, sub) not in M.CONTRASTS:
        return None
    rel, case, control, reason = M.CONTRASTS[(study, sub)]
    f = M.DATA / rel
    if not f.exists():
        return None
    raw = pd.read_csv(f, dtype={"CHEBI": str})
    groups = raw.iloc[0, 2:]
    g = groups.value_counts()
    return dict(kind="metabolights", file=f, raw=raw, labels=groups, case=list(groups.index[groups == case]),
                control=list(groups.index[groups == control]), reason=reason,
                groups=[{"group": k, "role": "case" if k == case else "control" if k == control else "not used",
                         "n": int(n)} for k, n in g.items()])


def contrast_of(row):
    try:
        if row.collection.startswith("C"):
            return wb_groups(row.study, row.sub_unit)
        if row.collection.startswith("D"):
            return ml_groups(row.study, row.sub_unit)
    except Exception as e:       # a missing per-sample table must not stop the report
        print("contrast", row.label, e)
    return None


ROLE = {"case": ("#126b63", "white"), "control": ("#8fc4bf", INK), "not used": ("#eef0f2", INK2),
        "QC / blank": ("white", INK2)}
SKIP_FACTORS = {"Sample Group", "Injection order"}
# Contrasts where case and control differ in more than one factor, read by hand (DM, 2026-11-24).
REVIEWED = {"ST003412": ("same", "Drug and Drug Treatment record the same drug"),
            "ST000698": ("same", "the knockdown is recorded across three yes/no columns"),
            "ST003499": ("differs", "patients and controls also differ in sex and age")}
NOTES = {"ST004410": "case: antimycin blocks the electron transport chain, so DHODH has no electron acceptor; "
                     "control: the yeast enzyme ScURA1 bypasses that, on the same drug"}


def factors(lab):
    """A deposited group label as {factor: value}; a bare label (MetaboLights) is one factor, 'group'."""
    import html
    import workbench_daa as W
    d = {k: html.unescape(v) for k, v in W.parse_label(lab).items() if k not in SKIP_FACTORS}
    return d or {"group": html.unescape(str(lab).strip())}


def contrast_summary(r, g):
    """What separates case from control in one comparison, what they share, and how many groups were pooled."""
    cs = [factors(g["labels"][c]) for c in g["case"]]
    ct = [factors(g["labels"][c]) for c in g["control"]]
    keys = sorted(set().union(*cs, *ct))
    val = lambda ds, k: sorted({d.get(k, "–") for d in ds})          # noqa: E731
    differ = [k for k in keys if val(cs, k) != val(ct, k)]
    same = [k for k in keys if k not in differ and len(val(cs, k)) == 1]
    rev = REVIEWED.get(r.study)
    status = "one factor" if len(differ) == 1 else ("reviewed: one perturbation" if rev and rev[0] == "same"
                                                    else "several factors")
    return {"label": r.label, "study": r.study, "gene": r.causal_genes.split(";")[0], "primary": r.primary == "True",
            "differ": [{"factor": k, "case": val(cs, k), "control": val(ct, k)} for k in differ],
            "matched": [{"factor": k, "value": val(cs, k)[0]} for k in same],
            "n_case_groups": len({str(sorted(d.items())) for d in cs}),
            "n_control_groups": len({str(sorted(d.items())) for d in ct}),
            "n_case": len(g["case"]), "n_control": len(g["control"]), "status": status,
            "note": (rev[1] if rev else NOTES.get(r.study, "")), "reason": g.get("reason", "")}


def short(s, n):
    s = str(s)
    return s if len(s) <= n else s[:n - 1] + "…"


def contrast_table(u, groups, out, figs):
    ts = u[(u.testable == "True") & u.collection.str[0].isin(["C", "D"])].drop_duplicates("label")
    rows = [contrast_summary(r, groups[r.label]) for r in ts.itertuples() if groups.get(r.label)]
    order = {"several factors": 0, "reviewed: one perturbation": 1, "one factor": 2}
    rows.sort(key=lambda x: (order[x["status"]], x["gene"], x["label"]))
    MARK = {"one factor": ("✓", "#126b63"), "reviewed: one perturbation": ("✓*", "#126b63"),
            "several factors": ("!", "#b7791f")}
    h = 0.142
    H = h * (len(rows) + 1.2) + 0.9
    fig = plt.figure(figsize=(10.4, H))
    ax = fig.add_axes([0.005, 0.45 / H, 0.99, h * (len(rows) + 1.2) / H])
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(len(rows), -1.2)
    X = [0.0, 0.025, 0.215, 0.6, 0.86]
    for x, t in zip(X, ["", "comparison · gene", "what differs:  case → control", "matched in both",
                        "groups pooled  case / ctrl"]):
        ax.text(x, -0.6, t, fontsize=5.8, color=INK2, weight="bold")
    for i, x in enumerate(rows):
        if i % 2 == 0:
            ax.add_patch(Rectangle((0, i), 1, 1, color="#f5f6f7", lw=0))
        m, c = MARK[x["status"]]
        ax.text(X[0] + 0.006, i + 0.55, m, fontsize=6, color=c, weight="bold", va="center")
        ax.text(X[1], i + 0.55, f"{short(x['label'], 30)} · {x['gene']}", fontsize=5.4, va="center", color=INK)
        d = "; ".join(f"{e['factor']}: {'/'.join(e['case'])} → {'/'.join(e['control'])}" for e in x["differ"])
        ax.text(X[2], i + 0.55, short(d, 92), fontsize=5.4, va="center", color=INK)
        mt = ", ".join(f"{e['factor']}={e['value']}" for e in x["matched"]) or "—"
        ax.text(X[3], i + 0.55, short(mt, 58), fontsize=5.2, va="center", color=INK2)
        pool = f"{x['n_case_groups']} / {x['n_control_groups']}   (n {x['n_case']} / {x['n_control']})"
        ax.text(X[4], i + 0.55, pool, fontsize=5.4, va="center", color=INK)
    n = {k: sum(x["status"] == k for x in rows) for k in MARK}
    title = (f"All {len(rows)} case-vs-control contrasts: {n['one factor']} differ in exactly one factor, "
             f"{n['reviewed: one perturbation']} record one perturbation in several columns, "
             f"{n['several factors']} {'differs' if n['several factors'] == 1 else 'differ'} in more")
    fig.suptitle(title, x=0.008, ha="left", fontsize=9, weight="bold", y=0.995)
    flagged = [x for x in rows if x["note"] and x["status"] != "one factor"]
    fig.text(0.008, 0.008, "✓ case and control differ in one factor only.  ✓* several columns, one perturbation "
             "(read by hand).  ! a real second difference.  Matched: factors with one shared value.  "
             "Collections C (Workbench) and D (MetaboLights).\n" +
             "  ".join(f"{x['study']}: {x['note']}." for x in {x['study']: x for x in flagged}.values()),
             fontsize=5.8, color=INK2)
    save(fig, out, "contrast_table", figs, title,
         "One row per comparison: the middle column is the only thing that should separate case from control; "
         "everything in 'matched in both' is held equal. Amber rows need a reader's judgement.")
    return rows


def contrast_diagrams(u, groups, rows, out, figs):
    """Every contrast the table does not mark clean (case and control differ in more than one factor), one study
    per panel, with all its deposited groups."""
    studies = list(dict.fromkeys(x["study"] for x in rows if x["status"] != "one factor"))
    ts = u[(u.testable == "True")].drop_duplicates("label")
    panels = []
    for s in studies:
        r = next((r for r in ts.itertuples() if r.study == s and groups.get(r.label)), None)
        if r is not None:
            panels.append((r, groups[r.label], contrast_summary(r, groups[r.label])))
    import textwrap
    nshow, LH = 7, 0.19                                     # groups shown per panel; inches per line
    lay = []
    for r, g, cs in panels:
        gs = sorted(g["groups"], key=lambda x: ["case", "control", "not used", "QC / blank"].index(x["role"]))
        note = textwrap.wrap(cs["note"], 220)
        lay.append((gs, note, 2 + len(note) + min(len(gs), nshow) + (len(gs) > nshow) + 0.6))
    H = sum(n for *_, n in lay) * LH + 0.45
    fig, axs = plt.subplots(len(panels), 1, figsize=(10.4, H), squeeze=False,
                            gridspec_kw={"height_ratios": [n for *_, n in lay], "hspace": 0.0})
    for ax, (r, g, cs), (gs, note, n) in zip(np.ravel(axs), panels, lay):
        ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(n, 0)
        parsed = [factors(x["group"].replace(" | ", "|")) if ":" in x["group"] else {"group": x["group"]} for x in gs]
        vary = [k for k in sorted({k for d in parsed for k in d}) if len({d.get(k, "") for d in parsed}) > 1]
        ax.text(0, 0.5, f"{r.study} · {cs['gene']}", fontsize=7.4, weight="bold", va="center", color=INK)
        ax.text(1.0, 0.5, f"{len(gs)} groups deposited", fontsize=6, va="center", ha="right", color=INK2)
        dif = "; ".join(f"{e['factor']}: {'/'.join(e['case'])} → {'/'.join(e['control'])}" for e in cs["differ"])
        ax.text(0, 1.5, "differs: " + short(dif, 175), fontsize=6, va="center", color="#126b63", weight="bold")
        for j, line in enumerate(note):
            ax.text(0, 2.45 + j, line, fontsize=5.6, va="center", color=INK2, style="italic")
        top = 2 + len(note)
        dk = {e["factor"] for e in cs["differ"]}
        for i, (x, d) in enumerate(zip(gs[:nshow], parsed[:nshow])):
            y = top + i
            fc, tc = ROLE[x["role"]]
            ax.add_patch(FancyBboxPatch((0.0, y + 0.1), 0.99, 0.8, boxstyle="round,pad=0,rounding_size=0.012",
                                        fc=fc, ec=INK2 if x["role"] == "QC / blank" else "none", lw=0.5, ls=(0, (2, 2))))
            lab = "  ·  ".join(f"{k}: {d[k]}" for k in sorted(vary, key=lambda k: k not in dk) if k in d) or x["group"]
            ax.text(0.015, y + 0.5, f"{x['role']} · n={x['n']}", fontsize=5.5, va="center", color=tc,
                    weight="bold" if x["role"] in ("case", "control") else "normal")
            ax.text(0.085, y + 0.5, short(lab, 190), fontsize=5.5, va="center", color=tc)
        if len(gs) > nshow:
            ax.text(0.015, top + nshow + 0.5, f"+ {len(gs) - nshow} more unused groups", fontsize=5.5,
                    color=INK2, va="center")
    nrow = sum(x["status"] != "one factor" for x in rows)
    title = (f"The {nrow} contrasts where case and control differ in more than one factor ({len(panels)} studies), "
             "with every deposited group")
    fig.suptitle(title, x=0.012, ha="left", fontsize=9.5, weight="bold")
    fig.subplots_adjust(left=0.012, right=0.988, top=1 - 0.4 / H, bottom=0.01)
    save(fig, out, "contrast_diagrams", figs, title,
         "Chosen by rule, not by hand: every study with a contrast the table does not mark clean. Dark = case, "
         "light = control, grey = deposited but not used; the green line says what separates them.")


# ---- inclusion margin --------------------------------------------------------------
def margins(r, g):
    """Samples per side (the smaller arm; collection A: patients against a reference population) and significantly
    changed metabolites in the network, the two quantities the inclusion rules (D26/D28, D29) put a floor on."""
    n = pd.to_numeric(r.n_samples, errors="coerce")
    if g is not None:
        npg = min(len(g["case"]), len(g["control"]))
    elif r.collection.startswith("A"):
        npg = n
    else:
        npg = (n / 2) if pd.notna(n) and n else np.nan
    return npg, int(r.n_inc or 0) + int(r.n_dec or 0)


def rag(r, g):
    """Only the written rules: red fails D26 (>= 3 samples per side; D28 applies it to every unit) or D29 (>= 3
    significantly changed metabolites in the network), amber meets them with a documented caveat, green meets them."""
    npg, ch = margins(r, g)
    why = []
    if pd.notna(npg) and npg < 3:
        why.append(f"{int(npg)} per side, under 3 (D26)")
    if ch < 3:
        why.append(f"{ch} changed metabolites in the network, under 3 (D29)")
    if why:
        return "red", "; ".join(why)
    amb = []
    if pd.isna(npg):
        amb.append("sample size not recorded, rule cannot be checked")
    if "other mitochondrial" in (g or {}).get("reason", "") or r.study == "ST001441":
        amb.append("controls are patients with other diseases (D26)")
    if amb:
        return "amber", "; ".join(amb)
    return "green", f"{int(npg)} per side, {ch} changed metabolites in the network"


def inclusion_margin(u, groups, out, figs):
    """Where each testable comparison sits above the two inclusion floors. Only the written rules are drawn."""
    ts = u[u.testable == "True"].drop_duplicates("label").copy()
    res = [rag(r, groups.get(r.label)) for r in ts.itertuples()]
    ts["rag"], ts["rag_why"] = [x[0] for x in res], [x[1] for x in res]
    mg = [margins(r, groups.get(r.label)) for r in ts.itertuples()]
    ts["npg"], ts["ch"] = [m[0] for m in mg], [m[1] for m in mg]
    ts["c"] = ts.collection.str[0]
    rng = np.random.default_rng(26)
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    for c in "ABCD":
        d = ts[ts.c == c]
        if not len(d):
            continue
        jx = np.exp(rng.uniform(0, 0.12, len(d)))            # spread ties right/up only, never below a floor
        jy = np.exp(rng.uniform(0, 0.12, len(d)))
        prim = d.primary.astype(str).eq("True").to_numpy()
        for mask, mk in ((prim, "o"), (~prim, "^")):
            ax.scatter(d.npg.to_numpy(float)[mask] * jx[mask], d.ch.to_numpy(float)[mask] * jy[mask], s=26, marker=mk,
                       color=CC[c], edgecolor="white", lw=0.6, zorder=3)
        amb = d[d.rag == "amber"]
        ax.scatter(amb.npg.astype(float), amb.ch.astype(float), s=110, facecolor="none", edgecolor=WARN, lw=1.4,
                   zorder=4)
    for v in (3,):
        ax.axvline(v, color=CRIT, lw=1, ls=(0, (3, 2)))
        ax.axhline(v, color=CRIT, lw=1, ls=(0, (3, 2)))
    ax.text(3.05, ts.ch.max() * 1.25, "3 per side (D26, D28)", fontsize=6.2, color=CRIT, va="top")
    ax.text(ts.npg.max() * 1.1, 3.1, "3 changed (D29)", fontsize=6.2, color=CRIT, ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(2.4, ts.npg.max() * 1.4); ax.set_ylim(2.4, ts.ch.max() * 1.5)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xticks([t for t in (3, 5, 10, 20, 50, 100) if t <= ts.npg.max() * 1.4])
    ax.set_yticks([t for t in (3, 5, 10, 20, 50, 100) if t <= ts.ch.max() * 1.5])
    ax.set_xlabel("samples per side (smaller arm; patients for A)", fontsize=7)
    ax.set_ylabel("significantly changed metabolites\nin the network", fontsize=7)
    ax.tick_params(labelsize=6.5)
    h = [plt.Line2D([], [], ls="", marker="o", color=CC[c], ms=5, label=B.COLL[c].split(" — ")[1])
         for c in "ACD" if (ts.c == c).any()]
    h += [plt.Line2D([], [], ls="", marker="o", color=INK2, ms=5, label="gene loss"),
          plt.Line2D([], [], ls="", marker="^", color=INK2, ms=5, label="inhibitor or gain"),
          plt.Line2D([], [], ls="", marker="o", mfc="none", mec=WARN, mew=1.4, ms=9,
                     label="caveat: controls are\nother patients")]
    ax.legend(handles=h, fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    at3 = int((ts.npg == 3).sum())
    low = int((ts.ch < 6).sum())
    cnt = ts.rag.value_counts()
    title = (f"All {len(ts)} testable comparisons clear both floors;\n{at3} sit at exactly 3 per side, "
             f"{low} have fewer than 6 changed metabolites")
    fig.suptitle(title, x=0.012, ha="left", fontsize=9, weight="bold")
    fig.text(0.012, 0.005, "Dashed lines are the only thresholds: >= 3 samples per side (D26, every unit by D28) and >= 3 "
             "significantly changed metabolites\nin the network (D29). Points are nudged up/right so ties stay visible. "
             f"{cnt.get('amber', 0)} documented caveat(s), {cnt.get('red', 0)} below a floor.",
             fontsize=5.8, color=INK2)
    fig.subplots_adjust(left=0.1, right=0.72, top=0.84, bottom=0.22)
    save(fig, out, "inclusion_margin", figs, title,
         "One dot per testable comparison: how far it sits above each inclusion floor. Dots near the dashed lines are "
         "the thinnest evidence in the set; the ringed ones carry a documented caveat.")
    return dict(zip(ts.label, zip(ts.rag, ts.rag_why))), {
        "testable": len(ts), "pass": int(cnt.get("green", 0)), "caveat": int(cnt.get("amber", 0)),
        "fail": int(cnt.get("red", 0)),
        "caveats": [f"{r.label}: {r.rag_why}" for r in ts[ts.rag == "amber"].itertuples()],
        "fails": [f"{r.label}: {r.rag_why}" for r in ts[ts.rag == "red"].itertuples()]}


# ---- DAA recheck -------------------------------------------------------------------
TOL = 1e-9


def recheck_unit(r, g):
    """Every metabolite of one comparison, re-tested one row at a time (a loop of scipy Welch tests, not the DAA
    scripts' code) and reconciled with the DAA output: same rows, same p, same log2FC."""
    if g["kind"] == "workbench":
        import workbench_daa as W
        raw = g["raw"]
        body = raw[~raw.iloc[:, 0].astype(str).str.startswith("#")]
        names = body.iloc[:, 0].astype(str).map(W.clean_name)
        vals = body.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
        named = ~names.str.fullmatch(r"(?i)istd|internal standard.*|")
        key = names.str.lower()
        daa = pd.read_csv(ROOT / "results" / "workbench_daa" / f"{r.study}_{r.sub_unit}.csv")
    else:
        raw = g["raw"]
        body = raw.iloc[1:]
        names = body["metabolite"].astype(str)
        vals = body.iloc[:, 2:].apply(pd.to_numeric, errors="coerce")
        named = pd.Series(True, index=body.index)
        key = names
        daa = pd.read_csv(ROOT / "results" / "metabolights_daa" / f"{r.study}_{r.sub_unit}.csv")
    vals = vals.where(vals > 0)
    med = vals.median(axis=1).fillna(-np.inf)
    here, n_short, n_dup, tied = {}, 0, 0, set()
    for k in key[named].unique():
        idx = key.index[(key == k) & named]
        n_dup += len(idx) - 1
        if len(idx) > 1 and (med.loc[idx] == med.loc[idx].max()).sum() > 1:
            tied.add(names.loc[idx[0]])                     # the duplicate rule has no unique answer
        i = med.loc[idx].idxmax()                            # a name listed twice: the higher-median row
        x = np.log2(vals.loc[i, g["case"]].dropna().astype(float))
        y = np.log2(vals.loc[i, g["control"]].dropna().astype(float))
        if len(x) < 2 or len(y) < 2:
            n_short += 1
            continue
        here[names.loc[i]] = (x.mean() - y.mean(), stats.ttest_ind(x, y, equal_var=False).pvalue)
    d = daa.set_index("name")
    both = [n for n in here if n in d.index]
    dfc = np.array([abs(here[n][0] - d.at[n, "log2FC"]) for n in both])
    p_h = np.array([here[n][1] for n in both]); p_d = d.loc[both, "p"].to_numpy(float)
    same_p = np.isclose(p_h, p_d, rtol=TOL, atol=0) | (np.isnan(p_h) & np.isnan(p_d))
    same_fc = dfc <= TOL * np.maximum(1, np.abs(d.loc[both, "log2FC"].to_numpy(float)))
    ok = same_p & same_fc
    differ = [{"name": n, "p_here": here[n][1], "p_daa": float(d.at[n, "p"]), "tied_duplicates": n in tied}
              for n, o in zip(both, ok) if not o]
    only_here = [n for n in here if n not in d.index and np.isfinite(here[n][1])]
    only_daa = [n for n in d.index if n not in here]
    return {"n_table": int(named.sum()), "n_dup_merged": n_dup, "n_too_few": n_short, "n_daa": len(d),
            "n_checked": len(both), "n_match": int(ok.sum()), "differ": differ, "ok": ok.tolist(),
            "only_here": only_here, "only_daa": only_daa,
            "p_here": p_h.tolist(), "p_daa": p_d.tolist(),
            "fc_here": [here[n][0] for n in both], "fc_daa": d.loc[both, "log2FC"].astype(float).tolist(),
            "max_rel_p_diff": float(np.nanmax(np.abs(p_h - p_d) / np.maximum(p_d, 1e-300))) if len(both) else 0.0,
            "max_fc_diff": float(dfc.max()) if len(dfc) else 0.0}


def daa_recheck(u, groups, out, figs):
    ts = u[(u.testable == "True") & u.collection.str[0].isin(["C", "D"])].drop_duplicates("label")
    res = {r.label: recheck_unit(r, groups[r.label]) for r in ts.itertuples() if groups.get(r.label)}
    n_units = len(res)
    n_rows = sum(v["n_checked"] for v in res.values())
    n_ok = sum(v["n_match"] for v in res.values())
    n_miss = sum(len(v["only_here"]) + len(v["only_daa"]) for v in res.values())

    fig, axs = plt.subplots(1, 2, figsize=(7.4, 4.0))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.37, wspace=0.3)
    ph = np.concatenate([v["p_here"] for v in res.values()]); pd_ = np.concatenate([v["p_daa"] for v in res.values()])
    fh = np.concatenate([v["fc_here"] for v in res.values()]); fd = np.concatenate([v["fc_daa"] for v in res.values()])
    okm = np.concatenate([v["ok"] for v in res.values()]).astype(bool)
    for ax, xh, xd, lab in ((axs[0], -np.log10(np.clip(pd_, 1e-300, 1)), -np.log10(np.clip(ph, 1e-300, 1)),
                             "−log10 p"), (axs[1], fd, fh, "log2 fold change")):
        lim = (np.nanmin([xh.min(), xd.min()]), np.nanmax([xh.max(), xd.max()]))
        ax.plot(lim, lim, color=RULE, lw=3, zorder=1)
        ax.scatter(xh[okm], xd[okm], s=3, color="#126b63", lw=0, alpha=0.5, zorder=2, rasterized=True)
        ax.scatter(xh[~okm], xd[~okm], s=26, facecolor="none", edgecolor="#b3261e", lw=1.2, zorder=3)
        ax.set_xlabel(f"{lab}, pipeline (DAA output file)", fontsize=7)
        ax.set_ylabel(f"{lab}, recomputed", fontsize=7)
        ax.tick_params(labelsize=6)
    axs[0].set_title("a  p-values", loc="left", fontsize=8)
    axs[1].set_title("b  fold changes", loc="left", fontsize=8)
    dif = [(k, x) for k, v in res.items() for x in v["differ"]]
    nt = sum(x["tied_duplicates"] for _, x in dif)
    title = (f"Every metabolite re-tested: {n_ok:,} of {n_rows:,} tests in {n_units} comparisons match the pipeline"
             + (f",\n{len(dif)} differ" + (" only where two duplicate rows tie" if nt == len(dif) else "") if dif else "")
             + (f"; {n_miss} rows missing on one side" if n_miss else "; none missing"))
    fig.suptitle(title, x=0.012, y=0.985, ha="left", va="top", fontsize=9, weight="bold")
    foot = ("Each dot is one metabolite in one comparison; on the grey line = identical to 1e-9; red rings = differ. "
            "Recomputed row by row from the per-sample tables\n(Workbench C, MetaboLights D) with the DAA's rules: "
            "log2 of positive values, >= 2 per group, Welch t-test, duplicate names keep the higher-median row.\n"
            "The case/control split is the pipeline's own (checked by eye in contrast_diagrams). "
            "Not covered: inborn errors (A) and B, which start from published z-scores or tables.")
    if dif:
        foot += "\nDiffering (the pipeline's pick between tied rows is arbitrary): " + ";\n".join(f"{x['name']} in {k}: p {x['p_daa']:.2g} (pipeline) vs {x['p_here']:.2g} "
                                           "(the other tied row)" for k, x in dif) + "."
    fig.text(0.012, 0.02, foot, fontsize=6.1, color=INK2, va="bottom")
    save(fig, out, "daa_recheck", figs, title.replace("\n", " "),
         "Each dot is a metabolite: if every one sits on the diagonal, the p-values and fold changes the rankings start "
         "from are exactly what the sample values give; red rings and rows missing on either side are counted in the "
         "title.")
    return {k: {f: v[f] for f in ("n_table", "n_dup_merged", "n_too_few", "n_daa", "n_checked", "n_match", "differ",
                                  "only_here", "only_daa", "max_rel_p_diff", "max_fc_diff")} for k, v in res.items()}


# ---- cards -------------------------------------------------------------------------
def candidate_cards(c, s, k, n=40):
    scr = {(r.id, r.gene): r for r in s.itertuples()}
    cur = {r.id: r for r in k.itertuples()}
    rep = c[c.source != "literature"].copy()
    rep["score_n"] = pd.to_numeric(rep.score, errors="coerce")
    top = rep.sort_values(["score_n", "id"], ascending=[False, True]).head(n)
    lit = c[c.source == "literature"].copy()
    lit["score_n"] = pd.to_numeric(lit.score, errors="coerce")
    top = pd.concat([top, lit.sort_values("score_n", ascending=False).head(10)])
    out = []
    for r in top.itertuples():
        if r.source == "workbench":
            urls = [(f"{MW}/{r.id}/factors", "json"), (f"{MW}/{r.id}/mwtab/txt", "txt")]
        elif r.source == "metabolights":
            urls = [(f"{ML}/{r.id}/{p}", "json") for p in ("organisms", "title", "description", "factors")]
        else:
            urls = []
        receipts = []
        for url, kind in urls:
            f = cached(url, kind)
            receipts.append({"url": url, "cache_file": str(f.relative_to(ROOT)) if f.exists() else "",
                             "fetched": datetime.fromtimestamp(f.stat().st_mtime).isoformat(timespec="minutes")
                             if f.exists() else "", "bytes": f.stat().st_size if f.exists() else 0})
        words = [w.strip() for w in r.evidence.split(",") if w.strip() and w.strip() != "sample factor"]
        sv = scr.get((r.id, r.gene))
        cv = cur.get(r.id)
        out.append({"id": r.id, "gene": r.gene, "source": r.source, "score": r.score, "title": r.title, "url": r.url,
                    "snippet": r.snippet, "gene_highlight": r.gene, "perturbation_words": words,
                    "perturbation": r.perturbation, "in_network": r.in_network == "True", "design_flags": r.design_flags,
                    "in_registry": r.in_registry, "factors": r.factors[:400],
                    "screen": {"verdict": sv.verdict, "case": sv.case_groups, "control": sv.control_groups,
                               "metabolites": sv.best_assay_named_metabolites, "reason": sv.reason or sv.contrast_note}
                    if sv else None,
                    "curated": {"verdict": cv.verdict, "reason": cv.reason, "case": cv.case_levels,
                                "control": cv.control_levels} if cv else None,
                    "cache_receipts": receipts})
    return out


def unit_cards(u, groups, flags, D, audit):
    added = d26_added()
    reg = pd.read_csv(ROOT / "benchmarks" / "data_registry.tsv", sep="\t", dtype=str).fillna("")
    disorder = pd.read_csv(ROOT / "benchmarks" / "cases.tsv", sep="\t", dtype=str).set_index("case_id").disorder
    cards = {}
    dup = set(u.label[u.label.duplicated()])
    for r in u.itertuples():
        g = groups.get(r.label) if r.label not in dup else None
        ds = []
        d, rule = FATE_RULE.get(r.fate, ("", ""))
        if d:
            ds.append(d)
        if r.study in added:
            ds.append("D26")
        if r.study == "MTBLS3628":
            ds.append("D23")
        if r.collection.startswith("A"):
            ds += ["D12", "D21"]
        if r.collection[0] in "CD":
            ds.append("D22")
        ds = sorted(set(ds), key=lambda x: int(x[1:]))
        rg = reg[(reg.study == r.study) & (reg.sub_unit == r.sub_unit)]
        if r.collection.startswith("A"):
            contrast = f"{disorder.get(r.label, r.label)} patients vs reference population (group-mean z)"
        elif g:
            cs = sorted({x["group"] for x in g["groups"] if x["role"] == "case"})
            ct = sorted({x["group"] for x in g["groups"] if x["role"] == "control"})
            contrast = f"{' + '.join(cs)}  vs  {' + '.join(ct)}"
        else:
            contrast = r.sub_unit
        # unprocessed registry rows share placeholder labels such as "(study)"; key them by study
        key = r.label if r.label not in dup else f"{r.study} {r.label}"
        cards[key] = {
            "label": r.label, "collection": r.collection, "study": r.study, "sub_unit": r.sub_unit,
            "name": disorder.get(r.label, "") if r.collection.startswith("A") else r.sample_detail,
            "url": (f"https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?StudyID={r.study}"
                    if r.study.startswith("ST") else f"https://www.ebi.ac.uk/metabolights/{r.study}"
                    if r.study.startswith("MTBLS") else ""),
            "contrast": contrast, "contrast_reason": g["reason"] if g else "",
            "n_case": len(g["case"]) if g else None, "n_control": len(g["control"]) if g else None,
            "n_samples": r.n_samples, "groups": g["groups"] if g else None,
            "causal_genes": r.causal_genes, "genes_in_network": r.genes_in_net, "perturbation": r.perturbation,
            "primary": r.primary == "True", "sample_matrix": r.sample_matrix, "data_format": r.data_format,
            "metabolites_measured": r.n_metabolites, "metabolites_in_network": r.n_in_network,
            "significant_decreased": r.n_dec, "significant_increased": r.n_inc,
            "sources_down": r.n_sources, "sources_up": r.n_sources_up,
            "viable_down": r.viable_down == "True", "viable_up": r.viable_up == "True",
            "cluster": r.cluster, "testable": r.testable == "True",
            "fate": r.fate, "fate_label": FL.get(r.fate, r.fate), "rule": rule,
            "registry_status": r.reg_status, "drop_reason": r.drop_reason,
            "registry_row": rg.iloc[0].to_dict() if len(rg) else None,
            "quality": {"flag": flags[r.label][0], "why": flags[r.label][1]} if r.label in flags else None,
            "decisions": [{"id": x, "title": D[x][0], "url": D[x][1]} for x in ds if x in D],
            "audit": audit.get(r.label)}
    return cards


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(ROOT / "results" / "transparency" / "data"))
    p.add_argument("--seed", type=int, default=26)
    args = p.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    B.style()
    sys.path.insert(0, str(ROOT / "benchmarks"))
    c, s, k, r, u = load()
    D = decisions()
    figs = []
    rec = recall_scoreboard(c, r, out, figs)
    source_flow(c, s, k, out, figs)
    ov = overlap(c, r, out, figs)
    ledger, failed = failure_ledger(out, figs)
    drops = drop_funnel(u, D, out, figs)
    groups = {row.label: contrast_of(row) for row in u.itertuples()}
    crow = contrast_table(u, groups, out, figs)
    contrast_diagrams(u, groups, crow, out, figs)
    flags, incl = inclusion_margin(u, groups, out, figs)
    (out / "contrasts.json").write_text(json.dumps(crow, indent=1, ensure_ascii=False), encoding="utf-8")
    audit = daa_recheck(u, groups, out, figs)
    cards = unit_cards(u, groups, flags, D, audit)
    (out / "cards.json").write_text(json.dumps(cards, indent=1, default=str), encoding="utf-8")
    (out / "candidates.json").write_text(json.dumps(candidate_cards(c, s, k), indent=1, default=str), encoding="utf-8")
    (out / "figures.json").write_text(json.dumps(figs, indent=1), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps({"recall": rec, "overlap": ov, "failure_ledger": ledger,
                                                  "failed_fetches": failed, "drops": drops,
                                                  "daa_recheck": {k: v for k, v in audit.items()},
                                                  "inclusion_check": incl},
                                                 indent=1, default=str), encoding="utf-8")
    for f in figs:
        print(f"{f['file']}: {f['title']}")


if __name__ == "__main__":
    main()
