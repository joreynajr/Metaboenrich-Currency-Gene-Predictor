"""Are the causal-gene ranks that moved between the correction_eval run and the head-to-head recompute ties?

For every comparison where reafect_bench's recomputed rank differs from correction_eval's, rescore the comparison,
take the gene table and compare the causal gene's score with the scores of the genes it swapped places with (the
rank window between the two runs). A tie means the window's scores agree to TOL relative; then the order inside it
is floating-point noise and either rank is right.

    python tools/report/tie_check.py

Reads results/reafect_bench/units/per_unit.tsv; writes results/transparency/bench/tie_check.tsv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
TOL = 1e-9
METHODS = {"walk_up": ("walk", "increased"), "walk_down": ("walk", "decreased"), "flow": ("flow", "decreased")}


def main():
    import correction_eval as CE
    from metaboenrich import benchmark as B
    from metaboenrich.daa import load_name_bridge
    from null_calibration import Calibrator, registry_units
    from reafect_bench import _a_group, _network, test_units

    u = pd.read_csv(ROOT / "results/reafect_bench/units/per_unit.tsv", sep="\t")
    todo = [(m, r) for m in METHODS for r in u.itertuples()
            if pd.notna(getattr(r, f"{m}_rank_reported")) and getattr(r, f"{m}_rank") != getattr(r, f"{m}_rank_reported")]
    if not todo:
        return
    seen = []                                             # every gene table built, last one wins
    for mod in (CE, B):
        f = mod.gene_table
        mod.gene_table = lambda *a, _f=f: seen.append(_f(*a)) or seen[-1]
    model, graph = _network()
    bridge = load_name_bridge(str(ROOT / "data/id_translation.csv"))
    units_tsv = {r.label: r for r in test_units().itertuples()}
    cases, data, rows = pd.read_csv(B.CASES, sep="\t").fillna("").set_index("case_id"), {}, []
    for m in sorted({m for m, _ in todo}):
        ev = CE.Evaluator(model, graph, bridge, Calibrator(graph, None, "flow"), *METHODS[m])
        reg = {x["label"]: x for x in registry_units(model, graph, bridge, METHODS[m][1])[0]}
        for _, r in [t for t in todo if t[0] == m]:
            seen.clear()
            ev.reference(reg[r.label])
            if units_tsv[r.label].collection == "A_inborn_errors":
                z, ann, samples = _a_group(B, cases, data, units_tsv[r.label].sub_unit)
                ev.ranker.run(B.group_daa(z, ann, samples), ROOT / "results/_tie_daa.csv", 0.05, 1.0)
            g = seen[-1]
            g = g[g.ensembl_id.isin(ev.ranker._universe_set)].reset_index(drop=True)
            hit = g[g.symbol.isin(r.causal_genes.split(";"))]
            pos = int(hit.index.min()) + 1
            lo, hi = sorted((int(getattr(r, f"{m}_rank")), int(getattr(r, f"{m}_rank_reported"))))
            win = g.iloc[lo - 1:hi]
            s0 = float(hit.best_score.iloc[0])
            rel = float((win.best_score - s0).abs().max() / max(abs(s0), 1e-300))
            rows.append({"label": r.label, "method": m, "rank": getattr(r, f"{m}_rank"),
                         "rank_reported": getattr(r, f"{m}_rank_reported"), "rank_now": pos,
                         "window_genes": len(win), "causal_score": s0, "max_rel_score_diff": rel, "tie": rel <= TOL})
            print(rows[-1], flush=True)
    (ROOT / "results/_tie_daa.csv").unlink(missing_ok=True)
    out = ROOT / "results/transparency/bench"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "tie_check.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
