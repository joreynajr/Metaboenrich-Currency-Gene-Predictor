"""Do the headline comparisons depend on the inclusion threshold? (D29)

Re-scores the main comparisons on every unit that was testable before D28 (the archived 95-unit run),
minus the D28 sample-size drops, under metabolite floors 0-100 and under the D29 signal rule (>= 3
significantly changed metabolites in the network). Reports every threshold; it does not pick one.

    python tools/report/rule_sensitivity.py [--archive results/_sensitivity_95_units_2026-11-24]

Writes results/rule_sensitivity.md.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]


def wins(d, a, b):
    """Cluster sign test: clusters where a ranks the causal gene higher than b, and the reverse."""
    g = d.groupby("cluster")[[a, b]].median()
    w, l = int((g[a] < g[b]).sum()), int((g[a] > g[b]).sum())
    return f"{w}–{l} (p = {binomtest(w, w + l).pvalue:.2g})" if w + l else "–"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--archive", default=str(ROOT / "results/_sensitivity_95_units_2026-11-24"))
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    arc = Path(args.archive)
    u = pd.read_csv(arc / "reafect_bench/units/per_unit.tsv", sep="\t")
    t = pd.read_csv(arc / "data_figures/units.tsv", sep="\t", dtype=str).drop_duplicates("label").set_index("label")
    u["n_met"] = u.label.map(t.n_metabolites.astype(float))
    u["changed"] = u.label.map(t.n_inc.astype(int) + t.n_dec.astype(int))
    reg = pd.read_csv(ROOT / "benchmarks/data_registry.tsv", sep="\t", dtype=str).fillna("")
    small = reg[reg.drop_reason.str.contains("minimum of 3 per side")]
    keys = ["collection", "study", "sub_unit"]
    lab = t.reset_index().merge(small[keys], on=keys).label
    u = u[~u.label.isin(set(lab))]
    u["primary"] = u.primary.astype(bool)
    rules = [(f">= {f} metabolites", u.n_met >= f) for f in (0, 10, 15, 20, 30, 50, 100)]
    rules.append((">= 3 changed in the network (D29)", u.changed >= 3))
    rows = ["| rule | units | clusters | walk ↑src vs Reafect, all | gene loss | inhibitor / gain | "
            "walk ↑src vs one-hop, gene loss (own list) |", "|---|---|---|---|---|---|---|"]
    for name, m in rules:
        d = u[m]
        rows.append(f"| {name} | {len(d)} | {d.cluster.nunique()} | "
                    f"{wins(d, 'walk_up_common_pct', 'reafect_common_pct')} | "
                    f"{wins(d[d.primary], 'walk_up_common_pct', 'reafect_common_pct')} | "
                    f"{wins(d[~d.primary], 'walk_up_common_pct', 'reafect_common_pct')} | "
                    f"{wins(d[d.primary], 'walk_up_own_pct', 'neighbour_own_pct')} |")
    txt = ["# Inclusion threshold sensitivity (D29)", "",
           f"Units testable before D28 ({arc.name}), minus the D28 sample-size drops: {len(u)} units. Cells are "
           "cluster wins first method – second method (two-sided sign test on cluster medians, ties dropped); walk "
           "vs Reafect on the shared gene list. Percentiles come from the 95-unit run, so the shared-list universe "
           "is that run's; the current run's numbers differ slightly.", ""] + rows + [""]
    (ROOT / "results/rule_sensitivity.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
