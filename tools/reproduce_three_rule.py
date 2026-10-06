"""Rebuild everything for 3-rule pruning, in order, from the inputs in data/.

    python tools/reproduce_three_rule.py
    python tools/reproduce_three_rule.py --skip-benchmark      # network, workbook and atlas only

Run from the repository (or package code) folder. Steps:
  1. carbon pairs: KEGG pairs + cleaned atom maps   -> data/atom_pairs_union.json
  2. structure of the nine compared networks         -> results/structure_three_rule/
  3. linked diagrams (Cytoscape .cyjs)               -> results/three_rule_diagrams/
  4. benchmark, 20 inborn errors, every network      -> results/benchmark_lookup/three_rule_<network>/
  5. the final network as tables + manifest          -> results/three_rule/network/
  6. spreadsheet: before vs after, every step        -> results/three_rule/network_before_after.xlsx
  7. interactive atlas                               -> docs/three_rule_atlas.html
  8. tests

Inputs (data/): Human-GEM.xml, Human-GEM.yml, Human-GEM_metabolites.tsv (downloaded from the pinned
Human-GEM release if missing), atom_pairs_raw.json (RXNMapper output; recomputed if missing, about
10 min, needs requirements-atoms.txt), kegg_rclass_pairs.json (KEGG REST cache; refetched if missing,
which needs internet and may differ from the original download), benchmark/ (CTD exports, see
benchmarks/get_benchmark_data.py) and id_translation.csv (the lab's name table).
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ["raw", "carbon_method", "carbon_cleaned", "carbon_carriers", "three_rule", "rules12_keep_rarest",
            "acyl_merged", "acyl_merged_keep_rarest", "first_method"]


def run(step, cmd):
    print(f"\n=== {step}\n    {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    subprocess.run([str(c) for c in cmd], cwd=ROOT, check=True)
    print(f"    done in {time.time() - t0:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id-translation", default=str(ROOT / "data/id_translation.csv"))
    ap.add_argument("--skip-benchmark", action="store_true")
    ap.add_argument("--jobs", type=int, default=3, help="benchmarks run in parallel (default: %(default)s)")
    args = ap.parse_args()
    py = sys.executable
    missing = [f for f in ("Human-GEM.xml", "Human-GEM_metabolites.tsv", "kegg_rclass_pairs.json",
                           "atom_pairs_raw.json") if not (ROOT / "data" / f).exists()]
    if missing:
        print("Not in data/ (will be downloaded or recomputed):", ", ".join(missing))
    if not Path(args.id_translation).exists():
        sys.exit(f"id_translation table not found: {args.id_translation}")

    run("1. carbon pairs", [py, "-m", "metaboenrich.atoms", "--combine", "union", "--out", "data/atom_pairs_union.json"])
    run("2. structure", [py, "-m", "metaboenrich.structure", "--stage-set", "three_rule",
                         "--out", "results/structure_three_rule"])
    run("3. diagrams", [py, "-m", "metaboenrich.stage_diagrams", "--structure",
                        "results/structure_three_rule/structure.json", "--out", "results/three_rule_diagrams"])
    if not args.skip_benchmark:
        print("\n=== 4. benchmark (each network takes 1-3 min)", flush=True)
        pending = list(NETWORKS)
        running = []
        while pending or running:
            while pending and len(running) < args.jobs:
                key = pending.pop(0)
                log = open(ROOT / f"results/benchmark_{key}.log", "w")
                running.append((key, log, subprocess.Popen(
                    [py, "-m", "metaboenrich.benchmark", "--levels", "group", "--stage", f"three_rule:{key}",
                     "--id-translation", args.id_translation, "--out", f"results/benchmark_lookup/three_rule_{key}"],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)))
            time.sleep(2)
            for item in list(running):
                key, log, proc = item
                if proc.poll() is not None:
                    log.close()
                    running.remove(item)
                    if proc.returncode:
                        sys.exit(f"benchmark {key} failed; see results/benchmark_{key}.log")
                    print(f"    {key} done", flush=True)
    run("5. network export", [py, "tools/export_three_rule.py", "--out", "results/three_rule/network"])
    run("6. spreadsheet", [py, "tools/rules_audit.py", "--out", "results/three_rule/network_before_after.xlsx"])
    if not args.skip_benchmark:
        run("7. atlas", [py, "tools/atlas/build_three_rule_atlas.py", "--id-translation", args.id_translation,
                         "--out", "docs/three_rule_atlas.html"])
    run("8. tests", [py, "-m", "unittest", "discover", "-s", "tests"])
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
