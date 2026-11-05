"""Evaluate the current algorithm on the lab's published datasets.

Each analysis unit of a previous run (default predictions/published_v1) is
rebuilt from its daa_mapping.tsv: the same mapped metabolites, log2FC, p and
significance rule. Every unit is then run with current flow, the walk with
sources = up and the walk with sources = down (D21), and the neighbour
baseline, exactly as in the inborn-error benchmark (metaboenrich.benchmark):
the causal gene's rank among all genes on network reactions, raw scoring, an
unscored gene given the mean of the remaining ranks.

Causal genes: the master table's Gene column, completed and corrected by
benchmarks/published_answer_key.tsv (units whose disorder is named but whose
gene column is empty). Units from a single person (Alaimo 2020) are left out
by default (D21); --include-single-subject adds them, labelled.

    python benchmarks/published_eval.py --out results/published_eval
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metaboenrich.benchmark import Ranker, best_rank                                           # noqa: E402
from metaboenrich.gem import load_sbml                                                         # noqa: E402
from metaboenrich.network import DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits, resolve_currency  # noqa: E402
from published_predictions import rule_settings                                                # noqa: E402

KEY = Path(__file__).with_name("published_answer_key.tsv")
ALIASES = {"MUT": "MMUT"}                       # old symbols in the master table -> Human-GEM symbols
METHODS = [("flow", "flow", "decreased"), ("walk_up", "walk", "increased"), ("walk_down", "walk", "decreased")]


def causal_genes(units, key):
    """unit -> (symbols, kind): the answer key where it has the unit, else the data's Gene column."""
    k = key.set_index("unit")
    out = {}
    for r in units.itertuples(index=False):
        if r.unit in k.index:
            genes, kind = k.at[r.unit, "causal_genes"], k.at[r.unit, "kind"]
        else:
            genes, kind = r.diagnosed_gene_in_data, "disorder"
        syms = [ALIASES.get(g.strip(), g.strip()) for g in str(genes).split(";") if g.strip()]
        out[r.unit] = (syms, kind if syms else "")
    return out


def unit_daa(path):
    """A unit's saved mapping as a DAA table the Ranker reads (met_id in the name column)."""
    m = pd.read_csv(path, sep="\t")
    return pd.DataFrame({"name": m.met_id, "HMDB": "", "KEGG": "", "log2FC": m.log2fc, "p": m.p}), len(m)


def fmt(x):
    return "-" if x is None or pd.isna(x) else f"{x:g}"


def summarise(df, label):
    lines = [f"### {label}", "", "| method | scored | median rank | top 10 | top 100 | not scored |", "|---|---|---|---|---|---|"]
    for name, _, _ in METHODS + [("neighbour", None, None)]:
        r = pd.to_numeric(df[f"rank_{name}"], errors="coerce")
        s = r.dropna()
        lines.append(f"| {name} | {len(s)} | {fmt(s.median()) if len(s) else '-'} | {(s <= 10).sum()} | "
                     f"{(s <= 100).sum()} | {r.isna().sum()} |")
    s = pd.to_numeric(df.loc[df.rank_flow.notna(), "rank_neighbour"], errors="coerce")
    lines.append(f"| neighbour, units flow scores | {len(s)} | {fmt(s.median()) if len(s) else '-'} | "
                 f"{(s <= 10).sum()} | {(s <= 100).sum()} | - |")
    return lines + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="predictions/published_v1", help="run whose units are rebuilt (default: %(default)s)")
    ap.add_argument("--key", default=str(KEY))
    ap.add_argument("--model", default="data/Human-GEM.xml")
    ap.add_argument("--out", default="results/published_eval")
    ap.add_argument("--include-single-subject", action="store_true",
                    help="also run the per-patient units (one person's z-scores; not a differential analysis, D21)")
    args = ap.parse_args()
    t0 = time.time()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_daa_tmp.csv"

    units = pd.read_csv(run / "units.tsv", sep="\t", dtype={"diagnosed_gene_in_data": str, "Subject_ID": str}).fillna("")
    genes_of = causal_genes(units, pd.read_csv(args.key, sep="\t").fillna(""))
    model = load_sbml(args.model)
    rules, _ = resolve_currency(model, "role")
    rules.edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, model)
    graph = build_graph(model, rules)
    rankers = {name: Ranker(model, graph, method, sources=src) for name, method, src in METHODS}
    U = len(rankers["flow"].universe)
    print(f"Network: {graph.n} nodes; {U} genes on network reactions (the ranking universe)")

    rows = []
    for u in units.itertuples(index=False):
        level = "single subject" if u.Subject_ID else "group"
        syms, kind = genes_of[u.unit]
        row = {"unit": u.unit, "source": u.source, "level": level, "rule": u.rule, "kind": kind,
               "causal_genes": ";".join(syms), "universe": U}
        if u.status != "ok":
            rows.append({**row, "status": u.status})
            continue
        if level != "group" and not args.include_single_subject:
            rows.append({**row, "status": "left out: single subject (D21)"})
            continue
        in_model, in_net = rankers["flow"].causal(syms)
        row.update(status="ok", causal_in_network=";".join(model.genes[e] for e in in_net))
        daa, n_mapped = unit_daa(run / u.unit / "daa_mapping.tsv")
        alpha, min_fc = rule_settings(u.rule)
        for name, method, src in METHODS:
            info, ranks, nb, _ = rankers[name].run(daa, tmp, alpha, min_fc)
            if info["mapped"] != n_mapped:
                raise SystemExit(f"{u.unit}: {info['mapped']} mapped metabolites, the saved mapping has {n_mapped}")
            row[f"rank_{name}"] = best_rank(ranks["raw"], in_net)
            row[f"pairs_{name}"] = info["pairs"]
            if name == "flow":
                row.update(mapped=info["mapped"], sources_down=info["sources"], targets_up=info["targets"],
                           one_sided=info["one_sided"], rank_neighbour=best_rank(nb, in_net))
                top = ranks["raw"].sort_values().head(10).index if ranks["raw"] is not None else []
                row["top_genes_flow"] = ", ".join(model.genes.get(g, g) for g in top)
            if method == "walk":
                row[f"walk_success_{name}"] = info.get("walk_success_fraction")
        print(f"{u.unit[:60]:60s} {row['causal_in_network'] or '-':14s} flow={fmt(row['rank_flow']):>6s} "
              f"up={fmt(row['rank_walk_up']):>6s} down={fmt(row['rank_walk_down']):>6s} nb={fmt(row['rank_neighbour']):>6s}")
        rows.append(row)
    tmp.unlink(missing_ok=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "units.tsv", sep="\t", index=False)
    ok = df[df.status == "ok"]
    gene_ok = ok[ok.causal_in_network.fillna("") != ""]
    lines = [f"# Published datasets: evaluation of the current algorithm", "",
             f"Units rebuilt from `{run.as_posix()}/*/daa_mapping.tsv`; causal genes from the master table's Gene column "
             f"and `{Path(args.key).name}`. Rank of the causal gene among {U} genes on network reactions, raw scoring; "
             "a gene the method does not score gets the mean of the remaining ranks. flow = current flow (identical under "
             "either orientation, D21); walk_up / walk_down = absorbing walk with sources = up / down.", "",
             f"Units: {len(df)} in the run; {(df.status == 'ok').sum()} evaluated; "
             f"{df.status.str.startswith('skipped').sum()} had fewer than 3 metabolites; "
             f"{(df.status == 'left out: single subject (D21)').sum()} single-subject units left out (D21). "
             f"Of the evaluated units, {len(gene_ok)} have a causal gene in the network.", ""]
    lines += summarise(gene_ok[gene_ok.kind == "disorder"], "One disorder vs controls")
    if (gene_ok.kind == "contrast").any():
        lines += summarise(gene_ok[gene_ok.kind == "contrast"], "Two-disorder contrasts (best rank over both disorders' genes)")
    lines += ["### Per unit", "", "| unit | causal gene | down / up | flow | walk up | walk down | neighbour |", "|---|---|---|---|---|---|---|"]
    for r in ok.itertuples(index=False):
        lines.append(f"| {r.unit[:70]} | {r.causal_in_network or '(' + r.causal_genes + ', not in network)'} | "
                     f"{r.sources_down} / {r.targets_up}{' one-sided' if r.one_sided else ''} | {fmt(r.rank_flow)} | "
                     f"{fmt(r.rank_walk_up)} | {fmt(r.rank_walk_down)} | {fmt(r.rank_neighbour)} |")
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n{len(ok)} units evaluated in {time.time() - t0:.0f}s -> {out}/")


if __name__ == "__main__":
    main()
