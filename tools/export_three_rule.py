"""Write the 3-rule pruned network as plain tables, with a manifest.

    python tools/export_three_rule.py --out network/three_rule

Writes:
    nodes.tsv      node_id, name, type (metabolite | reaction), Human-GEM reaction, genes
    edges.tsv      metabolite, reaction node, side (substrate | product)
    manifest.json  method and parameters, SHA-256 of every input file, software
                   versions, counts, and a fingerprint of the edge list
A reaction split into carbon links appears as one reaction node per link
("MAR03827#1", "#2"); its genes are on every link.
"""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metaboenrich import structure                                   # noqa: E402
from metaboenrich.atoms import CARRIER_MIN_LOADED, COUPLE_MIN_REACTIONS  # noqa: E402
from metaboenrich.gem import load_sbml                               # noqa: E402
from metaboenrich.network import build_graph                         # noqa: E402

INPUTS = ["Human-GEM.xml", "Human-GEM.yml", "Human-GEM_metabolites.tsv", "atom_pairs_raw.json",
          "kegg_rclass_pairs.json", "atom_pairs_union.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def versions():
    out = {"python": platform.python_version()}
    for mod in ("numpy", "scipy", "pandas", "rdkit", "rxnmapper", "torch", "openpyxl"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data"))
    ap.add_argument("--out", default=str(ROOT / "network/three_rule"))
    args = ap.parse_args()
    data, out = Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    model = load_sbml(str(data / "Human-GEM.xml"))
    stage = {s[0]: s for s in structure.stages(model, "three_rule", atom_dir=str(data))}["three_rule"]
    g = build_graph(model, stage[3], stage[4], stage[5], **stage[6])
    sym = model.genes

    with open(out / "nodes.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("node_id\tname\ttype\thuman_gem_reaction\tgenes\n")
        for i, nid in enumerate(g.node_ids):
            if g.is_reaction[i]:
                f.write(f"{nid.replace('R_', '')}\t{g.node_names[i]}\treaction\t{g.base_reaction[i].replace('R_', '')}\t"
                        f"{';'.join(sym.get(x, x) for x in g.reaction_genes[i])}\n")
            else:
                f.write(f"{nid}\t{g.node_names[i]}\tmetabolite\t\t\n")
    lines = []
    for (a, b), side in zip(g.edges, g.edge_side):
        m_i, r_i = (a, b) if g.is_reaction[b] else (b, a)
        lines.append(f"{g.node_ids[m_i]}\t{g.node_ids[r_i].replace('R_', '')}\t{side}\n")
    with open(out / "edges.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("metabolite\treaction_node\tside\n")
        f.writelines(lines)
    edges_sha = hashlib.sha256("".join(sorted(lines)).encode()).hexdigest()

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = ""
    n_rxn = int(g.is_reaction.sum())
    manifest = {
        "method": "3-rule pruning of Human-GEM",
        "steps": [
            "Merge compartments; drop transport and exchange reactions (one metabolite after merging)",
            "Step 1: drop biomass reactions, reactions with a pool pseudo-metabolite, and transporters "
            "carrying a carbon-containing metabolite across a membrane",
            "Rule 1: split each reaction into substrate->product links that share carbon atoms "
            "(KEGG RCLASS pairs plus cleaned RXNMapper atom maps, union); molecules without carbon link nothing",
            f"Rule 2: remove cofactor couples, pairs found in {COUPLE_MIN_REACTIONS} or more reactions",
            f"Rule 3: remove carrier links; a carrier is found whole inside at least {CARRIER_MIN_LOADED} partners, "
            "always attached at the same atom through a heteroatom",
            "Safeguard: a reaction the rules would leave with no link keeps its rarest link, unless it carries "
            "any metabolite across a membrane",
            "Drop reactions still listing more than 20 metabolites",
        ],
        "parameters": {"couple_min_reactions": COUPLE_MIN_REACTIONS, "carrier_min_loaded": CARRIER_MIN_LOADED,
                       "max_reaction_size": 20, "pair_source": "union"},
        "inputs_sha256": {f: sha256(data / f) for f in INPUTS if (data / f).exists()},
        "software": versions(),
        "code_commit": commit,
        "date": date.today().isoformat(),
        "counts": {"metabolites": int(g.n - n_rxn), "reaction_nodes": n_rxn,
                   "human_gem_reactions": len(set(g.base_reaction.values())), "edges": len(g.edges),
                   "genes": len({x for v in g.reaction_genes.values() for x in v})},
        "carriers": sorted({g_name for g_name in (model.species[s].name for s in model.species
                                                   if model.species[s].met_id in structure.CARRIERS_FOUND)}),
        "edges_sha256": edges_sha,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps(manifest["counts"]), "\nedges_sha256", edges_sha, "\nwrote", out)


if __name__ == "__main__":
    main()
