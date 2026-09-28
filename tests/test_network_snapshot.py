"""The committed network/ must match what the code builds from Human-GEM.

If this fails after an intentional change to currency handling or reaction
edits, regenerate it with `python -m metaboenrich.export_network --out network`
and commit the result. Skipped when the model has not been downloaded.
"""
import unittest
from pathlib import Path

import pandas as pd

from metaboenrich.gem import load_sbml
from metaboenrich.network import DEFAULT_REACTION_EDITS, build_graph, load_reaction_edits, resolve_currency

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "Human-GEM.xml"
SNAPSHOT = ROOT / "network" / "edges.tsv"


@unittest.skipUnless(MODEL.exists() and SNAPSHOT.exists(), "Human-GEM model or network snapshot not present")
class NetworkSnapshotTest(unittest.TestCase):
    def test_edges_match_committed_network(self):
        model = load_sbml(MODEL)
        rules, _ = resolve_currency(model, "role")
        rules.edits, _ = load_reaction_edits(DEFAULT_REACTION_EDITS, model)
        g = build_graph(model, rules)
        built = {(g.node_ids[m], g.node_ids[r], side) for (m, r), side in zip(g.edges, g.edge_side)}
        snap = pd.read_csv(SNAPSHOT, sep="\t")
        committed = set(zip(snap.metabolite, snap.reaction, snap.side))
        self.assertEqual(len(built - committed), 0, f"edges not in snapshot: {sorted(built - committed)[:5]}")
        self.assertEqual(len(committed - built), 0, f"snapshot edges not built: {sorted(committed - built)[:5]}")


if __name__ == "__main__":
    unittest.main()
