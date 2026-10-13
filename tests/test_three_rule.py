"""Keep-rarest safeguard (metaboenrich.atoms.carbon_method_pairs, keep_rarest).

The model-based tests need data/Human-GEM.xml and data/atom_pairs_union.json
(python -m metaboenrich.atoms) and are skipped without them.
"""
import hashlib
import json
import random
import unittest
from pathlib import Path
from types import SimpleNamespace

from metaboenrich.atoms import carbon_method_pairs

ROOT = Path(__file__).resolve().parents[1]
HAVE_DATA = (ROOT / "data/Human-GEM.xml").exists() and (ROOT / "data/atom_pairs_union.json").exists()


def fake_model(formulas):
    return SimpleNamespace(species={m: SimpleNamespace(met_id=m, formula=f) for m, f in formulas.items()})


class TieBreakTest(unittest.TestCase):
    def test_tie_break_stable(self):
        # R3 has two couples seen in the same number of reactions; the choice
        # must not depend on the order pairs or reactions arrive in.
        model = fake_model({x: "C2H4O2" for x in "abcdwxyz"})
        pairs = {"R1": [("a", "b")], "R2": [("a", "b")], "R4": [("c", "d")], "R5": [("c", "d")],
                 "R3": [("c", "d"), ("a", "b")]}
        choices = set()
        for seed in range(20):
            rng = random.Random(seed)
            items = list(pairs.items())
            rng.shuffle(items)
            shuffled = {k: rng.sample(v, len(v)) for k, v in items}
            out, _ = carbon_method_pairs(fake_model({x: "C2H4O2" for x in "abcdwxyz"}), shuffled,
                                         couple_min_reactions=2, keep_rarest=True)
            choices.add(tuple(out["R3"]))
        self.assertEqual(choices, {(("a", "b"),)})

    def test_rarest_pair_kept_and_logged(self):
        model = fake_model({x: "C2H4O2" for x in "abcd"})
        pairs = {f"A{i}": [("a", "b")] for i in range(5)}
        pairs.update({f"C{i}": [("c", "d")] for i in range(3)})
        pairs["R"] = [("a", "b"), ("c", "d")]
        log = []
        out, _ = carbon_method_pairs(model, pairs, couple_min_reactions=3, keep_rarest=True, log=log)
        self.assertEqual(out["R"], [("c", "d")])
        self.assertEqual([x["reaction"] for x in log if x["reaction"] == "R"], ["R"])

    def test_no_rescue_respected(self):
        model = fake_model({x: "C2H4O2" for x in "ab"})
        pairs = {f"A{i}": [("a", "b")] for i in range(3)}
        out, _ = carbon_method_pairs(model, pairs, couple_min_reactions=3, keep_rarest=True, no_rescue={"A0"})
        self.assertEqual(out["A0"], [])
        self.assertEqual(out["A1"], [("a", "b")])


@unittest.skipUnless(HAVE_DATA, "needs data/Human-GEM.xml and data/atom_pairs_union.json")
class HumanGemThreeRuleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from metaboenrich import structure
        from metaboenrich.gem import load_sbml
        from metaboenrich.network import build_graph
        cls.model = load_sbml(str(ROOT / "data/Human-GEM.xml"))
        cls.symbol_of = cls.model.genes
        cls.stages = staticmethod(structure.stages)
        cls.build = staticmethod(build_graph)
        st = {s[0]: s for s in structure.stages(cls.model, "three_rule", atom_dir=str(ROOT / "data"))}
        s = st["rules12_keep_rarest"]
        cls.g = build_graph(cls.model, s[3], s[4], s[5], **s[6])
        cls.names = {sp.met_id: sp.name for sp in cls.model.species.values()}

    def links(self, gene):
        """{reaction node: set of metabolite names} for the gene's nodes."""
        g, out = self.g, {}
        for i in range(g.n):
            if g.is_reaction[i] and any(self.symbol_of.get(x) == gene for x in g.reaction_genes[i]):
                out[g.node_ids[i]] = {g.node_names[a] for a, b in g.edges if b == i}
        return out

    def test_regression_off(self):
        # the six carbon networks are byte-identical to before the change
        ref = json.loads((ROOT / "tests/carbon_stage_hashes.json").read_text())
        for k, _, _, r, ms, do, ex in self.stages(self.model, "carbon", atom_dir=str(ROOT / "data")):
            g = self.build(self.model, r, ms, do, **ex)
            h = hashlib.sha256(json.dumps([g.node_ids, g.edges.tolist(), g.edge_side, sorted(g.unmapped_reactions),
                                           sorted(g.no_pair_reactions)]).encode()).hexdigest()
            self.assertEqual(h, ref[k], k)

    def test_glud1_rescued(self):
        links = self.links("GLUD1")
        self.assertTrue(links)
        for mets in links.values():
            self.assertEqual(mets, {"AKG", "glutamate"})

    def test_got1_unchanged(self):
        # aspartate transaminase keeps aspartate/OAA and still loses AKG/glutamate
        links = self.links("GOT1")
        self.assertIn({"aspartate", "OAA"}, links.values())
        self.assertNotIn({"AKG", "glutamate"}, links.values())

    def test_gls_single_pair(self):
        links = self.links("GLS")
        self.assertTrue(links)
        self.assertIn({"glutamine", "glutamate"}, links.values())

    def test_carriers_found_from_structure(self):
        from metaboenrich import structure
        found = {self.names[c] for c in structure.CARRIERS_FOUND}
        self.assertTrue({"CoA", "L-carnitine", "GSH"} <= found)
        self.assertNotIn("glycine", found)          # 36 of 37 conjugates at one atom: not every time
        self.assertNotIn("acetyl-CoA", found)       # thiolase extends the carbon chain, not a carrier load

    def test_carbon_carbon_chemistry_kept(self):
        from metaboenrich.atoms import find_carriers, load_atom_pairs
        _, remove = find_carriers(self.model, load_atom_pairs(ROOT / "data/atom_pairs_union.json"),
                                  ROOT / "data/Human-GEM_metabolites.tsv")
        ids = {n: m for m, n in self.names.items()}
        for a, b in (("acetyl-CoA", "acetoacetyl-CoA"), ("glycine", "5-aminolevulinate")):
            self.assertNotIn(tuple(sorted((ids[a], ids[b]))), remove, (a, b))
        self.assertIn(tuple(sorted((ids["CoA"], ids["(R)-methylmalonyl-CoA"]))), remove)

    def test_atp_not_a_hub_again(self):
        g = self.g
        atp = g.index[next(m for m, n in self.names.items() if n == "ATP")]
        self.assertLess(int(((g.edges[:, 0] == atp) | (g.edges[:, 1] == atp)).sum()), 60)


if __name__ == "__main__":
    unittest.main()
