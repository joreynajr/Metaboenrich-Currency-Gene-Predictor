import unittest

from metaboenrich.atoms import clean_pairs
from metaboenrich.network import build_graph, resolve_currency
from tests.test_currency import toy_model


class CleanPairsTest(unittest.TestCase):
    def test_transaminase_cross_pairs_removed(self):
        # RXNMapper's output for GABA transaminase: the true pairs plus two
        # 1-carbon cross-pairs, which the rule removes.
        raw = [["4-aminobutyrate", "glutamate", 1], ["4-aminobutyrate", "succinate semialdehyde", 3],
               ["AKG", "glutamate", 4], ["AKG", "succinate semialdehyde", 1]]
        kept = {(a, b) for a, b, _ in clean_pairs(raw)}
        self.assertEqual(kept, {("4-aminobutyrate", "succinate semialdehyde"), ("AKG", "glutamate")})

    def test_ties_are_kept(self):
        # Aspartate aminotransferase came back with every competing pair tied at
        # 2 carbons; carbon counts cannot resolve that, so the rule keeps them
        # and KEGG's curated pairs (used where available) do the resolving.
        raw = [["AKG", "glutamate", 3], ["aspartate", "OAA", 2],
               ["AKG", "OAA", 2], ["aspartate", "glutamate", 2]]
        self.assertEqual(len(clean_pairs(raw)), 4)

    def test_minor_but_real_donor_kept(self):
        # OTC: carbamoyl-phosphate gives citrulline one carbon; it is that
        # substrate's main destination, so the pair is kept.
        raw = [["ornithine", "citrulline", 5], ["carbamoyl-phosphate", "citrulline", 1]]
        self.assertEqual(len(clean_pairs(raw)), 2)


class CarbonBalanceTest(unittest.TestCase):
    def test_stoichiometry_counts(self):
        # alpha-mannosidase (MAR00055) releases two mannoses: balanced only if
        # the coefficient is applied (26 = 14 + 2 x 6)
        from types import SimpleNamespace as NS
        from metaboenrich.atoms import carbon_balance, carbon_unbalanced
        sp = {"tri": NS(formula="C26H45NO21"), "w": NS(formula="H2O"), "di": NS(formula="C14H25NO11"),
              "man": NS(formula="C6H12O6"), "acyl": NS(formula="C52H92N3O27RCO")}
        model = NS(species=sp, reactions={
            "ok": NS(substrates={"tri": 1, "w": 2}, products={"di": 1, "man": 2}),
            "off": NS(substrates={"tri": 1}, products={"di": 1, "man": 1}),
            "rco": NS(substrates={"acyl": 1}, products={})})
        self.assertEqual(carbon_balance(model)["ok"], (26, 26))
        self.assertEqual(carbon_balance(model)["rco"], (53, 0))   # the C of 'RCO' is carbon too
        self.assertEqual(carbon_unbalanced(model), {"off", "rco"})


class CarbonChannelGraphTest(unittest.TestCase):
    def test_reaction_split_into_channels(self):
        model, ids = toy_model()
        rules, _ = resolve_currency(model, "role")
        # hexokinase: glucose -> G6P is the only carbon pair once ATP/ADP are handled
        pairs = {"HK": [(ids["glucose"], ids["glucose-6-phosphate"]), (ids["ATP"], ids["ADP"])]}
        g = build_graph(model, rules, atom_pairs=pairs)
        hk = [nid for nid in g.node_ids if nid.startswith("HK")]
        self.assertEqual(hk, ["HK"])                      # ATP/ADP pair dropped by the currency rules
        names = {g.node_names[m] for m, r in g.edges if r == g.index["HK"]}
        self.assertEqual(names, {"glucose", "glucose-6-phosphate"})
        self.assertIn("ADSL", g.unmapped_reactions)       # no pairs given: kept whole


if __name__ == "__main__":
    unittest.main()
